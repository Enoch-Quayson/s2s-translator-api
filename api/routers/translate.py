"""
routers/translate.py — translation endpoints

Routes:
  POST /translate/audio   Upload English audio → translated audio + text
  POST /translate/text    Send English text → translated text + audio
  POST /translate/transcribe  ASR only: audio → English text (no MT/TTS)
"""

import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing   import Optional

from fastapi  import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from core.config  import settings
from core.state   import app_state
from core.auth    import get_current_user, get_optional_user
from schemas.models import (
    TranslateTextRequest,
    TranslationResponse,
    TranscribeResponse,
)

logger = APIRouter.__class__  # just for type hints
router = APIRouter()
log    = logging.getLogger("s2s.translate")

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/ogg", "audio/flac", "audio/aac",
    "audio/x-m4a", "audio/webm",
}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".webm", ".mp4"}


# ── helpers ───────────────────────────────────────────────────────────
async def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to disk and return its local path."""
    ext = Path(upload.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. "
                                 f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    filename = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(settings.AUDIO_UPLOAD_DIR, filename)
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)

    content = await upload.read()
    max_bytes = settings.MAX_AUDIO_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, f"File too large. Max {settings.MAX_AUDIO_MB}MB.")

    with open(save_path, "wb") as f:
        f.write(content)
    return save_path


async def _run_pipeline(pipeline, fn_name: str, **kwargs) -> object:
    """Run synchronous pipeline method in a thread pool."""
    loop = asyncio.get_event_loop()
    fn   = getattr(pipeline, fn_name)
    return await loop.run_in_executor(None, lambda: fn(**kwargs))


async def _save_session(user_id: Optional[str], result, input_mode: str) -> uuid.UUID:
    """Persist translation session + history row. Returns session_id."""
    if not user_id:
        return uuid.uuid4()  # anonymous — no DB write

    session_id = uuid.uuid4()
    status     = "completed" if result.success else "failed"

    # Resolve model IDs from registry
    asr_row = await app_state.db_pool.fetchrow(
        "SELECT id FROM asr_models WHERE language_code='en' AND is_active ORDER BY id LIMIT 1"
    )
    mt_row = await app_state.db_pool.fetchrow(
        "SELECT id FROM mt_models WHERE source_language='en' AND target_language=$1 AND is_active LIMIT 1",
        result.target_language,
    )
    tts_row = await app_state.db_pool.fetchrow(
        "SELECT id FROM tts_models WHERE language_code=$1 AND is_active LIMIT 1",
        result.target_language,
    )

    uid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

    async with app_state.db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                INSERT INTO translation_sessions (
                    id, user_id,
                    asr_model_id, mt_model_id, tts_model_id,
                    source_language_code, target_language_code,
                    input_mode, source_text,
                    asr_transcript, asr_confidence, asr_duration_ms,
                    translated_text, mt_duration_ms,
                    translated_audio_url, tts_duration_ms,
                    total_duration_ms, status, error_message
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
            """,
                session_id, uid,
                asr_row["id"] if asr_row else None,
                mt_row["id"]  if mt_row  else None,
                tts_row["id"] if tts_row else None,
                result.source_language, result.target_language,
                input_mode, result.asr_transcript,
                result.asr_transcript, result.asr_confidence, result.asr_duration_ms,
                result.translated_text, result.mt_duration_ms,
                result.audio_path, result.tts_duration_ms,
                result.total_duration_ms, status, result.error,
            )

            await conn.execute(
                "INSERT INTO translation_history (user_id, session_id) VALUES ($1,$2)",
                uid, session_id,
            )

            # Upsert stats
            await conn.execute("""
                INSERT INTO user_stats (user_id, total_translations, last_active_at)
                VALUES ($1, 1, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    total_translations = user_stats.total_translations + 1,
                    last_active_at = NOW(), updated_at = NOW()
            """, uid)

    return session_id


def _audio_url(audio_path: Optional[str]) -> Optional[str]:
    """Convert local file path to public URL path."""
    if not audio_path:
        return None
    return f"/audio/{Path(audio_path).name}"


# ══════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@router.post(
    "/audio",
    response_model=TranslationResponse,
    summary="Translate audio file (full pipeline: ASR → MT → TTS)",
    description="Upload an English audio file. Returns translated text + audio in French or Twi.",
)
async def translate_audio(
    file:            UploadFile = File(..., description="English audio file (.wav, .mp3, .ogg, etc.)"),
    target_language: str        = Form("fr", description="Target language: fr or tw"),
    generate_audio:  bool       = Form(True, description="Also generate TTS audio output"),
    current_user:    Optional[dict] = Depends(get_optional_user),
):
    if target_language not in ("fr", "tw"):
        raise HTTPException(400, "target_language must be 'fr' or 'tw'")

    # Save upload
    audio_path = await _save_upload(file)

    # Get pipeline
    pipeline = await app_state.get_pipeline(target_language)

    # Run ASR → MT → TTS in thread pool
    result = await _run_pipeline(
        pipeline, "translate_audio",
        audio_path  = audio_path,
        output_dir  = settings.AUDIO_OUTPUT_DIR,
    )

    if not result.success:
        raise HTTPException(500, f"Pipeline error: {result.error}")

    # Persist to DB
    user_id    = str(current_user["id"]) if current_user else None
    session_id = await _save_session(user_id, result, input_mode="speech")

    return TranslationResponse(
        session_id        = session_id,
        source_language   = result.source_language,
        target_language   = result.target_language,
        asr_transcript    = result.asr_transcript,
        asr_confidence    = result.asr_confidence,
        translated_text   = result.translated_text,
        audio_url         = _audio_url(result.audio_path) if generate_audio else None,
        asr_duration_ms   = result.asr_duration_ms,
        mt_duration_ms    = result.mt_duration_ms,
        tts_duration_ms   = result.tts_duration_ms,
        total_duration_ms = result.total_duration_ms,
        status            = "completed",
    )


@router.post(
    "/text",
    response_model=TranslationResponse,
    summary="Translate English text (MT → TTS)",
    description="Send English text. Returns translated text + optional audio.",
)
async def translate_text(
    request:      TranslateTextRequest,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    pipeline = await app_state.get_pipeline(request.target_language)

    kwargs = dict(
        text       = request.text,
        output_dir = settings.AUDIO_OUTPUT_DIR,
    )
    # If audio not requested, we still call translate_text (it always runs TTS)
    result = await _run_pipeline(pipeline, "translate_text", **kwargs)

    if not result.success:
        raise HTTPException(500, f"Pipeline error: {result.error}")

    user_id    = str(current_user["id"]) if current_user else None
    session_id = await _save_session(user_id, result, input_mode="text")

    return TranslationResponse(
        session_id        = session_id,
        source_language   = result.source_language,
        target_language   = result.target_language,
        asr_transcript    = result.asr_transcript,
        asr_confidence    = result.asr_confidence,
        translated_text   = result.translated_text,
        audio_url         = _audio_url(result.audio_path) if request.generate_audio else None,
        asr_duration_ms   = result.asr_duration_ms,
        mt_duration_ms    = result.mt_duration_ms,
        tts_duration_ms   = result.tts_duration_ms,
        total_duration_ms = result.total_duration_ms,
        status            = "completed",
    )


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="ASR only — transcribe English audio to text",
    description="Upload audio, get back English transcript only. No translation.",
)
async def transcribe_audio(
    file: UploadFile = File(..., description="English audio file"),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    audio_path = await _save_upload(file)

    # Load ASR module directly
    pipeline = await app_state.get_pipeline("fr")  # any pipeline — we only use ASR part
    asr_module = pipeline._get_asr()

    import time
    start = time.time()
    loop  = asyncio.get_event_loop()
    transcript, confidence = await loop.run_in_executor(
        None, asr_module.transcribe, audio_path
    )
    duration_ms = int((time.time() - start) * 1000)

    return TranscribeResponse(
        transcript  = transcript,
        confidence  = confidence,
        duration_ms = duration_ms,
    )


@router.get(
    "/audio/{filename}",
    summary="Download / stream a translated audio file",
)
async def get_audio(filename: str):
    path = os.path.join(settings.AUDIO_OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)
