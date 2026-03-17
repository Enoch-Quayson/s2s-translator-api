"""
routers/models.py — model registry endpoints

Routes:
  GET  /models              All models (ASR, MT, TTS)
  GET  /models/asr          ASR models
  GET  /models/mt           MT models
  GET  /models/tts          TTS models
  POST /models/download     Trigger model download (admin)
"""

import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from core.state  import app_state
from core.auth   import get_current_user
from schemas.models import (
    ModelRegistryResponse,
    ASRModelResponse, MTModelResponse, TTSModelResponse,
    MessageResponse,
)

router = APIRouter()
log    = logging.getLogger("s2s.models")


@router.get("", response_model=ModelRegistryResponse, summary="All registered models")
async def get_all_models():
    asr_rows = await app_state.db_pool.fetch(
        "SELECT * FROM asr_models ORDER BY id"
    )
    mt_rows  = await app_state.db_pool.fetch(
        "SELECT * FROM mt_models ORDER BY id"
    )
    tts_rows = await app_state.db_pool.fetch(
        "SELECT * FROM tts_models ORDER BY id"
    )
    return ModelRegistryResponse(
        asr=[ASRModelResponse(**dict(r)) for r in asr_rows],
        mt =[MTModelResponse(
                **{k: v for k, v in dict(r).items()
                   if k not in ("source_language","target_language")},
                source_language=r["source_language"],
                target_language=r["target_language"],
             ) for r in mt_rows],
        tts=[TTSModelResponse(**dict(r)) for r in tts_rows],
    )


@router.get("/asr", response_model=list[ASRModelResponse], summary="ASR models (English)")
async def get_asr_models():
    rows = await app_state.db_pool.fetch("SELECT * FROM asr_models ORDER BY id")
    return [ASRModelResponse(**dict(r)) for r in rows]


@router.get("/mt", response_model=list[MTModelResponse],
            summary="MT models (EN→FR, EN→Twi)")
async def get_mt_models():
    rows = await app_state.db_pool.fetch("SELECT * FROM mt_models ORDER BY id")
    return [MTModelResponse(**dict(r)) for r in rows]


@router.get("/tts", response_model=list[TTSModelResponse],
            summary="TTS models (French, Twi)")
async def get_tts_models():
    rows = await app_state.db_pool.fetch("SELECT * FROM tts_models ORDER BY id")
    return [TTSModelResponse(**dict(r)) for r in rows]


# ── background download ───────────────────────────────────────────────
def _run_download(hf_model_id: str, model_type: str):
    """Download a single model in background."""
    import subprocess, sys
    subprocess.run([
        sys.executable,
        "scripts/download_models.py",
        "--model", hf_model_id,
    ], check=True)


@router.post("/download", response_model=MessageResponse,
             summary="Trigger model download (runs in background)")
async def trigger_download(
    hf_model_id:     str,
    background_tasks: BackgroundTasks,
    current_user:    dict = Depends(get_current_user),
):
    """Queue a model for background download. Admin use."""
    # check model exists in registry
    for table in ("asr_models", "mt_models", "tts_models"):
        row = await app_state.db_pool.fetchrow(
            f"SELECT id, model_type FROM {table} WHERE hf_model_id=$1", hf_model_id
        )
        if row:
            model_type = table.replace("_models", "")
            break
    else:
        raise HTTPException(404, f"Model {hf_model_id!r} not found in registry")

    background_tasks.add_task(_run_download, hf_model_id, model_type)
    return MessageResponse(message=f"Download queued for {hf_model_id}")
