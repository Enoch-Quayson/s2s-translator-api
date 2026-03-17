"""
db.py
-----
Database helper for the S2S Translator.
Saves translation sessions and marks models as downloaded.

Usage:
    from db import TranslationDB
    db = TranslationDB("postgresql://user:pass@localhost/s2s")
    db.log_session(user_id, result, asr_model_id=1, mt_model_id=1, tts_model_id=1)
"""

import os
import uuid
from typing import Optional
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class TranslationDB:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not set. Pass db_url or set DATABASE_URL env var.")
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")

    def _conn(self):
        return psycopg2.connect(self.db_url)

    # ── Model registry ─────────────────────────────────────────────
    def mark_model_downloaded(self, hf_model_id: str, model_type: str, local_path: str):
        """Update is_downloaded=TRUE and local_path after successful download."""
        table = {"asr": "asr_models", "mt": "mt_models", "tts": "tts_models"}[model_type]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {table} SET is_downloaded=TRUE, local_path=%s WHERE hf_model_id=%s",
                    (local_path, hf_model_id)
                )

    def get_model_registry(self) -> list[dict]:
        """Return all models and their download status."""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM v_model_registry ORDER BY type, id")
                return [dict(r) for r in cur.fetchall()]

    # ── Translation sessions ─────────────────────────────────────────
    def log_session(
        self,
        user_id:             str,
        result,                            # TranslationResult from pipeline.py
        asr_model_id:        Optional[int] = None,
        mt_model_id:         Optional[int]  = None,
        tts_model_id:        Optional[int]  = None,
        input_mode:          str = "speech",
        source_audio_url:    Optional[str]  = None,
        translated_audio_url: Optional[str] = None,
    ) -> str:
        """
        Save a completed translation session to the database.
        Returns the new session UUID.
        """
        session_id = str(uuid.uuid4())
        status = "completed" if result.success else "failed"

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO translation_sessions (
                        id, user_id,
                        asr_model_id, mt_model_id, tts_model_id,
                        source_language_code, target_language_code,
                        input_mode,
                        source_text, source_audio_url,
                        asr_transcript, asr_confidence, asr_duration_ms,
                        translated_text, mt_duration_ms,
                        translated_audio_url, tts_duration_ms,
                        total_duration_ms, status, error_message
                    ) VALUES (
                        %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s, %s
                    )
                """, (
                    session_id, user_id,
                    asr_model_id, mt_model_id, tts_model_id,
                    result.source_language, result.target_language,
                    input_mode,
                    result.asr_transcript, source_audio_url,
                    result.asr_transcript, result.asr_confidence, result.asr_duration_ms,
                    result.translated_text, result.mt_duration_ms,
                    translated_audio_url or result.audio_path, result.tts_duration_ms,
                    result.total_duration_ms, status, result.error,
                ))

                # Also create history record
                cur.execute("""
                    INSERT INTO translation_history (user_id, session_id)
                    VALUES (%s, %s)
                """, (user_id, session_id))

                # Update user stats
                cur.execute("""
                    INSERT INTO user_stats (user_id, total_translations, last_active_at)
                    VALUES (%s, 1, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        total_translations = user_stats.total_translations + 1,
                        last_active_at = NOW(),
                        updated_at = NOW()
                """, (user_id,))

                # Log model performance
                if asr_model_id and result.asr_duration_ms:
                    cur.execute("""
                        INSERT INTO model_performance_log (session_id, model_type, model_id, duration_ms, success)
                        VALUES (%s, 'asr', %s, %s, %s)
                    """, (session_id, asr_model_id, result.asr_duration_ms, result.success))

                if mt_model_id and result.mt_duration_ms:
                    cur.execute("""
                        INSERT INTO model_performance_log (session_id, model_type, model_id, duration_ms, success)
                        VALUES (%s, 'mt', %s, %s, %s)
                    """, (session_id, mt_model_id, result.mt_duration_ms, result.success))

                if tts_model_id and result.tts_duration_ms:
                    cur.execute("""
                        INSERT INTO model_performance_log (session_id, model_type, model_id, duration_ms, success)
                        VALUES (%s, 'tts', %s, %s, %s)
                    """, (session_id, tts_model_id, result.tts_duration_ms, result.success))

        return session_id

    def get_user_history(self, user_id: str, limit: int = 50) -> list[dict]:
        """Fetch translation history feed for a user."""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM v_history_feed
                    WHERE user_id = %s
                    LIMIT %s
                """, (user_id, limit))
                return [dict(r) for r in cur.fetchall()]
