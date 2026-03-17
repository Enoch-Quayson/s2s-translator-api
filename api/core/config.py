"""
core/config.py — centralised settings loaded from environment / .env
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/s2s_translator"

    # ── Models ───────────────────────────────────────────────────────
    MODELS_DIR:   str  = "./models_cache"
    WARM_MODELS:  bool = False          # set True to load all models at startup

    # ── Audio ────────────────────────────────────────────────────────
    AUDIO_OUTPUT_DIR: str = "./audio_outputs"
    AUDIO_UPLOAD_DIR: str = "./audio_uploads"
    MAX_AUDIO_MB:     int = 25          # max upload size in MB

    # ── Auth ─────────────────────────────────────────────────────────
    SECRET_KEY:      str = "change-me-in-production"
    JWT_ALGORITHM:   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24   # 24 hours

    # ── CORS ─────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "*"]

    # ── Server ───────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
