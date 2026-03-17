"""
main.py — S2S Translator FastAPI Server
========================================
Startup order:
  1. Load environment / config
  2. Connect to database
  3. Warm-up ML pipelines (lazy by default, eager if WARM_MODELS=true)
  4. Mount routers
  5. Serve

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ── add project root to path so pipeline/ is importable ─────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config   import settings
from core.state    import app_state
from routers       import translate, users, history, phrasebook, models, health

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("s2s.main")


# ── lifespan (startup / shutdown) ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────────
    logger.info("🚀 Starting S2S Translator API...")

    # Create output dirs
    os.makedirs(settings.AUDIO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)

    # Connect to DB
    await app_state.connect_db(settings.DATABASE_URL)
    logger.info("✓ Database connected")

    # Warm up pipelines if requested
    if settings.WARM_MODELS:
        logger.info("Warming up ML pipelines (WARM_MODELS=true)...")
        await app_state.warmup_pipelines()
        logger.info("✓ Pipelines ready")
    else:
        logger.info("Models will load lazily on first request")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────────
    logger.info("Shutting down...")
    await app_state.close_db()
    logger.info("✓ Clean shutdown complete")


# ── app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "S2S Translator API",
    description = "Speech-to-Speech Translation: English → French / Twi\n\n"
                  "**Pipeline:** Whisper ASR → OPUS-MT → Meta MMS TTS",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── static files (serve translated audio) ────────────────────────────
app.mount("/audio", StaticFiles(directory=settings.AUDIO_OUTPUT_DIR, html=False), name="audio")

# ── routers ───────────────────────────────────────────────────────────
app.include_router(health.router,      prefix="/health",      tags=["Health"])
app.include_router(translate.router,   prefix="/translate",   tags=["Translation"])
app.include_router(users.router,       prefix="/users",       tags=["Users"])
app.include_router(history.router,     prefix="/history",     tags=["History"])
app.include_router(phrasebook.router,  prefix="/phrasebook",  tags=["Phrasebook"])
app.include_router(models.router,      prefix="/models",      tags=["Models"])


# ── global exception handler ──────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )