import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.state  import app_state
from routers     import translate, users, history, phrasebook, models, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("s2s.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting S2S Translator API...")

    os.makedirs(settings.AUDIO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)

    await app_state.connect_db(settings.DATABASE_URL)
    logger.info("✓ Database connected")

    yield

    logger.info("Shutting down...")
    await app_state.close_db()
    logger.info("✓ Clean shutdown complete")


app = FastAPI(
    title       = "S2S Translator API",
    description = "Speech-to-Speech Translation: English → French / Twi",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(health.router,     prefix="/health",     tags=["Health"])
app.include_router(translate.router,  prefix="/translate",  tags=["Translation"])
app.include_router(users.router,      prefix="/users",      tags=["Users"])
app.include_router(history.router,    prefix="/history",    tags=["History"])
app.include_router(phrasebook.router, prefix="/phrasebook", tags=["Phrasebook"])
app.include_router(models.router,     prefix="/models",     tags=["Models"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )