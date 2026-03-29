"""routers/health.py — health & readiness endpoints"""

from fastapi import APIRouter
from pydantic import BaseModel
from core.state import app_state

router = APIRouter()

class HealthResponse(BaseModel):
    status:    str
    database:  str
    pipelines: dict

@router.get("", response_model=HealthResponse, summary="Health check")
async def health_check():
    db_ok = False
    try:
        await app_state.db_pool.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        pass

    pipelines = {
        lang: ("loaded" if lang in app_state._pipelines else "not_loaded")
        for lang in ("fr", "tw")
    }
    overall = "ok" if db_ok else "degraded"
    return HealthResponse(status=overall, database="ok" if db_ok else "error", pipelines=pipelines)

@router.get("/ping", summary="Simple ping")
async def ping():
    return {"pong": True}
