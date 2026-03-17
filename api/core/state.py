"""
core/state.py — global app state: DB pool + pipeline singletons
"""
import asyncio
import logging
import re
from urllib.parse import unquote
from typing import Optional, Dict
import asyncpg
logger = logging.getLogger("s2s.state")
class AppState:
    """
    Single shared instance holding:
      - asyncpg connection pool
      - S2SPipeline instances (one per target language, loaded lazily)
    """
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self._pipelines: Dict[str, object] = {}    # {"fr": S2SPipeline, "tw": S2SPipeline}
        self._pipeline_locks: Dict[str, asyncio.Lock] = {
            "fr": asyncio.Lock(),
            "tw": asyncio.Lock(),
        }
    # ── Database ─────────────────────────────────────────────────────
    async def connect_db(self, dsn: str):
        pattern = r"postgresql://([^:]+):(.+)@([^:@/]+):(\d+)/(.+)"
        match = re.match(pattern, dsn)
        if match:
            self.db_pool = await asyncpg.create_pool(
                host=match.group(3),
                port=int(match.group(4)),
                user=match.group(1),
                password=unquote(match.group(2)),
                database=match.group(5),
                min_size=2,
                max_size=10,
                command_timeout=60,
                ssl="require",
            )
    async def close_db(self):
        if self.db_pool:
            await self.db_pool.close()
    # ── Pipelines ────────────────────────────────────────────────────
    async def get_pipeline(self, target_language: str):
        """Return (or lazily create) a pipeline for the given target language."""
        # Fast path: already loaded
        if target_language in self._pipelines:
            return self._pipelines[target_language]
        # Slow path: load in a thread so we don't block the event loop
        async with self._pipeline_locks[target_language]:
            # Double-check after acquiring lock
            if target_language in self._pipelines:
                return self._pipelines[target_language]
            logger.info(f"Loading pipeline for target={target_language}...")
            pipeline = await asyncio.get_event_loop().run_in_executor(
                None, self._load_pipeline, target_language
            )
            self._pipelines[target_language] = pipeline
            logger.info(f"✓ Pipeline ready for target={target_language}")
            return pipeline
    def _load_pipeline(self, target_language: str):
        """Synchronous pipeline loader (runs in thread pool)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
        from pipeline.pipeline import S2SPipeline
        from core.config import settings
        return S2SPipeline(
            target_language=target_language,
            lazy_load=True,
        )
    async def warmup_pipelines(self):
        """Eagerly load both pipelines at startup."""
        await asyncio.gather(
            self.get_pipeline("fr"),
            self.get_pipeline("tw"),
        )
# ── singleton ─────────────────────────────────────────────────────────
app_state = AppState()