"""
core/state.py — global app state: DB pool + pipeline singletons
"""

import asyncio
import logging
from typing import Optional, Dict
import asyncpg

logger = logging.getLogger("s2s.state")


class AppState:
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self._pipelines: Dict[str, object] = {}
        self._pipeline_locks: Dict[str, asyncio.Lock] = {
            "fr": asyncio.Lock(),
            "tw": asyncio.Lock(),
        }

    async def connect_db(self, dsn: str):
        self.db_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=2,
            max_size=10,
            command_timeout=60,
            ssl="require",
        )
        logger.info("Database connected")

    async def close_db(self):
        if self.db_pool:
            await self.db_pool.close()

    async def get_pipeline(self, target_language: str):
        if target_language in self._pipelines:
            return self._pipelines[target_language]

        async with self._pipeline_locks[target_language]:
            if target_language in self._pipelines:
                return self._pipelines[target_language]

            pipeline = await asyncio.get_event_loop().run_in_executor(
                None, self._load_pipeline, target_language
            )

            self._pipelines[target_language] = pipeline
            return pipeline

    def _load_pipeline(self, target_language: str):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

        from pipeline.pipeline import S2SPipeline

        return S2SPipeline(
            target_language=target_language,
            lazy_load=True,
        )

    async def warmup_pipelines(self):
        await asyncio.gather(
            self.get_pipeline("fr"),
            self.get_pipeline("tw"),
        )


app_state = AppState()