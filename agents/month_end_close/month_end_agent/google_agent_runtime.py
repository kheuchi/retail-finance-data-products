"""Google Agent Runtime (Vertex AI Agent Engine) adapter for the month-end agents (D-032).

Agent Engine pickles an instance of this class and calls its methods. Nothing is created at
construction time, so the instance pickles cleanly; tools, model and credentials are built per
query by ``month_end_agent.main``, exactly as on AgentCore.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


class MonthEndAgent:
    def set_up(self) -> None:
        """Called once per container start; nothing to prepare."""

    async def async_query(self, month: str | None = None) -> dict:
        from month_end_agent.main import invoke

        return await invoke({"month": month} if month else {})

    def query(self, month: str | None = None) -> dict:
        # Agent Runtime calls this from inside its own event loop: run the agent in a fresh
        # thread with its own loop (asyncio.run cannot nest).
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.async_query(month)).result()
