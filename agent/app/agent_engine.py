"""Google Agent Runtime (Vertex AI Agent Engine) adapter for the month-end agents (D-032).

Agent Engine pickles an instance of this class and calls its methods. Nothing is created at
construction time, so the instance pickles cleanly; tools, model and credentials are built per
query by ``app.main``, exactly as on AgentCore.
"""

from __future__ import annotations

import asyncio


class MonthEndAgent:
    def set_up(self) -> None:
        """Called once per container start; nothing to prepare."""

    async def async_query(self, month: str | None = None) -> dict:
        from app.main import invoke

        return await invoke({"month": month} if month else {})

    def query(self, month: str | None = None) -> dict:
        return asyncio.run(self.async_query(month))
