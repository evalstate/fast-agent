"""Progress reporting for tool-runner loops."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler


class ToolLoopProgressEmitter:
    def __init__(self, handler: ToolExecutionHandler, agent_name: str) -> None:
        self._handler = handler
        self._agent_name = agent_name
        self._tool_call_id: str | None = None
        self._step = 0
        self._finished = False
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> str | None:
        if self._tool_call_id:
            return self._tool_call_id
        try:
            self._tool_call_id = await self._handler.on_tool_start(
                "agent_loop", self._agent_name, None
            )
        except Exception:
            self._tool_call_id = None
        return self._tool_call_id

    async def step(self, label: str) -> None:
        async with self._lock:
            if self._finished:
                return
            self._step += 1
            tool_call_id = await self._ensure_started()
            if not tool_call_id:
                return
            message = f"step {self._step}"
            if label:
                message = f"{message} ({label})"
            with suppress(Exception):
                await self._handler.on_tool_progress(tool_call_id, float(self._step), None, message)

    async def finish(self, success: bool, error: str | None = None) -> None:
        async with self._lock:
            if self._finished:
                return
            self._finished = True
            if not self._tool_call_id:
                return
            with suppress(Exception):
                await self._handler.on_tool_complete(self._tool_call_id, success, None, error)
