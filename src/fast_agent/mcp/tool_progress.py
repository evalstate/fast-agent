"""Tool progress reporting for MCP server clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import uuid4

from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler

if TYPE_CHECKING:
    from mcp.types import ContentBlock
else:
    ContentBlock = Any


class MCPToolProgressManager(ToolExecutionHandler):
    """Forward tool execution progress to MCP `notifications/progress`."""

    def __init__(
        self,
        reporter: Callable[[float, float | None, str | None], Awaitable[None]],
    ) -> None:
        self._reporter = reporter
        self._tool_labels: dict[str, str] = {}
        self._tool_use_map: dict[str, str] = {}
        self._tool_use_by_call: dict[str, set[str]] = {}

    def _format_message(
        self, tool_call_id: str | None, message: str | None = None, status: str | None = None
    ) -> str | None:
        label = self._tool_labels.get(tool_call_id or "")
        if message and label:
            return f"{label}: {message}"
        if message:
            return message
        if status and label:
            return f"{label}: {status}"
        return label or status

    async def _report_progress(
        self,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        try:
            await self._reporter(progress, total, message)
        except Exception:
            pass

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None = None,
    ) -> str:
        tool_call_id = tool_use_id or str(uuid4())
        self._tool_labels[tool_call_id] = f"{server_name}/{tool_name}"
        if tool_use_id:
            existing_call_id = self._tool_use_map.get(tool_use_id)
            if existing_call_id and existing_call_id != tool_call_id:
                self._tool_use_by_call.get(existing_call_id, set()).discard(tool_use_id)
            self._tool_use_map[tool_use_id] = tool_call_id
            self._tool_use_by_call.setdefault(tool_call_id, set()).add(tool_use_id)

        await self._report_progress(
            0.0,
            None,
            self._format_message(tool_call_id, status="started"),
        )
        return tool_call_id

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        await self._report_progress(
            progress,
            total,
            self._format_message(tool_call_id, message=message),
        )

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[ContentBlock] | None,
        error: str | None,
    ) -> None:
        status = "completed" if success else "failed"
        detail = error if error else status
        await self._report_progress(
            1.0,
            1.0,
            self._format_message(tool_call_id, message=detail),
        )
        self._tool_labels.pop(tool_call_id, None)
        tool_use_ids = self._tool_use_by_call.pop(tool_call_id, set())
        for tool_use_id in tool_use_ids:
            self._tool_use_map.pop(tool_use_id, None)

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        tool_call_id = tool_use_id or f"{server_name}/{tool_name}"
        message = error or "permission denied"
        await self._report_progress(
            0.0,
            None,
            self._format_message(tool_call_id, message=message),
        )

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        return self._tool_use_map.get(tool_use_id)

    async def ensure_tool_call_exists(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
    ) -> str:
        existing_id = self._tool_use_map.get(tool_use_id)
        if existing_id:
            return existing_id
        return await self.on_tool_start(
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            tool_use_id=tool_use_id,
        )
