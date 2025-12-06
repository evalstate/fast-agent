"""
Workflow telemetry helpers for emitting virtual tool progress.

This module provides a pluggable abstraction that workflows (router, parallel)
can use to announce delegation steps without knowing which transport consumes
the events. Transports that care about surfacing these events (e.g. ACP) can
install a telemetry implementation that forwards them to tool progress
notifications, while the default implementation is a no-op.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from mcp.types import ContentBlock, TextContent

if TYPE_CHECKING:
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler


# Plan entry types for ACP plan mode
PlanEntryStatus = Literal["pending", "in_progress", "completed"]
PlanEntryPriority = Literal["high", "medium", "low"]


@dataclass
class PlanEntry:
    """A task within a plan."""

    content: str
    priority: PlanEntryPriority
    status: PlanEntryStatus


class PlanTelemetryProvider(Protocol):
    """Provider capable of sending plan updates."""

    async def update_plan(self, entries: list[PlanEntry]) -> None:
        """Send a plan update with the current list of plan entries."""
        ...


class NoOpPlanTelemetryProvider:
    """Provider that does nothing with plan updates."""

    async def update_plan(self, entries: list[PlanEntry]) -> None:
        pass


class WorkflowStepHandle(Protocol):
    """Represents a virtual workflow step that can emit progress and completion."""

    async def update(
        self,
        *,
        message: str | None = None,
        progress: float | None = None,
        total: float | None = None,
    ) -> None:
        """Send an incremental update about this workflow step."""

    async def finish(
        self,
        success: bool,
        *,
        text: str | None = None,
        content: list[ContentBlock] | None = None,
        error: str | None = None,
    ) -> None:
        """Complete the workflow step with optional success text/content."""


class WorkflowTelemetry(AbstractAsyncContextManager, WorkflowStepHandle):
    """
    Base async context manager returned by telemetry providers.

    Implementations should override __aenter__/__aexit__ along with update/finish.
    """

    async def __aenter__(self) -> WorkflowStepHandle:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        # Default no-op exit (override in subclasses to auto-complete)
        return False

    async def update(  # pragma: no cover - default no-op
        self,
        *,
        message: str | None = None,
        progress: float | None = None,
        total: float | None = None,
    ) -> None:
        return None

    async def finish(  # pragma: no cover - default no-op
        self,
        success: bool,
        *,
        text: str | None = None,
        content: list[ContentBlock] | None = None,
        error: str | None = None,
    ) -> None:
        return None


class NullWorkflowTelemetry(WorkflowTelemetry):
    """No-op telemetry implementation used when no transport wants workflow updates."""

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class WorkflowTelemetryProvider(Protocol):
    """Provider capable of starting workflow steps."""

    def start_step(
        self,
        tool_name: str,
        *,
        server_name: str = "workflow",
        arguments: dict[str, Any] | None = None,
    ) -> WorkflowTelemetry: ...


class NoOpWorkflowTelemetryProvider:
    """Provider that always returns a no-op workflow step."""

    def start_step(
        self,
        tool_name: str,
        *,
        server_name: str = "workflow",
        arguments: dict[str, Any] | None = None,
    ) -> WorkflowTelemetry:
        return NullWorkflowTelemetry()


@dataclass
class _ToolHandlerWorkflowStep(WorkflowTelemetry):
    handler: ToolExecutionHandler
    tool_name: str
    server_name: str
    arguments: dict[str, Any] | None

    _tool_call_id: str | None = None
    _finished: bool = False
    _lock: asyncio.Lock = asyncio.Lock()

    async def __aenter__(self) -> WorkflowStepHandle:
        self._tool_call_id = await self.handler.on_tool_start(
            self.tool_name, self.server_name, self.arguments
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if not self._finished:
            success = exc_type is None
            error_message = str(exc) if exc else None
            await self.finish(success, error=error_message)
        return False

    async def update(
        self,
        *,
        message: str | None = None,
        progress: float | None = None,
        total: float | None = None,
    ) -> None:
        if not self._tool_call_id or (message is None and progress is None):
            return
        await self.handler.on_tool_progress(
            self._tool_call_id,
            progress if progress is not None else 0.0,
            total,
            message,
        )

    async def finish(
        self,
        success: bool,
        *,
        text: str | None = None,
        content: list[ContentBlock] | None = None,
        error: str | None = None,
    ) -> None:
        if self._finished or not self._tool_call_id:
            self._finished = True
            return

        final_content = content
        if final_content is None and text:
            final_content: list[ContentBlock] = [TextContent(type="text", text=text)]

        await self.handler.on_tool_complete(
            self._tool_call_id,
            success,
            final_content,
            error,
        )
        self._finished = True


class ToolHandlerWorkflowTelemetry(NoOpWorkflowTelemetryProvider):
    """
    Telemetry provider that forwards workflow steps to a ToolExecutionHandler.
    """

    def __init__(self, handler: ToolExecutionHandler, *, server_name: str = "workflow") -> None:
        self._handler = handler
        self._server_name = server_name

    def start_step(
        self,
        tool_name: str,
        *,
        server_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> WorkflowTelemetry:
        effective_server = server_name or self._server_name
        return _ToolHandlerWorkflowStep(
            handler=self._handler,
            tool_name=tool_name,
            server_name=effective_server,
            arguments=arguments,
        )


class ACPPlanTelemetryProvider:
    """
    Telemetry provider that sends plan updates via ACP session/update notifications.
    """

    def __init__(self, connection: Any, session_id: str) -> None:
        self._connection = connection
        self._session_id = session_id

    async def update_plan(self, entries: list[PlanEntry]) -> None:
        """Send a plan update with the current list of plan entries."""
        if not self._connection:
            return

        # Convert PlanEntry to dict format expected by ACP
        plan_entries = [
            {
                "content": entry.content,
                "priority": entry.priority,
                "status": entry.status,
            }
            for entry in entries
        ]

        plan_update = {
            "sessionUpdate": "plan",
            "entries": plan_entries,
        }

        await self._connection.session_update(session_id=self._session_id, update=plan_update)
