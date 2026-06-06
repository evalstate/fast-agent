from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import TextContent

from fast_agent.workflow_telemetry import ToolHandlerWorkflowTelemetry

if TYPE_CHECKING:
    from mcp.types import ContentBlock


class _RecordingToolHandler:
    def __init__(self) -> None:
        self.starts: list[tuple[str, str, dict[str, Any] | None]] = []
        self.progress: list[tuple[str, float, float | None, str | None]] = []
        self.completions: list[tuple[str, bool, list[ContentBlock] | None, str | None]] = []

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None = None,
    ) -> str:
        del tool_use_id
        self.starts.append((tool_name, server_name, arguments))
        return f"call-{len(self.starts)}"

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        self.progress.append((tool_call_id, progress, total, message))

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[ContentBlock] | None,
        error: str | None,
    ) -> None:
        self.completions.append((tool_call_id, success, content, error))

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        del tool_name, server_name, tool_use_id, error

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        del tool_use_id
        return None

    async def ensure_tool_call_exists(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
    ) -> str:
        del tool_name, server_name, arguments
        return tool_use_id


@pytest.mark.asyncio
async def test_tool_handler_workflow_step_emits_lifecycle_events() -> None:
    handler = _RecordingToolHandler()
    telemetry = ToolHandlerWorkflowTelemetry(handler, server_name="workflow-test")

    async with telemetry.start_step("route", arguments={"agent": "writer"}) as step:
        await step.update(message="delegating", progress=1, total=2)
        await step.finish(True, text="done")
        await step.finish(False, error="ignored")

    assert handler.starts == [("route", "workflow-test", {"agent": "writer"})]
    assert handler.progress == [("call-1", 1, 2, "delegating")]
    assert len(handler.completions) == 1
    tool_call_id, success, content, error = handler.completions[0]
    assert (tool_call_id, success, error) == ("call-1", True, None)
    assert content is not None
    assert isinstance(content[0], TextContent)
    assert content[0].type == "text"
    assert content[0].text == "done"


@pytest.mark.asyncio
async def test_tool_handler_workflow_step_auto_finishes_on_exception() -> None:
    handler = _RecordingToolHandler()
    telemetry = ToolHandlerWorkflowTelemetry(handler)

    with pytest.raises(RuntimeError, match="boom"):
        async with telemetry.start_step("parallel"):
            raise RuntimeError("boom")

    assert handler.completions == [("call-1", False, None, "boom")]
