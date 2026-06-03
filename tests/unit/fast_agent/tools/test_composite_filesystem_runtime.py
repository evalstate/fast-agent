from __future__ import annotations

from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.tools.composite_filesystem_runtime import CompositeFilesystemRuntime
from fast_agent.types import RequestParams


class _Runtime:
    def __init__(self, *tool_names: str) -> None:
        self.tools = [Tool(name=name, inputSchema={}) for name in tool_names]
        self.calls: list[tuple[str, dict[str, Any] | None, str | None, RequestParams | None]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult:
        self.calls.append((name, arguments, tool_use_id, request_params))
        return CallToolResult(content=[TextContent(type="text", text=name)], isError=False)

    def metadata(self) -> dict[str, Any]:
        return {"tools": [tool.name for tool in self.tools]}


@pytest.mark.asyncio
async def test_composite_runtime_routes_to_first_runtime_with_tool() -> None:
    primary = _Runtime("read_text_file")
    fallback = _Runtime("read_text_file", "apply_patch")
    runtime = CompositeFilesystemRuntime(primary=primary, fallback=fallback)
    request_params = RequestParams(use_history=False)

    result = await runtime.call_tool(
        "read_text_file",
        {"path": "sample.txt"},
        "tool-use-1",
        request_params=request_params,
    )

    assert result.isError is False
    assert primary.calls == [
        ("read_text_file", {"path": "sample.txt"}, "tool-use-1", request_params)
    ]
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_composite_runtime_routes_to_fallback_for_unique_tool() -> None:
    primary = _Runtime("read_text_file")
    fallback = _Runtime("apply_patch")
    runtime = CompositeFilesystemRuntime(primary=primary, fallback=fallback)

    result = await runtime.call_tool("apply_patch", {"input": "*** Begin Patch"})

    assert result.isError is False
    assert primary.calls == []
    assert fallback.calls == [("apply_patch", {"input": "*** Begin Patch"}, None, None)]


@pytest.mark.asyncio
async def test_composite_runtime_routes_attach_resource_alias_to_attach_media() -> None:
    primary = _Runtime("read_text_file")
    fallback = _Runtime("attach_media")
    runtime = CompositeFilesystemRuntime(primary=primary, fallback=fallback)

    result = await runtime.call_tool("attach_resource", {"path": "image.png"})

    assert result.isError is False
    assert primary.calls == []
    assert fallback.calls == [("attach_resource", {"path": "image.png"}, None, None)]


@pytest.mark.asyncio
async def test_composite_runtime_reports_unsupported_tool() -> None:
    runtime = CompositeFilesystemRuntime(
        primary=_Runtime("read_text_file"),
        fallback=_Runtime("apply_patch"),
    )

    result = await runtime.call_tool("missing")

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "Error: unsupported filesystem tool 'missing'"


def test_composite_runtime_tools_are_deduplicated_in_priority_order() -> None:
    runtime = CompositeFilesystemRuntime(
        primary=_Runtime("read_text_file", "write_text_file"),
        fallback=_Runtime("write_text_file", "apply_patch"),
    )

    assert [tool.name for tool in runtime.tools] == [
        "read_text_file",
        "write_text_file",
        "apply_patch",
    ]
