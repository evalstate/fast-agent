from __future__ import annotations

from mcp.types import Tool

from fast_agent.tools.tool_sources import FAST_AGENT_TOOL_SOURCE_META, set_tool_source, tool_source


def test_set_tool_source_adds_metadata() -> None:
    tool = set_tool_source(Tool(name="read_text_file", inputSchema={}), "shell")

    assert tool.meta == {FAST_AGENT_TOOL_SOURCE_META: "shell"}


def test_tool_source_reads_metadata() -> None:
    tool = Tool(
        name="read_text_file",
        inputSchema={},
        _meta={FAST_AGENT_TOOL_SOURCE_META: "acp_filesystem"},
    )

    assert tool_source(tool) == "acp_filesystem"


def test_tool_source_ignores_unknown_metadata_value() -> None:
    tool = Tool(
        name="read_text_file",
        inputSchema={},
        _meta={FAST_AGENT_TOOL_SOURCE_META: "unknown"},
    )

    assert tool_source(tool) is None
