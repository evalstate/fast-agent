"""Tests for tool summary suffix classification."""

from __future__ import annotations

from mcp.types import Tool

from fast_agent.commands.tool_summaries import build_tool_summaries


def _tool(
    name: str,
    *,
    meta: dict | None = None,
    description: str = "",
    input_schema: dict | None = None,
):
    return Tool(
        name=name,
        title=None,
        description=description,
        meta=meta or {},
        inputSchema=input_schema or {},
    )


class _AgentStub:
    def __init__(
        self,
        *,
        card_tool_names=(),
        smart_tool_names=(),
        agent_backed_tools: dict[str, object] | None = None,
    ) -> None:
        self._card_tool_names = set(card_tool_names)
        self._smart_tool_names = set(smart_tool_names)
        self._agent_backed_tools = agent_backed_tools or {}

    @property
    def card_tool_names(self) -> set[str]:
        return self._card_tool_names

    @property
    def smart_tool_names(self) -> set[str]:
        return self._smart_tool_names

    @smart_tool_names.setter
    def smart_tool_names(self, value) -> None:
        self._smart_tool_names = set(value)

    @property
    def parallel_smart_tool_calls(self) -> bool:
        return False

    @parallel_smart_tool_calls.setter
    def parallel_smart_tool_calls(self, value: bool) -> None:
        del value

    @property
    def agent_backed_tools(self) -> dict[str, object]:
        return self._agent_backed_tools


def test_build_tool_summaries_marks_smart_tools() -> None:
    agent = _AgentStub(smart_tool_names={"smart", "smart_with_resource"})

    summaries = build_tool_summaries(agent, [_tool("smart"), _tool("smart_with_resource")])

    assert summaries[0].suffix == "(Smart)"
    assert summaries[1].suffix == "(Smart)"


def test_build_tool_summaries_preserves_non_smart_suffixes() -> None:
    agent = _AgentStub(smart_tool_names={"smart"})

    summaries = build_tool_summaries(agent, [_tool("demo__search")])

    assert summaries[0].suffix == "(MCP)"


def test_build_tool_summaries_marks_smart_skybridge_tools() -> None:
    agent = _AgentStub(smart_tool_names={"smart_with_resource"})

    summaries = build_tool_summaries(
        agent,
        [_tool("smart_with_resource", meta={"openai/skybridgeEnabled": True})],
    )

    assert summaries[0].suffix == "(Smart) (skybridge)"
