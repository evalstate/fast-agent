"""Tests for tool summary suffix classification."""

from __future__ import annotations

from types import SimpleNamespace

from fast_agent.commands.tool_summaries import build_tool_summaries


def _tool(
    name: str,
    *,
    meta: dict | None = None,
    description: str = "",
    input_schema: dict | None = None,
):
    return SimpleNamespace(
        name=name,
        title=None,
        description=description,
        meta=meta or {},
        inputSchema=input_schema,
    )


def _agent_stub(**overrides):
    base = {
        "_card_tool_names": set(),
        "_smart_tool_names": set(),
        "_agent_tools": {},
        "_child_agents": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_tool_summaries_marks_smart_tools() -> None:
    agent = _agent_stub(_smart_tool_names={"smart", "smart_with_resource"})

    summaries = build_tool_summaries(agent, [_tool("smart"), _tool("smart_with_resource")])

    assert summaries[0].suffix == "(Smart)"
    assert summaries[1].suffix == "(Smart)"


def test_build_tool_summaries_preserves_non_smart_suffixes() -> None:
    agent = _agent_stub(_smart_tool_names={"smart"})

    summaries = build_tool_summaries(agent, [_tool("demo__search")])

    assert summaries[0].suffix == "(MCP)"


def test_build_tool_summaries_marks_smart_skybridge_tools() -> None:
    agent = _agent_stub(_smart_tool_names={"smart_with_resource"})

    summaries = build_tool_summaries(
        agent,
        [_tool("smart_with_resource", meta={"openai/skybridgeEnabled": True})],
    )

    assert summaries[0].suffix == "(Smart) (skybridge)"

