from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.workflow.agents_as_tools_agent import AgentsAsToolsAgent
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.plugins.manifest import load_plugin_manifest
from fast_agent.plugins.models import PluginToolCall, PluginToolCallReviewSpec
from fast_agent.plugins.tool_call_review import (
    load_plugin_tool_call_reviewers,
    review_plugin_tool_calls,
)
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.plugins.tool_call_review import LoadedPluginToolCallReview


def _manifest(plugin_dir: Path, schema_version: int) -> None:
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        f"schema_version: {schema_version}\n"
        "name: guard\n"
        "hooks:\n"
        "  tool_call_review: ./guard.py:review\n",
        encoding="utf-8",
    )


def test_manifest_v3_declares_tool_call_review(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "guard"
    _manifest(plugin_dir, 3)

    manifest = load_plugin_manifest(plugin_dir)

    assert manifest.tool_call_review is not None
    assert manifest.tool_call_review.plugin_name == "guard"
    assert manifest.tool_call_review.handler == f"{(plugin_dir / 'guard.py').as_posix()}:review"


def test_manifest_v2_rejects_tool_call_review(tmp_path: Path) -> None:
    _manifest(tmp_path / "guard", 2)

    with pytest.raises(AgentConfigError, match="schema_version: 3"):
        load_plugin_manifest(tmp_path / "guard")


def _reviewers(tmp_path: Path, source: str, *names: str) -> tuple[LoadedPluginToolCallReview, ...]:
    module = tmp_path / "review.py"
    module.write_text(source, encoding="utf-8")
    return load_plugin_tool_call_reviewers(
        tuple(PluginToolCallReviewSpec(name, f"{module}:{name}") for name in names)
    )


async def _review(reviewers, calls: list[PluginToolCall], config=None) -> dict[str, str]:
    return await review_plugin_tool_calls(
        reviewers,
        agent_name="agent",
        model=None,
        instruction="",
        tools=(),
        history=(),
        request=PromptMessageExtended(role="assistant", content=[]),
        tool_calls=calls,
        config=config or {},
    )


CALLS = [PluginToolCall("a", "run", {"cmd": "ls"}), PluginToolCall("b", "run", {"cmd": "rm"})]


@pytest.mark.asyncio
async def test_reviewers_chain_and_only_see_calls_still_running(tmp_path: Path) -> None:
    reviewers = _reviewers(
        tmp_path,
        "def first(ctx):\n"
        "    return {c.id: ctx.config['why'] for c in ctx.tool_calls if c.arguments['cmd'] == 'rm'}\n"
        "async def second(ctx):\n"
        "    ctx.config['seen'].extend(c.id for c in ctx.tool_calls)\n"
        "    return None\n",
        "first",
        "second",
    )
    seen: list[str] = []

    denials = await _review(reviewers, CALLS, {"first": {"why": "no rm"}, "second": {"seen": seen}})

    assert denials == {"b": "no rm"}
    assert seen == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def deny(ctx):\n    raise RuntimeError('boom')\n", "failed: boom"),
        ("def deny(ctx):\n    return ['a']\n", "returned list"),
    ],
)
async def test_failing_reviewer_denies_every_pending_call(
    tmp_path: Path, source: str, expected: str
) -> None:
    denials = await _review(_reviewers(tmp_path, source, "deny"), CALLS)

    assert set(denials) == {"a", "b"}
    assert all(expected in message for message in denials.values())


@pytest.mark.asyncio
async def test_reviewer_that_fails_to_load_denies_calls(tmp_path: Path) -> None:
    reviewers = load_plugin_tool_call_reviewers(
        (PluginToolCallReviewSpec("missing", f"{tmp_path / 'nope.py'}:review"),)
    )

    denials = await _review(reviewers, CALLS)

    assert set(denials) == {"a", "b"}
    assert "failed to load" in denials["a"]


@pytest.mark.asyncio
async def test_agent_skips_denied_calls_and_returns_their_message(tmp_path: Path) -> None:
    ran: list[str] = []

    def shell(cmd: str) -> str:
        ran.append(cmd)
        return f"ran {cmd}"

    module = tmp_path / "review.py"
    module.write_text(
        "def review(ctx):\n"
        "    tools = sorted(t.name for t in ctx.tools if t.name == 'shell')\n"
        "    why = f'{ctx.agent_name} {ctx.instruction} {tools} {len(ctx.tool_calls)}'\n"
        "    return {c.id: why for c in ctx.tool_calls if c.arguments['cmd'] == 'rm'}\n",
        encoding="utf-8",
    )
    settings = Settings(ptc={"enabled": False})
    settings._plugin_tool_call_review = (PluginToolCallReviewSpec("guard", f"{module}:review"),)
    agent = McpAgent(
        config=AgentConfig(name="agent", instruction="be careful", servers=[]),
        connection_persistence=False,
        context=Context(config=settings),
        tools=[shell],
    )
    request = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="running")],
        tool_calls={
            "a": CallToolRequest(
                params=CallToolRequestParams(name="shell", arguments={"cmd": "ls"})
            ),
            "b": CallToolRequest(
                params=CallToolRequestParams(name="shell", arguments={"cmd": "rm"})
            ),
        },
    )

    result = await agent.run_tools(request)

    assert ran == ["ls"]
    assert result.tool_results is not None
    assert result.tool_results["b"].is_error
    assert get_text(result.tool_results["b"].content[0]) == "agent be careful ['shell'] 2"
    assert not result.tool_results["a"].is_error
    await agent._aggregator.close()


class _ChildAgent(LlmAgent):
    def __init__(self, name: str) -> None:
        super().__init__(AgentConfig(name))
        self.runs = 0

    async def generate(self, messages, request_params=None, tools=None) -> PromptMessageExtended:
        self.runs += 1
        return PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="ok")]
        )

    async def spawn_detached_instance(self, name: str | None = None) -> _ChildAgent:
        return self


def _call(name: str, **arguments: str) -> CallToolRequest:
    return CallToolRequest(params=CallToolRequestParams(name=name, arguments=arguments))


@pytest.mark.asyncio
async def test_agents_as_tools_reviews_child_and_local_calls_as_one_batch(
    tmp_path: Path,
) -> None:
    ran: list[str] = []

    def shell(cmd: str) -> str:
        ran.append(cmd)
        return f"ran {cmd}"

    module = tmp_path / "review.py"
    module.write_text(
        "def review(ctx):\n"
        "    names = ','.join(c.name for c in ctx.tool_calls)\n"
        "    blocked = {'agent__danger', 'shell'}\n"
        "    return {\n"
        "        c.id: f'{names} of {len(ctx.request.tool_calls)}'\n"
        "        for c in ctx.tool_calls\n"
        "        if c.name in blocked and c.arguments.get('cmd', 'rm') == 'rm'\n"
        "    }\n",
        encoding="utf-8",
    )
    settings = Settings(ptc={"enabled": False})
    settings._plugin_tool_call_review = (PluginToolCallReviewSpec("guard", f"{module}:review"),)
    safe, danger = _ChildAgent("safe"), _ChildAgent("danger")
    agent = AgentsAsToolsAgent(
        AgentConfig(name="parent", servers=[]),
        [safe, danger],
        context=Context(config=settings),
        connection_persistence=False,
        tools=[shell],
    )
    request = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={
            "1": _call("SHELL", cmd="rm"),
            "2": _call("agent__safe", message="hi"),
            "3": _call("danger", message="hi"),
            "4": _call("shell", cmd="ls"),
        },
    )

    result = await agent.run_tools(request)

    assert result.tool_results is not None
    assert list(result.tool_results) == ["1", "2", "3", "4"]
    reason = "shell,agent__safe,agent__danger,shell of 4"
    assert [get_text(result.tool_results[cid].content[0]) for cid in ("1", "3")] == [reason] * 2
    assert result.tool_results["1"].is_error and result.tool_results["3"].is_error
    assert (safe.runs, danger.runs, ran) == (1, 0, ["ls"])
    await agent._aggregator.close()
