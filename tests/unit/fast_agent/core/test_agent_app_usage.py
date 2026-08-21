from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.core.agent_app import AgentApp
from fast_agent.integrations import herdr_lifecycle
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)
from fast_agent.plugins.models import PluginPostUserTurnSpec
from fast_agent.ui.turn_usage_display import (
    TurnUsageDisplay,
    format_regular_turn_usage,
    format_regular_turn_usage_with_subagents,
    format_turn_usage,
)

if TYPE_CHECKING:
    from pathlib import Path


def _turn(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: int = 0,
    cache_read: int | None = None,
    cache_write: int | None = None,
    provider: Provider = Provider.OPENAI,
    usage_schema: UsageSchema = UsageSchema.OPENAI_CHAT,
    model: str = "gpt-test",
) -> TurnUsage:
    return TurnUsage(
        provider=provider,
        usage_schema=usage_schema,
        model=model,
        prompt=PromptTokenUsage(
            total=prompt_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
        ),
        completion=CompletionTokenUsage(total=completion_tokens),
        tool_calls=tool_calls,
    )


def _agent(usage_accumulator: UsageAccumulator):
    return SimpleNamespace(name="assistant", usage_accumulator=usage_accumulator)


def test_regular_agent_usage_displays_last_turn_when_no_turn_start_index() -> None:
    usage = UsageAccumulator()
    usage.add_turn(_turn(prompt_tokens=100, completion_tokens=20, tool_calls=1))
    app = AgentApp({"assistant": _agent(usage)})

    display = app._collect_agent_turn_usage(app["assistant"], None)

    assert display is not None
    output = format_regular_turn_usage(display)
    assert "Last:[/dim] [blue]▶ 100[/blue] input" in output
    assert "[green]◀ 20[/green] output" in output
    assert "· 1 tool call" in output


def test_regular_agent_usage_displays_turn_delta_with_context_percentage() -> None:
    usage = UsageAccumulator()
    usage.add_turn(_turn(prompt_tokens=100, completion_tokens=20, tool_calls=1))
    usage.add_turn(_turn(prompt_tokens=50, completion_tokens=10, tool_calls=2))
    usage.set_context_window_size(200)
    app = AgentApp({"assistant": _agent(usage)})

    display = app._collect_agent_turn_usage(app["assistant"], 1)

    assert display is not None
    output = format_regular_turn_usage(display)
    assert "[blue]▶ 50[/blue] input" in output
    assert "[green]◀ 10[/green] output" in output
    assert "· 2 tool calls · context 30.0%" in output


def test_regular_agent_usage_breaks_out_subagents_used_during_turn() -> None:
    usage = UsageAccumulator()
    subagents = UsageAccumulator()
    agent = _agent(usage)
    agent.subagent_usage_accumulator = subagents
    app = AgentApp({"assistant": agent})
    start = app._capture_turn_start_indices("assistant")

    child_turn = _turn(
        prompt_tokens=300,
        completion_tokens=20,
        tool_calls=2,
        cache_read=270,
    )
    subagents.add_turn(child_turn)
    usage.add_turn(child_turn.model_copy(deep=True))
    usage.add_turn(_turn(prompt_tokens=100, completion_tokens=10, tool_calls=1, cache_read=50))

    total = app._collect_agent_turn_usage(agent, start["assistant"])
    delegated = app._collect_subagent_turn_usage(
        agent,
        start["assistant::subagents"],
    )

    assert total is not None
    assert delegated is not None
    assert total.input_tokens == 400
    assert delegated.input_tokens == 300
    assert delegated.context_percentage is None
    lines = format_regular_turn_usage_with_subagents(total, delegated)
    assert lines[0].startswith("[dim]Last:[/dim]")
    assert "▶ 400" in lines[0]
    assert "└─ subagents:" in lines[1]
    assert "▶ 300" in lines[1]


def test_plugin_usage_uses_parent_total_without_double_counting_subagents() -> None:
    usage = UsageAccumulator()
    subagents = UsageAccumulator()
    agent = _agent(usage)
    agent.subagent_usage_accumulator = subagents
    app = AgentApp({"assistant": agent})
    start = app._capture_turn_start_indices("assistant")
    child = _turn(prompt_tokens=300, completion_tokens=20)
    subagents.add_turn(child)
    usage.add_turn(child.model_copy(deep=True))
    usage.add_turn(_turn(prompt_tokens=100, completion_tokens=10))

    turn, session = app._collect_plugin_usage(agent, start)

    assert len(turn) == 2
    assert len(session) == 2
    assert sum(item.prompt.total or 0 for item in turn) == 400

    completed = app.complete_user_turn("assistant", start)

    assert completed is not None
    assert len(completed.attempts) == 2
    assert len(completed.ledgers) == 1
    assert completed.ledgers[0].label == "subagents"
    assert len(completed.ledgers[0].attempts) == 1


def test_plugin_usage_collects_parallel_turn_and_session_attempts() -> None:
    children = []
    for name in ("first", "second", "fan-in"):
        usage = UsageAccumulator()
        usage.add_turn(_turn(prompt_tokens=10, completion_tokens=1))
        child = _agent(usage)
        child.name = name
        children.append(child)
    parallel = ParallelAgent(
        AgentConfig("parallel"),
        children[2],
        children[:2],
    )
    app = AgentApp({"parallel": parallel})
    start = app._capture_turn_start_indices("parallel")
    for child in children:
        child.usage_accumulator.add_turn(_turn(prompt_tokens=20, completion_tokens=2))

    turn, session = app._collect_plugin_usage(parallel, start)

    assert len(turn) == 3
    assert len(session) == 6
    assert sum(item.prompt.total or 0 for item in turn) == 60

    completed = app.complete_user_turn("parallel", start)

    assert completed is not None
    assert len(completed.attempts) == 3
    assert [ledger.label for ledger in completed.ledgers] == ["first", "second", "fan-in"]
    assert all(len(ledger.attempts) == 1 for ledger in completed.ledgers)


@pytest.mark.asyncio
async def test_interactive_send_runs_post_user_turn_plugin_once_and_quiet_send_skips_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "calls.txt"
    hook = tmp_path / "hook.py"
    hook.write_text(
        "from fast_agent.plugins import PluginPostUserTurnOutput\n"
        "\n"
        "def display(ctx):\n"
        f"    with open({marker.as_posix()!r}, 'a', encoding='utf-8') as stream:\n"
        "        stream.write(f'{len(ctx.turn_usage)}:{len(ctx.session_usage)}\\n')\n"
        "    return PluginPostUserTurnOutput(session_usage='$0.0123')\n",
        encoding="utf-8",
    )
    reported_usage: list[str] = []
    monkeypatch.setattr(
        herdr_lifecycle,
        "report_session_usage",
        reported_usage.append,
    )

    usage = UsageAccumulator()
    usage_agent = _agent(usage)

    app = AgentApp(
        {"assistant": usage_agent},
        plugin_post_user_turn=[
            PluginPostUserTurnSpec("marker", f"{hook}:display"),
        ],
    )

    async def send(_message, _request_params) -> str:
        usage.add_turn(_turn(prompt_tokens=10, completion_tokens=1))
        return "done"

    usage_agent.send = send

    await app._send_interactive_message(
        "shown",
        "assistant",
        request_params=None,
        show_usage=True,
    )
    await app._send_interactive_message(
        "quiet",
        "assistant",
        request_params=None,
        show_usage=False,
    )

    assert marker.read_text(encoding="utf-8") == "1:1\n"
    assert len(app.user_turn_usage) == 1
    assert reported_usage == ["$0.0123"]


def test_regular_agent_usage_displays_cache_percentage_and_ttl() -> None:
    usage = UsageAccumulator()
    usage.add_turn(
        _turn(
            prompt_tokens=100,
            completion_tokens=20,
            cache_write=10,
            cache_read=5,
        )
    )
    usage.last_cache_activity_time = time.time()
    agent = _agent(usage)
    agent.llm = SimpleNamespace(resolved_model=SimpleNamespace(cache_ttl="5m"))
    app = AgentApp({"assistant": agent})

    display = app._collect_agent_turn_usage(app["assistant"], None)

    assert display is not None
    output = format_regular_turn_usage(display)
    assert "(cache 5%, wrote 10)" in output
    assert "· cache TTL " in output
    assert "^" not in output
    assert "*" not in output


def test_usage_display_omits_cache_percentage_when_provider_does_not_report_it() -> None:
    display = format_turn_usage(
        TurnUsageDisplay(
            input_tokens=100,
            output_tokens=20,
            tool_calls=0,
            cache_percentage=None,
            cache_write_tokens=None,
            context_percentage=None,
            cache_ttl=None,
        )
    )

    assert "cache" not in display


def test_usage_display_compacts_counts_at_one_million() -> None:
    display = format_turn_usage(
        TurnUsageDisplay(
            input_tokens=1_234_567,
            output_tokens=1_000_000,
            tool_calls=0,
            cache_percentage=None,
            cache_write_tokens=None,
            context_percentage=None,
            cache_ttl=None,
        )
    )

    assert "▶ 1.23M" in display
    assert "◀ 1.00M" in display


def test_openai_responses_usage_displays_documented_minimum_cache_ttl() -> None:
    usage = UsageAccumulator()
    usage.add_turn(
        _turn(
            prompt_tokens=100,
            completion_tokens=20,
            cache_read=50,
            provider=Provider.RESPONSES,
            usage_schema=UsageSchema.OPENAI_RESPONSES,
            model="gpt-5.6",
        )
    )
    app = AgentApp({"assistant": _agent(usage)})

    display = app._collect_agent_turn_usage(app["assistant"], None)

    assert display is not None
    assert "cache TTL 30m+" in format_regular_turn_usage(display)


@pytest.mark.parametrize(
    ("model", "uses_minimum_ttl"),
    [
        ("gpt-5.5", False),
        ("gpt-5.6-mini", True),
        ("gpt-6", True),
    ],
)
def test_openai_minimum_cache_ttl_model_range(
    model: str,
    uses_minimum_ttl: bool,
) -> None:
    assert AgentApp._uses_openai_minimum_cache_ttl(model) is uses_minimum_ttl


def test_older_openai_responses_usage_omits_org_dependent_cache_ttl() -> None:
    usage = UsageAccumulator()
    usage.add_turn(
        _turn(
            prompt_tokens=100,
            completion_tokens=20,
            cache_read=50,
            provider=Provider.RESPONSES,
            usage_schema=UsageSchema.OPENAI_RESPONSES,
            model="gpt-5.5",
        )
    )
    app = AgentApp({"assistant": _agent(usage)})

    display = app._collect_agent_turn_usage(app["assistant"], None)

    assert display is not None
    assert "cache TTL" not in format_regular_turn_usage(display)
