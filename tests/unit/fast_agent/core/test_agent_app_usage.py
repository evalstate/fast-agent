from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from fast_agent.core.agent_app import AgentApp
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)
from fast_agent.ui.turn_usage_display import (
    TurnUsageDisplay,
    format_regular_turn_usage,
    format_regular_turn_usage_with_subagents,
    format_turn_usage,
)


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
