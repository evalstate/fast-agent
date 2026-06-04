from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from fast_agent.core import agent_app as agent_app_module
from fast_agent.core.agent_app import AgentApp
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import CacheUsage, TurnUsage, UsageAccumulator

if TYPE_CHECKING:
    import pytest


def _turn(
    *,
    input_tokens: int,
    output_tokens: int,
    tool_calls: int = 0,
    cache_usage: CacheUsage | None = None,
) -> TurnUsage:
    return TurnUsage(
        provider=Provider.OPENAI,
        model="gpt-test",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        tool_calls=tool_calls,
        cache_usage=cache_usage or CacheUsage(),
    )


def _agent(usage_accumulator: UsageAccumulator):
    return SimpleNamespace(name="assistant", usage_accumulator=usage_accumulator)


def test_regular_agent_usage_displays_last_turn_when_no_turn_start_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = UsageAccumulator()
    usage.add_turn(_turn(input_tokens=100, output_tokens=20, tool_calls=1))
    app = AgentApp({"assistant": _agent(usage)})

    printed: list[str] = []
    monkeypatch.setattr(
        agent_app_module,
        "rich_print",
        lambda *values: printed.append(" ".join(str(value) for value in values)),
    )

    app._show_regular_agent_usage(app["assistant"], None)

    output = "\n".join(printed)
    assert "Last turn: 100 Input, 20 Output, 1 tool calls" in output


def test_regular_agent_usage_displays_turn_delta_with_context_percentage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = UsageAccumulator()
    usage.add_turn(_turn(input_tokens=100, output_tokens=20, tool_calls=1))
    usage.add_turn(_turn(input_tokens=50, output_tokens=10, tool_calls=2))
    usage.set_context_window_size(200)
    app = AgentApp({"assistant": _agent(usage)})

    printed: list[str] = []
    monkeypatch.setattr(
        agent_app_module,
        "rich_print",
        lambda *values: printed.append(" ".join(str(value) for value in values)),
    )

    app._show_regular_agent_usage(app["assistant"], 0)

    output = "\n".join(printed)
    assert "Last turn: 50 Input, 10 Output, 2 tool calls (30.0%)" in output


def test_regular_agent_usage_displays_cache_markers_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = UsageAccumulator()
    usage.add_turn(
        _turn(
            input_tokens=100,
            output_tokens=20,
            cache_usage=CacheUsage(cache_write_tokens=10, cache_hit_tokens=5),
        )
    )
    usage.last_cache_activity_time = time.time()
    agent = _agent(usage)
    agent.llm = SimpleNamespace(resolved_model=SimpleNamespace(cache_ttl="5m"))
    app = AgentApp({"assistant": agent})

    printed: list[str] = []
    monkeypatch.setattr(
        agent_app_module,
        "rich_print",
        lambda *values: printed.append(" ".join(str(value) for value in values)),
    )

    app._show_regular_agent_usage(app["assistant"], None)

    output = "\n".join(printed)
    assert "[bright_yellow]^[/bright_yellow]" in output
    assert "[bright_green]*[/bright_green]" in output
    assert "[dim](" in output
