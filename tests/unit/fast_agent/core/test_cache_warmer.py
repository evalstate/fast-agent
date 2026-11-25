from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.config import CacheWarmSettings
from fast_agent.core.cache_warmer import (
    CacheWarmer,
    WarmableAgent,
    format_status_summary,
    ping_interval_seconds,
    status_level,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.commands.context import CommandContext
    from fast_agent.interfaces import FastAgentLLMProtocol


def _turn(*, age_seconds: float = 0.0, cache_read: int | None = 990) -> TurnUsage:
    return TurnUsage(
        provider=Provider.ANTHROPIC,
        usage_schema=UsageSchema.OPENAI_CHAT,
        model="test-model",
        prompt=PromptTokenUsage(total=1000, cache_read=cache_read),
        completion=CompletionTokenUsage(total=0),
        timestamp=time.time() - age_seconds,
    )


class _Agent:
    """An agent whose last real request happened ``idle_seconds`` ago."""

    name = "agent"

    def __init__(
        self,
        *,
        idle_seconds: float | None = 600,
        retention: int | None = 300,
        mid_turn: bool = True,
        settings: CacheWarmSettings | None = None,
        on_request: Callable[[], Awaitable[None]] | None = None,
        cache_read: int | None = 990,
    ) -> None:
        self.accumulator = UsageAccumulator()
        if idle_seconds is not None:
            self.accumulator.add_turn(_turn(age_seconds=idle_seconds))
        self.llm = SimpleNamespace(
            usage_accumulator=self.accumulator,
            cache_retention_seconds=retention,
            warms_cache_mid_turn=mid_turn,
            warm_cache_reports_cache_use=True,
        )
        self.context = SimpleNamespace(
            config=SimpleNamespace(cache_warm=settings or CacheWarmSettings())
        )
        self.requests = 0
        self._on_request = on_request
        self._cache_read = cache_read

    async def warm_prompt_cache(self, request_params=None, tools=None) -> TurnUsage:
        self.requests += 1
        if self._on_request is not None:
            await self._on_request()
        turn = _turn(cache_read=self._cache_read)
        self.accumulator.add_turn(turn)
        return turn


def _warmer(agent: _Agent) -> CacheWarmer:
    return CacheWarmer(cast("WarmableAgent", agent))


async def _until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0)


@pytest.mark.parametrize("retention", [300, 1800, 3600])
def test_requests_are_spaced_inside_the_retention_period(retention: int) -> None:
    assert 0 < ping_interval_seconds(retention) < retention


@pytest.mark.asyncio
async def test_armed_prompt_refreshes_a_lapsing_cache_and_reports_the_hit() -> None:
    agent = _Agent()
    warmer = _warmer(agent)
    warmer.arm(600)
    lines: list[str] = []

    async with warmer.keep_warm(
        "/warm", automatic=False, armed=True, report=lambda line: lines.append(line or "")
    ):
        await _until(lambda: agent.requests == 1)
        await _until(lambda: warmer.status().next_ping_at is not None)
        assert "✓ 1/1" in (format_status_summary(warmer.status()) or "")

    status = warmer.status()
    assert agent.requests == 1
    assert status.last_ping is not None and status.last_ping.outcome == "hit"
    assert any("HIT" in line for line in lines)


@pytest.mark.asyncio
async def test_nothing_is_sent_without_a_reason_to_warm() -> None:
    agent = _Agent()
    warmer = _warmer(agent)

    async with warmer.keep_warm("hc", automatic=False, armed=True):
        await asyncio.sleep(0.01)

    assert agent.requests == 0


def test_idle_summary_says_how_long_the_cache_lasts_and_warns_once_it_has_expired() -> None:
    fresh = _warmer(_Agent(idle_seconds=60, retention=300)).status()
    lapsed = _warmer(_Agent(idle_seconds=600, retention=300)).status()
    unknown = _warmer(_Agent(retention=None)).status()

    assert (format_status_summary(fresh) or "").startswith("cache until ")
    assert status_level(fresh) == "ok"
    assert format_status_summary(lapsed) == "cache expired"
    assert status_level(lapsed) == "warning"
    assert format_status_summary(unknown) is None


@pytest.mark.asyncio
async def test_retention_longer_than_the_window_needs_no_requests() -> None:
    # Codex-like: the cache outlives the automatic window, so nothing is sent.
    agent = _Agent(idle_seconds=60, retention=50 * 60)
    warmer = _warmer(agent)
    lines: list[str] = []

    async with warmer.keep_warm("hc", automatic=True, report=lambda line: lines.append(line or "")):
        await asyncio.sleep(0.01)

    assert agent.requests == 0
    assert any("no further requests" in line for line in lines)
    assert (format_status_summary(warmer.status()) or "").startswith("cache until ")


@pytest.mark.asyncio
async def test_automatic_warming_ends_a_fixed_time_after_the_agents_own_last_request() -> None:
    settings = CacheWarmSettings(max_minutes=20)
    agent = _Agent(idle_seconds=600, settings=settings)
    warmer = _warmer(agent)

    async with warmer.keep_warm("hc", automatic=True):
        await _until(lambda: agent.requests == 1)
    first_window_end = warmer.status().until
    # The warmer's own request must not push the window out.
    async with warmer.keep_warm("hc", automatic=True):
        await asyncio.sleep(0.01)
    assert warmer.status().until == first_window_end

    stale = _Agent(idle_seconds=21 * 60, settings=settings)
    async with _warmer(stale).keep_warm("hc", automatic=True):
        await asyncio.sleep(0.01)
    assert stale.requests == 0


@pytest.mark.asyncio
async def test_automatic_warming_can_be_disabled() -> None:
    agent = _Agent(settings=CacheWarmSettings(auto=False))

    async with _warmer(agent).keep_warm("tools", automatic=True, mid_turn=True):
        await asyncio.sleep(0.01)

    assert agent.requests == 0


@pytest.mark.asyncio
async def test_model_with_unknown_retention_is_warmed_only_at_a_configured_interval() -> None:
    unknown = _Agent(retention=None)
    warmer = _warmer(unknown)
    warmer.arm(600)
    async with warmer.keep_warm("/warm", automatic=False, armed=True):
        await asyncio.sleep(0.01)
    assert unknown.requests == 0

    configured = _Agent(retention=None, settings=CacheWarmSettings(interval_seconds=60))
    warmer = _warmer(configured)
    warmer.arm(600)
    async with warmer.keep_warm("/warm", automatic=False, armed=True):
        await _until(lambda: configured.requests == 1)


@pytest.mark.asyncio
async def test_provider_that_cannot_warm_mid_turn_is_left_alone_while_tools_run() -> None:
    agent = _Agent(mid_turn=False)

    async with _warmer(agent).keep_warm("tools", automatic=True, mid_turn=True):
        await asyncio.sleep(0.01)

    assert agent.requests == 0


@pytest.mark.asyncio
async def test_failed_request_stops_warming_and_is_reported() -> None:
    async def fail() -> None:
        raise RuntimeError("provider said no")

    agent = _Agent(on_request=fail)
    warmer = _warmer(agent)
    lines: list[str] = []

    async with warmer.keep_warm("hc", automatic=True, report=lambda line: lines.append(line or "")):
        await _until(lambda: warmer.status().state == "failed")
        await asyncio.sleep(0.01)

    assert agent.requests == 1
    assert warmer.status().state == "failed"
    assert any("FAILED" in line and "provider said no" in line for line in lines)
    assert "FAILED" in (format_status_summary(warmer.status()) or "")


@pytest.mark.asyncio
async def test_display_fault_does_not_reach_the_caller() -> None:
    def broken_display(line: str | None) -> None:
        raise RuntimeError("display failed")

    agent = _Agent()
    warmer = _warmer(agent)

    async with warmer.keep_warm("tools", automatic=True, mid_turn=True, report=broken_display):
        await _until(lambda: agent.requests == 1)

    assert warmer.status().last_ping is not None


@pytest.mark.asyncio
async def test_no_request_starts_once_the_scope_is_left() -> None:
    agent = _Agent()  # a request is already due

    async with _warmer(agent).keep_warm("hc", automatic=True):
        pass
    await asyncio.sleep(0.01)

    assert agent.requests == 0


@pytest.mark.asyncio
async def test_nothing_is_sent_before_the_agents_first_request() -> None:
    agent = _Agent(idle_seconds=None)
    warmer = _warmer(agent)
    warmer.arm(600)

    async with warmer.keep_warm("/warm", automatic=True, armed=True):
        await asyncio.sleep(0.01)

    assert agent.requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_read", [0, None])
async def test_cold_cache_is_reported_as_a_miss(cache_read: int | None) -> None:
    agent = _Agent(cache_read=cache_read)
    warmer = _warmer(agent)

    async with warmer.keep_warm("hc", automatic=True):
        await _until(lambda: agent.requests == 1)
        await _until(lambda: warmer.status().last_ping is not None)

    last = warmer.status().last_ping
    assert last is not None and last.outcome == "miss"


@pytest.mark.asyncio
async def test_leaving_the_scope_lets_a_request_in_flight_finish() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        started.set()
        await release.wait()

    agent = _Agent(on_request=hold)
    warmer = _warmer(agent)

    async def scope() -> None:
        async with warmer.keep_warm("hc", automatic=True):
            await started.wait()

    leaving = asyncio.create_task(scope())
    await started.wait()
    await asyncio.sleep(0.01)
    assert not leaving.done()  # still waiting for the request

    release.set()
    await leaving
    last = warmer.status().last_ping
    assert last is not None and last.outcome == "hit"
    assert agent.requests == 1


@pytest.mark.asyncio
async def test_cancelled_caller_still_lets_the_request_finish_and_stays_cancelled() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        started.set()
        await release.wait()

    agent = _Agent(on_request=hold)
    warmer = _warmer(agent)

    async def scope() -> None:
        async with warmer.keep_warm("tools", automatic=True, mid_turn=True):
            await asyncio.Event().wait()

    turn = asyncio.create_task(scope())
    await started.wait()
    turn.cancel()
    await asyncio.sleep(0.01)
    turn.cancel()  # a second interrupt while the request drains
    await asyncio.sleep(0.01)
    assert not turn.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await turn
    last = warmer.status().last_ping
    assert last is not None and last.outcome == "hit"


@pytest.mark.asyncio
async def test_manual_ping_refreshes_expired_window_and_clears_failure() -> None:
    agent = _Agent(idle_seconds=600, retention=300, cache_read=0)
    warmer = _warmer(agent)
    before = warmer.status()
    assert before.lapses_at is not None and before.lapses_at < time.time()

    # Simulate a prior failure state to verify a successful manual ping clears it.
    warmer._state, warmer._detail = "failed", "prior error"

    ping = await warmer.ping("/warm")
    assert ping.outcome == "miss"
    assert agent.requests == 1

    after = warmer.status()
    assert after.state == "off"
    assert after.detail is None
    assert after.last_ping is ping
    assert after.lapses_at is not None and after.lapses_at > time.time() + 200


@pytest.mark.asyncio
async def test_handle_warm_ping_reports_expiry_state_and_new_window() -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.llm_decorator import LlmDecorator
    from fast_agent.commands.handlers.warm import handle_warm
    from fast_agent.config import Settings
    from fast_agent.context import Context

    class _FakeDecorator(LlmDecorator):
        def __init__(self, *, idle_seconds: float, cache_read: int) -> None:
            super().__init__(
                AgentConfig(name="agent"),
                context=Context(config=Settings(cache_warm=CacheWarmSettings())),
            )
            self._accumulator = UsageAccumulator()
            self._accumulator.add_turn(_turn(age_seconds=idle_seconds))
            self._llm = cast(
                "FastAgentLLMProtocol",
                SimpleNamespace(
                    usage_accumulator=self._accumulator,
                    cache_retention_seconds=300,
                    warms_cache_mid_turn=True,
                    warm_cache_reports_cache_use=True,
                ),
            )
            self._cache_read = cache_read

        async def warm_prompt_cache(self, request_params=None, tools=None) -> TurnUsage:
            turn = _turn(cache_read=self._cache_read)
            self._accumulator.add_turn(turn)
            return turn

    # Case 1: expired cache -> cold miss -> refreshed new window
    expired_agent = _FakeDecorator(idle_seconds=600, cache_read=0)
    ctx = cast(
        "CommandContext",
        SimpleNamespace(agent_provider=SimpleNamespace(_agent=lambda _n: expired_agent)),
    )
    outcome = await handle_warm(ctx, agent_name="agent", action="ping")
    assert len(outcome.messages) == 1
    assert outcome.messages[0].channel == "info"
    text = str(outcome.messages[0].text)
    assert "MISS" in text
    assert "already expired" in text
    assert "New cache window" in text

    # Case 2: warm cache -> hit -> refreshed new window
    warm_agent = _FakeDecorator(idle_seconds=60, cache_read=990)
    ctx_warm = cast(
        "CommandContext",
        SimpleNamespace(agent_provider=SimpleNamespace(_agent=lambda _n: warm_agent)),
    )
    outcome_warm = await handle_warm(ctx_warm, agent_name="agent", action="ping")
    text_warm = str(outcome_warm.messages[0].text)
    assert "HIT" in text_warm
    assert "still warm" in text_warm
    assert "New cache window" in text_warm
