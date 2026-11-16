"""Keep an agent's provider prompt cache warm while nothing else is refreshing it.

A provider drops an unused prompt cache entry after its retention period, so a
conversation that pauses for longer pays full input price on its next request.
Managed-process polling avoids this by returning to the model before the cache
lapses. Other pauses have no model request to lean on: a turn blocked on a
subagent or slow tool, a coordinator idle while herdr-core children work, or a
person who steps away. For those, ``CacheWarmer`` re-sends the conversation as a
keep-warm request shortly before each lapse. Nothing is added to history.

One warmer belongs to one agent. Callers hold a ``keep_warm`` scope for exactly
as long as the agent makes no requests of its own; leaving the scope waits for
a request in flight, so a keep-warm request never overlaps a real one.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

from fast_agent.config import CacheWarmSettings
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from mcp import Tool

    from fast_agent.context import Context
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.request_params import RequestParams
    from fast_agent.llm.usage_tracking import TurnUsage

logger = get_logger(__name__)

WarmReason = Literal["tools", "hc", "/warm"]
WarmState = Literal["off", "active", "failed"]

# A request that reads less than this share of its prompt from cache found the
# cache cold (or the prompt too small to cache) and is reported as a miss.
_HIT_RATIO = 0.5
_PING_HISTORY = 50
_PING_TIMEOUT_SECONDS = 120


def ping_interval_seconds(retention_seconds: int) -> int:
    """Seconds between request starts that keep a cache of this retention alive."""
    return retention_seconds - max(45, retention_seconds // 10)


class WarmableAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def context(self) -> Context | None: ...

    @property
    def llm(self) -> FastAgentLLMProtocol | None: ...

    async def warm_prompt_cache(
        self,
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> TurnUsage: ...


@dataclass(frozen=True, slots=True)
class WarmPing:
    """The outcome of one keep-warm request."""

    at: float
    reason: WarmReason
    duration_seconds: float
    prompt_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None

    @property
    def hit_ratio(self) -> float | None:
        if not self.prompt_tokens or self.cache_read_tokens is None:
            return None
        return self.cache_read_tokens / self.prompt_tokens

    @property
    def outcome(self) -> Literal["hit", "miss", "sent", "failed"]:
        """``sent`` is a request whose provider does not say how much it read from cache."""
        if self.error is not None:
            return "failed"
        ratio = self.hit_ratio
        if ratio is None:
            return "sent"
        return "hit" if ratio >= _HIT_RATIO else "miss"


@dataclass(frozen=True, slots=True)
class WarmStatus:
    """What a renderer needs to show whether warming is working."""

    state: WarmState
    reason: WarmReason | None
    next_ping_at: float | None
    until: float | None
    detail: str | None
    pings: tuple[WarmPing, ...]
    lapses_at: float | None
    """When the provider is expected to drop the cache; None when that is unknown."""

    @property
    def last_ping(self) -> WarmPing | None:
        return self.pings[-1] if self.pings else None


def _clock(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def _request_started_at(turn: TurnUsage) -> float:
    """Cache lifetimes run from the start of a request, not from its completion."""
    duration_ms = turn.timing.duration_ms if turn.timing is not None else 0.0
    return turn.timestamp - (duration_ms or 0.0) / 1000


def format_ping(ping: WarmPing) -> str:
    """One line describing a keep-warm request, for scrollback and ``/warm status``."""
    prefix = f"{_clock(ping.at)} {ping.reason}"
    if ping.error is not None:
        return f"{prefix}: FAILED - {ping.error}"
    ratio = ping.hit_ratio
    cached = (
        f"{ping.cache_read_tokens or 0:,} of {ping.prompt_tokens or 0:,} prompt tokens "
        f"read from cache ({ratio:.1%})"
        if ratio is not None
        else "cache usage not reported"
    )
    parts = [f"{prefix}: {ping.outcome.upper()} - {cached}"]
    if ping.cache_write_tokens:
        parts.append(f"wrote {ping.cache_write_tokens:,}")
    if ping.output_tokens:
        parts.append(f"{ping.output_tokens:,} output")
    parts.append(f"{ping.duration_seconds:.1f}s")
    if ping.cost_usd is not None:
        parts.append(f"${ping.cost_usd:.4f}")
    return " · ".join(parts)


def format_status_summary(status: WarmStatus) -> str | None:
    """A short phrase for the toolbar and terminal status; None when there is nothing to say.

    While warming it describes the keep-warm requests; otherwise it says how long
    the cache is expected to last, or that it has expired.
    """
    last = status.last_ping
    if status.state == "active":
        parts = ["warm"]
        if last is not None:
            ratio = last.hit_ratio
            verified = sum(1 for ping in status.pings if ping.outcome != "miss")
            mark = "✗ MISS" if last.outcome == "miss" else "✓"
            parts.append(
                f"{mark} {verified}/{len(status.pings)}"
                + (f" {ratio:.0%}" if ratio is not None else "")
            )
        if status.next_ping_at is not None:
            parts.append(f"next {_clock(status.next_ping_at)[:5]}")
        if status.until is not None:
            parts.append(f"until {_clock(status.until)[:5]}")
        return " ".join(parts)
    parts = []
    if status.state == "failed" and last is not None:
        parts.append(f"warm FAILED {_clock(last.at)}")
    if status.lapses_at is not None:
        parts.append(
            "cache expired"
            if status.lapses_at <= time.time()
            else f"cache until {_clock(status.lapses_at)[:5]}"
        )
    return " · ".join(parts) or None


def status_level(status: WarmStatus) -> Literal["ok", "warning", "error"]:
    """How a renderer should colour the summary."""
    last = status.last_ping
    if status.state == "failed" or (
        status.state == "active" and last is not None and last.outcome == "miss"
    ):
        return "error"
    if status.state != "active" and status.lapses_at is not None:
        return "warning" if status.lapses_at <= time.time() else "ok"
    return "ok"


class CacheWarmer:
    """Schedules keep-warm requests for one agent's conversation."""

    def __init__(self, agent: WarmableAgent) -> None:
        self._agent = agent
        self._armed_until: float | None = None
        # Automatic warming is measured from the last request the agent made
        # itself, so the warmer remembers which recorded request was its own.
        self._activity_at: float | None = None
        self._own_request: TurnUsage | None = None
        self._pings: deque[WarmPing] = deque(maxlen=_PING_HISTORY)
        self._state: WarmState = "off"
        self._reason: WarmReason | None = None
        self._next_ping_at: float | None = None
        self._until: float | None = None
        self._detail: str | None = None

    @property
    def settings(self) -> CacheWarmSettings:
        context = self._agent.context
        config = context.config if context is not None else None
        return config.cache_warm if config is not None else CacheWarmSettings()

    @property
    def retention_seconds(self) -> int | None:
        llm = self._agent.llm
        return llm.cache_retention_seconds if llm is not None else None

    @property
    def interval_seconds(self) -> int | None:
        """Seconds between keep-warm requests; None when this model cannot be warmed."""
        configured = self.settings.interval_seconds
        if configured is not None:
            return configured
        retention = self.retention_seconds
        return ping_interval_seconds(retention) if retention is not None else None

    @property
    def armed_until(self) -> float | None:
        """When a ``/warm`` window ends, while it is still open."""
        if self._armed_until is not None and self._armed_until <= time.time():
            self._armed_until = None
        return self._armed_until

    def arm(self, seconds: float) -> float:
        """Keep the cache warm at the idle prompt for ``seconds`` from now."""
        self._armed_until = time.time() + seconds
        return self._armed_until

    def disarm(self) -> None:
        self._armed_until = None
        if self._state != "active":
            self._state, self._detail = "off", None

    def status(self) -> WarmStatus:
        last = self._last_request_started_at()
        retention = self.retention_seconds
        return WarmStatus(
            state=self._state,
            reason=self._reason,
            next_ping_at=self._next_ping_at,
            until=self._until,
            detail=self._detail,
            pings=tuple(self._pings),
            lapses_at=last + retention if last is not None and retention is not None else None,
        )

    def _turns(self) -> list[TurnUsage]:
        llm = self._agent.llm
        accumulator = llm.usage_accumulator if llm is not None else None
        return accumulator.turns if accumulator is not None else []

    def _last_request_started_at(self) -> float | None:
        turns = self._turns()
        if not turns:
            self._activity_at = None
            return None
        if turns[-1] is not self._own_request:
            self._activity_at = _request_started_at(turns[-1])
        return _request_started_at(turns[-1])

    def _window_end(self, *, automatic: bool, armed: bool) -> float | None:
        self._last_request_started_at()
        ends: list[float] = []
        settings = self.settings
        if automatic and settings.auto and self._activity_at is not None:
            ends.append(self._activity_at + settings.max_minutes * 60)
        if armed and (armed_until := self.armed_until) is not None:
            ends.append(armed_until)
        return max(ends, default=None)

    @asynccontextmanager
    async def keep_warm(
        self,
        reason: WarmReason,
        *,
        automatic: bool,
        armed: bool = False,
        mid_turn: bool = False,
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        report: Callable[[str | None], None] | None = None,
    ) -> AsyncGenerator[None]:
        """Keep the cache warm for as long as the caller stays inside this scope.

        ``automatic`` warms for ``max_minutes`` after the agent's last request;
        ``armed`` honours a ``/warm`` window. ``mid_turn`` says the last response's
        tool calls are still running. ``report`` is called whenever the status
        changes, with a line for scrollback when there is one.
        """
        llm = self._agent.llm
        interval = self.interval_seconds
        until = self._window_end(automatic=automatic, armed=armed)
        if (
            llm is None
            or interval is None
            or until is None
            or until <= time.time()
            or (mid_turn and not llm.warms_cache_mid_turn)
        ):
            yield
            return
        stop = asyncio.Event()
        task = asyncio.create_task(
            self._run(reason, interval, until, stop, request_params, tools, report)
        )
        try:
            yield
        finally:
            # A request in flight is allowed to finish, even if the caller is
            # being cancelled: abandoning it drops the provider connection, and
            # with it the cache this scope protected. It is bounded by its timeout.
            stop.set()
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            if self._state == "active":
                self._state = "off"
            self._next_ping_at = None
            self._report(report, None)
            if cancelled:
                raise asyncio.CancelledError

    async def _run(
        self,
        reason: WarmReason,
        interval: int,
        until: float,
        stop: asyncio.Event,
        request_params: RequestParams | None,
        tools: list[Tool] | None,
        report: Callable[[str | None], None] | None,
    ) -> None:
        self._state, self._reason, self._until, self._detail = "active", reason, until, None
        try:
            await self._schedule(reason, interval, until, stop, request_params, tools, report)
        except Exception as exc:
            # Warming is an optimisation: its faults stay out of the caller's turn.
            logger.exception("Cache keep-warm scheduling failed")
            self._state, self._next_ping_at, self._detail = "failed", None, str(exc)

    async def _schedule(
        self,
        reason: WarmReason,
        interval: int,
        until: float,
        stop: asyncio.Event,
        request_params: RequestParams | None,
        tools: list[Tool] | None,
        report: Callable[[str | None], None] | None,
    ) -> None:
        while not stop.is_set():
            last = self._last_request_started_at()
            if last is None:
                return
            now = time.time()
            due = max(now, last + interval)
            if due >= until:
                self._state, self._next_ping_at = "off", None
                self._report(
                    report,
                    f"cache warm ({reason}): no further requests before the window ends at "
                    f"{_clock(until)}",
                )
                return
            self._next_ping_at = due
            self._report(report, None)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=due - now)
            if stop.is_set():
                return
            ping = await self._ping(reason, request_params, tools)
            self._pings.append(ping)
            logger.info("Cache keep-warm request", agent_name=self._agent.name, data=asdict(ping))
            if ping.error is not None:
                self._state, self._next_ping_at, self._detail = "failed", None, ping.error
                self._report(report, f"cache warm {format_ping(ping)}; warming stopped")
                return
            self._report(report, f"cache warm {format_ping(ping)}")

    @staticmethod
    def _report(report: Callable[[str | None], None] | None, line: str | None) -> None:
        """Tell the caller's display; a display fault must not disturb the turn."""
        if report is None:
            return
        try:
            report(line)
        except Exception:
            logger.exception("Cache keep-warm status could not be displayed")

    async def _ping(
        self,
        reason: WarmReason,
        request_params: RequestParams | None,
        tools: list[Tool] | None,
    ) -> WarmPing:
        started = time.time()
        try:
            usage = await asyncio.wait_for(
                self._agent.warm_prompt_cache(request_params, tools),
                timeout=_PING_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            # Whatever the failed request recorded was not the agent's own activity.
            turns = self._turns()
            self._own_request = turns[-1] if turns else None
            return WarmPing(
                at=time.time(),
                reason=reason,
                duration_seconds=time.time() - started,
                error=f"{type(exc).__name__}: {exc}".splitlines()[0],
            )
        self._own_request = usage
        llm = self._agent.llm
        reports_cache_use = llm is not None and llm.warm_cache_reports_cache_use
        return WarmPing(
            at=time.time(),
            reason=reason,
            duration_seconds=time.time() - started,
            prompt_tokens=usage.prompt.total,
            cache_read_tokens=usage.prompt.cache_read if reports_cache_use else None,
            cache_write_tokens=usage.prompt.cache_write if reports_cache_use else None,
            output_tokens=usage.completion.total,
            cost_usd=usage.cost_usd,
        )
