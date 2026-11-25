"""Shared /warm command handlers."""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from rich.text import Text

from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.commands.results import CommandOutcome
from fast_agent.core.cache_warmer import CacheWarmer, format_ping
from fast_agent.utils.time import format_two_unit_duration

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext

WarmAction = Literal["start", "off", "status", "ping"]

# Shown with the estimate: what a keep-warm request costs relative to the prompt.
_CACHE_READ_PRICE = "about 0.1x the input price"
# A five-minute cache needs roughly this many requests an hour; past that an
# hour-long cache entry is the cheaper way to cover the same wait.
_LONG_WINDOW_SECONDS = 50 * 60
_STATUS_PINGS = 10


def _clock(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%H:%M")


def _schedule_lines(warmer: CacheWarmer, agent: LlmDecorator, window_seconds: int) -> list[str]:
    interval = warmer.interval_seconds
    assert interval is not None
    retention = warmer.retention_seconds
    requests = window_seconds // interval + 1
    lines = [
        (
            f"Cache retention {_retention_label(retention)}: one keep-warm request every "
            f"{_interval_label(warmer)}, up to {requests} in this window."
        )
    ]
    accumulator = agent.usage_accumulator
    context_tokens = accumulator.current_context_tokens if accumulator is not None else None
    if context_tokens:
        lines.append(
            f"Each request re-reads the conversation (~{context_tokens:,} tokens now) from "
            f"cache, billed at {_CACHE_READ_PRICE}: up to ~{requests * context_tokens:,} "
            "cached tokens for the window."
        )
    if retention == 300 and window_seconds > _LONG_WINDOW_SECONDS:
        lines.append(
            "For waits this long a 1-hour cache is usually cheaper: one request every "
            f"{format_two_unit_duration(54 * 60)} instead of {requests}, though cache writes "
            "then bill at 2x instead of 1.25x (anthropic.cache_ttl: 1h)."
        )
    return lines


def _retention_label(retention: int | None) -> str:
    return format_two_unit_duration(retention) if retention is not None else "unknown"


def _interval_label(warmer: CacheWarmer) -> str:
    interval = warmer.interval_seconds
    assert interval is not None
    configured = " (cache_warm.interval_seconds)" if warmer.settings.interval_seconds else ""
    return f"{format_two_unit_duration(interval)}{configured}"


def _warmer(ctx: "CommandContext", agent_name: str) -> tuple[LlmDecorator, CacheWarmer] | None:
    agent = ctx.agent_provider._agent(agent_name)
    return (agent, agent.cache_warmer) if isinstance(agent, LlmDecorator) else None


async def handle_warm(
    ctx: "CommandContext",
    *,
    agent_name: str,
    action: WarmAction,
    minutes: int | None = None,
) -> CommandOutcome:
    """Start, stop or describe prompt-cache warming at the idle prompt."""
    outcome = CommandOutcome()
    resolved = _warmer(ctx, agent_name)
    if resolved is None:
        outcome.add_message(
            "This agent does not talk to a model, so it has no prompt cache to keep warm.",
            channel="warning",
            agent_name=agent_name,
        )
        return outcome
    agent, warmer = resolved

    if action == "off":
        warmer.disarm()
        automatic = "stays on" if warmer.settings.auto else "is off"
        outcome.add_message(
            "Cache warming at the idle prompt is off. Automatic warming while tools, "
            f"subagents or herdr-core children run {automatic} (cache_warm.auto).",
            channel="info",
            agent_name=agent_name,
        )
        return outcome

    if action == "status":
        outcome.add_message(
            Text("\n").join(Text(line) for line in _status_lines(agent, warmer)),
            channel="info",
            title="Cache warming",
            agent_name=agent_name,
        )
        return outcome

    if action == "ping":
        previous = warmer.status()
        was_expired = previous.lapses_at is not None and previous.lapses_at <= time.time()
        ping = await warmer.ping("/warm")
        if ping.error is not None:
            outcome.add_message(
                Text(format_ping(ping)),
                channel="warning",
                title="Cache warming",
                agent_name=agent_name,
            )
            return outcome
        current = warmer.status()
        lines = [format_ping(ping)]
        if ping.outcome == "miss":
            lines.append("Cache had already expired (cold miss; cache refreshed).")
        elif ping.outcome == "hit":
            if was_expired:
                lines.append("Cache was still warm on the provider (hit past expected expiry).")
            else:
                lines.append("Cache was still warm (hit).")
        elif was_expired:
            lines.append(
                "Keep-warm request sent after expected expiry "
                "(provider does not report cache hits)."
            )
        else:
            lines.append("Keep-warm request sent (provider does not report cache hits).")
        if current.lapses_at is not None:
            lines.append(
                f"New cache window: expected to last until {_clock(current.lapses_at)} "
                f"({_retention_label(warmer.retention_seconds)})."
            )
        if (armed_until := warmer.armed_until) is not None:
            lines.append(f"Idle prompt warming remains on until {_clock(armed_until)}.")
        outcome.add_message(
            Text("\n").join(Text(line) for line in lines),
            channel="info",
            title="Cache warming",
            agent_name=agent_name,
        )
        return outcome

    if warmer.interval_seconds is None:
        outcome.add_message(
            "fast-agent does not know how long this model's provider keeps a prompt cache, "
            "so there is nothing to schedule. Set cache_warm.interval_seconds to warm it anyway.",
            channel="warning",
            agent_name=agent_name,
        )
        return outcome

    window_seconds = (minutes or warmer.settings.max_minutes) * 60
    until = warmer.arm(window_seconds)
    lines = [
        (
            f"Keeping the prompt cache warm while this prompt is idle until {_clock(until)} "
            f"({format_two_unit_duration(window_seconds)})."
        ),
        *_schedule_lines(warmer, agent, window_seconds),
        (
            "Each request is reported here and in the toolbar. /warm status shows the history; "
            "/warm off stops."
        ),
    ]
    outcome.add_message(
        Text("\n").join(Text(line) for line in lines),
        channel="info",
        title="Cache warming",
        agent_name=agent_name,
    )
    return outcome


def _status_lines(agent: LlmDecorator, warmer: CacheWarmer) -> list[str]:
    settings = warmer.settings
    mid_turn = agent.llm is None or agent.llm.warms_cache_mid_turn
    status = warmer.status()
    interval = warmer.interval_seconds
    retention = warmer.retention_seconds
    armed_until = warmer.armed_until
    lines = [
        "Idle prompt (/warm): "
        + (f"on until {_clock(armed_until)}" if armed_until is not None else "off"),
        (
            "Automatic (tools, subagents, herdr-core children): "
            f"{'on' if settings.auto else 'off'}, for {settings.max_minutes}m after the last "
            "model request"
            + ("" if mid_turn else "; not while tools or subagents run on this connection")
        ),
        f"Cache retention: {_retention_label(retention)} · keep-warm interval: "
        + (_interval_label(warmer) if interval is not None else "none (not warmable)"),
    ]
    if status.lapses_at is not None:
        lines.append(
            "Cache: expired"
            if status.lapses_at <= time.time()
            else f"Cache: expected to last until {_clock(status.lapses_at)}"
        )
    if status.state == "failed" and status.detail:
        lines.append(f"Last failure: {status.detail}")
    pings = status.pings
    if not pings:
        lines.append("No keep-warm requests yet.")
        return lines
    counts = Counter(ping.outcome for ping in pings)
    cached = sum(ping.cache_read_tokens or 0 for ping in pings)
    lines.append(
        f"Recent requests: {len(pings)} ("
        + ", ".join(f"{count} {outcome}" for outcome, count in sorted(counts.items()))
        + f") · {cached:,} tokens read from cache · last {int(time.time() - pings[-1].at)}s ago"
    )
    lines.extend(f"  {format_ping(ping)}" for ping in pings[-_STATUS_PINGS:])
    return lines
