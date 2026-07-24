"""Compact post-turn usage rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rich import print as rich_print

from fast_agent.ui.progress_display import progress_display
from fast_agent.utils.count_display import format_compact_count
from fast_agent.utils.time import format_two_unit_duration

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class CacheTTLExpiry:
    expires_at: float


@dataclass(frozen=True, slots=True)
class CacheTTLMinimum:
    seconds: int


type CacheTTLDisplay = CacheTTLExpiry | CacheTTLMinimum


@dataclass(frozen=True, slots=True)
class TurnUsageDisplay:
    """UI projection of canonical usage consumed by one user-initiated turn."""

    input_tokens: int
    output_tokens: int
    tool_calls: int
    cache_percentage: float | None
    cache_write_tokens: int | None
    context_percentage: float | None
    cache_ttl: CacheTTLDisplay | None


@dataclass(frozen=True, slots=True)
class NamedTurnUsageDisplay:
    name: str
    usage: TurnUsageDisplay


def _format_cache_percentage(percentage: float) -> str:
    if percentage < 100 and round(percentage) == 100:
        return ">99%"
    return f"{percentage:.0f}%"


def format_turn_usage(usage: TurnUsageDisplay) -> str:
    cache_parts: list[str] = []
    if usage.cache_percentage is not None:
        cache_parts.append(_format_cache_percentage(usage.cache_percentage))
    if usage.cache_write_tokens:
        cache_parts.append(f"wrote {format_compact_count(usage.cache_write_tokens)}")
    cache_info = f" [dim](cache {', '.join(cache_parts)})[/dim]" if cache_parts else ""
    details: list[str] = []
    if usage.tool_calls > 0:
        details.append(
            f"{usage.tool_calls} tool {'call' if usage.tool_calls == 1 else 'calls'}"
        )
    if usage.context_percentage is not None:
        details.append(f"context {usage.context_percentage:.1f}%")
    if isinstance(usage.cache_ttl, CacheTTLExpiry):
        expiry = datetime.fromtimestamp(usage.cache_ttl.expires_at).strftime("%H:%M")
        details.append(f"cache TTL {expiry}")
    elif isinstance(usage.cache_ttl, CacheTTLMinimum):
        details.append(f"cache TTL ≥{format_two_unit_duration(usage.cache_ttl.seconds)}")

    detail_info = f" [dim]· {' · '.join(details)}[/dim]" if details else ""
    return (
        f"[blue]▶ {format_compact_count(usage.input_tokens)}[/blue] input{cache_info}  "
        f"[green]◀ {format_compact_count(usage.output_tokens)}[/green] output{detail_info}"
    )


def format_regular_turn_usage(usage: TurnUsageDisplay) -> str:
    return f"[dim]Last:[/dim] {format_turn_usage(usage)}"


def format_parallel_turn_usage(children: Sequence[NamedTurnUsageDisplay]) -> list[str]:
    total_input = sum(child.usage.input_tokens for child in children)
    cache_percentages = [child.usage.cache_percentage for child in children]
    cache_percentage = None
    if total_input > 0 and all(value is not None for value in cache_percentages):
        cached_tokens = sum(
            child.usage.input_tokens * child.usage.cache_percentage / 100
            for child in children
            if child.usage.cache_percentage is not None
        )
        cache_percentage = cached_tokens / total_input * 100
    cache_writes = [child.usage.cache_write_tokens for child in children]
    cache_write_tokens = (
        sum(value for value in cache_writes if value is not None)
        if all(value is not None for value in cache_writes)
        else None
    )

    total = TurnUsageDisplay(
        input_tokens=total_input,
        output_tokens=sum(child.usage.output_tokens for child in children),
        tool_calls=sum(child.usage.tool_calls for child in children),
        cache_percentage=cache_percentage,
        cache_write_tokens=cache_write_tokens,
        context_percentage=None,
        cache_ttl=None,
    )
    lines = [f"[dim]Last (parallel):[/dim] {format_turn_usage(total)}"]
    for index, child in enumerate(children):
        prefix = "└─" if index == len(children) - 1 else "├─"
        lines.append(
            f"[dim]  {prefix} {child.name}:[/dim] {format_turn_usage(child.usage)}"
        )
    return lines


def display_regular_turn_usage(usage: TurnUsageDisplay) -> None:
    with progress_display.paused():
        rich_print()
        rich_print(format_regular_turn_usage(usage))


def display_parallel_turn_usage(children: Sequence[NamedTurnUsageDisplay]) -> None:
    with progress_display.paused():
        for line in format_parallel_turn_usage(children):
            rich_print(line)
