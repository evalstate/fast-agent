"""Compact Rich table for concurrent subagent monitoring."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from fast_agent.event_progress import SubagentMonitorSnapshot
from fast_agent.utils.count_display import format_compact_count
from fast_agent.utils.time import format_compact_duration

_MODEL_WIDTH = 25
_MODEL_META_SEPARATOR = " · "
_MODEL_DETAIL_GAP = "  "
_DETAIL_HEADER = f"{'model':<{_MODEL_WIDTH + len(_MODEL_DETAIL_GAP)}}detail"

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rich.progress import Task


def render_subagent_table(
    subagent_tasks: Sequence[Task],
    process_tasks: Iterable[Task],
    *,
    console_width: int,
    spinner_frame: Text,
) -> Table:
    """Render one stable row per active subagent."""
    show_processes = console_width >= 98
    show_output = console_width >= 84
    show_input = console_width >= 74
    show_turn = console_width >= 64
    detail_width = (
        44
        if show_processes
        else 38
        if show_output
        else 34
        if show_input
        else 33
        if show_turn
        else max(8, console_width - 21)
    )
    table = Table(
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        pad_edge=False,
        expand=False,
    )
    table.add_column(
        "subagent",
        width=16,
        overflow="ellipsis",
        no_wrap=True,
    )
    table.add_column(
        "",
        width=3,
        justify="center",
        no_wrap=True,
    )
    table.add_column(
        _DETAIL_HEADER,
        width=detail_width,
        overflow="ellipsis",
        no_wrap=True,
    )
    if show_input:
        table.add_column("in", width=7, justify="right", style="blue", no_wrap=True)
        table.add_column("cache", width=5, justify="right", style="blue", no_wrap=True)
    if show_output:
        table.add_column("out", width=8, justify="right", style="green", no_wrap=True)
    if show_processes:
        table.add_column(
            "processes",
            width=9,
            justify="right",
            style="magenta",
            no_wrap=True,
        )

    process_list = list(process_tasks)
    for task in subagent_tasks:
        snapshot = task.fields.get("subagent_monitor")
        if not isinstance(snapshot, SubagentMonitorSnapshot):
            continue
        task_name = str(task.fields.get("task_name") or "")
        row: list[str | Text] = [
            Text(str(task.fields.get("target") or task_name), style="bold white"),
            spinner_frame,
            _detail_text(snapshot, task, show_turn=show_turn),
        ]
        if show_input:
            row.append(format_compact_count(snapshot.input_tokens, significant_digits=4))
            row.append(_cache_percentage_text(snapshot.cache_percentage))
        if show_output:
            row.append(
                f"~{format_compact_count(snapshot.output_tokens, significant_digits=4)}"
                if snapshot.output_estimated
                else format_compact_count(snapshot.output_tokens, significant_digits=4)
            )
        if show_processes:
            row.append(_process_summary(process_list, owner_row=task_name))
        table.add_row(*row)
    return table


def _process_summary(tasks: Sequence[Task], *, owner_row: str) -> str:
    owned = [
        task
        for task in tasks
        if bool(task.fields.get("is_process_poll"))
        and task.fields.get("process_owner_row") == owner_row
    ]
    if not owned:
        return "—"

    elapsed_values = [elapsed for task in owned if (elapsed := _process_elapsed(task)) is not None]
    elapsed = max(elapsed_values) if elapsed_values else None
    elapsed_label = format_compact_duration(elapsed)
    return f"{len(owned)} · {elapsed_label}" if elapsed_label is not None else str(len(owned))


def _process_elapsed(task: Task) -> float | None:
    value = task.fields.get("process_elapsed_seconds")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    snapshot_elapsed = task.fields.get("process_snapshot_task_elapsed")
    local_tick = 0.0
    if isinstance(snapshot_elapsed, (int, float)) and not isinstance(snapshot_elapsed, bool):
        local_tick = max((task.elapsed or 0.0) - float(snapshot_elapsed), 0.0)
    return float(value) + local_tick


def _detail_text(snapshot: SubagentMonitorSnapshot, task: Task, *, show_turn: bool) -> Text:
    metadata = Text()
    if show_turn:
        turn = Text(str(snapshot.turn), style="cyan")
        turn.truncate(3, overflow="ellipsis")
        metadata.append(_MODEL_META_SEPARATOR, style="dim")
        metadata.append_text(turn)
    if context_label := _context_label(snapshot.context_percentage):
        if show_turn:
            metadata.append(" ")
        else:
            metadata.append(_MODEL_META_SEPARATOR, style="dim")
        metadata.append(f"({context_label})", style="blue")

    text = Text(snapshot.model or "—", style="cyan" if snapshot.model else "dim")
    text.truncate(max(1, _MODEL_WIDTH - metadata.cell_len), overflow="ellipsis")
    text.append_text(metadata)
    text.truncate(_MODEL_WIDTH, overflow="ellipsis", pad=True)
    text.append(_MODEL_DETAIL_GAP)
    text.append(snapshot.state, style=_state_style(snapshot.state))
    elapsed = task.fields.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
        snapshot_elapsed = task.fields.get("elapsed_snapshot_task_elapsed")
        local_tick = 0.0
        if isinstance(snapshot_elapsed, (int, float)) and not isinstance(snapshot_elapsed, bool):
            local_tick = max((task.elapsed or 0.0) - float(snapshot_elapsed), 0.0)
        elapsed_label = format_compact_duration(float(elapsed) + local_tick)
        if elapsed_label is not None:
            text.append(f" · {elapsed_label}", style="dim")
    return text


def _context_label(context_percentage: float | None) -> str:
    if context_percentage is None or not math.isfinite(context_percentage):
        return ""
    safe_percentage = max(context_percentage, 0.0)
    return "100%+" if safe_percentage >= 100 else f"{min(round(safe_percentage), 99)}%"


def _cache_percentage_text(cache_percentage: float | None) -> str:
    if cache_percentage is None or not math.isfinite(cache_percentage):
        return "—"
    safe_percentage = max(cache_percentage, 0.0)
    if safe_percentage < 100 and round(safe_percentage) == 100:
        return ">99%"
    return f"{safe_percentage:.0f}%"


def _state_style(state: str) -> str:
    normalized = state.casefold()
    if normalized.startswith("tool:"):
        return "magenta"
    return {
        "starting": "green",
        "thinking": "bold yellow",
        "processing": "cyan",
        "finalizing": "blue",
    }.get(normalized, "white")


__all__ = ["render_subagent_table"]
