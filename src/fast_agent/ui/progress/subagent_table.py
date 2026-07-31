"""Compact Rich table for concurrent subagent monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from fast_agent.event_progress import SubagentMonitorSnapshot
from fast_agent.utils.count_display import format_compact_count
from fast_agent.utils.time import format_compact_duration

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rich.progress import Task


def render_subagent_table(
    subagent_tasks: Sequence[Task],
    process_tasks: Iterable[Task],
    *,
    console_width: int,
) -> Table:
    """Render one stable row per active subagent."""
    narrow = console_width < 72
    table = Table(
        title=f"Subagents ({len(subagent_tasks)})",
        title_justify="left",
        title_style="bold",
        box=None,
        padding=(0, 1),
        pad_edge=False,
        expand=True,
    )
    table.add_column(
        "subagent",
        min_width=8,
        max_width=14 if narrow else 24,
        ratio=2,
        overflow="ellipsis",
        no_wrap=True,
    )
    table.add_column(
        "state",
        min_width=8,
        max_width=14 if narrow else 28,
        ratio=2,
        overflow="ellipsis",
        no_wrap=True,
    )
    table.add_column("turn", width=4, justify="right", style="cyan", no_wrap=True)
    table.add_column("in", width=7, justify="right", style="blue", no_wrap=True)
    table.add_column("out", width=7, justify="right", style="green", no_wrap=True)
    table.add_column(
        "processes",
        min_width=4,
        max_width=11,
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
        table.add_row(
            Text(str(task.fields.get("target") or task_name), style="bold white"),
            _state_text(snapshot, task),
            str(snapshot.turn),
            format_compact_count(snapshot.input_tokens, significant_digits=4),
            (
                f"~{format_compact_count(snapshot.output_tokens, significant_digits=4)}"
                if snapshot.output_estimated
                else format_compact_count(snapshot.output_tokens, significant_digits=4)
            ),
            _process_summary(process_list, owner_row=task_name),
        )
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

    elapsed_values = [
        elapsed for task in owned if (elapsed := _process_elapsed(task)) is not None
    ]
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


def _state_text(snapshot: SubagentMonitorSnapshot, task: Task) -> Text:
    text = Text(snapshot.state, style=_state_style(snapshot.state))
    elapsed = task.fields.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
        snapshot_elapsed = task.fields.get("elapsed_snapshot_task_elapsed")
        local_tick = 0.0
        if isinstance(snapshot_elapsed, (int, float)) and not isinstance(
            snapshot_elapsed, bool
        ):
            local_tick = max((task.elapsed or 0.0) - float(snapshot_elapsed), 0.0)
        elapsed_label = format_compact_duration(float(elapsed) + local_tick)
        if elapsed_label is not None:
            text.append(f" · {elapsed_label}", style="dim")
    return text


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
