#!/usr/bin/env python3
"""Event Replay Script

Replays events from a JSONL log file using rich_progress display.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

import typer

from fast_agent.core.logging.events import Event
from fast_agent.core.logging.listeners import convert_log_event
from fast_agent.ui.rich_progress import RichProgressDisplay

if TYPE_CHECKING:
    from fast_agent.event_progress import ProgressEvent


class ProgressDisplayProtocol(Protocol):
    """Minimal display contract used by replay logic."""

    def update(self, event: "ProgressEvent") -> None: ...


SleepFunc = Callable[[float], None]


@dataclass(frozen=True)
class ReplayStats:
    """Summary produced by replay_events."""

    total_events: int
    progress_events: int


def _compute_replay_delay(previous: datetime, current: datetime, *, speed: float) -> float:
    """Compute delay for an event pair, scaled by replay speed."""
    if speed <= 0:
        raise ValueError("speed must be > 0")

    elapsed_seconds = (current - previous).total_seconds()
    if elapsed_seconds <= 0:
        return 0.0
    return elapsed_seconds / speed


def _select_events(
    events: Sequence[Event],
    *,
    start_at: int,
    end_at: int | None,
) -> list[Event]:
    """Return event window requested by the CLI."""
    if start_at < 0:
        raise ValueError("start_at must be >= 0")
    if end_at is not None and end_at < start_at:
        raise ValueError("end_at must be >= start_at")
    return list(events[start_at:end_at])


def replay_events(
    events: Sequence[Event],
    *,
    display: ProgressDisplayProtocol,
    speed: float,
    sleep_func: SleepFunc = time.sleep,
) -> ReplayStats:
    """Replay events to a progress display and preserve source timing."""
    if speed <= 0:
        raise ValueError("speed must be > 0")

    progress_events = 0
    previous_timestamp: datetime | None = None

    for event in events:
        if previous_timestamp is not None:
            delay = _compute_replay_delay(previous_timestamp, event.timestamp, speed=speed)
            if delay > 0:
                sleep_func(delay)
        previous_timestamp = event.timestamp

        progress_event = convert_log_event(event)
        if progress_event is None:
            continue
        display.update(progress_event)
        progress_events += 1

    return ReplayStats(total_events=len(events), progress_events=progress_events)


def load_events(path: Path) -> list[Event]:
    """Load events from JSONL file."""
    events = []
    with open(path) as f:
        for line in f:
            if line.strip():
                raw_event = json.loads(line)
                # Convert from log format to event format
                event = Event(
                    type=raw_event.get("level", "info").lower(),
                    namespace=raw_event.get("namespace", ""),
                    message=raw_event.get("message", ""),
                    timestamp=datetime.fromisoformat(raw_event["timestamp"]),
                    data=raw_event.get("data", {}),  # Get data directly
                )
                events.append(event)
    return events


def main(
    log_file: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    speed: float = typer.Option(
        1.0,
        "--speed",
        "-s",
        help="Replay speed multiplier (2.0 = 2x faster, 0.5 = 2x slower).",
    ),
    start_at: int = typer.Option(
        0,
        "--start-at",
        help="Start replay at this zero-based event index.",
    ),
    end_at: int | None = typer.Option(
        None,
        "--end-at",
        help="Stop replay before this zero-based event index.",
    ),
) -> None:
    """Replay MCP Agent events from a log file with progress display."""
    if speed <= 0:
        raise typer.BadParameter("--speed must be greater than zero.")
    if start_at < 0:
        raise typer.BadParameter("--start-at must be >= 0.")
    if end_at is not None and end_at < start_at:
        raise typer.BadParameter("--end-at must be >= --start-at.")

    # Load events from file
    events = load_events(log_file)
    selected_events = _select_events(events, start_at=start_at, end_at=end_at)
    if not selected_events:
        typer.echo("No events selected for replay.")
        return

    # Initialize progress display
    progress = RichProgressDisplay()
    progress.start()

    stats: ReplayStats | None = None
    try:
        stats = replay_events(selected_events, display=progress, speed=speed)
    except KeyboardInterrupt:
        typer.echo("Replay interrupted by user.")
    finally:
        progress.stop()

    if stats is not None:
        typer.echo(
            "Replayed "
            f"{stats.progress_events} progress event(s) "
            f"from {stats.total_events} selected event(s)."
        )


if __name__ == "__main__":
    typer.run(main)
