"""Compact managed-process polling display helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import ByteSize

from fast_agent.utils.time import format_compact_duration

# Braille dots:
#   1 4
#   2 5
#   3 6
#   7 8
# Within a cell, remaining fill drains top→bottom on the left column, then
# top→bottom on the right: 1, 2, 3, 7, 4, 5, 6, 8. Across cells, drain is
# right→left (left cells stay full longest; the right edge empties first).
#
# Each sweep drains 8–24 physical dots based on the wait budget, displayed in a
# fixed three-cell track. A dot remains visible for at most 10 seconds; longer
# waits use more dots, then additional full sweeps rather than slowing the
# drain. The next dot blinks before staying off. Blank is reserved for the poll
# deadline.
_CELL_DOT_BITS: tuple[int, ...] = (
    0x01,  # 1
    0x02,  # 2
    0x04,  # 3
    0x40,  # 7
    0x08,  # 4
    0x10,  # 5
    0x20,  # 6
    0x80,  # 8
)
_MAX_POLL_COUNTDOWN_CELLS = 3
_DOTS_PER_CELL = len(_CELL_DOT_BITS)  # 8
_MAX_SECONDS_PER_STEP = 10.0
_MAX_SECONDS_PER_CELL = _DOTS_PER_CELL * _MAX_SECONDS_PER_STEP


def _cell_glyph(level: int) -> str:
    """Glyph for cell fill level 0 (empty) .. 8 (full)."""
    if level <= 0:
        return " "
    if level >= _DOTS_PER_CELL:
        return chr(0x28FF)  # ⣿
    bits = 0
    for dot_bit in _CELL_DOT_BITS[:level]:
        bits |= dot_bit
    return chr(0x2800 + bits)


def _track_from_remaining_units(
    remaining_units: int,
    *,
    cell_count: int = _MAX_POLL_COUNTDOWN_CELLS,
) -> str:
    """Build a Braille track; drain order is top→bottom, right→left."""
    unit_count = cell_count * _DOTS_PER_CELL
    remaining_units = max(0, min(unit_count, remaining_units))
    # Prefer filling left cells so the right edge drains first.
    cells: list[str] = []
    left_to_assign = remaining_units
    for _ in range(cell_count):
        level = min(_DOTS_PER_CELL, left_to_assign)
        cells.append(_cell_glyph(level))
        left_to_assign -= level
    return "".join(cells)


def _countdown_cell_count(wait_seconds: int) -> int:
    """Use the fewest cells that keep each dot visible for at most 10s."""
    return min(
        _MAX_POLL_COUNTDOWN_CELLS,
        max(1, math.ceil(wait_seconds / _MAX_SECONDS_PER_CELL)),
    )


def _countdown_cycle_count(wait_seconds: int) -> int:
    """How many sweeps keep each dot interval at most 10s."""
    unit_count = _countdown_cell_count(wait_seconds) * _DOTS_PER_CELL
    max_one_cycle = unit_count * _MAX_SECONDS_PER_STEP
    return max(1, math.ceil(wait_seconds / max_one_cycle))


def _remaining_slots_for_wait(*, wait_seconds: int, elapsed_seconds: float) -> int:
    """Return remaining dots in the active adaptive-width sweep."""
    elapsed = max(elapsed_seconds, 0.0)
    if elapsed >= wait_seconds:
        return 0

    unit_count = _countdown_cell_count(wait_seconds) * _DOTS_PER_CELL
    cycles = _countdown_cycle_count(wait_seconds)
    total_steps = cycles * unit_count
    step_seconds = wait_seconds / total_steps
    completed_steps = min(math.floor(elapsed / step_seconds), total_steps - 1)
    remaining_global = total_steps - completed_steps
    return (remaining_global - 1) % unit_count + 1


def _slots_to_fill_units(remaining_slots: int, *, unit_count: int) -> int:
    """Clamp timing slots to the physical Braille dots."""
    return max(0, min(unit_count, remaining_slots))


def _remaining_units_for_wait(
    *,
    wait_seconds: int,
    elapsed_seconds: float,
    blink_next: bool = False,
) -> int:
    """Map wait progress onto the blinking adaptive-width glyph ladder."""
    unit_count = _countdown_cell_count(wait_seconds) * _DOTS_PER_CELL
    remaining = _slots_to_fill_units(
        _remaining_slots_for_wait(
            wait_seconds=wait_seconds,
            elapsed_seconds=elapsed_seconds,
        ),
        unit_count=unit_count,
    )
    return max(0, remaining - int(blink_next))


@dataclass(frozen=True, slots=True)
class ProcessOutputActivity:
    """Compact output-activity chip for process monitoring displays."""

    text: str
    style: str | None = None


def format_process_output_activity(
    *,
    has_observed_output: bool | None,
    seconds_since_last_output: float | None,
) -> ProcessOutputActivity | None:
    """Return a short activity chip with output recency.

    Recent output is emphasized for a short window, then fades. Missing or
    never-seen output stays silent.
    """
    if has_observed_output is None or seconds_since_last_output is None:
        return None
    if not has_observed_output:
        return None

    age = max(seconds_since_last_output, 0.0)
    age_label = format_compact_duration(age) or "0s"
    text = f"output {age_label} ago"
    if age <= 5:
        return ProcessOutputActivity(text, "bold bright_green")
    if age < 30:
        return ProcessOutputActivity(text, "green")
    return ProcessOutputActivity(text)


def format_process_poll_countdown_track(
    *,
    wait_seconds: int | None,
    elapsed_seconds: float,
    blink_next: bool = False,
) -> str | None:
    """Return a braille track that empties as a poll wait approaches its deadline.

    Drain order is top→bottom within a cell (left column, then right) and
    right→left across cells. The next dot blinks before staying off. Timing uses
    eight dots per 80s of wait budget, up to 24 dots, while the rendered track
    remains three cells wide. Waits longer than 240s run multiple sweeps so a
    dot never remains unchanged for more than 10s. ``None`` falls back to the
    pulse spinner.
    """
    if type(wait_seconds) is not int or wait_seconds <= 0:
        return None

    remaining_units = _remaining_units_for_wait(
        wait_seconds=wait_seconds,
        elapsed_seconds=elapsed_seconds,
        blink_next=blink_next,
    )
    return _track_from_remaining_units(remaining_units)


def format_process_output_size(total_bytes: int | None) -> str | None:
    """Return a compact cumulative output size for active-process displays."""
    if type(total_bytes) is not int or total_bytes <= 0:
        return None

    return ByteSize(total_bytes).human_readable(decimal=True)
