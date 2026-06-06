"""
Enhanced notification tracker for prompt_toolkit toolbar display.
Tracks both active events (sampling/elicitation) and completed notifications.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias

from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_casefold


@dataclass(frozen=True, slots=True)
class EventDisplay:
    singular: str
    plural: str
    compact: str


# Display metadata for toolbar summaries (singular, plural, compact label)
_EVENT_ORDER = ("tool_update", "sampling", "elicitation", "warning")
_EVENT_DISPLAY = {
    "tool_update": EventDisplay("tool update", "tool updates", "tool"),
    "sampling": EventDisplay("sample", "samples", "samp"),
    "elicitation": EventDisplay("elicitation", "elicitations", "elic"),
    "warning": EventDisplay("warning", "warnings", "warn"),
}

# Active events currently in progress
ActiveEventKey: TypeAlias = tuple[str, str]
active_events: dict[ActiveEventKey, dict[str, str]] = {}

# Completed notifications history
notifications: list[dict[str, str]] = []

# Startup warnings are tracked separately so they can be emitted once
# without polluting toolbar warning counters.
startup_warnings: list[str] = []
_startup_warning_seen: set[str] = set()


def _notification(event_type: str, **payload: str) -> dict[str, str]:
    return {"type": event_type, **payload}


def add_tool_update(server_name: str) -> None:
    """Add a tool update notification.

    Args:
        server_name: Name of the server that had tools updated
    """
    notifications.append(_notification("tool_update", server=server_name))


def add_warning(
    message: str, *, surface: Literal["runtime_toolbar", "startup_once"] = "runtime_toolbar"
) -> None:
    """Add a deferred warning notification.

    Args:
        message: Warning text to track.
        surface: Where warning should appear.
            - runtime_toolbar: contributes to toolbar warning counters
            - startup_once: queued for one-time startup digest emission
    """
    normalized_message = message.strip()
    if not normalized_message:
        return

    if surface == "startup_once":
        if normalized_message not in _startup_warning_seen:
            startup_warnings.append(normalized_message)
            _startup_warning_seen.add(normalized_message)
        return

    notifications.append(_notification("warning", message=normalized_message))


def pop_startup_warnings() -> list[str]:
    """Return startup warnings queued for one-time emission and clear the queue."""
    if not startup_warnings:
        return []

    queued = list(startup_warnings)
    startup_warnings.clear()
    return queued


def remove_startup_warnings_containing(fragment: str) -> int:
    """Remove queued startup warnings containing a fragment (case-insensitive)."""
    needle = strip_casefold(fragment)
    if not needle:
        return 0

    to_remove = [warning for warning in startup_warnings if needle in strip_casefold(warning)]
    if not to_remove:
        return 0

    startup_warnings[:] = [
        warning for warning in startup_warnings if needle not in strip_casefold(warning)
    ]
    for warning in to_remove:
        _startup_warning_seen.discard(warning)
    return len(to_remove)


def _invalidate_prompt() -> None:
    """Ask prompt_toolkit to redraw when an interactive app is active."""
    try:
        from prompt_toolkit.application.current import get_app

        get_app().invalidate()
    except Exception:
        pass


def _start_active_event(event_type: str, server_name: str) -> None:
    active_events[(event_type, server_name)] = {
        "server": server_name,
        "start_time": datetime.now().isoformat(),
    }
    _invalidate_prompt()


def _complete_active_event(event_type: str, server_name: str) -> None:
    completed = active_events.pop((event_type, server_name), None)
    if completed is None:
        return
    notifications.append(_notification(event_type, server=server_name))
    _invalidate_prompt()


def start_sampling(server_name: str) -> None:
    """Start tracking a sampling operation.

    Args:
        server_name: Name of the server making the sampling request
    """
    _start_active_event("sampling", server_name)


def end_sampling(server_name: str) -> None:
    """End tracking a sampling operation and add to completed notifications.

    Args:
        server_name: Name of the server that made the sampling request
    """
    _complete_active_event("sampling", server_name)


def start_elicitation(server_name: str) -> None:
    """Start tracking an elicitation operation.

    Args:
        server_name: Name of the server making the elicitation request
    """
    _start_active_event("elicitation", server_name)


def end_elicitation(server_name: str) -> None:
    """End tracking an elicitation operation and add to completed notifications.

    Args:
        server_name: Name of the server that made the elicitation request
    """
    _complete_active_event("elicitation", server_name)


def get_active_status() -> dict[str, str] | None:
    """Get currently active operation, if any.

    Returns:
        Dict with 'type' and 'server' keys, or None if nothing active
    """
    for event_type in ("sampling", "elicitation"):
        for active_type, server_name in active_events:
            if active_type == event_type:
                return {"type": event_type, "server": server_name}
    return None


def clear() -> None:
    """Clear all notifications and active events."""
    notifications.clear()
    active_events.clear()
    startup_warnings.clear()
    _startup_warning_seen.clear()


def get_count() -> int:
    """Get the current completed notification count."""
    return len(notifications)


def get_latest() -> dict[str, str] | None:
    """Get the most recent completed notification."""
    return notifications[-1] if notifications else None


def get_counts_by_type() -> dict[str, int]:
    """Aggregate completed notifications by event type."""
    counts = Counter(notification["type"] for notification in notifications)
    return _ordered_event_counts(counts)


def _ordered_event_counts(counts: Mapping[str, int]) -> dict[str, int]:
    if not counts:
        return {}
    ordered = {
        event_type: counts[event_type] for event_type in _EVENT_ORDER if event_type in counts
    }
    ordered.update(
        (event_type, count) for event_type, count in counts.items() if event_type not in ordered
    )
    return ordered


def format_event_label(event_type: str, count: int, *, compact: bool = False) -> str:
    """Format a human-readable label for an event count."""
    event_display = _EVENT_DISPLAY.get(event_type)

    if event_display is None:
        base = event_type.replace("_", " ")
        if compact:
            return f"{base[:1]}:{count}"
        return format_count(count, base)

    if compact:
        return f"{event_display.compact}:{count}"

    return format_count(count, event_display.singular, event_display.plural)


def get_summary(*, compact: bool = False) -> str:
    """Get a summary of completed notifications by type.

    Args:
        compact: When True, use short-form labels for constrained UI areas.

    Returns:
        String like "3 tool updates, 2 samples" or "tool:3 samp:2" when compact.
    """
    counts = get_counts_by_type()
    if not counts:
        return ""

    parts = [
        format_event_label(event_type, count, compact=compact)
        for event_type, count in counts.items()
    ]

    separator = " " if compact else ", "
    return separator.join(parts)
