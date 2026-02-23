"""
Enhanced notification tracker for prompt_toolkit toolbar display.
Tracks both active events (sampling/elicitation) and completed notifications.
"""

from datetime import datetime
from typing import Literal

# Display metadata for toolbar summaries (singular, plural, compact label)
_EVENT_ORDER = ("tool_update", "sampling", "elicitation", "warning")
_EVENT_DISPLAY = {
    "tool_update": {"singular": "tool update", "plural": "tool updates", "compact": "tool"},
    "sampling": {"singular": "sample", "plural": "samples", "compact": "samp"},
    "elicitation": {"singular": "elicitation", "plural": "elicitations", "compact": "elic"},
    "warning": {"singular": "warning", "plural": "warnings", "compact": "warn"},
}

# Active events currently in progress
active_events: dict[str, dict[str, str]] = {}

# Completed notifications history
notifications: list[dict[str, str]] = []

# Startup warnings are tracked separately so they can be emitted once
# without polluting toolbar warning counters.
startup_warnings: list[str] = []
_startup_warning_seen: set[str] = set()


def add_tool_update(server_name: str) -> None:
    """Add a tool update notification.

    Args:
        server_name: Name of the server that had tools updated
    """
    notifications.append({
        'type': 'tool_update',
        'server': server_name
    })


def add_warning(message: str, *, surface: Literal["runtime_toolbar", "startup_once"] = "runtime_toolbar") -> None:
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

    notifications.append({
        'type': 'warning',
        'message': normalized_message,
    })


def pop_startup_warnings() -> list[str]:
    """Return startup warnings queued for one-time emission and clear the queue."""
    if not startup_warnings:
        return []

    queued = list(startup_warnings)
    startup_warnings.clear()
    return queued


def remove_startup_warnings_containing(fragment: str) -> int:
    """Remove queued startup warnings containing a fragment (case-insensitive)."""
    needle = fragment.strip().casefold()
    if not needle:
        return 0

    to_remove = [
        warning
        for warning in startup_warnings
        if needle in warning.casefold()
    ]
    if not to_remove:
        return 0

    startup_warnings[:] = [
        warning
        for warning in startup_warnings
        if needle not in warning.casefold()
    ]
    for warning in to_remove:
        _startup_warning_seen.discard(warning)
    return len(to_remove)


def start_sampling(server_name: str) -> None:
    """Start tracking a sampling operation.

    Args:
        server_name: Name of the server making the sampling request
    """
    active_events['sampling'] = {
        'server': server_name,
        'start_time': datetime.now().isoformat()
    }

    # Force prompt_toolkit to redraw if active
    try:
        from prompt_toolkit.application.current import get_app
        get_app().invalidate()
    except Exception:
        pass


def end_sampling(server_name: str) -> None:
    """End tracking a sampling operation and add to completed notifications.

    Args:
        server_name: Name of the server that made the sampling request
    """
    if 'sampling' in active_events:
        del active_events['sampling']

    notifications.append({
        'type': 'sampling',
        'server': server_name
    })

    # Force prompt_toolkit to redraw if active
    try:
        from prompt_toolkit.application.current import get_app
        get_app().invalidate()
    except Exception:
        pass


def start_elicitation(server_name: str) -> None:
    """Start tracking an elicitation operation.

    Args:
        server_name: Name of the server making the elicitation request
    """
    active_events['elicitation'] = {
        'server': server_name,
        'start_time': datetime.now().isoformat()
    }

    # Force prompt_toolkit to redraw if active
    try:
        from prompt_toolkit.application.current import get_app
        get_app().invalidate()
    except Exception:
        pass


def end_elicitation(server_name: str) -> None:
    """End tracking an elicitation operation and add to completed notifications.

    Args:
        server_name: Name of the server that made the elicitation request
    """
    if 'elicitation' in active_events:
        del active_events['elicitation']

    notifications.append({
        'type': 'elicitation',
        'server': server_name
    })

    # Force prompt_toolkit to redraw if active
    try:
        from prompt_toolkit.application.current import get_app
        get_app().invalidate()
    except Exception:
        pass


def get_active_status() -> dict[str, str] | None:
    """Get currently active operation, if any.

    Returns:
        Dict with 'type' and 'server' keys, or None if nothing active
    """
    if 'sampling' in active_events:
        return {'type': 'sampling', 'server': active_events['sampling']['server']}
    if 'elicitation' in active_events:
        return {'type': 'elicitation', 'server': active_events['elicitation']['server']}
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
    counts: dict[str, int] = {}
    for notification in notifications:
        event_type = notification['type']
        counts[event_type] = counts.get(event_type, 0) + 1

    if not counts:
        return {}

    ordered: dict[str, int] = {}
    for event_type in _EVENT_ORDER:
        if event_type in counts:
            ordered[event_type] = counts[event_type]

    for event_type, count in counts.items():
        if event_type not in ordered:
            ordered[event_type] = count

    return ordered


def format_event_label(event_type: str, count: int, *, compact: bool = False) -> str:
    """Format a human-readable label for an event count."""
    event_display = _EVENT_DISPLAY.get(event_type)

    if event_display is None:
        base = event_type.replace('_', ' ')
        if compact:
            return f"{base[:1]}:{count}"
        label = base if count == 1 else f"{base}s"
        return f"{count} {label}"

    if compact:
        return f"{event_display['compact']}:{count}"

    label = event_display['singular'] if count == 1 else event_display['plural']
    return f"{count} {label}"


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
