"""
Enhanced notification tracker for prompt_toolkit toolbar display.
Tracks both active events (sampling/elicitation) and completed notifications.
"""

from datetime import datetime
from typing import Dict, List, Optional

# Active events currently in progress
active_events: Dict[str, Dict[str, str]] = {}

# Completed notifications history
notifications: List[Dict[str, str]] = []


def add_tool_update(server_name: str) -> None:
    """Add a tool update notification.

    Args:
        server_name: Name of the server that had tools updated
    """
    notifications.append({
        'type': 'tool_update',
        'server': server_name
    })


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


def get_active_status() -> Optional[Dict[str, str]]:
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


def get_count() -> int:
    """Get the current completed notification count."""
    return len(notifications)


def get_latest() -> Dict[str, str] | None:
    """Get the most recent completed notification."""
    return notifications[-1] if notifications else None


def get_summary() -> str:
    """Get a summary of completed notifications by type.

    Returns:
        String like "3 tools, 1 sampling, 2 elicitations" or "1 tool update"
    """
    if not notifications:
        return ""

    counts = {}
    for notification in notifications:
        event_type = notification['type']
        if event_type == 'tool_update':
            counts['tools'] = counts.get('tools', 0) + 1
        else:
            # For sampling/elicitation, use the type directly
            counts[event_type] = counts.get(event_type, 0) + 1

    # Build summary string
    parts = []
    for event_type, count in sorted(counts.items()):
        if event_type == 'tools':
            parts.append(f"{count} tool{'s' if count != 1 else ''}")
        elif event_type == 'sampling':
            parts.append(f"{count} sample{'s' if count != 1 else ''}")
        else:
            parts.append(f"{count} {event_type}{'s' if count != 1 else ''}")

    return ", ".join(parts)