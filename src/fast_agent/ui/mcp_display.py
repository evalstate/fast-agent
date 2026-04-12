"""Rendering helpers for MCP status information in the enhanced prompt UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Literal, Protocol, runtime_checkable

from rich.text import Text

from fast_agent.ui import console

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import ServerStatus
    from fast_agent.mcp.transport_tracking import ChannelSnapshot


@runtime_checkable
class _ConfigWithInstruction(Protocol):
    instruction: str | None


@runtime_checkable
class _HasConfig(Protocol):
    config: object | None


@runtime_checkable
class _ServerStatusProvider(Protocol):
    async def get_server_status(self) -> dict[str, ServerStatus]: ...


type CapabilityState = bool | Literal["blue", "red", "warn"]


@dataclass(frozen=True, slots=True)
class _ChannelSummaryEntry:
    label: str
    arrow: str
    channel: ChannelSnapshot | None


@dataclass(frozen=True, slots=True)
class _ChannelSummaryLayout:
    transport_display: str
    default_bucket_seconds: int
    default_bucket_count: int
    metrics_prefix_width: int
    is_stdio: bool
    health_insert_label: str | None


# Centralized color configuration
class Colours:
    """Color constants for MCP status display elements."""

    # Timeline activity colors (Option A: Mixed Intensity)
    ERROR = "bright_red"  # Keep error bright
    DISABLED = "bright_blue"  # Keep disabled bright
    RESPONSE = "blue"  # Normal blue instead of bright
    REQUEST = "yellow"  # Normal yellow instead of bright
    NOTIFICATION = "cyan"  # Normal cyan instead of bright
    PING = "dim green"  # Keep ping dim
    IDLE = "white dim"
    NONE = "dim"

    # Channel arrow states
    ARROW_ERROR = "bright_red"
    ARROW_DISABLED = "bright_yellow"  # For explicitly disabled/off
    ARROW_METHOD_NOT_ALLOWED = "cyan"  # For 405 method not allowed (notification color)
    ARROW_OFF = "black dim"
    ARROW_IDLE = "bright_cyan"  # Connected but no activity
    ARROW_ACTIVE = "bright_green"  # Connected with activity

    # Capability token states
    TOKEN_ERROR = "bright_red"
    TOKEN_WARNING = "bright_cyan"
    TOKEN_CAUTION = "bright_yellow"
    TOKEN_DISABLED = "dim"
    TOKEN_HIGHLIGHTED = "bright_yellow"
    TOKEN_ENABLED = "bright_green"

    # MCP capability token states (reverse for visibility across themes)
    CAP_TOKEN_CAUTION = "reverse bright_yellow"
    CAP_TOKEN_HIGHLIGHTED = "reverse bright_yellow"
    CAP_TOKEN_ENABLED = "reverse bright_green"

    # Text elements
    TEXT_DIM = "dim"
    TEXT_DEFAULT = "default"  # Use terminal's default text color
    TEXT_BRIGHT = "bright_white"
    TEXT_ERROR = "bright_red"
    TEXT_WARNING = "bright_yellow"
    TEXT_SUCCESS = "bright_green"
    TEXT_INFO = "bright_blue"
    TEXT_CYAN = "cyan"


# Symbol definitions for timelines and legends
SYMBOL_IDLE = "·"
SYMBOL_ERROR = "●"
SYMBOL_RESPONSE = "▼"
SYMBOL_NOTIFICATION = "●"
SYMBOL_REQUEST = "◆"
SYMBOL_STDIO_ACTIVITY = "●"
SYMBOL_PING = "●"
SYMBOL_DISABLED = "▽"


# Color mappings for different contexts
TIMELINE_COLORS = {
    "error": Colours.ERROR,
    "disabled": Colours.DISABLED,
    "response": Colours.RESPONSE,
    "request": Colours.REQUEST,
    "notification": Colours.NOTIFICATION,
    "ping": Colours.PING,
    "none": Colours.IDLE,
}

TIMELINE_COLORS_STDIO = {
    "error": Colours.ERROR,
    "request": Colours.TOKEN_ENABLED,  # All activity shows as bright green
    "response": Colours.TOKEN_ENABLED,
    "notification": Colours.TOKEN_ENABLED,
    "ping": Colours.PING,
    "none": Colours.IDLE,
}


def _format_compact_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total = int(seconds)
    if total < 1:
        return "<1s"
    mins, secs = divmod(total, 60)
    if mins == 0:
        return f"{secs}s"
    hours, mins = divmod(mins, 60)
    if hours == 0:
        return f"{mins}m{secs:02d}s"
    days, hours = divmod(hours, 24)
    if days == 0:
        return f"{hours}h{mins:02d}m"
    return f"{days}d{hours:02d}h"


def _format_timeline_label(total_seconds: int) -> str:
    total = max(0, int(total_seconds))
    if total == 0:
        return "0s"

    days, remainder = divmod(total, 86400)
    if days:
        if remainder == 0:
            return f"{days}d"
        hours = remainder // 3600
        if hours == 0:
            return f"{days}d"
        return f"{days}d{hours}h"

    hours, remainder = divmod(total, 3600)
    if hours:
        if remainder == 0:
            return f"{hours}h"
        minutes = remainder // 60
        if minutes == 0:
            return f"{hours}h"
        return f"{hours}h{minutes:02d}m"

    minutes, seconds = divmod(total, 60)
    if minutes:
        if seconds == 0:
            return f"{minutes}m"
        return f"{minutes}m{seconds:02d}s"

    return f"{seconds}s"


def _summarise_call_counts(call_counts: dict[str, int]) -> str | None:
    if not call_counts:
        return None
    ordered = sorted(call_counts.items(), key=lambda item: item[0])
    return ", ".join(f"{name}:{count}" for name, count in ordered)


def _format_session_id(session_id: str | None) -> Text:
    text = Text()
    if not session_id:
        text.append("None", style="yellow")
        return text
    if session_id == "local":
        text.append("local", style="cyan")
        return text

    value = _truncate_middle(session_id, max_length=24, edge_length=10)
    text.append(value, style="green")
    return text


def _truncate_middle(value: str, *, max_length: int, edge_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:edge_length]}...{value[-edge_length:]}"


def _cookie_string_field(cookie: dict[str, object] | None, key: str) -> str | None:
    if not isinstance(cookie, dict):
        return None
    raw_value = cookie.get(key)
    if not isinstance(raw_value, str):
        return None
    value = raw_value.strip()
    return value or None


def _cookie_timestamp_field(cookie: dict[str, object] | None, *keys: str) -> str | None:
    if not isinstance(cookie, dict):
        return None

    for key in keys:
        value = _cookie_string_field(cookie, key)
        if value:
            return value

    data = cookie.get("data")
    if isinstance(data, dict):
        key_set = set(keys)
        for data_key, raw_value in data.items():
            if data_key not in key_set or not isinstance(raw_value, str):
                continue
            if raw_value.strip():
                return raw_value.strip()

    return None


def _format_cookie_timestamp_local(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return _truncate_middle(timestamp, max_length=32, edge_length=14)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone()

    return parsed.strftime("%d/%m/%y %H:%M")


def _format_experimental_session_status(status: ServerStatus) -> Text:
    text = Text()
    supported = status.experimental_session_supported
    if supported is True:
        cookie_id = _cookie_string_field(status.session_cookie, "sessionId")
        created = _cookie_timestamp_field(
            status.session_cookie,
            "created",
            "created_at",
            "createdAt",
        )
        expiry = _cookie_timestamp_field(
            status.session_cookie,
            "expiry",
            "expires",
            "expires_at",
            "expiresAt",
        )

        if cookie_id:
            text.append(
                _truncate_middle(cookie_id, max_length=32, edge_length=14),
                style=Colours.TEXT_SUCCESS,
            )
        else:
            text.append("none", style=Colours.TEXT_DIM)

        if created or expiry:
            text.append(" (", style=Colours.TEXT_DIM)
            if created:
                text.append(_format_cookie_timestamp_local(created), style=Colours.TEXT_DEFAULT)
            if created and expiry:
                text.append(" → ", style=Colours.TEXT_DIM)
            if expiry:
                text.append(_format_cookie_timestamp_local(expiry), style=Colours.TEXT_DEFAULT)
            text.append(")", style=Colours.TEXT_DIM)
        else:
            text.append(" (unknown)", style=Colours.TEXT_DIM)
        return text

    if supported is False:
        text.append("not advertised", style=Colours.TEXT_DIM)
        return text

    text.append("unknown", style=Colours.TEXT_DIM)
    return text

def _build_aligned_field(
    label: str, value: Text | str, *, label_width: int = 9, value_style: str = Colours.TEXT_DEFAULT
) -> Text:
    field = Text()
    field.append(f"{label:<{label_width}}: ", style="dim")
    if isinstance(value, Text):
        field.append_text(value)
    else:
        field.append(value, style=value_style)
    return field


def _instruction_capability_state(
    status: ServerStatus,
    *,
    template_expected: bool,
) -> CapabilityState:
    if not status.instructions_available:
        return False
    if status.instructions_enabled is False:
        return "red"
    if status.instructions_enabled is None and not template_expected:
        return "warn"
    if status.instructions_enabled is None:
        return True
    if template_expected:
        return True
    return "blue"


def _skybridge_capability_state(status: ServerStatus) -> CapabilityState:
    skybridge_config = status.skybridge
    if not skybridge_config:
        return False
    if skybridge_config.warnings:
        return "warn"
    if skybridge_config.enabled:
        return True
    return False


def _elicitation_capability_state(mode: str | None) -> CapabilityState:
    normalized_mode = (mode or "").lower()
    if normalized_mode == "auto-cancel":
        return "red"
    if normalized_mode and normalized_mode != "none":
        return True
    return False


def _sampling_capability_state(mode: str | None) -> CapabilityState:
    normalized_mode = (mode or "").lower()
    if normalized_mode == "configured":
        return "blue"
    if normalized_mode == "auto":
        return True
    return False


def _capability_token_style(supported: CapabilityState, highlighted: bool) -> str:
    if supported == "red":
        return Colours.TOKEN_ERROR
    if supported == "blue":
        return Colours.TOKEN_WARNING
    if supported == "warn":
        return Colours.CAP_TOKEN_CAUTION
    if not supported:
        return Colours.TOKEN_DISABLED
    if highlighted:
        return Colours.CAP_TOKEN_HIGHLIGHTED
    return Colours.CAP_TOKEN_ENABLED


def _format_capability_shorthand(
    status: ServerStatus, template_expected: bool
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    caps = status.server_capabilities
    tools = caps.tools if caps else None
    prompts = caps.prompts if caps else None
    resources = caps.resources if caps else None
    logging_caps = caps.logging if caps else None
    completion_caps = caps.completions if caps else None
    experimental_caps = caps.experimental if caps else None

    entries: list[tuple[str, CapabilityState, bool]] = [
        ("To", bool(tools), bool(tools and tools.listChanged)),
        ("Pr", bool(prompts), bool(prompts and prompts.listChanged)),
        (
            "Re",
            bool(resources),
            bool(resources and resources.listChanged),
        ),
        ("Rs", bool(resources and resources.subscribe), bool(resources and resources.subscribe)),
        ("Lo", bool(logging_caps), False),
        ("Co", bool(completion_caps), False),
        ("Ex", bool(experimental_caps), False),
        ("In", _instruction_capability_state(status, template_expected=template_expected), False),
        ("Sk", _skybridge_capability_state(status), False),
        ("Ro", bool(status.roots_configured), False),
        ("El", _elicitation_capability_state(status.elicitation_mode), False),
        ("Sa", _sampling_capability_state(status.sampling_mode), False),
        ("Sp", bool(status.spoofing_enabled), False),
    ]

    tokens = [
        (_label, _capability_token_style(supported, highlighted))
        for _label, supported, highlighted in entries
    ]
    return tokens[:8], tokens[8:]


def _build_capability_text(tokens: list[tuple[str, str]]) -> Text:
    line = Text()
    host_boundary_inserted = False
    for idx, (label, style) in enumerate(tokens):
        if idx:
            line.append(" ")
        if not host_boundary_inserted and label == "Ro":
            line.append("• ", style="dim")
            host_boundary_inserted = True
        line.append(label, style=style)
    return line


def _format_relative_time(dt: datetime | None) -> str:
    if dt is None:
        return "never"
    now = datetime.now(timezone.utc)
    seconds = max(0, (now - dt).total_seconds())
    return _format_compact_duration(seconds) or "<1s"


def _truncate_detail(value: str, max_len: int = 48) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _build_health_text(status: ServerStatus) -> Text | None:
    interval = status.ping_interval_seconds
    if interval is None:
        return None

    health = Text()
    state_label, state_style = _get_health_state(status)
    if interval <= 0:
        health.append(state_label, style=state_style)
        return health

    max_missed = status.ping_max_missed or 0
    misses = _compute_display_misses(status)

    health.append(state_label, style=state_style)
    health.append(f" | interval: {interval}s", style=Colours.TEXT_DIM)

    misses_text = f"{misses}/{max_missed}" if max_missed else str(misses)
    misses_style = Colours.TEXT_WARNING if misses > 0 else Colours.TEXT_DIM
    health.append(f" | misses: {misses_text}", style=misses_style)

    last_ok = _format_relative_time(status.ping_last_ok_at)
    health.append(f" | last ok: {last_ok}", style=Colours.TEXT_DIM)

    if misses > 0:
        last_fail = _format_relative_time(status.ping_last_fail_at)
        health.append(f" | last fail: {last_fail}", style=Colours.TEXT_DIM)
        if status.ping_last_error:
            err = _truncate_detail(status.ping_last_error)
            health.append(f" | last err: {err}", style=Colours.TEXT_ERROR)

    return health


def _offline_health_state(status: ServerStatus) -> tuple[str, str] | None:
    if status.is_connected is not False:
        return None
    if status.error_message and "initializing" in status.error_message:
        return ("pending", Colours.TEXT_DIM)
    return ("offline", Colours.TEXT_ERROR)


def _stale_health_state(
    status: ServerStatus,
    *,
    interval: int,
    max_missed: int,
) -> tuple[str, str] | None:
    last_ping_at = status.ping_last_ok_at or status.ping_last_fail_at
    if last_ping_at is None or max_missed <= 0:
        return None
    if last_ping_at.tzinfo is None:
        last_ping_at = last_ping_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if (now - last_ping_at).total_seconds() > interval * max_missed:
        return ("stale", Colours.TEXT_ERROR)
    return None


def _get_health_state(status: ServerStatus) -> tuple[str, str]:
    interval = status.ping_interval_seconds
    if interval is None:
        return ("unknown", Colours.TEXT_DIM)
    if interval <= 0:
        return ("disabled", Colours.TEXT_DIM)

    offline_state = _offline_health_state(status)
    if offline_state is not None:
        return offline_state

    if _has_transport_error(status):
        return ("error", Colours.TEXT_ERROR)

    max_missed = status.ping_max_missed or 0
    misses = _compute_display_misses(status)
    has_activity = bool(status.ping_last_ok_at or status.ping_last_fail_at)
    stale_state = _stale_health_state(status, interval=interval, max_missed=max_missed)
    if stale_state is not None:
        return stale_state

    if not has_activity:
        return ("pending", Colours.TEXT_DIM)
    if max_missed and misses >= max_missed:
        return ("failed", Colours.TEXT_ERROR)
    if misses > 0:
        return ("missed", Colours.TEXT_WARNING)
    return ("ok", Colours.TEXT_SUCCESS)


def _has_transport_error(status: ServerStatus) -> bool:
    snapshot = status.transport_channels
    if snapshot is None:
        return False
    channels = [
        snapshot.get,
        snapshot.post_json,
        snapshot.post_sse,
        snapshot.post,
        snapshot.resumption,
        snapshot.stdio,
    ]
    for channel in channels:
        if channel is None:
            continue
        if channel.last_status_code == 405 or channel.state == "disabled":
            continue
        if channel.last_error and "405" in channel.last_error:
            continue
        if channel.state == "error":
            return True
    return False


def _compute_display_misses(status: ServerStatus) -> int:
    interval = status.ping_interval_seconds
    if interval is None or interval <= 0:
        return status.ping_consecutive_failures or 0

    last_ping_at = status.ping_last_ok_at or status.ping_last_fail_at
    if last_ping_at is None:
        return status.ping_consecutive_failures or 0

    if last_ping_at.tzinfo is None:
        last_ping_at = last_ping_at.replace(tzinfo=timezone.utc)

    elapsed = (datetime.now(timezone.utc) - last_ping_at).total_seconds()
    if elapsed <= 0:
        return status.ping_consecutive_failures or 0

    derived = int(elapsed // interval)
    recorded = status.ping_consecutive_failures or 0
    return max(recorded, derived)


def _get_ping_attempts(status: ServerStatus) -> int:
    ok = status.ping_ok_count or 0
    fail = status.ping_fail_count or 0
    return ok + fail


def _format_label(label: str, width: int = 10) -> str:
    return f"{label:<{width}}" if len(label) < width else label


def _build_inline_timeline(
    buckets: Iterable[str],
    *,
    bucket_seconds: int | None = None,
    bucket_count: int | None = None,
) -> str:
    """Build a compact timeline string for inline display."""
    bucket_list = list(buckets)
    count = bucket_count or len(bucket_list)
    if count <= 0:
        count = len(bucket_list) or 1

    seconds = bucket_seconds or 30
    total_window = seconds * count
    timeline = f"  [dim]{_format_timeline_label(total_window)}[/dim] "

    if len(bucket_list) < count:
        bucket_list.extend(["none"] * (count - len(bucket_list)))
    elif len(bucket_list) > count:
        bucket_list = bucket_list[-count:]

    for state in bucket_list:
        color = TIMELINE_COLORS.get(state, Colours.NONE)
        symbol = _timeline_symbol_for_state(state)
        timeline += f"[bold {color}]{symbol}[/bold {color}]"
    timeline += " [dim]now[/dim]"
    return timeline


def _timeline_symbol_for_state(state: str, *, is_stdio: bool = False) -> str:
    if state in {"idle", "none"}:
        return SYMBOL_IDLE
    if state == "error":
        return SYMBOL_ERROR
    if state == "ping":
        return SYMBOL_PING
    if state == "disabled":
        return SYMBOL_DISABLED
    if is_stdio:
        return SYMBOL_STDIO_ACTIVITY
    if state == "request":
        return SYMBOL_REQUEST
    if state == "notification":
        return SYMBOL_NOTIFICATION
    return SYMBOL_RESPONSE


def _timeline_color_map(*, is_stdio: bool) -> dict[str, str]:
    return TIMELINE_COLORS_STDIO if is_stdio else TIMELINE_COLORS


def _normalise_timeline_states(
    bucket_states: Iterable[str] | None,
    bucket_count: int,
) -> list[str]:
    states = list(bucket_states or [])
    if len(states) < bucket_count:
        states.extend(["none"] * (bucket_count - len(states)))
    elif len(states) > bucket_count:
        states = states[-bucket_count:]
    return states


def _build_channel_entries(status: ServerStatus) -> list[_ChannelSummaryEntry]:
    snapshot = status.transport_channels
    if snapshot is None:
        return []

    transport_lower = (status.transport or "").lower()
    http_channels = [snapshot.get, snapshot.post_sse, snapshot.post_json]
    stdio_channel = snapshot.stdio

    if any(channel is not None for channel in http_channels):
        entries = [
            _ChannelSummaryEntry("GET (SSE)", "◀", snapshot.get),
            _ChannelSummaryEntry("POST (SSE)", "▶", snapshot.post_sse),
        ]
        if transport_lower != "sse":
            entries.append(_ChannelSummaryEntry("POST (JSON)", "▶", snapshot.post_json))
        return entries

    if stdio_channel is None:
        return []

    return [_ChannelSummaryEntry("STDIO", "⇄", stdio_channel)]


def _get_channel_health_insert_label(
    status: ServerStatus,
    entries: list[_ChannelSummaryEntry],
) -> str | None:
    if status.ping_interval_seconds is None:
        return None

    label_names = [entry.label for entry in entries]
    if "POST (JSON)" in label_names:
        return "POST (JSON)"
    if "POST (SSE)" in label_names:
        return "POST (SSE)"
    if "STDIO" in label_names:
        return "STDIO"
    if label_names:
        return label_names[-1]
    return None


def _build_channel_summary_layout(
    status: ServerStatus,
    entries: list[_ChannelSummaryEntry],
) -> _ChannelSummaryLayout:
    snapshot = status.transport_channels
    default_bucket_seconds = snapshot.activity_bucket_seconds if snapshot else None
    default_bucket_count = snapshot.activity_bucket_count if snapshot else None
    default_bucket_seconds = default_bucket_seconds or 30
    default_bucket_count = default_bucket_count or 20
    timeline_header_label = _format_timeline_label(default_bucket_seconds * default_bucket_count)
    metrics_prefix_width = 22 + len(timeline_header_label) + default_bucket_count
    transport = status.transport or "unknown"
    transport_display = transport.upper() if transport != "unknown" else "Channels"
    is_stdio = len(entries) == 1 and entries[0].label == "STDIO"
    return _ChannelSummaryLayout(
        transport_display=transport_display,
        default_bucket_seconds=default_bucket_seconds,
        default_bucket_count=default_bucket_count,
        metrics_prefix_width=metrics_prefix_width,
        is_stdio=is_stdio,
        health_insert_label=_get_channel_health_insert_label(status, entries),
    )


def _render_channel_summary_header(indent: str, layout: _ChannelSummaryLayout) -> None:
    console.console.print()

    header = Text(indent)
    header_intro = f"┌ {layout.transport_display} "
    header.append(header_intro, style="dim")
    dash_count = max(1, layout.metrics_prefix_width - len(header_intro) + 2)
    header.append("─" * dash_count, style="dim")
    header.append(
        "  activity" if layout.is_stdio else "  req  resp notif  ping",
        style="dim",
    )
    console.console.print(header)

    empty_header = Text(indent)
    empty_header.append("│", style="dim")
    console.console.print(empty_header)


def _channel_arrow_style(channel: ChannelSnapshot | None) -> str:
    if channel is None:
        return Colours.ARROW_OFF

    state = (channel.state or "open").lower()
    if channel.last_status_code == 405:
        return Colours.ARROW_METHOD_NOT_ALLOWED
    if state == "error":
        return Colours.ARROW_ERROR
    if state in {"off", "disabled"}:
        return Colours.ARROW_OFF
    if channel.request_count == 0 and channel.response_count == 0:
        return Colours.ARROW_IDLE
    if state in {"open", "connected"}:
        return Colours.ARROW_ACTIVE
    return Colours.ARROW_IDLE


def _display_channel_arrow(arrow: str, channel: ChannelSnapshot | None) -> str:
    if channel is None or channel.last_status_code != 405:
        return arrow
    return {"◀": "◁", "▶": "▷", "⇄": "⇄"}.get(arrow, arrow)


def _channel_error_entry(
    label: str,
    channel: ChannelSnapshot | None,
) -> tuple[str, str] | None:
    if channel is None:
        return None
    if (channel.state or "").lower() != "error" or channel.last_status_code == 405:
        return None
    if not channel.last_error:
        return None

    error_message = channel.last_error
    if channel.last_status_code:
        error_message = f"{error_message} ({channel.last_status_code})"
    return label.split()[0], error_message


def _channel_label_style(
    label: str,
    channel: ChannelSnapshot | None,
    arrow_style: str,
) -> str:
    if channel is None:
        return Colours.TEXT_DIM
    if channel.last_status_code == 405 and "GET" in label:
        return Colours.TEXT_DIM
    if arrow_style == Colours.ARROW_ERROR and "GET" in label:
        return Colours.TEXT_ERROR
    if (
        channel.request_count == 0
        and channel.response_count == 0
        and channel.notification_count == 0
        and (channel.ping_count or 0) == 0
    ):
        return Colours.TEXT_DIM
    return Colours.TEXT_DEFAULT


def _append_channel_timeline(
    line: Text,
    channel: ChannelSnapshot | None,
    *,
    layout: _ChannelSummaryLayout,
) -> None:
    channel_bucket_seconds = (
        channel.activity_bucket_seconds if channel else None
    ) or layout.default_bucket_seconds
    bucket_count = (
        len(channel.activity_buckets)
        if channel is not None and channel.activity_buckets
        else channel.activity_bucket_count if channel else None
    )
    if not bucket_count or bucket_count <= 0:
        bucket_count = layout.default_bucket_count

    line.append(
        f"{_format_timeline_label(channel_bucket_seconds * bucket_count)} ",
        style="dim",
    )

    bucket_states = channel.activity_buckets if channel is not None and channel.activity_buckets else []
    if bucket_states:
        color_map = _timeline_color_map(is_stdio=layout.is_stdio)
        for bucket_state in bucket_states:
            color = color_map.get(bucket_state, "dim")
            symbol = _timeline_symbol_for_state(bucket_state, is_stdio=layout.is_stdio)
            line.append(symbol, style=f"bold {color}")
    else:
        for _ in range(bucket_count):
            line.append(SYMBOL_IDLE, style="black dim")

    line.append(" now", style="dim")


def _append_channel_metrics(
    line: Text,
    channel: ChannelSnapshot | None,
    *,
    is_stdio: bool,
) -> None:
    if is_stdio:
        if channel is not None and channel.message_count > 0:
            activity = str(channel.message_count).rjust(8)
            activity_style = Colours.TEXT_DEFAULT
        else:
            activity = "-".rjust(8)
            activity_style = Colours.TEXT_DIM
        line.append(f"  {activity}", style=activity_style)
        return

    if channel is None:
        req = resp = notif = ping = "-".rjust(5)
        metrics_style = Colours.TEXT_DIM
    else:
        channel_state = (channel.state or "open").lower()
        is_shut = channel.last_status_code == 405 or channel_state in {"off", "disabled"}
        if is_shut:
            req = resp = notif = ping = "-".rjust(5)
            metrics_style = Colours.TEXT_DIM
        else:
            req = str(channel.request_count).rjust(5)
            resp = str(channel.response_count).rjust(5)
            notif = str(channel.notification_count).rjust(5)
            ping = str(channel.ping_count).rjust(5) if channel.ping_count else "-".rjust(5)
            metrics_style = Colours.TEXT_DEFAULT

    if metrics_style == Colours.TEXT_DIM:
        line.append(f"  {req} {resp} {notif} {ping}", style=metrics_style)
        return

    ping_style = Colours.TEXT_DEFAULT if channel is not None and channel.ping_count else Colours.TEXT_DIM
    line.append("  ", style="dim")
    line.append(req, style=metrics_style)
    line.append(" ", style="dim")
    line.append(resp, style=metrics_style)
    line.append(" ", style="dim")
    line.append(notif, style=metrics_style)
    line.append(" ", style="dim")
    line.append(ping, style=ping_style)


def _render_channel_health_row(
    status: ServerStatus,
    indent: str,
    *,
    layout: _ChannelSummaryLayout,
) -> None:
    line = Text(indent)
    line.append("│ ", style="dim")
    _, state_style = _get_health_state(status)
    line.append(SYMBOL_PING, style=state_style)
    line.append(f" {'HEALTH':<13}", style=state_style)

    bucket_seconds = status.ping_activity_bucket_seconds or layout.default_bucket_seconds
    bucket_count = status.ping_activity_bucket_count or layout.default_bucket_count
    line.append(f"{_format_timeline_label(bucket_seconds * bucket_count)} ", style="dim")

    color_map = _timeline_color_map(is_stdio=layout.is_stdio)
    for bucket_state in _normalise_timeline_states(status.ping_activity_buckets, bucket_count):
        color = color_map.get(bucket_state, "dim")
        symbol = _timeline_symbol_for_state(bucket_state, is_stdio=layout.is_stdio)
        line.append(symbol, style=f"bold {color}")

    line.append(" now", style="dim")
    ping_attempts = _get_ping_attempts(status)
    if layout.is_stdio:
        activity = str(ping_attempts).rjust(8) if ping_attempts > 0 else "-".rjust(8)
        activity_style = Colours.TEXT_DEFAULT if ping_attempts > 0 else Colours.TEXT_DIM
        line.append(f"  {activity}", style=activity_style)
    else:
        line.append("  ", style="dim")
        line.append("-".rjust(5), style=Colours.TEXT_DIM)
        line.append(" ", style="dim")
        line.append("-".rjust(5), style=Colours.TEXT_DIM)
        line.append(" ", style="dim")
        line.append("-".rjust(5), style=Colours.TEXT_DIM)
        line.append(" ", style="dim")
        ping = str(ping_attempts).rjust(5) if ping_attempts > 0 else "-".rjust(5)
        ping_style = Colours.TEXT_DEFAULT if ping_attempts > 0 else Colours.TEXT_DIM
        line.append(ping, style=ping_style)

    console.console.print(line)


def _render_single_channel_row(
    entry: _ChannelSummaryEntry,
    indent: str,
    *,
    layout: _ChannelSummaryLayout,
) -> tuple[str, str] | None:
    line = Text(indent)
    line.append("│ ", style="dim")

    arrow_style = _channel_arrow_style(entry.channel)
    line.append(_display_channel_arrow(entry.arrow, entry.channel), style=arrow_style)
    line.append(
        f" {entry.label:<13}",
        style=_channel_label_style(entry.label, entry.channel, arrow_style),
    )

    _append_channel_timeline(line, entry.channel, layout=layout)
    _append_channel_metrics(line, entry.channel, is_stdio=layout.is_stdio)
    console.console.print(line)
    return _channel_error_entry(entry.label, entry.channel)


def _render_channel_errors(errors: list[tuple[str, str]], indent: str) -> None:
    if not errors:
        return

    empty_line = Text(indent)
    empty_line.append("│", style="dim")
    console.console.print(empty_line)

    for channel_type, error_message in errors:
        error_line = Text(indent)
        error_line.append("│ ", style=Colours.TEXT_DIM)
        error_line.append("▲ ", style=Colours.TEXT_WARNING)
        error_line.append(f"{channel_type}: ", style=Colours.TEXT_DEFAULT)
        error_line.append(_truncate_detail(error_message, max_len=60), style=Colours.TEXT_ERROR)
        console.console.print(error_line)


def _render_channel_footer(
    entries: list[_ChannelSummaryEntry],
    indent: str,
    *,
    is_stdio: bool,
) -> None:
    has_timelines = any(entry.channel is not None and entry.channel.activity_buckets for entry in entries)
    if has_timelines:
        empty_before = Text(indent)
        empty_before.append("│", style="dim")
        console.console.print(empty_before)

    footer = Text(indent)
    footer.append("└", style="dim")
    if has_timelines:
        footer.append(" legend: ", style="dim")
        if is_stdio:
            legend_map = [
                ("activity", f"bold {Colours.TOKEN_ENABLED}"),
                ("idle", Colours.IDLE),
            ]
        else:
            legend_map = [
                ("error", f"bold {Colours.ERROR}"),
                ("response", f"bold {Colours.RESPONSE}"),
                ("request", f"bold {Colours.REQUEST}"),
                ("notification", f"bold {Colours.NOTIFICATION}"),
                ("ping", Colours.PING),
                ("idle", Colours.IDLE),
            ]

        for index, (name, color) in enumerate(legend_map):
            if index > 0:
                footer.append(" ", style="dim")
            symbol = SYMBOL_STDIO_ACTIVITY if is_stdio and name == "activity" else _timeline_symbol_for_state(name, is_stdio=is_stdio)
            footer.append(symbol, style=color)
            footer.append(f" {name}", style="dim")

    console.console.print(footer)


def _render_channel_summary(status: ServerStatus, indent: str, total_width: int) -> None:
    del total_width

    entries = _build_channel_entries(status)
    if not any(entry.channel is not None for entry in entries):
        return

    layout = _build_channel_summary_layout(status, entries)
    _render_channel_summary_header(indent, layout)

    errors: list[tuple[str, str]] = []
    health_inserted = False
    for entry in entries:
        error = _render_single_channel_row(entry, indent, layout=layout)
        if error is not None:
            errors.append(error)

        if layout.health_insert_label == entry.label and not health_inserted:
            _render_channel_health_row(status, indent, layout=layout)
            health_inserted = True

    _render_channel_errors(errors, indent)
    _render_channel_footer(entries, indent, is_stdio=layout.is_stdio)
    console.console.print()


async def _load_server_status_map(agent: object) -> dict[str, ServerStatus]:
    if not isinstance(agent, _ServerStatusProvider):
        return {}

    try:
        status_map = await agent.get_server_status()
    except Exception:
        return {}

    return status_map if isinstance(status_map, dict) else {}


def _template_expects_server_instructions(agent: object) -> bool:
    if not isinstance(agent, _HasConfig):
        return False

    config = agent.config
    if config is None or not isinstance(config, _ConfigWithInstruction):
        return False
    return "{{serverInstructions}}" in str(config.instruction or "")


def _console_width() -> int:
    try:
        return console.console.size.width
    except Exception:
        return 80


def _render_mcp_status_header(label: Text, total_width: int, right: Text | None = None) -> None:
    line = Text()
    line.append_text(label)
    line.append(" ")

    separator_width = total_width - line.cell_len
    if right is not None and right.cell_len > 0:
        separator_width -= right.cell_len
        separator_width = max(1, separator_width)
        line.append("─" * separator_width, style="dim")
        line.append_text(right)
    else:
        line.append("─" * max(1, separator_width), style="dim")

    console.console.print()
    console.console.print(line)
    console.console.print()


def _render_server_header(server: str, index: int, *, indent: str, total_width: int) -> None:
    header_label = Text(indent)
    header_label.append("▎", style=Colours.TEXT_CYAN)
    header_label.append(SYMBOL_RESPONSE, style=f"dim {Colours.TEXT_CYAN}")
    header_label.append(f" [{index:2}] ", style=Colours.TEXT_CYAN)
    header_label.append(server, style=f"{Colours.TEXT_INFO} bold")
    _render_mcp_status_header(header_label, total_width)


def _build_client_display(status: ServerStatus) -> str:
    client_parts: list[str] = []
    if status.client_info_name:
        client_parts.append(status.client_info_name)
    if status.client_info_version:
        client_parts.append(status.client_info_version)
    return _truncate_detail(" ".join(client_parts), max_len=24)


def _render_server_metadata(status: ServerStatus, *, indent: str) -> None:
    meta_line = Text(indent + "  ")
    meta_fields = [
        _build_aligned_field(
            "name",
            _truncate_detail(status.implementation_name or status.server_name or "unknown", max_len=30),
        )
    ]

    version_display = status.implementation_version or ""
    if version_display:
        meta_fields.append(_build_aligned_field("version", _truncate_detail(version_display, max_len=12)))

    for index, field in enumerate(meta_fields):
        if index:
            meta_line.append("  ", style="dim")
        meta_line.append_text(field)

    client_display = _build_client_display(status)
    if client_display:
        meta_line.append(" | ", style="dim")
        meta_line.append_text(_build_aligned_field("client", client_display))

    console.console.print(meta_line)

    session_line = Text(indent + "  ")
    session_line.append_text(_build_aligned_field("session", _format_session_id(status.session_id)))
    console.console.print(session_line)

    experimental_session_line = Text(indent + "  ")
    experimental_session_line.append_text(
        _build_aligned_field("sessions", _format_experimental_session_status(status))
    )
    console.console.print(experimental_session_line)

    health_text = _build_health_text(status)
    if health_text is not None:
        health_line = Text(indent + "  ")
        health_line.append_text(_build_aligned_field("health", health_text))
        console.console.print(health_line)

    console.console.print()


def _build_server_state_segments(
    status: ServerStatus,
    *,
    template_expected: bool,
) -> list[Text]:
    state_segments: list[Text] = []

    duration = _format_compact_duration(status.staleness_seconds)
    if duration:
        last_text = Text("last activity: ", style=Colours.TEXT_DIM)
        last_text.append(duration, style=Colours.TEXT_DEFAULT)
        last_text.append(" ago", style=Colours.TEXT_DIM)
        state_segments.append(last_text)

    if status.error_message and status.is_connected is False:
        state_segments.append(Text(status.error_message, style=Colours.TEXT_ERROR))

    instructions_available = bool(status.instructions_available)
    if instructions_available and status.instructions_enabled is False:
        state_segments.append(Text("instructions disabled", style=Colours.TEXT_ERROR))
    elif instructions_available and not template_expected:
        state_segments.append(Text("instr. not in sysprompt", style=Colours.TEXT_WARNING))

    if status.spoofing_enabled:
        state_segments.append(Text("client spoof", style=Colours.TEXT_WARNING))

    return state_segments


def _render_server_state(status: ServerStatus, *, indent: str, template_expected: bool) -> None:
    state_segments = _build_server_state_segments(status, template_expected=template_expected)
    if not state_segments:
        return

    status_line = Text(indent + "  ")
    for index, segment in enumerate(state_segments):
        if index:
            status_line.append("  |  ", style="dim")
        status_line.append_text(segment)
    console.console.print(status_line)


def _render_server_calls(status: ServerStatus, *, indent: str) -> None:
    calls = _summarise_call_counts(status.call_counts)
    if calls:
        calls_line = Text(indent + "  ")
        calls_line.append("mcp calls: ", style=Colours.TEXT_DIM)
        calls_line.append(calls, style=Colours.TEXT_DEFAULT)
        if status.reconnect_count > 0:
            calls_line.append("  |  ", style="dim")
            calls_line.append("reconnects: ", style=Colours.TEXT_DIM)
            calls_line.append(str(status.reconnect_count), style=Colours.TEXT_WARNING)
        console.console.print(calls_line)
        return

    if status.reconnect_count > 0:
        reconnect_line = Text(indent + "  ")
        reconnect_line.append("reconnects: ", style=Colours.TEXT_DIM)
        reconnect_line.append(str(status.reconnect_count), style=Colours.TEXT_WARNING)
        console.console.print(reconnect_line)


def _render_capability_banner(
    tokens: list[tuple[str, str]],
    *,
    indent: str,
    total_width: int,
) -> None:
    prefix = Text(indent)
    prefix.append("─| ", style="dim")
    suffix = Text(" |", style="dim")

    caps_content = _build_capability_text(tokens) if tokens else Text("none", style="dim")
    caps_display = caps_content.copy()
    available = max(0, total_width - prefix.cell_len - suffix.cell_len)
    if caps_display.cell_len > available:
        caps_display.truncate(available)

    banner_line = Text()
    banner_line.append_text(prefix)
    banner_line.append_text(caps_display)
    banner_line.append_text(suffix)
    remaining = total_width - banner_line.cell_len
    if remaining > 0:
        banner_line.append("─" * remaining, style="dim")

    console.console.print(banner_line)


def _render_server_status_block(
    server: str,
    status: ServerStatus,
    *,
    index: int,
    total_count: int,
    indent: str,
    total_width: int,
    template_expected: bool,
) -> None:
    primary_caps, secondary_caps = _format_capability_shorthand(status, template_expected)
    _render_server_header(server, index, indent=indent, total_width=total_width)
    _render_server_metadata(status, indent=indent)
    _render_server_state(status, indent=indent, template_expected=template_expected)
    _render_server_calls(status, indent=indent)
    _render_channel_summary(status, indent, total_width)
    _render_capability_banner(
        primary_caps + secondary_caps,
        indent=indent,
        total_width=total_width,
    )

    if index != total_count:
        console.console.print()


async def render_mcp_status(agent, indent: str = "") -> None:
    server_status_map = await _load_server_status_map(agent)
    if not server_status_map:
        console.console.print(f"{indent}[dim]•[/dim] [dim]No MCP status available[/dim]")
        return

    template_expected = _template_expects_server_instructions(agent)
    total_width = _console_width()
    server_items = list(sorted(server_status_map.items()))

    for index, (server, status) in enumerate(server_items, start=1):
        _render_server_status_block(
            server,
            status,
            index=index,
            total_count=len(server_items),
            indent=indent,
            total_width=total_width,
            template_expected=template_expected,
        )

    console.console.print()
