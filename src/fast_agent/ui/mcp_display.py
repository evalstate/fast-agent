"""Rendering helpers for MCP status information in the enhanced prompt UI."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from rich.console import Console
from rich.text import Text

from fast_agent.mcp.transport_tracking import ChannelSnapshot
from fast_agent.ui import console
from fast_agent.utils.text import strip_casefold
from fast_agent.utils.time import format_compact_duration, format_two_unit_duration

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import ServerStatus


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

_STATUS_CONSOLE: ContextVar[Console | None] = ContextVar("_STATUS_CONSOLE", default=None)


def _status_console() -> Console:
    return _STATUS_CONSOLE.get() or console.console


_ELICITATION_MODE_STATES: dict[str, CapabilityState] = {
    "auto-cancel": "red",
    "none": False,
}
_SAMPLING_MODE_STATES: dict[str, CapabilityState] = {
    "auto": True,
    "configured": "blue",
}


@dataclass(frozen=True, slots=True)
class _ChannelSummaryEntry:
    label: str
    arrow: str
    channel: ChannelSnapshot | None


@dataclass(frozen=True, slots=True)
class _ChannelErrorEntry:
    label: str
    message: str


@dataclass(frozen=True, slots=True)
class _HealthState:
    label: str
    style: str


@dataclass(frozen=True, slots=True)
class _ChannelSummaryLayout:
    transport_display: str
    default_bucket_seconds: int
    default_bucket_count: int
    metrics_prefix_width: int
    is_stdio: bool
    show_ping: bool


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
    ARROW_METHOD_NOT_ALLOWED = "cyan"  # For 405 method not allowed (notification color)
    ARROW_OFF = "black dim"
    ARROW_IDLE = "bright_cyan"  # Connected but no activity
    ARROW_ACTIVE = "bright_green"  # Connected with activity

    # Capability token states
    TOKEN_ERROR = "bright_red"
    TOKEN_WARNING = "bright_cyan"
    TOKEN_DISABLED = "dim"
    TOKEN_ENABLED = "bright_green"

    # MCP capability token states (reverse for visibility across themes)
    CAP_TOKEN_CAUTION = "reverse bright_yellow"
    CAP_TOKEN_HIGHLIGHTED = "reverse bright_yellow"
    CAP_TOKEN_ENABLED = "reverse bright_green"

    # Text elements
    TEXT_DIM = "dim"
    TEXT_DEFAULT = "default"  # Use terminal's default text color
    TEXT_ERROR = "bright_red"
    TEXT_WARNING = "bright_yellow"
    TEXT_SUCCESS = "bright_green"
    TEXT_INFO = "bright_blue"
    TEXT_CYAN = "cyan"


_ARROW_PRE_ACTIVITY_STYLE_BY_STATE = {
    "error": Colours.ARROW_ERROR,
    "off": Colours.ARROW_OFF,
    "disabled": Colours.ARROW_OFF,
}
_ARROW_ACTIVE_STYLE_BY_STATE = {
    "open": Colours.ARROW_ACTIVE,
    "connected": Colours.ARROW_ACTIVE,
}
METHOD_NOT_ALLOWED_STATUS = 405


# Symbol definitions for timelines and legends
SYMBOL_IDLE = "·"
SYMBOL_ERROR = "●"
SYMBOL_RESPONSE = "▼"
SYMBOL_NOTIFICATION = "●"
SYMBOL_REQUEST = "◆"
SYMBOL_STDIO_ACTIVITY = "●"
SYMBOL_PING = "●"
SYMBOL_DISABLED = "▽"

_TIMELINE_BASE_SYMBOLS = {
    "idle": SYMBOL_IDLE,
    "none": SYMBOL_IDLE,
    "error": SYMBOL_ERROR,
    "ping": SYMBOL_PING,
    "disabled": SYMBOL_DISABLED,
}
_TIMELINE_HTTP_SYMBOLS = {
    "request": SYMBOL_REQUEST,
    "notification": SYMBOL_NOTIFICATION,
}


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

_CAPABILITY_STRING_STYLES: dict[Literal["blue", "red", "warn"], str] = {
    "red": Colours.TOKEN_ERROR,
    "blue": Colours.TOKEN_WARNING,
    "warn": Colours.CAP_TOKEN_CAUTION,
}


_format_compact_duration = format_compact_duration
_format_timeline_label = format_two_unit_duration


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


def _app_integration_capability_state(status: ServerStatus) -> CapabilityState:
    app_integration_config = status.app_integration_config
    if not app_integration_config:
        return False
    if app_integration_config.warnings:
        return "warn"
    return bool(app_integration_config.enabled)


def _elicitation_capability_state(mode: str | None) -> CapabilityState:
    normalized_mode = strip_casefold(mode or "")
    return _ELICITATION_MODE_STATES.get(normalized_mode, bool(normalized_mode))


def _sampling_capability_state(mode: str | None) -> CapabilityState:
    normalized_mode = strip_casefold(mode or "")
    return _SAMPLING_MODE_STATES.get(normalized_mode, False)


def _capability_token_style(supported: CapabilityState, highlighted: bool) -> str:
    if isinstance(supported, str):
        return _CAPABILITY_STRING_STYLES[supported]
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
        ("To", bool(tools), bool(tools and tools.list_changed)),
        ("Pr", bool(prompts), bool(prompts and prompts.list_changed)),
        (
            "Re",
            bool(resources),
            bool(resources and resources.list_changed),
        ),
        ("Rs", bool(resources and resources.subscribe), bool(resources and resources.subscribe)),
        ("Lo", bool(logging_caps), False),
        ("Co", bool(completion_caps), False),
        ("Ex", bool(experimental_caps), False),
        ("In", _instruction_capability_state(status, template_expected=template_expected), False),
        ("Sk", bool(status.mcp_skills_enabled), False),
        ("Ui", _app_integration_capability_state(status), False),
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
    if status.protocol_era == "modern":
        return None
    interval = status.ping_interval_seconds
    if interval is None:
        return None

    health = Text()
    state = _get_health_state(status)
    if interval <= 0:
        health.append(state.label, style=state.style)
        return health

    max_missed = status.ping_max_missed or 0
    misses = _compute_display_misses(status)

    health.append(state.label, style=state.style)
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


def _offline_health_state(status: ServerStatus) -> _HealthState | None:
    if status.is_connected is not False:
        return None
    if status.error_message and "initializing" in status.error_message:
        return _HealthState(label="pending", style=Colours.TEXT_DIM)
    return _HealthState(label="offline", style=Colours.TEXT_ERROR)


def _stale_health_state(
    status: ServerStatus,
    *,
    interval: int,
    max_missed: int,
) -> _HealthState | None:
    last_ping_at = _latest_ping_at(status)
    if last_ping_at is None or max_missed <= 0:
        return None
    now = datetime.now(timezone.utc)
    if (now - last_ping_at).total_seconds() > interval * max_missed:
        return _HealthState(label="stale", style=Colours.TEXT_ERROR)
    return None


def _get_health_state(status: ServerStatus) -> _HealthState:
    interval = status.ping_interval_seconds
    if interval is None:
        return _HealthState(label="unknown", style=Colours.TEXT_DIM)
    if interval <= 0:
        return _HealthState(label="disabled", style=Colours.TEXT_DIM)

    offline_state = _offline_health_state(status)
    if offline_state is not None:
        return offline_state

    if _has_transport_error(status):
        return _HealthState(label="error", style=Colours.TEXT_ERROR)

    return _active_ping_health_state(status, interval=interval)


def _active_ping_health_state(status: ServerStatus, *, interval: int) -> _HealthState:
    max_missed = status.ping_max_missed or 0
    misses = _compute_display_misses(status)
    has_activity = bool(status.ping_last_ok_at or status.ping_last_fail_at)
    stale_state = _stale_health_state(status, interval=interval, max_missed=max_missed)
    if stale_state is not None:
        return stale_state

    if not has_activity:
        return _HealthState(label="pending", style=Colours.TEXT_DIM)
    if max_missed and misses >= max_missed:
        return _HealthState(label="failed", style=Colours.TEXT_ERROR)
    if misses > 0:
        return _HealthState(label="missed", style=Colours.TEXT_WARNING)
    return _HealthState(label="ok", style=Colours.TEXT_SUCCESS)


def _has_transport_error(status: ServerStatus) -> bool:
    snapshot = status.transport_channels
    if snapshot is None:
        return False
    channels = [
        snapshot.get,
        snapshot.listen,
        snapshot.post_json,
        snapshot.post_sse,
        snapshot.post,
        snapshot.resumption,
        snapshot.stdio,
    ]
    for channel in channels:
        if channel is None:
            continue
        channel_state = strip_casefold(channel.state or "")
        if _channel_is_method_not_allowed(channel) or channel_state == "disabled":
            continue
        if channel.last_error and str(METHOD_NOT_ALLOWED_STATUS) in channel.last_error:
            continue
        if channel_state == "error":
            return True
    return False


def _compute_display_misses(status: ServerStatus) -> int:
    interval = status.ping_interval_seconds
    if interval is None or interval <= 0:
        return status.ping_consecutive_failures or 0

    last_ping_at = _latest_ping_at(status)
    if last_ping_at is None:
        return status.ping_consecutive_failures or 0

    elapsed = (datetime.now(timezone.utc) - last_ping_at).total_seconds()
    if elapsed <= 0:
        return status.ping_consecutive_failures or 0

    derived = int(elapsed // interval)
    recorded = status.ping_consecutive_failures or 0
    return max(recorded, derived)


def _latest_ping_at(status: ServerStatus) -> datetime | None:
    pings = [
        _utc_datetime_or_none(status.ping_last_ok_at),
        _utc_datetime_or_none(status.ping_last_fail_at),
    ]
    ping_times = [ping_at for ping_at in pings if ping_at is not None]
    if not ping_times:
        return None
    return max(ping_times)


def _utc_datetime_or_none(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timeline_symbol_for_state(state: str, *, is_stdio: bool = False) -> str:
    if state in _TIMELINE_BASE_SYMBOLS:
        return _TIMELINE_BASE_SYMBOLS[state]
    if is_stdio:
        return SYMBOL_STDIO_ACTIVITY
    return _TIMELINE_HTTP_SYMBOLS.get(state, SYMBOL_RESPONSE)


def _timeline_color_map(*, is_stdio: bool) -> dict[str, str]:
    return TIMELINE_COLORS_STDIO if is_stdio else TIMELINE_COLORS


def _build_channel_entries(status: ServerStatus) -> list[_ChannelSummaryEntry]:
    snapshot = status.transport_channels
    if snapshot is None:
        return []

    transport_lower = strip_casefold(status.transport or "")
    entries: list[_ChannelSummaryEntry] = []
    if snapshot.get is not None:
        entries.append(_ChannelSummaryEntry("GET (SSE)", "◀", snapshot.get))
    listen_channel = snapshot.listen
    if (
        listen_channel is None
        and transport_lower == "http"
        and strip_casefold(status.subscription_state or "") == "disabled"
    ):
        listen_channel = ChannelSnapshot(state="disabled")
    if listen_channel is not None:
        entries.append(_ChannelSummaryEntry("LISTEN (SSE)", "◀", listen_channel))

    post_sse_channel = snapshot.post_sse
    post_json_channel = snapshot.post_json
    if post_sse_channel is None and post_json_channel is None and snapshot.post is not None:
        if snapshot.post.mode == "sse":
            post_sse_channel = snapshot.post
        else:
            post_json_channel = snapshot.post

    has_http_channels = any(
        channel is not None
        for channel in (
            snapshot.get,
            snapshot.listen,
            snapshot.post,
            snapshot.post_sse,
            snapshot.post_json,
        )
    )
    if transport_lower == "http" or (transport_lower != "sse" and has_http_channels):
        entries.append(_ChannelSummaryEntry("POST (SSE)", "▶", post_sse_channel))
        entries.append(_ChannelSummaryEntry("POST (JSON)", "▶", post_json_channel))
    elif post_sse_channel is not None:
        entries.append(_ChannelSummaryEntry("POST (SSE)", "▶", post_sse_channel))

    if entries:
        return entries

    if snapshot.stdio is None:
        return []

    return [_ChannelSummaryEntry("STDIO", "⇄", snapshot.stdio)]


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
        show_ping=status.protocol_era != "modern",
    )


def _render_channel_summary_header(indent: str, layout: _ChannelSummaryLayout) -> None:
    _status_console().print()

    header = Text(indent)
    header_intro = f"┌ {layout.transport_display} "
    header.append(header_intro, style="dim")
    dash_count = max(1, layout.metrics_prefix_width - len(header_intro) + 2)
    header.append("─" * dash_count, style="dim")
    if layout.is_stdio:
        metrics_header = "  activity"
    elif layout.show_ping:
        metrics_header = "  req  resp notif  ping"
    else:
        metrics_header = "  req  resp notif"
    header.append(metrics_header, style="dim")
    _status_console().print(header)

    empty_header = Text(indent)
    empty_header.append("│", style="dim")
    _status_console().print(empty_header)


def _channel_arrow_style(channel: ChannelSnapshot | None) -> str:
    if channel is None:
        return Colours.ARROW_OFF

    state = strip_casefold(channel.state or "open")
    if _channel_is_method_not_allowed(channel):
        return Colours.ARROW_METHOD_NOT_ALLOWED
    if pre_activity_style := _ARROW_PRE_ACTIVITY_STYLE_BY_STATE.get(state):
        return pre_activity_style
    if channel.request_count == 0 and channel.response_count == 0:
        return Colours.ARROW_IDLE
    return _ARROW_ACTIVE_STYLE_BY_STATE.get(state, Colours.ARROW_IDLE)


def _channel_is_method_not_allowed(channel: ChannelSnapshot | None) -> bool:
    return channel is not None and channel.last_status_code == METHOD_NOT_ALLOWED_STATUS


def _display_channel_arrow(arrow: str, channel: ChannelSnapshot | None) -> str:
    if channel is None:
        return arrow
    state = strip_casefold(channel.state or "")
    if not _channel_is_method_not_allowed(channel) and state not in {
        "closed",
        "disabled",
        "idle",
        "off",
    }:
        return arrow
    return {"◀": "◁", "▶": "▷", "⇄": "⇄"}.get(arrow, arrow)


def _channel_error_entry(
    label: str,
    channel: ChannelSnapshot | None,
) -> _ChannelErrorEntry | None:
    if channel is None:
        return None
    if strip_casefold(channel.state or "") != "error" or _channel_is_method_not_allowed(channel):
        return None
    if not channel.last_error:
        return None

    error_message = channel.last_error
    if channel.last_status_code:
        error_message = f"{error_message} ({channel.last_status_code})"
    return _ChannelErrorEntry(label=label.split()[0], message=error_message)


def _channel_label_style(
    label: str,
    channel: ChannelSnapshot | None,
    arrow_style: str,
) -> str:
    if channel is None:
        return Colours.TEXT_DIM
    incoming_channel = label.startswith(("GET ", "LISTEN "))
    if incoming_channel and (
        _channel_is_method_not_allowed(channel) or arrow_style == Colours.ARROW_OFF
    ):
        return Colours.TEXT_DIM
    if arrow_style == Colours.ARROW_ERROR and incoming_channel:
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
        else channel.activity_bucket_count
        if channel
        else None
    )
    if not bucket_count or bucket_count <= 0:
        bucket_count = layout.default_bucket_count

    line.append(
        f"{_format_timeline_label(channel_bucket_seconds * bucket_count)} ",
        style="dim",
    )

    bucket_states = (
        channel.activity_buckets if channel is not None and channel.activity_buckets else []
    )
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
    show_ping: bool,
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
        channel_state = strip_casefold(channel.state or "open")
        is_shut = _channel_is_method_not_allowed(channel) or channel_state in {"off", "disabled"}
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
        metrics = f"  {req} {resp} {notif}"
        if show_ping:
            metrics += f" {ping}"
        line.append(metrics, style=metrics_style)
        return

    ping_style = (
        Colours.TEXT_DEFAULT if channel is not None and channel.ping_count else Colours.TEXT_DIM
    )
    line.append("  ", style="dim")
    line.append(req, style=metrics_style)
    line.append(" ", style="dim")
    line.append(resp, style=metrics_style)
    line.append(" ", style="dim")
    line.append(notif, style=metrics_style)
    if show_ping:
        line.append(" ", style="dim")
        line.append(ping, style=ping_style)


def _render_single_channel_row(
    entry: _ChannelSummaryEntry,
    indent: str,
    *,
    layout: _ChannelSummaryLayout,
) -> _ChannelErrorEntry | None:
    line = Text(indent)
    line.append("│ ", style="dim")

    arrow_style = _channel_arrow_style(entry.channel)
    line.append(_display_channel_arrow(entry.arrow, entry.channel), style=arrow_style)
    line.append(
        f" {entry.label:<13}",
        style=_channel_label_style(entry.label, entry.channel, arrow_style),
    )

    _append_channel_timeline(line, entry.channel, layout=layout)
    _append_channel_metrics(
        line,
        entry.channel,
        is_stdio=layout.is_stdio,
        show_ping=layout.show_ping,
    )
    _status_console().print(line)
    return _channel_error_entry(entry.label, entry.channel)


def _render_channel_errors(errors: list[_ChannelErrorEntry], indent: str) -> None:
    if not errors:
        return

    empty_line = Text(indent)
    empty_line.append("│", style="dim")
    _status_console().print(empty_line)

    for error in errors:
        error_line = Text(indent)
        error_line.append("│ ", style=Colours.TEXT_DIM)
        error_line.append("▲ ", style=Colours.TEXT_WARNING)
        error_line.append(f"{error.label}: ", style=Colours.TEXT_DEFAULT)
        error_line.append(_truncate_detail(error.message, max_len=60), style=Colours.TEXT_ERROR)
        _status_console().print(error_line)


def _render_channel_footer(
    entries: list[_ChannelSummaryEntry],
    indent: str,
    *,
    is_stdio: bool,
    show_ping: bool,
) -> None:
    has_timelines = any(
        entry.channel is not None and entry.channel.activity_buckets for entry in entries
    )
    if has_timelines:
        empty_before = Text(indent)
        empty_before.append("│", style="dim")
        _status_console().print(empty_before)

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
            ]
            if show_ping:
                legend_map.append(("ping", Colours.PING))
            legend_map.append(("idle", Colours.IDLE))

        for index, (name, color) in enumerate(legend_map):
            if index > 0:
                footer.append(" ", style="dim")
            symbol = (
                SYMBOL_STDIO_ACTIVITY
                if is_stdio and name == "activity"
                else _timeline_symbol_for_state(name, is_stdio=is_stdio)
            )
            footer.append(symbol, style=color)
            footer.append(f" {name}", style="dim")

    _status_console().print(footer)


def _render_channel_summary(status: ServerStatus, indent: str, total_width: int) -> None:
    del total_width

    entries = _build_channel_entries(status)
    if not entries:
        return

    layout = _build_channel_summary_layout(status, entries)
    _render_channel_summary_header(indent, layout)

    errors: list[_ChannelErrorEntry] = []
    for entry in entries:
        error = _render_single_channel_row(entry, indent, layout=layout)
        if error is not None:
            errors.append(error)

    _render_channel_errors(errors, indent)
    _render_channel_footer(
        entries,
        indent,
        is_stdio=layout.is_stdio,
        show_ping=layout.show_ping,
    )
    _status_console().print()


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
        return _status_console().size.width
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

    _status_console().print()
    _status_console().print(line)
    _status_console().print()


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
            _truncate_detail(
                status.implementation_name or status.server_name or "unknown", max_len=30
            ),
        )
    ]

    version_display = status.implementation_version or ""
    if version_display:
        meta_fields.append(
            _build_aligned_field("version", _truncate_detail(version_display, max_len=12))
        )

    for index, field in enumerate(meta_fields):
        if index:
            meta_line.append("  ", style="dim")
        meta_line.append_text(field)

    client_display = _build_client_display(status)
    if client_display:
        meta_line.append(" | ", style="dim")
        meta_line.append_text(_build_aligned_field("client", client_display))

    _status_console().print(meta_line)

    protocol_line = Text(indent + "  ")
    protocol = status.protocol_version or "unknown"
    if status.protocol_era:
        era = status.protocol_era
        if status.protocol_mode == era:
            era = f"forced {era}"
        protocol += f" ({era})"
    protocol_line.append_text(_build_aligned_field("protocol", protocol))
    if status.protocol_era != "modern":
        protocol_line.append("  ", style="dim")
        protocol_line.append_text(
            _build_aligned_field("session", _format_session_id(status.session_id))
        )
    _status_console().print(protocol_line)

    health_text = _build_health_text(status)
    if health_text is not None:
        health_line = Text(indent + "  ")
        health_line.append_text(_build_aligned_field("health", health_text))
        _status_console().print(health_line)

    _status_console().print()


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
    _status_console().print(status_line)


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
        _status_console().print(calls_line)
        return

    if status.reconnect_count > 0:
        reconnect_line = Text(indent + "  ")
        reconnect_line.append("reconnects: ", style=Colours.TEXT_DIM)
        reconnect_line.append(str(status.reconnect_count), style=Colours.TEXT_WARNING)
        _status_console().print(reconnect_line)


def _render_mcp_skills_hint(server: str, status: ServerStatus, *, indent: str) -> None:
    if not status.mcp_skills_enabled:
        return

    skills_line = Text(indent + "  ")
    skills_line.append(
        f"Skills over MCP are available: use `/skills registry {server}` to select them",
        style=Colours.TEXT_SUCCESS,
    )
    _status_console().print(skills_line)


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

    _status_console().print(banner_line)


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
    _render_mcp_skills_hint(server, status, indent=indent)
    _render_capability_banner(
        primary_caps + secondary_caps,
        indent=indent,
        total_width=total_width,
    )

    if index != total_count:
        _status_console().print()


async def render_mcp_status(
    agent,
    indent: str = "",
    *,
    output_console: Console | None = None,
) -> None:
    token = _STATUS_CONSOLE.set(output_console) if output_console is not None else None
    try:
        server_status_map = await _load_server_status_map(agent)
        if not server_status_map:
            _status_console().print(f"{indent}[dim]•[/dim] [dim]No MCP status available[/dim]")
            return

        template_expected = _template_expects_server_instructions(agent)
        total_width = _console_width()
        server_items = sorted(server_status_map.items())

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

        _status_console().print()
    finally:
        if token is not None:
            _STATUS_CONSOLE.reset(token)


async def render_mcp_status_text(agent, *, width: int = 100) -> str:
    buffer = StringIO()
    output_console = Console(
        file=buffer,
        width=width,
        color_system=None,
        force_terminal=False,
    )
    await render_mcp_status(agent, output_console=output_console)
    return buffer.getvalue().strip()
