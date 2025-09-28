"""Rendering helpers for MCP status information in the enhanced prompt UI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

from rich.text import Text

from fast_agent.ui import console

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import ServerStatus
    from fast_agent.mcp.transport_tracking import ChannelSnapshot


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


def _summarise_call_counts(call_counts: dict[str, int]) -> str | None:
    if not call_counts:
        return None
    ordered = sorted(call_counts.items(), key=lambda item: item[0])
    return ", ".join(f"{name}:{count}" for name, count in ordered)


def _format_session_id(session_id: str | None) -> Text:
    text = Text()
    if not session_id:
        text.append("none", style="yellow")
        return text
    if session_id == "local":
        text.append("local", style="cyan")
        return text

    # Only trim if excessively long (>24 chars)
    value = session_id
    if len(session_id) > 24:
        # Trim middle to preserve start and end
        value = f"{session_id[:10]}...{session_id[-10:]}"
    text.append(value, style="green")
    return text


def _build_aligned_field(
    label: str, value: Text | str, *, label_width: int = 9, value_style: str = "white"
) -> Text:
    field = Text()
    field.append(f"{label:<{label_width}}: ", style="dim")
    if isinstance(value, Text):
        field.append_text(value)
    else:
        field.append(value, style=value_style)
    return field


def _cap_attr(source, attr: str | None) -> bool:
    if source is None:
        return False
    target = source
    if attr:
        if isinstance(source, dict):
            target = source.get(attr)
        else:
            target = getattr(source, attr, None)
    if isinstance(target, bool):
        return target
    return bool(target)


def _format_capability_shorthand(
    status: ServerStatus, template_expected: bool
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    caps = status.server_capabilities
    tools = getattr(caps, "tools", None)
    prompts = getattr(caps, "prompts", None)
    resources = getattr(caps, "resources", None)
    logging_caps = getattr(caps, "logging", None)
    completion_caps = (
        getattr(caps, "completion", None)
        or getattr(caps, "completions", None)
        or getattr(caps, "respond", None)
    )
    experimental_caps = getattr(caps, "experimental", None)

    instructions_available = bool(status.instructions_available)
    instructions_enabled = status.instructions_enabled

    entries = [
        ("To", _cap_attr(tools, None), _cap_attr(tools, "listChanged")),
        ("Pr", _cap_attr(prompts, None), _cap_attr(prompts, "listChanged")),
        (
            "Re",
            _cap_attr(resources, "read") or _cap_attr(resources, None),
            _cap_attr(resources, "listChanged"),
        ),
        ("Rs", _cap_attr(resources, "subscribe"), _cap_attr(resources, "subscribe")),
        ("Lo", _cap_attr(logging_caps, None), False),
        ("Co", _cap_attr(completion_caps, None), _cap_attr(completion_caps, "listChanged")),
        ("Ex", _cap_attr(experimental_caps, None), False),
    ]

    if not instructions_available:
        entries.append(("In", False, False))
    elif instructions_enabled is False:
        entries.append(("In", "red", False))
    elif instructions_enabled is None and not template_expected:
        entries.append(("In", "blue", False))
    elif instructions_enabled is None:
        entries.append(("In", True, False))
    elif template_expected:
        entries.append(("In", True, False))
    else:
        entries.append(("In", "blue", False))

    if status.roots_configured:
        entries.append(("Ro", True, False))
    else:
        entries.append(("Ro", False, False))

    mode = (status.elicitation_mode or "").lower()
    if mode == "auto_cancel":
        entries.append(("El", "red", False))
    elif mode and mode != "none":
        entries.append(("El", True, False))
    else:
        entries.append(("El", False, False))

    sampling_mode = (status.sampling_mode or "").lower()
    if sampling_mode == "configured":
        entries.append(("Sa", "blue", False))
    elif sampling_mode == "auto":
        entries.append(("Sa", True, False))
    else:
        entries.append(("Sa", False, False))

    entries.append(("Sp", bool(status.spoofing_enabled), False))

    def token_style(supported, highlighted) -> str:
        if supported == "red":
            return "bright_red"
        if supported == "blue":
            return "bright_cyan"
        if not supported:
            return "dim"
        if highlighted:
            return "bright_yellow"
        return "bright_green"

    tokens = [
        (label, token_style(supported, highlighted)) for label, supported, highlighted in entries
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
    try:
        now = datetime.now(timezone.utc)
    except Exception:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
    seconds = max(0, (now - dt).total_seconds())
    return _format_compact_duration(seconds) or "<1s"


def _format_label(label: str, width: int = 10) -> str:
    return f"{label:<{width}}" if len(label) < width else label


def _build_channel_line_with_timeline(
    label: str,
    channel: ChannelSnapshot | None,
    indent: str
) -> tuple[list[Text], bool]:
    """Build channel display with metrics and timeline on same line."""
    lines = []

    # Skip if no channel
    if channel is None:
        return lines, False

    # Channel header - compact format without excessive padding
    header = Text(indent)
    header.append("│   ", style="dim")
    header.append(label, style="bright_white bold")

    state = (channel.state or "open").lower()
    # Only show state for GET channel, or if any channel is not open/connected
    if label == "GET" and state != "open":
        state_styles = {
            "off": "bright_yellow",
            "disabled": "bright_blue",
            "error": "bright_red",
            "idle": "dim",
        }
        header.append(" (", style="dim")
        header.append(state, style=state_styles.get(state, "dim"))
        # Don't show 405 status codes as they're expected for disabled endpoints
        if channel.last_status_code and state in {"off", "disabled", "error"} and channel.last_status_code != 405:
            header.append(f" {channel.last_status_code}", style="bright_white")
        header.append(")", style="dim")
    elif label.startswith("POST") and state not in {"open", "connected"}:
        # Only show non-open states for POST channels
        state_styles = {
            "off": "bright_yellow",
            "disabled": "bright_blue",
            "error": "bright_red",
            "idle": "dim",
        }
        header.append(" (", style="dim")
        header.append(state, style=state_styles.get(state, "dim"))
        # Don't show 405 status codes as they're expected for disabled endpoints
        if channel.last_status_code and state in {"off", "disabled", "error"} and channel.last_status_code != 405:
            header.append(f" {channel.last_status_code}", style="bright_white")
        header.append(")", style="dim")

    # Only show mode for POST-JSON when it's mixed
    if label == "POST-JSON" and channel.mode == "mixed" and channel.mode_counts:
        breakdown = ", ".join(
            f"{name}={count}" for name, count in sorted(channel.mode_counts.items())
        )
        header.append(f"  mixed: {breakdown}", style="bright_magenta")

    lines.append(header)

    # Metrics line with timeline shifted right
    metrics = Text(indent)
    metrics.append("│       ", style="dim")

    # Show breakdown with colons and left-aligned numbers first
    metrics.append("req:", style="dim")
    metrics.append(f"{channel.request_count:<5}", style="bright_yellow")
    metrics.append("resp:", style="dim")
    metrics.append(f"{channel.response_count:<5}", style="bright_blue")
    metrics.append("notif:", style="dim")
    metrics.append(f"{channel.notification_count:<5}", style="bright_cyan")

    # Add timeline after metrics (shifted right for more space)
    if hasattr(channel, "activity_buckets") and channel.activity_buckets:
        # Build timeline directly into the metrics line
        color_map = {
            "error": "bright_red",
            "disabled": "bright_blue",
            "response": "bright_blue",
            "request": "bright_yellow",
            "notification": "bright_cyan",
            "ping": "bright_green",
            "none": "dim",
        }
        metrics.append("    10m ", style="dim")  # Extra spacing
        for state in channel.activity_buckets:
            color = color_map.get(state, "dim")
            metrics.append("●", style=f"bold {color}")
        metrics.append(" now", style="dim")
        timeline = True
    else:
        timeline = False

    lines.append(metrics)

    # Ping health for GET channel
    if label == "GET" and (channel.ping_count or channel.ping_last_at):
        ping_line = Text(indent)
        ping_line.append("│       ", style="dim")
        ping_line.append("ping:", style="dim")
        ping_count = channel.ping_count or 0
        ping_line.append(f"{ping_count:<5}", style="bright_green" if ping_count else "dim")
        ping_line.append("last:", style="dim")
        ping_line.append(
            _format_relative_time(channel.ping_last_at),
            style="bright_green" if channel.ping_last_at else "dim",
        )
        lines.append(ping_line)

    # Error line only if there's an error (but suppress 405 Method Not Allowed)
    if channel.last_error and not ("405" in channel.last_error and "Method Not Allowed" in channel.last_error):
        error_line = Text(indent)
        error_line.append("│       ", style="dim")
        detail = channel.last_error
        if len(detail) > 60:
            detail = detail[:57] + "..."
        error_line.append("error: ", style="dim")
        error_line.append(detail, style="bright_red")
        lines.append(error_line)

    return lines, timeline


def _build_inline_timeline(buckets: Iterable[str]) -> str:
    """Build a compact timeline string for inline display."""
    color_map = {
        "error": "bright_red",
        "disabled": "bright_blue",
        "response": "bright_blue",
        "request": "bright_yellow",
        "notification": "bright_cyan",
        "ping": "bright_green",
        "none": "dim",
    }
    timeline = "  [dim]10m[/dim] "
    for state in buckets:
        color = color_map.get(state, "dim")
        timeline += f"[bold {color}]●[/bold {color}]"
    timeline += " [dim]now[/dim]"
    return timeline


def _build_activity_line(label: str, buckets: Iterable[str], indent: str) -> str:
    # Using markup string for consistent bright rendering
    line = f"{indent}[dim]│       [/dim]timeline [dim]{_format_label(label)}[/dim] "
    color_map = {
        "error": "bright_red",
        "disabled": "bright_blue",  # Consider removing this state
        "response": "bright_blue",  # Changed to blue for better visibility
        "request": "bright_yellow",
        "notification": "bright_cyan",
        "ping": "bright_green",
        "none": "dim",  # Keep idle/none as dim
    }
    for state in buckets:
        color = color_map.get(state, "dim")
        line += f"[bold {color}]●[/bold {color}]"
    line += "[dim]  (last 10m)[/dim]"
    return line


# Note: _build_activity_legend function removed - legend now built directly in footer


def _render_channel_summary(status: ServerStatus, indent: str, total_width: int) -> None:
    snapshot = getattr(status, "transport_channels", None)
    if snapshot is None:
        return

    # Always show all three channel types for consistency
    entries: list[tuple[str, ChannelSnapshot | None]] = [
        ("POST-JSON", getattr(snapshot, "post_json", None)),
        ("POST-SSE", getattr(snapshot, "post_sse", None)),
        ("GET", getattr(snapshot, "get", None)),
    ]

    # Skip if no channels have data
    if not any(channel is not None for _, channel in entries):
        return

    console.console.print()  # Add space before channels
    header = Text(indent)
    header.append("┌ Channels", style="dim")
    console.console.print(header)

    timelines_to_show = []
    for idx, (label, channel) in enumerate(entries):
        # Build and print channel lines
        lines, has_timeline = _build_channel_line_with_timeline(label, channel, indent)

        # Print all lines (timeline is now built into metrics line)
        if lines:
            for line in lines:
                console.console.print(line)

            if has_timeline:
                timelines_to_show.append(label)
        else:
            # Show placeholder for channels with no data to maintain box continuity
            placeholder = Text(indent)
            placeholder.append("│   ", style="dim")
            placeholder.append(label, style="bright_white bold")
            placeholder.append(" (idle)", style="dim")
            console.console.print(placeholder)

    # Add legend with empty rows around it if any timelines were shown
    if timelines_to_show:
        # Empty row before legend
        empty_before = Text(indent)
        empty_before.append("│", style="dim")
        console.console.print(empty_before)

        # Legend row
        legend_line = Text(indent)
        legend_line.append("│ legend ", style="dim")
        legend_map = [
            ("error", "bright_red"),
            ("response", "bright_blue"),
            ("request", "bright_yellow"),
            ("notification", "bright_cyan"),
            ("ping", "bright_green"),
            ("idle", "dim"),
        ]
        for name, color in legend_map:
            legend_line.append("●", style=f"bold {color}")
            legend_line.append(f" {name} ", style="dim")
            legend_line.append(" ", style="dim")
        console.console.print(legend_line)

        # Empty row after legend
        empty_after = Text(indent)
        empty_after.append("│", style="dim")
        console.console.print(empty_after)

    # Footer
    footer = Text(indent)
    footer.append("└", style="dim")
    console.console.print(footer)


async def render_mcp_status(agent, indent: str = "") -> None:
    server_status_map = {}
    if hasattr(agent, "get_server_status") and callable(getattr(agent, "get_server_status")):
        try:
            server_status_map = await agent.get_server_status()
        except Exception:
            server_status_map = {}

    if not server_status_map:
        console.console.print(f"{indent}[dim]•[/dim] [dim]No MCP status available[/dim]")
        return

    template_expected = False
    if hasattr(agent, "config"):
        template_expected = "{{serverInstructions}}" in str(
            getattr(agent.config, "instruction", "")
        )

    try:
        total_width = console.console.size.width
    except Exception:
        total_width = 80

    def render_header(label: Text, right: Text | None = None) -> None:
        line = Text()
        line.append_text(label)
        line.append(" ")

        separator_width = total_width - line.cell_len
        if right and right.cell_len > 0:
            separator_width -= right.cell_len
            separator_width = max(1, separator_width)
            line.append("─" * separator_width, style="dim")
            line.append_text(right)
        else:
            line.append("─" * max(1, separator_width), style="dim")

        console.console.print()
        console.console.print(line)
        console.console.print()

    server_items = list(sorted(server_status_map.items()))

    for index, (server, status) in enumerate(server_items, start=1):
        primary_caps, secondary_caps = _format_capability_shorthand(status, template_expected)

        impl_name = status.implementation_name or status.server_name or "unknown"
        impl_display = impl_name[:30]
        if len(impl_name) > 30:
            impl_display = impl_display[:27] + "..."

        version_display = status.implementation_version or ""
        if len(version_display) > 12:
            version_display = version_display[:9] + "..."

        header_label = Text(indent)
        header_label.append("▎", style="cyan")
        header_label.append("●", style="dim cyan")
        header_label.append(f" [{index:2}] ", style="cyan")
        header_label.append(server, style="bright_blue bold")
        render_header(header_label)

        meta_line = Text(indent + "  ")
        meta_fields: list[Text] = []
        meta_fields.append(_build_aligned_field("name", impl_display))
        if version_display:
            meta_fields.append(_build_aligned_field("version", version_display))

        for idx, field in enumerate(meta_fields):
            if idx:
                meta_line.append("  ", style="dim")
            meta_line.append_text(field)

        session_text = _format_session_id(status.session_id)
        meta_line.append(" | ", style="dim")
        meta_line.append_text(_build_aligned_field("session", session_text))

        client_parts = []
        if status.client_info_name:
            client_parts.append(status.client_info_name)
        if status.client_info_version:
            client_parts.append(status.client_info_version)
        client_display = " ".join(client_parts)
        if len(client_display) > 24:
            client_display = client_display[:21] + "..."

        if client_display:
            meta_line.append(" | ", style="dim")
            meta_line.append_text(_build_aligned_field("client", client_display))

        console.console.print(meta_line)
        console.console.print()

        # Build status segments
        state_segments: list[Text] = []
        if status.is_connected is True:
            state_segments.append(Text("connected", style="bright_green"))
        elif status.is_connected is False:
            state_segments.append(Text("offline", style="bright_red"))

        if status.roots_configured and (status.roots_count or 0) > 0:
            roots_text = Text("roots:", style="white")
            roots_text.append(str(status.roots_count), style="bright_white")
            state_segments.append(roots_text)

        duration = _format_compact_duration(status.staleness_seconds)
        if duration:
            last_text = Text("last activity: ", style="white")
            last_text.append(duration, style="bright_white")
            last_text.append(" ago", style="dim")
            state_segments.append(last_text)

        if status.error_message and status.is_connected is False:
            state_segments.append(Text(status.error_message, style="bright_red"))

        instr_available = bool(status.instructions_available)
        if instr_available and status.instructions_enabled is False:
            state_segments.append(Text("instructions disabled", style="bright_red"))
        elif instr_available and not template_expected:
            state_segments.append(Text("template missing", style="bright_yellow"))

        if status.spoofing_enabled:
            state_segments.append(Text("client spoof", style="bright_yellow"))

        # Transport and main status line
        transport = getattr(status, "transport", None) or "unknown"
        status_line = Text(indent + "  ")
        transport_value = Text(transport, style="bright_white" if transport != "unknown" else "dim")
        status_line.append_text(_build_aligned_field("transport", transport_value))
        for segment in state_segments:
            status_line.append("  |  ", style="dim")
            status_line.append_text(segment)

        console.console.print(status_line)

        # MCP protocol calls made (only shows calls that have actually been invoked)
        calls = _summarise_call_counts(status.call_counts)
        if calls:
            calls_line = Text(indent + "  ")
            calls_line.append("mcp calls: ", style="dim")
            calls_line.append(calls, style="bright_white")
            console.console.print(calls_line)
        _render_channel_summary(status, indent, total_width)

        combined_tokens = primary_caps + secondary_caps
        prefix = Text(indent)
        prefix.append("─| ", style="dim")
        suffix = Text(" |", style="dim")

        caps_content = (
            _build_capability_text(combined_tokens)
            if combined_tokens
            else Text("none", style="dim")
        )

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

        if index != len(server_items):
            console.console.print()
