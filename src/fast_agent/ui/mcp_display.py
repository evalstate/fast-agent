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


def _render_channel_summary(status: ServerStatus, indent: str, total_width: int) -> None:
    snapshot = getattr(status, "transport_channels", None)
    if snapshot is None:
        return

    # Show channel types based on what's available
    entries: list[tuple[str, str, ChannelSnapshot | None]] = []

    # Check if we have HTTP transport channels
    http_channels = [
        getattr(snapshot, "get", None),
        getattr(snapshot, "post_sse", None),
        getattr(snapshot, "post_json", None),
    ]

    # Check if we have stdio transport channel
    stdio_channel = getattr(snapshot, "stdio", None)

    if any(channel is not None for channel in http_channels):
        # HTTP transport - show the original three channels
        entries = [
            ("GET (SSE)", "◀", getattr(snapshot, "get", None)),
            ("POST (SSE)", "▶", getattr(snapshot, "post_sse", None)),
            ("POST (JSON)", "▶", getattr(snapshot, "post_json", None)),
        ]
    elif stdio_channel is not None:
        # STDIO transport - show single bidirectional channel
        entries = [
            ("STDIO", "⇄", stdio_channel),
        ]

    # Skip if no channels have data
    if not any(channel is not None for _, _, channel in entries):
        return

    console.console.print()  # Add space before channels

    # Determine if we're showing stdio or HTTP channels
    is_stdio = stdio_channel is not None

    # Get transport type for display
    transport = getattr(status, "transport", None) or "unknown"
    transport_display = transport.upper() if transport != "unknown" else "Channels"

    # Header with column labels
    header = Text(indent)
    header.append(f"┌ {transport_display} ", style="dim")

    # Calculate padding needed based on transport display length
    # Base structure: "┌ " (2) + transport_display + " " (1) + "─" padding to align with columns
    header_prefix_len = 3 + len(transport_display)

    if is_stdio:
        # Simplified header for stdio: just activity column
        # Need to align with "│ ⇄ STDIO        10m ●●●●●●●●●●●●●●●●●●●● now        29"
        # That's: "│ " + arrow + " " + label(13) + "10m " + dots(20) + " now" = 47 chars
        # Then: "  " + activity(8) = 10 chars
        # Total content width = 47 + 10 = 57 chars
        # So we need 47 - header_prefix_len dashes before "activity"
        dash_count = max(1, 47 - header_prefix_len)
        header.append("─" * dash_count, style="dim")
        header.append("  activity", style="dim")
    else:
        # Original header for HTTP channels
        # Need to align with the req/resp/notif/ping columns
        # Structure: "│ " + arrow + " " + label(13) + "10m " + dots(20) + " now" = 47 chars
        # Then: "  " + req(5) + " " + resp(5) + " " + notif(5) + " " + ping(5) = 25 chars
        # Total content width = 47 + 25 = 72 chars
        # So we need 47 - header_prefix_len dashes before the column headers
        dash_count = max(1, 47 - header_prefix_len)
        header.append("─" * dash_count, style="dim")
        header.append("  req  resp notif  ping", style="dim")

    console.console.print(header)

    # Empty row after header for cleaner spacing
    empty_header = Text(indent)
    empty_header.append("│", style="dim")
    console.console.print(empty_header)

    # Collect any errors to show at bottom
    errors = []

    # Build timeline color map
    if is_stdio:
        # Simplified color map for stdio: bright green for activity, dim for idle
        timeline_color_map = {
            "error": "bright_red",  # Keep error as red
            "request": "bright_green",  # All activity shows as bright green
            "response": "bright_green",  # (not used in stdio but just in case)
            "notification": "bright_green",  # (not used in stdio but just in case)
            "ping": "bright_green",  # (not used in stdio but just in case)
            "none": "white dim",
        }
    else:
        # Full color map for HTTP channels
        timeline_color_map = {
            "error": "bright_red",
            "disabled": "bright_blue",
            "response": "bright_blue",
            "request": "bright_yellow",
            "notification": "bright_cyan",
            "ping": "bright_green",
            "none": "white dim",
        }

    for label, arrow, channel in entries:
        line = Text(indent)
        line.append("│ ", style="dim")

        # Determine arrow color based on state
        arrow_style = "black dim"  # default no channel
        if channel:
            state = (channel.state or "open").lower()

            # Check for 405 status code (method not allowed = disabled endpoint)
            if channel.last_status_code == 405:
                arrow_style = "bright_yellow"
                # Don't add 405 to errors list - it's just disabled, not an error
            # Error state (non-405 errors)
            elif state == "error":
                arrow_style = "bright_red"
                if channel.last_error and channel.last_status_code != 405:
                    error_msg = channel.last_error
                    if channel.last_status_code:
                        errors.append(
                            (label.split()[0], f"{error_msg} ({channel.last_status_code})")
                        )
                    else:
                        errors.append((label.split()[0], error_msg))
            # Explicitly disabled or off
            elif state in {"off", "disabled"}:
                arrow_style = "black dim"
            # No activity (idle)
            elif channel.request_count == 0 and channel.response_count == 0:
                arrow_style = "bright_cyan"
            # Active/connected with activity
            elif state in {"open", "connected"}:
                arrow_style = "bright_green"
            # Fallback for other states
            else:
                arrow_style = "bright_cyan"

        # Arrow and label with better spacing
        line.append(arrow, style=arrow_style)
        line.append(f" {label:<13}", style="bright_white")

        # Always show timeline (dim black dots if no data)
        line.append("10m ", style="dim")
        if channel and channel.activity_buckets:
            # Show actual activity
            for bucket_state in channel.activity_buckets:
                color = timeline_color_map.get(bucket_state, "dim")
                line.append("●", style=f"bold {color}")
        else:
            # Show dim black dots for no activity
            for _ in range(20):
                line.append("●", style="black dim")
        line.append(" now", style="dim")

        # Metrics - different layouts for stdio vs HTTP
        if is_stdio:
            # Simplified activity column for stdio
            if channel and channel.message_count > 0:
                activity = str(channel.message_count).rjust(8)
                activity_style = "bright_white"
            else:
                activity = "-".rjust(8)
                activity_style = "dim"
            line.append(f"  {activity}", style=activity_style)
        else:
            # Original HTTP columns
            if channel:
                req = str(channel.request_count).rjust(5)
                resp = str(channel.response_count).rjust(5)
                notif = str(channel.notification_count).rjust(5)
                ping = str(channel.ping_count if channel.ping_count else "-").rjust(5)
            else:
                req = "-".rjust(5)
                resp = "-".rjust(5)
                notif = "-".rjust(5)
                ping = "-".rjust(5)
            line.append(
                f"  {req} {resp} {notif} {ping}", style="bright_white" if channel else "dim"
            )

        console.console.print(line)

        # Debug: print the raw line length
        # import sys
        # print(f"Line length: {len(line.plain)}", file=sys.stderr)

    # Show errors at bottom if any
    if errors:
        # Empty row before errors
        empty_line = Text(indent)
        empty_line.append("│", style="dim")
        console.console.print(empty_line)

        for channel_type, error_msg in errors:
            error_line = Text(indent)
            error_line.append("│ ", style="dim")
            error_line.append("⚠ ", style="bright_yellow")
            error_line.append(f"{channel_type}: ", style="bright_white")
            # Truncate long error messages
            if len(error_msg) > 60:
                error_msg = error_msg[:57] + "..."
            error_line.append(error_msg, style="bright_red")
            console.console.print(error_line)

    # Legend if any timelines shown
    has_timelines = any(channel and channel.activity_buckets for _, _, channel in entries)

    if has_timelines:
        # Empty row before footer with legend
        empty_before = Text(indent)
        empty_before.append("│", style="dim")
        console.console.print(empty_before)

    # Footer with legend
    footer = Text(indent)
    footer.append("└", style="dim")

    if has_timelines:
        footer.append(" legend: ", style="dim")

        if is_stdio:
            # Simplified legend for stdio: just activity vs idle
            legend_map = [
                ("activity", "bright_green"),
                ("idle", "white dim"),
            ]
        else:
            # Full legend for HTTP channels
            legend_map = [
                ("error", "bright_red"),
                ("response", "bright_blue"),
                ("request", "bright_yellow"),
                ("notification", "bright_cyan"),
                ("ping", "bright_green"),
                ("idle", "white dim"),
            ]

        for i, (name, color) in enumerate(legend_map):
            if i > 0:
                footer.append(" ", style="dim")
            footer.append("●", style=f"bold {color}")
            footer.append(f" {name}", style="dim")

    console.console.print(footer)

    # Add blank line for spacing before capabilities
    console.console.print()


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

        # First line: name and version
        meta_line = Text(indent + "  ")
        meta_fields: list[Text] = []
        meta_fields.append(_build_aligned_field("name", impl_display))
        if version_display:
            meta_fields.append(_build_aligned_field("version", version_display))

        for idx, field in enumerate(meta_fields):
            if idx:
                meta_line.append("  ", style="dim")
            meta_line.append_text(field)

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

        # Second line: session (on its own line)
        session_line = Text(indent + "  ")
        session_text = _format_session_id(status.session_id)
        session_line.append_text(_build_aligned_field("session", session_text))
        console.console.print(session_line)
        console.console.print()

        # Build status segments
        state_segments: list[Text] = []

        duration = _format_compact_duration(status.staleness_seconds)
        if duration:
            last_text = Text("last activity: ", style="dim")
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

        # Main status line (without transport and connected)
        if state_segments:
            status_line = Text(indent + "  ")
            for idx, segment in enumerate(state_segments):
                if idx:
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
