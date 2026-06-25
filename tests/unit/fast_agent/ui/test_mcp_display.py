import io
from datetime import datetime, timedelta, timezone

import pytest
from click.utils import strip_ansi
from rich.console import Console

from fast_agent.mcp.mcp_aggregator import ServerStatus
from fast_agent.mcp.skybridge import SkybridgeServerConfig
from fast_agent.mcp.transport_tracking import ChannelSnapshot, TransportSnapshot
from fast_agent.ui import console
from fast_agent.ui.mcp_display import (
    SYMBOL_NOTIFICATION,
    SYMBOL_REQUEST,
    SYMBOL_RESPONSE,
    SYMBOL_STDIO_ACTIVITY,
    Colours,
    _capability_token_style,
    _channel_arrow_style,
    _elicitation_capability_state,
    _format_compact_duration,
    _format_timeline_label,
    _get_health_state,
    _render_channel_summary,
    _sampling_capability_state,
    _skybridge_capability_state,
    _timeline_symbol_for_state,
    render_mcp_status,
)


def _set_console_size(width: int = 100, height: int = 24) -> Console:
    original_console = console.console
    console.console = Console(
        file=io.StringIO(),
        force_terminal=True,
        width=width,
        height=height,
    )
    return original_console


def _restore_console_size(original_console: Console) -> None:
    console.console = original_console


def test_health_state_marks_stale_when_last_ping_exceeds_window():
    now = datetime.now(timezone.utc)
    status = ServerStatus(
        server_name="test",
        is_connected=True,
        ping_interval_seconds=5,
        ping_max_missed=3,
        ping_last_ok_at=now - timedelta(seconds=16),
    )

    state = _get_health_state(status)

    assert state.label == "stale"


def test_health_state_uses_newer_failed_ping_when_ok_ping_is_older():
    now = datetime.now(timezone.utc)
    status = ServerStatus(
        server_name="test",
        is_connected=True,
        ping_interval_seconds=5,
        ping_max_missed=3,
        ping_consecutive_failures=1,
        ping_last_ok_at=now - timedelta(seconds=60),
        ping_last_fail_at=now - timedelta(seconds=2),
    )

    state = _get_health_state(status)

    assert state.label == "missed"


def test_health_state_uses_newer_ok_ping_when_failed_ping_is_older():
    now = datetime.now(timezone.utc)
    status = ServerStatus(
        server_name="test",
        is_connected=True,
        ping_interval_seconds=5,
        ping_max_missed=3,
        ping_consecutive_failures=0,
        ping_last_ok_at=now - timedelta(seconds=2),
        ping_last_fail_at=now - timedelta(seconds=60),
    )

    state = _get_health_state(status)

    assert state.label == "ok"


def test_format_compact_duration_omits_missing_and_non_finite_values() -> None:
    assert _format_compact_duration(None) is None
    assert _format_compact_duration(float("nan")) is None
    assert _format_compact_duration(float("inf")) is None


def test_format_compact_duration_formats_positive_values() -> None:
    assert _format_compact_duration(0.5) == "<1s"
    assert _format_compact_duration(65) == "1m05s"
    assert _format_compact_duration(3700) == "1h01m"


@pytest.mark.parametrize(
    ("total_seconds", "expected"),
    [
        (0, "0s"),
        (-5, "0s"),
        (5, "5s"),
        (60, "1m"),
        (65, "1m05s"),
        (3600, "1h"),
        (3660, "1h01m"),
        (86400, "1d"),
        (90000, "1d1h"),
        (86400 + 59 * 60, "1d"),
    ],
)
def test_format_timeline_label_uses_largest_two_units(
    total_seconds: int,
    expected: str,
) -> None:
    assert _format_timeline_label(total_seconds) == expected


def test_skybridge_capability_state_returns_false_when_config_disabled() -> None:
    status = ServerStatus(
        server_name="test",
        skybridge=SkybridgeServerConfig(server_name="test"),
    )

    assert _skybridge_capability_state(status) is False


def test_capability_mode_states_are_normalized() -> None:
    assert _elicitation_capability_state(None) is False
    assert _elicitation_capability_state(" NONE ") is False
    assert _elicitation_capability_state(" Auto-Cancel ") == "red"
    assert _elicitation_capability_state("forms") is True

    assert _sampling_capability_state(None) is False
    assert _sampling_capability_state(" AUTO ") is True
    assert _sampling_capability_state(" Configured ") == "blue"
    assert _sampling_capability_state("disabled") is False


def test_capability_token_style_maps_special_states_and_fallbacks() -> None:
    assert _capability_token_style("red", highlighted=False) == Colours.TOKEN_ERROR
    assert _capability_token_style("blue", highlighted=False) == Colours.TOKEN_WARNING
    assert _capability_token_style("warn", highlighted=False) == Colours.CAP_TOKEN_CAUTION
    assert _capability_token_style(False, highlighted=True) == Colours.TOKEN_DISABLED
    assert _capability_token_style(True, highlighted=True) == Colours.CAP_TOKEN_HIGHLIGHTED
    assert _capability_token_style(True, highlighted=False) == Colours.CAP_TOKEN_ENABLED


def test_timeline_symbol_for_state_uses_stdio_fallback_after_special_states() -> None:
    assert _timeline_symbol_for_state("request") == SYMBOL_REQUEST
    assert _timeline_symbol_for_state("notification") == SYMBOL_NOTIFICATION
    assert _timeline_symbol_for_state("response") == SYMBOL_RESPONSE
    assert _timeline_symbol_for_state("request", is_stdio=True) == SYMBOL_STDIO_ACTIVITY
    assert _timeline_symbol_for_state("response", is_stdio=True) == SYMBOL_STDIO_ACTIVITY


@pytest.mark.parametrize(
    ("channel", "expected_style"),
    [
        (None, Colours.ARROW_OFF),
        (ChannelSnapshot(state="open", last_status_code=405), Colours.ARROW_METHOD_NOT_ALLOWED),
        (ChannelSnapshot(state=" ERROR "), Colours.ARROW_ERROR),
        (ChannelSnapshot(state=" DISABLED "), Colours.ARROW_OFF),
        (ChannelSnapshot(state="open"), Colours.ARROW_IDLE),
        (
            ChannelSnapshot(state=" CONNECTED ", request_count=1, response_count=1),
            Colours.ARROW_ACTIVE,
        ),
        (
            ChannelSnapshot(state="closing", request_count=1, response_count=1),
            Colours.ARROW_IDLE,
        ),
    ],
)
def test_channel_arrow_style_preserves_status_precedence(
    channel: ChannelSnapshot | None,
    expected_style: str,
) -> None:
    assert _channel_arrow_style(channel) == expected_style


def test_render_channel_summary_shows_health_row_and_errors() -> None:
    status = ServerStatus(
        server_name="demo",
        transport="http",
        ping_interval_seconds=30,
        ping_ok_count=3,
        ping_fail_count=1,
        ping_activity_buckets=["ping", "error"],
        ping_activity_bucket_seconds=30,
        ping_activity_bucket_count=4,
        transport_channels=TransportSnapshot(
            activity_bucket_seconds=30,
            activity_bucket_count=4,
            get=ChannelSnapshot(
                state="error",
                last_status_code=500,
                last_error="gateway timeout",
                request_count=1,
                response_count=0,
                notification_count=0,
                ping_count=0,
                activity_buckets=["error", "none"],
            ),
            post_json=ChannelSnapshot(
                state="open",
                request_count=4,
                response_count=4,
                notification_count=1,
                ping_count=2,
                activity_buckets=["request", "response", "notification", "ping"],
            ),
        ),
    )

    original_console = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = capture.get()
    finally:
        _restore_console_size(original_console)

    assert "HTTP" in output
    assert "GET (SSE)" in output
    assert "POST (JSON)" in output
    assert "HEALTH" in output
    assert "gateway timeout (500)" in output
    assert "legend:" in output


def test_render_channel_summary_uses_legacy_post_channel() -> None:
    status = ServerStatus(
        server_name="demo",
        transport="http",
        transport_channels=TransportSnapshot(
            post=ChannelSnapshot(
                state="error",
                last_status_code=502,
                last_error="bad gateway",
                request_count=1,
                response_count=0,
            ),
        ),
    )

    original_console = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = capture.get()
    finally:
        _restore_console_size(original_console)

    assert "HTTP" in output
    assert "POST (JSON)" in output
    assert "bad gateway (502)" in output


class _FakeConfig:
    def __init__(self, instruction: str) -> None:
        self.instruction = instruction


class _FakeAgent:
    def __init__(self, status_map: dict[str, ServerStatus], instruction: str) -> None:
        self._status_map = status_map
        self.config = _FakeConfig(instruction)

    async def get_server_status(self) -> dict[str, ServerStatus]:
        return self._status_map


@pytest.mark.asyncio
async def test_render_mcp_status_renders_server_details_and_calls() -> None:
    now = datetime.now(timezone.utc)
    agent = _FakeAgent(
        {
            "demo-server": ServerStatus(
                server_name="demo-server",
                implementation_name="Demo MCP Server",
                implementation_version="2026.03.14-build7",
                client_info_name="fast-agent",
                client_info_version="1.2.3",
                session_id="sess-1234567890abcdefghijklmnop",
                is_connected=True,
                staleness_seconds=12,
                call_counts={"list_tools": 2},
                reconnect_count=1,
                instructions_available=True,
                instructions_enabled=True,
                ping_interval_seconds=30,
                ping_ok_count=2,
                ping_last_ok_at=now - timedelta(seconds=10),
                transport="stdio",
                transport_channels=TransportSnapshot(
                    activity_bucket_seconds=30,
                    activity_bucket_count=4,
                    stdio=ChannelSnapshot(
                        state="connected",
                        message_count=6,
                        request_count=2,
                        response_count=3,
                        notification_count=1,
                        activity_buckets=["request", "response", "notification", "ping"],
                    ),
                ),
            )
        },
        instruction="{{serverInstructions}}\nFollow the MCP status block.",
    )

    original_console = _set_console_size(width=110)
    try:
        with console.console.capture() as capture:
            await render_mcp_status(agent, indent="  ")
        output = capture.get()
    finally:
        _restore_console_size(original_console)

    assert "demo-server" in output
    assert "Demo MCP Server" in output
    assert "fast-agent 1.2.3" in output
    assert "mcp calls:" in output
    assert "reconnects:" in output
    assert "STDIO" in output
    assert "session" in output


@pytest.mark.asyncio
async def test_render_mcp_status_shows_skills_hint_above_capability_bar() -> None:
    agent = _FakeAgent(
        {
            "skills-server": ServerStatus(
                server_name="skills-server",
                is_connected=True,
                staleness_seconds=202,
                transport="stdio",
                mcp_skills_enabled=True,
                transport_channels=TransportSnapshot(
                    activity_bucket_seconds=30,
                    activity_bucket_count=4,
                    stdio=ChannelSnapshot(
                        state="connected",
                        message_count=1,
                        request_count=1,
                        response_count=1,
                        notification_count=0,
                        activity_buckets=["request", "response"],
                    ),
                ),
            )
        },
        instruction="",
    )

    original_console = _set_console_size(width=120)
    try:
        with console.console.capture() as capture:
            await render_mcp_status(agent, indent="  ")
        output = strip_ansi(capture.get())
    finally:
        _restore_console_size(original_console)

    assert "last activity:" in output
    assert "last activity:" in output and "Skills over MCP" in output

    lines = output.splitlines()
    transport_index = next(index for index, line in enumerate(lines) if "STDIO" in line)
    skills_index = next(index for index, line in enumerate(lines) if "Skills over MCP" in line)
    capability_index = next(index for index, line in enumerate(lines) if "─| " in line)

    assert transport_index < skills_index < capability_index
    assert " Sk " in lines[capability_index] or " Sk" in lines[capability_index]


@pytest.mark.asyncio
async def test_render_mcp_status_reports_when_no_server_status_is_available() -> None:
    class _NoStatusAgent:
        config = _FakeConfig("")

    with console.console.capture() as capture:
        await render_mcp_status(_NoStatusAgent(), indent="  ")
    output = capture.get()

    assert "No MCP status available" in output
