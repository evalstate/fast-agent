import io
from datetime import datetime, timedelta, timezone

import pytest
from click.utils import strip_ansi
from rich.console import Console

from fast_agent.mcp.app_integrations import AppServerConfig
from fast_agent.mcp.mcp_aggregator import ServerStatus
from fast_agent.mcp.transport_tracking import (
    ChannelSnapshot,
    DiscoverySnapshot,
    TransportSnapshot,
)
from fast_agent.ui import console
from fast_agent.ui.mcp_display import (
    SYMBOL_NOTIFICATION,
    SYMBOL_REQUEST,
    SYMBOL_RESPONSE,
    SYMBOL_STDIO_ACTIVITY,
    Colours,
    _app_integration_capability_state,
    _build_health_text,
    _capability_token_style,
    _channel_arrow_style,
    _elicitation_capability_state,
    _format_compact_duration,
    _format_timeline_label,
    _get_health_state,
    _render_channel_summary,
    _sampling_capability_state,
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


def test_modern_health_text_is_omitted_without_legacy_ping_loop() -> None:
    status = ServerStatus(
        server_name="modern",
        protocol_era="modern",
        ping_interval_seconds=30,
        ping_max_missed=3,
    )

    assert _build_health_text(status) is None


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


def test_app_integration_capability_state_returns_false_when_config_disabled() -> None:
    status = ServerStatus(
        server_name="test",
        app_integration_config=AppServerConfig(server_name="test"),
    )

    assert _app_integration_capability_state(status) is False


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


def test_render_channel_summary_shows_observed_channels_and_errors() -> None:
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
    assert "HEALTH" not in output
    assert "gateway timeout (500)" in output
    assert "legend:" in output


def test_render_channel_summary_shows_distinct_post_response_modes() -> None:
    post_sse = ChannelSnapshot(
        state="open",
        request_count=6,
        activity_buckets=["request"],
    )
    status = ServerStatus(
        server_name="demo",
        transport="http",
        ping_interval_seconds=30,
        transport_channels=TransportSnapshot(
            post=post_sse,
            post_sse=post_sse,
            activity_bucket_seconds=30,
            activity_bucket_count=4,
        ),
    )

    original_console = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = capture.get()
    finally:
        _restore_console_size(original_console)

    assert "POST (SSE)" in output
    assert "GET (SSE)" in output
    assert "not observed" in output
    assert "POST (JSON)" in output
    assert strip_ansi(output).count("     6     0     0") == 1
    assert "HEALTH" not in output


def test_render_idle_http_summary_shows_both_post_response_modes() -> None:
    status = ServerStatus(
        server_name="demo",
        transport="http",
        protocol_era="modern",
        transport_channels=TransportSnapshot(),
    )

    original_console = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = strip_ansi(capture.get())
    finally:
        _restore_console_size(original_console)

    assert "POST (SSE)" in output
    assert "POST (JSON)" in output


def test_stateless_legacy_http_explains_unavailable_channels_and_discovery() -> None:
    status = ServerStatus(
        server_name="demo",
        transport="http",
        protocol_mode="auto",
        protocol_era="legacy",
        session_id=None,
        is_connected=True,
        ping_interval_seconds=30,
        transport_channels=TransportSnapshot(
            discovery=DiscoverySnapshot(
                state="legacy-fallback",
                status_code=400,
                detail="HTTP 400",
            )
        ),
    )

    original_console = _set_console_size(width=160)
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=160)
        output = strip_ansi(capture.get())
    finally:
        _restore_console_size(original_console)

    assert output.count("GET (SSE)") == 1
    assert output.count("LISTEN (SSE)") == 1
    assert output.count("POST (SSE)") == 1
    assert output.count("POST (JSON)") == 1
    assert "unavailable (no session)" in output
    assert "unavailable (legacy protocol)" in output
    assert output.count("not observed") == 2
    assert "Discovery: HTTP 400; legacy fallback succeeded" in output
    assert "POST: HTTP 400" not in output
    assert _get_health_state(status).label == "pending"


def test_render_disabled_listen_channel_uses_disabled_style() -> None:
    status = ServerStatus(
        server_name="demo",
        transport="http",
        protocol_era="modern",
        subscription_state="disabled",
        transport_channels=TransportSnapshot(),
    )

    original_console = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = strip_ansi(capture.get())
    finally:
        _restore_console_size(original_console)

    assert "◁ LISTEN (SSE)" in output
    assert "    -     -     -" in output


def test_render_modern_listen_channel_shows_notifications_without_ping() -> None:
    status = ServerStatus(
        server_name="demo",
        transport="http",
        protocol_era="modern",
        transport_channels=TransportSnapshot(
            listen=ChannelSnapshot(
                state="open",
                request_count=1,
                notification_count=3,
                activity_buckets=["request", "notification"],
            ),
            activity_bucket_seconds=30,
            activity_bucket_count=2,
        ),
    )

    original_console = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = capture.get()
    finally:
        _restore_console_size(original_console)

    assert "LISTEN (SSE)" in output
    assert "◀ LISTEN (SSE)" in strip_ansi(output)
    assert "    1     0     3" in strip_ansi(output)
    assert "notification" in output
    assert "ping" not in output


def test_render_disconnected_listen_channel_uses_dimmed_incoming_arrow() -> None:
    status = ServerStatus(
        server_name="demo",
        transport="http",
        protocol_era="modern",
        transport_channels=TransportSnapshot(
            listen=ChannelSnapshot(
                state="off",
                request_count=1,
                disconnect_at=datetime.now(timezone.utc),
            ),
        ),
    )

    original_console = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = strip_ansi(capture.get())
    finally:
        _restore_console_size(original_console)

    assert "◁ LISTEN (SSE)" in output


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
                protocol_mode="modern",
                protocol_version="2026-07-28",
                protocol_era="modern",
                negotiation="adopt",
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
    assert "2026-07-28 (forced modern)" in output
    assert "session" not in output
    assert "health" not in output
    assert "adopt" not in output
    assert "discover" not in output


@pytest.mark.asyncio
async def test_render_forced_legacy_status_keeps_session_and_health() -> None:
    agent = _FakeAgent(
        {
            "legacy": ServerStatus(
                server_name="legacy",
                protocol_mode="legacy",
                protocol_version="2025-11-25",
                protocol_era="legacy",
                negotiation="initialize",
                session_id="legacy-session",
                is_connected=True,
                ping_interval_seconds=0,
            )
        },
        instruction="",
    )

    original_console = _set_console_size(width=110)
    try:
        with console.console.capture() as capture:
            await render_mcp_status(agent, indent="  ")
        output = capture.get()
    finally:
        _restore_console_size(original_console)

    assert "2025-11-25 (forced legacy)" in output
    assert "initialize" not in output
    assert "session" in output
    assert "legacy-session" in output
    assert "health" in output
    assert "disabled" in output


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
