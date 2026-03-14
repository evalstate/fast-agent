from datetime import datetime, timedelta, timezone

import pytest

from fast_agent.mcp.mcp_aggregator import ServerStatus
from fast_agent.mcp.transport_tracking import ChannelSnapshot, TransportSnapshot
from fast_agent.ui import console
from fast_agent.ui.mcp_display import (
    _format_experimental_session_status,
    _get_health_state,
    _render_channel_summary,
    render_mcp_status,
)


def _set_console_size(width: int = 100, height: int = 24) -> tuple[object | None, object | None]:
    original_width = getattr(console.console, "_width", None)
    original_height = getattr(console.console, "_height", None)
    console.console._width = width
    console.console._height = height
    return original_width, original_height


def _restore_console_size(original_width: object | None, original_height: object | None) -> None:
    console.console._width = original_width
    console.console._height = original_height


def test_health_state_marks_stale_when_last_ping_exceeds_window():
    now = datetime.now(timezone.utc)
    status = ServerStatus(
        server_name="test",
        is_connected=True,
        ping_interval_seconds=5,
        ping_max_missed=3,
        ping_last_ok_at=now - timedelta(seconds=16),
    )

    state, _style = _get_health_state(status)

    assert state == "stale"


def test_experimental_session_status_not_advertised_when_disabled() -> None:
    status = ServerStatus(server_name="test", experimental_session_supported=False)

    rendered = _format_experimental_session_status(status)

    assert rendered.plain == "not advertised"


def test_experimental_session_status_shows_created_to_expiry_range() -> None:
    created_iso = "2026-02-24T10:00:00+00:00"
    expiry_iso = "2026-02-24T12:34:56.000000+00:00"
    status = ServerStatus(
        server_name="test",
        experimental_session_supported=True,
        session_cookie={
            "sessionId": "sess-cookie-id-1234567890abcdefghijklmnop",
            "created": created_iso,
            "expiresAt": expiry_iso,
        },
    )

    rendered = _format_experimental_session_status(status)
    expected_created = datetime.fromisoformat(created_iso).astimezone().strftime("%d/%m/%y %H:%M")
    expected_expiry = datetime.fromisoformat(expiry_iso).astimezone().strftime("%d/%m/%y %H:%M")

    assert rendered.plain.startswith("sess-cookie-id")
    assert "(" in rendered.plain and ")" in rendered.plain
    assert " → " in rendered.plain
    assert expected_created in rendered.plain
    assert expected_expiry in rendered.plain
    assert "T12:34:56" not in rendered.plain
    assert "..." in rendered.plain


def test_experimental_session_status_uses_session_id_field() -> None:
    status = ServerStatus(
        server_name="test",
        experimental_session_supported=True,
        session_cookie={
            "sessionId": "sess-new-format-1234567890",
            "expiresAt": "2026-02-24T12:34:56.000000+00:00",
        },
    )

    rendered = _format_experimental_session_status(status)

    assert "sess-new-format" in rendered.plain


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

    original_width, original_height = _set_console_size()
    try:
        with console.console.capture() as capture:
            _render_channel_summary(status, indent="  ", total_width=100)
        output = capture.get()
    finally:
        _restore_console_size(original_width, original_height)

    assert "HTTP" in output
    assert "GET (SSE)" in output
    assert "POST (JSON)" in output
    assert "HEALTH" in output
    assert "gateway timeout (500)" in output
    assert "legend:" in output


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
                experimental_session_supported=True,
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

    original_width, original_height = _set_console_size(width=110)
    try:
        with console.console.capture() as capture:
            await render_mcp_status(agent, indent="  ")
        output = capture.get()
    finally:
        _restore_console_size(original_width, original_height)

    assert "demo-server" in output
    assert "Demo MCP Server" in output
    assert "fast-agent 1.2.3" in output
    assert "mcp calls:" in output
    assert "reconnects:" in output
    assert "STDIO" in output
    assert "session" in output


@pytest.mark.asyncio
async def test_render_mcp_status_reports_when_no_server_status_is_available() -> None:
    class _NoStatusAgent:
        config = _FakeConfig("")

    with console.console.capture() as capture:
        await render_mcp_status(_NoStatusAgent(), indent="  ")
    output = capture.get()

    assert "No MCP status available" in output
