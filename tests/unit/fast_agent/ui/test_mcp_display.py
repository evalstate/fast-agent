from datetime import datetime, timedelta, timezone

from fast_agent.mcp.mcp_aggregator import ServerStatus
from fast_agent.ui.mcp_display import _format_experimental_session_status, _get_health_state


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
