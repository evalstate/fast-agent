from datetime import datetime, timedelta, timezone

from fast_agent.mcp.mcp_aggregator import ServerStatus
from fast_agent.ui.mcp_display import _get_health_state


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
