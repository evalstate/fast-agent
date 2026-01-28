from datetime import datetime

from fast_agent.session.formatting import format_session_entries
from fast_agent.session.session_manager import SessionInfo


def test_format_session_entries_strips_timestamp_prefix() -> None:
    now = datetime(2026, 1, 18, 12, 0)
    session = SessionInfo(
        name="2601181200-AbCd12",
        created_at=now,
        last_activity=now,
        history_files=[],
        metadata={},
    )

    compact = format_session_entries([session], None, mode="compact")
    assert compact
    assert "AbCd12" in compact[0]
    assert "2601181200" not in compact[0]

    verbose = format_session_entries([session], None, mode="verbose")
    assert verbose
    assert "AbCd12" in verbose[0]
    assert "2601181200" not in verbose[0]


def test_format_session_entries_marks_pinned() -> None:
    now = datetime(2026, 1, 18, 12, 0)
    session = SessionInfo(
        name="2601181200-AbCd12",
        created_at=now,
        last_activity=now,
        history_files=[],
        metadata={"pinned": True},
    )

    compact = format_session_entries([session], None, mode="compact")
    assert compact
    assert "pin" in compact[0].lower()

    verbose = format_session_entries([session], None, mode="verbose")
    assert verbose
    assert "\U0001F4CC" in verbose[0]
