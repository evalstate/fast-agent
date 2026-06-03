from datetime import datetime

import pytest

from fast_agent.session.formatting import (
    SessionEntrySummary,
    SessionListMode,
    extract_session_title,
    format_history_summary,
    format_session_agent_label,
    format_session_entries,
)
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


def test_extract_session_title_uses_first_nonblank_metadata_value() -> None:
    assert extract_session_title({"title": "  ", "label": " Review session "}) == (
        "Review session"
    )
    assert extract_session_title({"title": "\n\t", "first_user_preview": " First prompt "}) == (
        "First prompt"
    )
    assert extract_session_title({"title": " \n "}) is None


def test_extract_session_title_ignores_structured_metadata_values() -> None:
    assert extract_session_title({"title": {"text": "Review"}, "label": ["fallback"]}) is None
    assert extract_session_title({"title": 123, "first_user_preview": " Review "}) == "Review"


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


def test_format_session_entries_accepts_named_mode() -> None:
    now = datetime(2026, 1, 18, 12, 0)
    session = SessionInfo(
        name="2601181200-AbCd12",
        created_at=now,
        last_activity=now,
        history_files=[],
        metadata={},
    )

    compact = format_session_entries([session], None, mode=SessionListMode.COMPACT)

    assert compact == format_session_entries([session], None, mode="compact")


def test_format_session_entries_ignores_non_string_agent_metadata_keys() -> None:
    now = datetime(2026, 1, 18, 12, 0)
    session = SessionInfo(
        name="2601181200-AbCd12",
        created_at=now,
        last_activity=now,
        history_files=[],
        metadata={
            "last_history_by_agent": {
                "alpha": 3,
                5: 2,
                " beta ": 1,
            }
        },
    )

    compact = format_session_entries([session], None, mode="compact")

    assert compact
    assert "2 agents: alpha, beta" in compact[0]
    assert "5" not in compact[0]


def test_format_session_entries_rejects_unknown_mode() -> None:
    now = datetime(2026, 1, 18, 12, 0)
    session = SessionInfo(
        name="2601181200-AbCd12",
        created_at=now,
        last_activity=now,
        history_files=[],
        metadata={},
    )

    with pytest.raises(ValueError, match="typo"):
        format_session_entries([session], None, mode="typo")  # ty: ignore[invalid-argument-type]


def test_format_history_summary_pluralizes_message_counts() -> None:
    assert (
        format_history_summary({"alpha": 1, "beta": 2})
        == "alpha (1 message), beta (2 messages)"
    )


def test_format_session_agent_label_pluralizes_agent_count() -> None:
    assert (
        format_session_agent_label(
            SessionEntrySummary(
                index=1,
                display_name="session",
                is_current=False,
                is_pinned=False,
                timestamp="Jan 18 12:00",
                agent_count=1,
                agent_label="alpha",
            )
        )
        == "1 agent: alpha"
    )
    assert (
        format_session_agent_label(
            SessionEntrySummary(
                index=1,
                display_name="session",
                is_current=False,
                is_pinned=False,
                timestamp="Jan 18 12:00",
                agent_count=2,
                agent_label="alpha, beta",
            )
        )
        == "2 agents: alpha, beta"
    )
