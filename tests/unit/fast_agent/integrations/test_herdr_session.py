from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from fast_agent.integrations import herdr_session

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import Session, SessionManager


def test_report_session_maps_persisted_and_active_metadata(monkeypatch) -> None:
    reports: list[dict[str, object]] = []
    session = SimpleNamespace(
        info=SimpleNamespace(
            name="session-1",
            metadata={
                "title": "Review auth",
                "agent_name": "saved-agent",
                "pinned": True,
                "forked_from": "session-0",
            },
            created_at=datetime.now(),
            last_activity=datetime.now(),
        ),
        directory=Path("/tmp/sessions/session-1"),
    )
    agent = SimpleNamespace(
        name="active-agent",
        config=SimpleNamespace(model="provider.model?reasoning=high"),
        context=None,
    )
    monkeypatch.setattr(
        herdr_session,
        "report_session_metadata",
        lambda **kwargs: reports.append(kwargs),
    )
    monkeypatch.setenv("HERDR_ENV", "1")

    herdr_session.report_session(
        cast("Session", session),
        agent=cast("AgentProtocol", agent),
    )

    assert reports == [
        {
            "session_id": "session-1",
            "title": "Review auth",
            "model": "provider.model?reasoning=high",
            "agent_name": "saved-agent",
            "pinned": True,
            "forked_from": "session-0",
        }
    ]


def test_report_current_session_clears_metadata_without_a_session(monkeypatch) -> None:
    reports: list[dict[str, object]] = []
    monkeypatch.setattr(
        herdr_session,
        "report_session_metadata",
        lambda **kwargs: reports.append(kwargs),
    )
    monkeypatch.setenv("HERDR_ENV", "1")
    manager = SimpleNamespace(current_session=None)

    herdr_session.report_current_session(cast("SessionManager", manager))

    assert reports == [
        {
            "session_id": None,
            "title": None,
            "model": None,
            "agent_name": None,
            "pinned": False,
            "forked_from": None,
        }
    ]
