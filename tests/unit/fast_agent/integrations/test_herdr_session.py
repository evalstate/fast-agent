from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from fast_agent.integrations import herdr_session
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)

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
                "last_user_prompt": "Check the latest auth changes",
            },
            created_at=datetime.now(),
            last_activity=datetime.now(),
        ),
        directory=Path("/tmp/sessions/session-1"),
    )
    usage = UsageAccumulator(
        turns=[
            TurnUsage(
                provider=Provider.RESPONSES,
                usage_schema=UsageSchema.OPENAI_RESPONSES,
                model="model",
                prompt=PromptTokenUsage(total=300),
                completion=CompletionTokenUsage(total=50),
            )
        ]
    )
    usage.set_context_window_size(1_000)
    agent = SimpleNamespace(
        name="active-agent",
        config=SimpleNamespace(model="codexresponses.gpt-5.6-sol?reasoning=high"),
        context=None,
        usage_accumulator=usage,
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
            "model": "codexresponses.gpt-5.6-sol?reasoning=high",
            "agent_name": "saved-agent",
            "pinned": True,
            "forked_from": "session-0",
            "context_usage": "35.0% - gpt-5.6-sol",
            "token_usage": "300 in · 50 out",
            "prompt": "Check the latest auth changes",
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
            "context_usage": None,
            "token_usage": None,
            "prompt": None,
        }
    ]


def test_session_presentation_prefers_manual_title_over_latest_prompt() -> None:
    prompt = "latest prompt"

    assert herdr_session._session_presentation(
        {
            "title": "  Manual session title ",
            "last_user_prompt": prompt,
            "first_user_preview": "first prompt",
        }
    ) == ("Manual session title", prompt)


def test_session_presentation_uses_normalized_capped_latest_prompt() -> None:
    prompt = "Please review\n\n" + ("the latest changes " * 20)
    normalized = " ".join(prompt.split())[:200]

    assert herdr_session._session_presentation(
        {
            "last_user_prompt": prompt,
            "first_user_preview": "first prompt",
        }
    ) == (normalized, normalized)


def test_session_presentation_falls_back_for_older_sessions() -> None:
    assert herdr_session._session_presentation({"first_user_preview": " Original prompt "}) == (
        "Original prompt",
        None,
    )


def test_report_current_session_does_not_resolve_agent_when_inactive(monkeypatch) -> None:
    monkeypatch.delenv("HERDR_ENV", raising=False)

    herdr_session.report_current_session(
        cast("SessionManager", SimpleNamespace(current_session=None)),
        agent_lookup=lambda: (_ for _ in ()).throw(AssertionError("unexpected lookup")),
    )
