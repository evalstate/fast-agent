"""Publish fast-agent session presentation through the optional Herdr integration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fast_agent.integrations.herdr_lifecycle import report_session_metadata
from fast_agent.session import extract_session_title

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import Session, SessionManager


def report_session(
    session: "Session | None",
    *,
    agent: "AgentProtocol | None" = None,
) -> None:
    if os.environ.get("HERDR_ENV") != "1":
        return
    metadata = session.info.metadata if session is not None else {}
    configured_agent_name = metadata.get("agent_name")
    agent_name = (
        configured_agent_name
        if isinstance(configured_agent_name, str) and configured_agent_name
        else agent.name
        if agent is not None
        else None
    )
    configured_model = metadata.get("model")
    model = (
        agent.config.model
        if agent is not None and agent.config.model
        else configured_model
        if isinstance(configured_model, str) and configured_model
        else None
    )
    forked_from = metadata.get("forked_from")
    report_session_metadata(
        session_id=session.info.name if session is not None else None,
        title=extract_session_title(metadata),
        model=model,
        agent_name=agent_name,
        pinned=metadata.get("pinned") is True,
        forked_from=forked_from if isinstance(forked_from, str) else None,
    )


def report_current_session(
    manager: "SessionManager | None",
    *,
    agent: "AgentProtocol | None" = None,
) -> None:
    if os.environ.get("HERDR_ENV") != "1" or manager is None:
        return
    report_session(manager.current_session, agent=agent)
