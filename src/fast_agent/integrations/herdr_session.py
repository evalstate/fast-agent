"""Publish fast-agent session presentation through the optional Herdr integration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fast_agent.integrations.herdr_lifecycle import report_session_metadata
from fast_agent.llm.model_display_name import format_model_display_name
from fast_agent.session import extract_session_title
from fast_agent.ui.context_usage_display import format_compact_context_usage_percent
from fast_agent.utils.count_display import format_compact_count

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import Session, SessionManager


def _usage_metadata(
    agent: "AgentProtocol | None",
    *,
    model: str | None,
) -> tuple[str | None, str | None]:
    usage = agent.usage_accumulator if agent is not None else None
    context_percentage = (
        format_compact_context_usage_percent(usage.context_usage_percentage)
        if usage is not None
        else None
    )
    context_parts = [
        part for part in (context_percentage, format_model_display_name(model)) if part is not None
    ]
    context_usage = " - ".join(context_parts) or None

    if usage is None:
        return context_usage, None
    summary = usage.summary
    token_parts: list[str] = []
    if summary.prompt.total is not None:
        token_parts.append(f"{format_compact_count(summary.prompt.total)} in")
    if summary.completion.total is not None:
        token_parts.append(f"{format_compact_count(summary.completion.total)} out")
    return context_usage, " · ".join(token_parts) or None


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
    context_usage, token_usage = _usage_metadata(agent, model=model)
    report_session_metadata(
        session_id=session.info.name if session is not None else None,
        title=extract_session_title(metadata),
        model=model,
        agent_name=agent_name,
        pinned=metadata.get("pinned") is True,
        forked_from=forked_from if isinstance(forked_from, str) else None,
        context_usage=context_usage,
        token_usage=token_usage,
    )


def report_current_session(
    manager: "SessionManager | None",
    *,
    agent: "AgentProtocol | None" = None,
) -> None:
    if os.environ.get("HERDR_ENV") != "1" or manager is None:
        return
    report_session(manager.current_session, agent=agent)
