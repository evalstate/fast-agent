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
    from collections.abc import Callable

    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import Session, SessionManager

_PROMPT_LIMIT = 200


def _metadata_text(
    metadata: dict[str, object], key: str, *, limit: int | None = None
) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized:
        return None
    return normalized[:limit] if limit is not None else normalized


def _session_presentation(metadata: dict[str, object]) -> tuple[str | None, str | None]:
    prompt = _metadata_text(metadata, "last_user_prompt", limit=_PROMPT_LIMIT)
    manual_title = _metadata_text(metadata, "title") or _metadata_text(metadata, "label")
    display_title = manual_title or prompt or extract_session_title(metadata)
    return display_title, prompt


def _reported_model(agent: "AgentProtocol | None", configured_model: object) -> str | None:
    if agent is not None:
        agent_model = agent.config.model
        if agent_model and agent_model.startswith("$") and agent.llm is not None:
            return agent.llm.resolved_model.wire_model_name
        if agent_model:
            return agent_model
    return configured_model if isinstance(configured_model, str) and configured_model else None


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
    costs = [turn.cost_usd for turn in usage.turns]
    if costs and all(cost is not None for cost in costs):
        total_cost_usd = sum(cost for cost in costs if cost is not None)
        return context_usage, f"${total_cost_usd:.4f}"

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
    model = _reported_model(agent, metadata.get("model"))
    forked_from = metadata.get("forked_from")
    display_title, prompt = _session_presentation(metadata)
    context_usage, token_usage = _usage_metadata(agent, model=model)
    report_session_metadata(
        session_id=session.info.name if session is not None else None,
        title=display_title,
        model=model,
        agent_name=agent_name,
        pinned=metadata.get("pinned") is True,
        forked_from=forked_from if isinstance(forked_from, str) else None,
        context_usage=context_usage,
        token_usage=token_usage,
        prompt=prompt,
    )


def report_current_session(
    manager: "SessionManager | None",
    *,
    agent: "AgentProtocol | None" = None,
    agent_lookup: "Callable[[], AgentProtocol | None] | None" = None,
) -> None:
    if os.environ.get("HERDR_ENV") != "1" or manager is None:
        return
    if agent is None and agent_lookup is not None:
        agent = agent_lookup()
    report_session(manager.current_session, agent=agent)
