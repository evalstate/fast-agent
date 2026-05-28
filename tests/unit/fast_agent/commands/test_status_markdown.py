from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.renderers.status_markdown import render_status_markdown
from fast_agent.commands.status_summaries import (
    ConversationStatsSummary,
    ErrorHandlingSummary,
    StatusSummary,
    build_status_summary,
)

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


def _summary(*, model_source: str | None) -> StatusSummary:
    return StatusSummary(
        fast_agent_version="1.2.3",
        client_info=None,
        model_summary=None,
        parallel_summary=None,
        model_source=model_source,
        conversation_stats=ConversationStatsSummary(
            agent_name="agent",
            turns=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            tool_calls=0,
            tool_successes=0,
            tool_errors=0,
            context_usage_line="Context Used: 0%",
        ),
        uptime_seconds=0.0,
        error_report=ErrorHandlingSummary(
            channel_label="Error Channel: fast-agent-error",
            recent_entries=[],
        ),
        warnings=[],
    )


def test_render_status_markdown_includes_model_source_when_present() -> None:
    rendered = render_status_markdown(_summary(model_source="last used model"), heading="status")

    assert "- Model Source: last used model" in rendered


def test_render_status_markdown_omits_model_source_when_missing() -> None:
    rendered = render_status_markdown(_summary(model_source=None), heading="status")

    assert "- Model Source:" not in rendered


def test_build_status_summary_prefers_agent_context_model_source() -> None:
    agent = SimpleNamespace(
        agent_type=AgentType.BASIC,
        name="agent",
        context=SimpleNamespace(config=SimpleNamespace(model_source="last used model")),
        llm=None,
        message_history=[],
    )

    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=cast("AgentProtocol", agent),
        client_info=None,
        client_capabilities=None,
        protocol_version=None,
        uptime_seconds=0.0,
        instance=None,
    )

    assert summary.model_source == "last used model"
