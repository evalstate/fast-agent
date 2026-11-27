from __future__ import annotations

import pytest
from mcp_types import ImageContent, TextContent

from fast_agent.acp.command_io import render_history_turn_text
from fast_agent.commands.history_summaries import build_history_overview, build_history_turn_report
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import TOOL_DEFINITIONS_META
from fast_agent.ui.history_display_rows import build_history_rows

ROSTER = "<fastagent:tool-sources>server: docs</fastagent:tool-sources>"
DEFINITIONS = "<fastagent:tool-definitions>def fetch(): ...</fastagent:tool-definitions>"


def _history() -> list[PromptMessageExtended]:
    return [
        PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="what changed?"),
                TextContent(type="text", text=DEFINITIONS, meta={TOOL_DEFINITIONS_META: ["fetch"]}),
                TextContent(type="text", text=ROSTER, meta={TOOL_SOURCES_FINGERPRINT_META: "abc"}),
            ],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="nothing")],
        ),
    ]


def _assert_user_text_only(text: str) -> None:
    assert "what changed?" in text
    assert "fastagent:tool" not in text


@pytest.mark.unit
def test_history_overview_and_turn_previews_exclude_generated_tool_context() -> None:
    history = _history()

    overview = build_history_overview(history)
    report = build_history_turn_report(history)

    assert overview.recent_messages[0].snippet == "what changed?"
    assert [turn.user_snippet for turn in report.turns] == ["what changed?"]


@pytest.mark.unit
def test_history_turn_text_and_rows_exclude_generated_tool_context() -> None:
    history = _history()

    image_only = PromptMessageExtended(
        role="user",
        content=[
            ImageContent(type="image", data="YWJj", mime_type="image/png"),
            TextContent(type="text", text=ROSTER, meta={TOOL_SOURCES_FINGERPRINT_META: "abc"}),
        ],
    )

    _assert_user_text_only(render_history_turn_text("agent", history))
    rows = build_history_rows([*history, image_only])
    assert rows[0].preview == "what changed?"
    assert rows[-1].preview == "<no text>"


@pytest.mark.unit
def test_acp_history_turn_text_renders_heading_for_mcp_events() -> None:
    from fast_agent.mcp.events_extension import EventOccurrence
    from fast_agent.mcp.events_subscriptions import EventInboxItem, build_event_prompt_message

    event_message = build_event_prompt_message(
        [
            EventInboxItem(
                "event",
                "sub1",
                "jobs",
                "job.status_changed",
                occurrence=EventOccurrence(
                    event_id="ev-1",
                    name="job.status_changed",
                    timestamp="2026-11-12T10:00:00Z",
                    data={"stage": "DONE"},
                ),
            )
        ]
    )

    rendered = render_history_turn_text("agent", [event_message])
    assert "- user: events · jobs/job.status_changed" in rendered
    assert '"stage": "DONE"' in rendered
    assert "<no text>" not in rendered
