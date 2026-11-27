from __future__ import annotations

import pytest
from mcp_types import ImageContent, TextContent

from fast_agent.agents.current_user_message import snapshot_current_user_message
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import TOOL_DEFINITIONS_META
from fast_agent.types import PromptMessageExtended


@pytest.mark.unit
def test_current_user_snapshot_excludes_generated_tool_context() -> None:
    user_text = TextContent(type="text", text="summarise this ^tool:fetch")
    image = ImageContent(type="image", data="YWJj", mime_type="image/png")

    snapshot = snapshot_current_user_message(
        [
            PromptMessageExtended(
                role="user",
                content=[
                    user_text,
                    image,
                    TextContent(
                        type="text",
                        text="<fastagent:tool-definitions>def fetch(): ...",
                        meta={TOOL_DEFINITIONS_META: ["fetch"]},
                    ),
                    TextContent(
                        type="text",
                        text="<fastagent:tool-sources>server: docs",
                        meta={TOOL_SOURCES_FINGERPRINT_META: "abc"},
                    ),
                ],
            )
        ]
    )

    assert snapshot is not None
    assert snapshot.content == (user_text, image)


@pytest.mark.unit
def test_current_user_snapshot_skips_mcp_events_only_wake_message() -> None:
    from fast_agent.mcp.events_extension import EventOccurrence
    from fast_agent.mcp.events_subscriptions import EventInboxItem, build_event_prompt_message

    human_text = TextContent(type="text", text="watch the build")
    human_message = PromptMessageExtended(role="user", content=[human_text])
    event_message = build_event_prompt_message(
        [
            EventInboxItem(
                "event",
                "k1",
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

    snapshot = snapshot_current_user_message([human_message, event_message])
    assert snapshot is not None
    assert snapshot.content == (human_text,)
    assert snapshot_current_user_message([event_message]) is None
