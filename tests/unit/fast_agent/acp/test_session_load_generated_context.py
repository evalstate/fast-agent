from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from acp.schema import AgentMessageChunk, ImageContentBlock, TextContentBlock, UserMessageChunk
from mcp_types import ImageContent, TextContent

from fast_agent.acp.server.session_store import ACPServerSessionStore, SessionStoreHost
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import TOOL_DEFINITIONS_META
from fast_agent.types import PromptMessageExtended


@pytest.mark.unit
def test_session_load_replays_only_user_authored_chunks() -> None:
    store = ACPServerSessionStore(cast("SessionStoreHost", SimpleNamespace()))

    updates = store.build_history_updates(
        [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="describe ^tool:fetch"),
                    ImageContent(type="image", data="YWJj", mime_type="image/png"),
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
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="done")],
            ),
        ]
    )

    assert [type(update) for update in updates] == [
        UserMessageChunk,
        UserMessageChunk,
        AgentMessageChunk,
    ]
    first, second, _ = (update.content for update in updates)
    assert isinstance(first, TextContentBlock)
    assert first.text == "describe ^tool:fetch"
    assert isinstance(second, ImageContentBlock)
