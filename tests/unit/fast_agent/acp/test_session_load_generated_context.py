from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from acp.schema import AgentMessageChunk, ImageContentBlock, TextContentBlock, UserMessageChunk
from mcp_types import ImageContent, TextContent, Tool

from fast_agent.acp.server.session_store import ACPServerSessionStore, SessionStoreHost
from fast_agent.config import ProgrammaticToolCallingSettings
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import expand_tool_mentions
from fast_agent.ptc.runtime import ProgrammaticToolRuntime
from fast_agent.session.session_manager import normalize_user_prompt
from fast_agent.types import PromptMessageExtended


@pytest.mark.unit
@pytest.mark.parametrize("typed", ["describe ^tool:fetch", "^tool:fetch"])
def test_session_load_replays_only_user_authored_chunks(typed: str) -> None:
    store = ACPServerSessionStore(cast("SessionStoreHost", SimpleNamespace()))
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    surface = runtime.surface(runtime.model_tools([Tool(name="fetch", input_schema={})]))
    message = PromptMessageExtended(
        role="user",
        content=[
            TextContent(type="text", text=typed),
            ImageContent(type="image", data="YWJj", mime_type="image/png"),
        ],
    )
    expand_tool_mentions(surface, message)
    message.content.append(
        TextContent(
            type="text",
            text="<fastagent:tool-sources>server: docs",
            meta={TOOL_SOURCES_FINGERPRINT_META: "abc"},
        )
    )
    updates = store.build_history_updates(
        [
            message,
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
    assert first.text == typed
    assert isinstance(second, ImageContentBlock)
    assert normalize_user_prompt(message) == typed
    assert isinstance(message.content[0], TextContent)
    assert message.content[0].text == typed.removesuffix("^tool:fetch").rstrip()
