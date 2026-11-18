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
