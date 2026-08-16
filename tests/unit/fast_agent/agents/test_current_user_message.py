import asyncio

import pytest
from mcp_types import CallToolResult, ImageContent, TextContent

from fast_agent.agents.current_user_message import (
    get_current_user_message,
    reset_current_user_message,
    set_current_user_message,
    snapshot_current_user_message,
)
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import PromptMessageExtended


@pytest.mark.unit
def test_current_user_snapshot_copies_only_latest_eligible_content() -> None:
    image = ImageContent(type="image", data="YWJj", mime_type="image/png")
    snapshot = snapshot_current_user_message(
        [
            Prompt.user("older"),
            PromptMessageExtended(
                role="user",
                content=[text_content("latest"), image],
                channels={"private": [text_content("ignored")]},
            ),
            PromptMessageExtended(
                role="user",
                content=[text_content("tool result")],
                tool_results={"call": CallToolResult(content=[text_content("done")])},
            ),
            PromptMessageExtended(
                role="user",
                content=[text_content("template")],
                is_template=True,
            ),
        ]
    )

    assert snapshot is not None
    assert snapshot.content[0] == text_content("latest")
    assert snapshot.content[1] == image
    assert snapshot.content[1] is not image


@pytest.mark.unit
@pytest.mark.asyncio
async def test_current_user_context_is_task_local_and_nested() -> None:
    first = snapshot_current_user_message([Prompt.user("first")])
    second = snapshot_current_user_message([Prompt.user("second")])
    assert first is not None
    assert second is not None

    async def read_in_task(message) -> str:
        token = set_current_user_message(message)
        try:
            await asyncio.sleep(0)
            current = get_current_user_message()
            assert current is not None
            content = current.content[0]
            assert isinstance(content, TextContent)
            return content.text
        finally:
            reset_current_user_message(token)

    outer_token = set_current_user_message(first)
    try:
        inner_token = set_current_user_message(second)
        reset_current_user_message(inner_token)
        assert get_current_user_message() == first
        assert await asyncio.gather(read_in_task(first), read_in_task(second)) == [
            "first",
            "second",
        ]
        assert get_current_user_message() == first
    finally:
        reset_current_user_message(outer_token)

    assert get_current_user_message() is None
