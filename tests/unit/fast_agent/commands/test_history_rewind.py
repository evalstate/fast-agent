from unittest.mock import Mock

import pytest
from mcp_types import TextContent

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.history import handle_history_rewind
from fast_agent.types import PromptMessageExtended


@pytest.mark.parametrize("original_text", ["  Inspect ^tool:jobs_list ^file:notes.md  ", None])
@pytest.mark.asyncio
async def test_rewind_prefills_the_original_user_text(original_text: str | None) -> None:
    prompt = PromptMessageExtended(
        role="user",
        content=[
            TextContent(
                type="text",
                text="Inspect",
                meta={"fast_agent_original_text": original_text}
                if original_text is not None
                else None,
            ),
            TextContent(
                type="text",
                text="<fastagent:tool-sources>sources</fastagent:tool-sources>",
                meta={"fast_agent_tool_sources_fingerprint": "fingerprint"},
            ),
        ],
    )
    agent = Mock(message_history=[prompt])
    ctx = CommandContext(
        agent_provider=Mock(_agent=Mock(return_value=agent)),
        current_agent_name="main",
        io=Mock(),
    )

    outcome = await handle_history_rewind(ctx, agent_name="main", turn_index=1)

    assert outcome.buffer_prefill == (original_text if original_text is not None else "Inspect")
    agent.load_message_history.assert_called_once_with([])
