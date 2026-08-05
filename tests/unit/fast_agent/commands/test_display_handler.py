from types import SimpleNamespace

import pytest
from mcp.types import TextContent

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.display import handle_show_markdown
from fast_agent.types import PromptMessageExtended


@pytest.mark.asyncio
async def test_show_markdown_returns_literal_assistant_source() -> None:
    source = (
        "## Heading\n\n"
        "- **bold**\n"
        "- [label](https://example.test)\n"
        "- [bold]literal Rich markup[/bold]"
    )
    agent = SimpleNamespace(
        llm=object(),
        message_history=[
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text=source)],
            )
        ],
    )
    ctx = CommandContext(
        agent_provider=StaticAgentProvider({"dev": agent}),
        current_agent_name="dev",
        io=NonInteractiveCommandIOBase(),
        no_home=True,
    )

    outcome = await handle_show_markdown(ctx, agent_name="dev")

    assert len(outcome.messages) == 1
    message = outcome.messages[0]
    assert message.plain_text() == source
    assert message.render_markdown is False
    assert message.verbatim is True
    assert message.title == "Last Assistant Response"
    assert message.agent_name == "dev"
