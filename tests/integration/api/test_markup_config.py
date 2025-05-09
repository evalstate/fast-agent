import pytest
from rich.errors import MarkupError

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_markup_raises_an_error(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
    )
    async def agent_function():
        async with fast.run() as agent:
            with pytest.raises(MarkupError):
                assert "test1" in await agent.agent1.send(Prompt.user("'[/]test1"))

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_markup_disabled_does_not_error(markup_fast_agent):
    @markup_fast_agent.agent(
        "agent2",
        instruction="You are a helpful AI Agent",
    )
    async def agent_function():
        async with markup_fast_agent.run() as agent:
            assert "test2" in await agent.agent2.send(Prompt.user("'[/]test2"))

    await agent_function()
