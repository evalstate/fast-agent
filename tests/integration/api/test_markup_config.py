import pytest

from fast_agent.core.prompt import Prompt


@pytest.mark.integration
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
