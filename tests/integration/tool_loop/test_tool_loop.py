import pytest

from mcp_agent.core.request_params import RequestParams


@pytest.mark.integration
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_loop(fast_agent):
    @fast_agent.agent(instruction="You are a helpful AI Agent")
    async def agent_function():
        async with fast_agent.run() as agent:
            await agent.default.generate("New implementation", RequestParams(max_iterations=0))

    await agent_function()
