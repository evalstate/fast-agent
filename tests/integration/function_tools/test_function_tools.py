import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_function_tools_from_card(fast_agent):
    fast_agent.load_agents("agent.md")
    async with fast_agent.run() as agent:
        tools = await agent.calc.list_tools()
        assert any(t.name == "add" for t in tools.tools)
