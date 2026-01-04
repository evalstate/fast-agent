import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_function_tools_from_card(fast_agent):
    fast_agent.load_agents("agent.md")
    async with fast_agent.run() as agent:
        tools = await agent.calc.list_tools()
        assert any(t.name == "add" for t in tools.tools)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_function_tools_from_decorator(fast_agent):
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @fast_agent.agent(name="calc_decorator", model="passthrough", function_tools=[add])
    async def calc_decorator():
        return None

    async with fast_agent.run() as agent:
        tools = await agent.calc_decorator.list_tools()
        assert any(t.name == "add" for t in tools.tools)
