import pytest

from fast_agent.agents import McpAgent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_function_tools_from_card(fast_agent):
    fast_agent.load_agents("agent.md")
    async with fast_agent.run() as agent:
        tools = await agent.calc.list_tools()
        assert any(t.name == "add" for t in tools.tools)
        result = await agent.calc.call_tool("add", {"a": 2, "b": 3})
        assert result.isError is False
        assert result.structuredContent is None
        assert result.content is not None
        assert result.content[0].text == "5"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_function_tools_from_decorator(fast_agent):
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def summarize() -> dict[str, str]:
        """Return a summary."""
        return {"status": "ok"}

    @fast_agent.agent(
        name="calc_decorator",
        model="passthrough",
        function_tools=[add, summarize],
    )
    async def calc_decorator():
        return None

    async with fast_agent.run() as agent:
        tools = await agent.calc_decorator.list_tools()
        assert any(t.name == "add" for t in tools.tools)
        assert any(t.name == "summarize" for t in tools.tools)

        add_result = await agent.calc_decorator.call_tool("add", {"a": 4, "b": 5})
        assert add_result.isError is False
        assert add_result.structuredContent is None
        assert add_result.content is not None
        assert add_result.content[0].text == "9"

        summarize_result = await agent.calc_decorator.call_tool("summarize", {})
        assert summarize_result.isError is False
        assert summarize_result.structuredContent is None
        assert summarize_result.content is not None
        assert summarize_result.content[0].text == '{"status":"ok"}'


@pytest.mark.integration
@pytest.mark.asyncio
async def test_function_tools_from_custom_decorator(fast_agent):
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @fast_agent.custom(
        McpAgent,
        name="custom_calc_decorator",
        model="passthrough",
        function_tools=[add],
    )
    async def custom_calc_decorator():
        return None

    async with fast_agent.run() as agent:
        tools = await agent.custom_calc_decorator.list_tools()
        assert any(t.name == "add" for t in tools.tools)

        add_result = await agent.custom_calc_decorator.call_tool("add", {"a": 6, "b": 7})
        assert add_result.isError is False
        assert add_result.structuredContent is None
        assert add_result.content is not None
        assert add_result.content[0].text == "13"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_global_function_tools_are_available_to_supported_custom_agents(fast_agent):
    @fast_agent.tool
    def ping() -> str:
        """Return a marker string."""
        return "pong"

    @fast_agent.custom(
        McpAgent,
        name="custom_global_tools",
        model="passthrough",
    )
    async def custom_global_tools():
        return None

    async with fast_agent.run() as agent:
        tools = await agent.custom_global_tools.list_tools()
        assert any(t.name == "ping" for t in tools.tools)

        ping_result = await agent.custom_global_tools.call_tool("ping", {})
        assert ping_result.isError is False
        assert ping_result.structuredContent is None
        assert ping_result.content is not None
        assert ping_result.content[0].text == "pong"
