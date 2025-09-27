import pytest
from mcp.types import TextContent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_tool_flow(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="async_agent",
        instruction="Exercise async MCP tools",
        model="passthrough",
        servers=["async_tools_demo"],
    )
    async def async_agent_definition():
        async with fast.run() as app:
            agent = app.async_agent

            tools = await agent.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert "async_tools_demo-sync_ping" in tool_names

            async_tool = next(tool for tool in tools.tools if tool.name.endswith("async_uppercase"))
            assert getattr(async_tool, "invocationMode", None) == "async"

            result = await agent.call_tool(async_tool.name, {"message": "hello"})
            assert result.operation is not None
            assert not result.isError

            token = result.operation.token
            status = await agent.get_operation_status(token)
            assert status.status in {"submitted", "working"}

            final = await agent.wait_for_operation_result(token, poll_interval=0.05, timeout=5)
            assert not final.isError
            assert any(
                isinstance(block, TextContent) and block.text == "HELLO"
                for block in final.content
            )

    await async_agent_definition()
