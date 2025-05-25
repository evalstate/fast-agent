import asyncio
import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mcp import ListToolsResult

# Enable debug logging for the test
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.timeout(30)  # 30 seconds timeout
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_list_changes(fast_agent):
    fast = fast_agent
    print("Starting tool list change test")

    @fast.agent(name="test", instruction="here are your instructions", servers=["dynamic_tool"])
    async def agent_function():
        print("Initializing agent")
        async with fast.run() as app:
            # Initially there should be one tool (check_weather)
            tools: ListToolsResult = await app.test.list_tools()
            assert 1 == len(tools.tools)
            assert "dynamic_tool-check_weather" == tools.tools[0].name

            # Calling check_weather will toggle the dynamic_tool and send a notification
            result = await app.test.send('***CALL_TOOL check_weather {"location": "New York"}')
            assert "sunny" in result

            # Wait for the tool list to be refreshed (with retry)
            await asyncio.sleep(0.5)

            tools = await app.test.list_tools()
            dynamic_tool_found = False
            # Check if dynamic_tool is in the list
            for tool in tools.tools:
                if tool.name == "dynamic_tool-dynamic_tool":
                    dynamic_tool_found = True
                    break

            # Verify the dynamic tool was added
            assert dynamic_tool_found, (
                "Dynamic tool was not added to the tool list after notification"
            )
            assert 2 == len(tools.tools), f"Expected 2 tools but found {len(tools.tools)}"

            # Call check_weather again to toggle the dynamic_tool off
            result = await app.test.send('***CALL_TOOL check_weather {"location": "Boston"}')
            assert "sunny" in result

            # Sleep between retries
            await asyncio.sleep(0.5)

            # Get the updated tool list
            tools = await app.test.list_tools()

            assert 1 == len(tools.tools)

    await agent_function()
