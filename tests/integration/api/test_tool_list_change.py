import asyncio
import logging

import pytest

# Enable debug logging for the test
logging.basicConfig(level=logging.DEBUG)


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
            tools_dict = await app.test.list_mcp_tools()
            # Check that we have the dynamic_tool server
            assert "dynamic_tool" in tools_dict
            assert 1 == len(tools_dict["dynamic_tool"])
            assert "check_weather" == tools_dict["dynamic_tool"][0].name

            # Calling check_weather will toggle the dynamic_tool and send a notification
            result = await app.test.send('***CALL_TOOL check_weather {"location": "New York"}')
            assert "sunny" in result

            # Wait for the tool list to be refreshed (with retry)
            await asyncio.sleep(0.5)

            tools_dict = await app.test.list_mcp_tools()
            dynamic_tool_found = False
            # Check if dynamic_tool is in the list
            if "dynamic_tool" in tools_dict:
                for tool in tools_dict["dynamic_tool"]:
                    if tool.name == "dynamic_tool":
                        dynamic_tool_found = True
                        break

            # Verify the dynamic tool was added
            assert dynamic_tool_found, (
                "Dynamic tool was not added to the tool list after notification"
            )
            total_tools = sum(len(tool_list) for tool_list in tools_dict.values())
            assert 2 == total_tools, f"Expected 2 tools but found {total_tools}"

            # Call check_weather again to toggle the dynamic_tool off
            result = await app.test.send('***CALL_TOOL check_weather {"location": "Boston"}')
            assert "sunny" in result

            # Sleep between retries
            await asyncio.sleep(0.5)

            # Get the updated tool list
            tools_dict = await app.test.list_mcp_tools()

            total_tools = sum(len(tool_list) for tool_list in tools_dict.values())
            assert 1 == total_tools

    await agent_function()
