#!/usr/bin/env python3


from mcp.server.fastmcp import FastMCP

# Create the FastMCP server
app = FastMCP(name="An MCP Server", instructions="Here is how to use this server")

# Track if our dynamic tool is registered
dynamic_tool_registered = False


@app.tool(
    name="check_weather",
    description="Returns the weather for a specified location.",
)
async def check_weather(location: str) -> str:
    """The location to check"""
    global dynamic_tool_registered

    # Get the current context which gives us access to the session
    context = app.get_context()

    # Toggle the dynamic tool
    if dynamic_tool_registered:
        # Remove the tool by recreating the tool manager's tool list
        # This is a simple approach for testing purposes
        app._tool_manager._tools = {
            name: tool for name, tool in app._tool_manager._tools.items() if name != "dynamic_tool"
        }
        dynamic_tool_registered = False
    else:
        # Add a new tool dynamically
        app.add_tool(
            lambda: "This is a dynamic tool",
            name="dynamic_tool",
            description="A tool that was added dynamically",
        )
        dynamic_tool_registered = True

    # Send notification that the tool list has changed
    await context.session.send_tool_list_changed()

    # Return weather condition
    return "It's sunny in " + location


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
