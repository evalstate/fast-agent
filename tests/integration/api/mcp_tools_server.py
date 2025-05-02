#!/usr/bin/env python3
"""
Simple MCP server that responds to tool calls with text and image content.
"""

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="An MCP Server", instructions="Here is how to use this server")


@app.tool(
    name="check_weather",
    description="Returns the weather for a specified location.",
)
def check_weather(location: str) -> str:
    """The location to check"""
    # Write the location to a text file
    with open("weather_location.txt", "w") as f:
        f.write(location)

    # Return sunny weather condition
    return "It's sunny in " + location


@app.tool(name="shirt-colour", description="Returns the colour of a shirt.")
def shirt_colour() -> str:
    return "blue polka dots"


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
