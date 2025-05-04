#!/usr/bin/env python3
"""
Simple MCP server that responds to tool calls with text and image content.
"""

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="Integration Server")


@app.tool(
    name="check_weather",
    description="Returns the weather for a specified location.",
)
def check_weather(location: str) -> str:
    # Write the location to a text file
    with open("weather_location.txt", "w") as f:
        f.write(location)

    # Return sunny weather condition
    return "It's sunny in " + location


@app.tool(name="shirt_colour", description="returns the colour of the shirt being worn")
def shirt_colour() -> str:
    return "blue polka dots"


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
