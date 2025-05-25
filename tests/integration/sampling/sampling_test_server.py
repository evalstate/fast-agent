"""
Enhanced test server for sampling functionality
"""

import logging
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, SamplingMessage, TextContent

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("sampling_server")

# Create MCP server
mcp = FastMCP("MCP Root Tester", log_level="DEBUG")


@mcp.tool()
async def sample(ctx: Context, to_sample: str | None = "hello, world") -> CallToolResult:
    """Tool that echoes back the input parameter"""
    try:
        logger.info(f"Sample tool called with to_sample={to_sample!r}")

        # Handle None value - use default if to_sample is None
        value = to_sample if to_sample is not None else "hello, world"

        # Send message to LLM but we don't use the response
        # This creates the sampling context
        await ctx.session.create_message(
            max_tokens=1024,
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=value))],
        )

        # Return the result directly, without nesting
        logger.info(f"Returning value: {value}")
        return CallToolResult(content=[TextContent(type="text", text=value)])
    except Exception as e:
        logger.error(f"Error in sample tool: {e}", exc_info=True)
        # Ensure we always include the content field in the error response
        return CallToolResult(isError=True, content=[TextContent(type="text", text=f"Error: {str(e)}")])


@mcp.tool()
async def sample_many(ctx: Context) -> CallToolResult:
    """Tool that echoes back the input parameter"""

    result = await ctx.session.create_message(
        max_tokens=1024,
        messages=[
            SamplingMessage(role="user", content=TextContent(type="text", text="message 1")),
            SamplingMessage(role="user", content=TextContent(type="text", text="message 2")),
        ],
    )

    # Return the result directly, without nesting
    return CallToolResult(content=[TextContent(type="text", text=str(result))])


if __name__ == "__main__":
    logger.info("Starting sampling test server...")
    mcp.run()
