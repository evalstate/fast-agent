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


@mcp.tool()
async def sample_parallel(ctx: Context, count: int = 5) -> CallToolResult:
    """Tool that makes multiple concurrent sampling requests to test parallel processing"""
    try:
        logger.info(f"Making {count} concurrent sampling requests")

        # Create multiple concurrent sampling requests
        import asyncio

        async def _send_sampling(request: int):
            return await ctx.session.create_message(
                max_tokens=100,
                messages=[SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=f"Parallel request {request+1}")
                )],
            )


        tasks = []
        for i in range(count):
            task = _send_sampling(i)
            tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*[_send_sampling(i) for i in range(count)])

        # Combine results
        response_texts = [result.content.text for result in results]
        combined_response = f"Completed {len(results)} parallel requests: " + ", ".join(response_texts[:3])
        if len(response_texts) > 3:
            combined_response += f"... and {len(response_texts) - 3} more"

        logger.info(f"Parallel sampling completed: {combined_response}")
        return CallToolResult(content=[TextContent(type="text", text=combined_response)])

    except Exception as e:
        logger.error(f"Error in sample_parallel tool: {e}", exc_info=True)
        return CallToolResult(isError=True, content=[TextContent(type="text", text=f"Error: {str(e)}")])


if __name__ == "__main__":
    logger.info("Starting sampling test server...")
    mcp.run()
