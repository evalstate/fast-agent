"""
Test server for sampling with tools integration tests.

This server provides tools that test the sampling with tools functionality.
"""

import logging
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import (
    CallToolResult,
    SamplingMessage,
    TextContent,
    Tool,
    ToolChoice,
    ToolResultContent,
    ToolUseContent,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("sampling_tools_test_server")

mcp = FastMCP("Sampling Tools Test Server", log_level="DEBUG")

# Simple test tool definitions
TEST_TOOLS = [
    Tool(
        name="echo",
        description="Echo back the input",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"},
            },
            "required": ["message"],
        },
    ),
]


@mcp.tool()
async def test_sampling_with_tools(ctx: Context, message: str) -> CallToolResult:
    """
    Test sampling with tools - sends a request with tools and checks the response.

    This tool verifies that:
    1. The sampling request with tools is properly sent
    2. The client processes tools correctly
    """
    logger.info(f"test_sampling_with_tools called with message: {message}")

    # Send sampling request with tools
    result = await ctx.session.create_message(
        max_tokens=256,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=message),
            )
        ],
        tools=TEST_TOOLS,
        tool_choice=ToolChoice(mode="auto"),
    )

    logger.info(f"Received result: stopReason={result.stopReason}")

    # Return info about what we received
    info = f"stopReason={result.stopReason}, model={result.model}"
    return CallToolResult(
        content=[TextContent(type="text", text=f"Sampling completed: {info}")]
    )


@mcp.tool()
async def test_sampling_without_tools(ctx: Context, message: str) -> CallToolResult:
    """
    Test sampling without tools - verifies backward compatibility.
    """
    logger.info(f"test_sampling_without_tools called with message: {message}")

    result = await ctx.session.create_message(
        max_tokens=256,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=message),
            )
        ],
    )

    logger.info(f"Received result: stopReason={result.stopReason}")

    # Extract text from result
    if isinstance(result.content, TextContent):
        response_text = result.content.text
    elif isinstance(result.content, list):
        response_text = " ".join(
            c.text for c in result.content if isinstance(c, TextContent)
        )
    else:
        response_text = str(result.content)

    return CallToolResult(
        content=[TextContent(type="text", text=f"Response: {response_text}")]
    )


@mcp.tool()
async def test_tool_result_handling(ctx: Context) -> CallToolResult:
    """
    Test a multi-turn tool conversation.

    This sends an initial request, receives a tool use response,
    then sends tool results back.
    """
    logger.info("test_tool_result_handling called")

    # First request - ask for tool use
    result = await ctx.session.create_message(
        max_tokens=256,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text="Use the echo tool to say hello"),
            )
        ],
        tools=TEST_TOOLS,
        tool_choice=ToolChoice(mode="required"),  # Force tool use
    )

    logger.info(f"First result: stopReason={result.stopReason}")

    # With passthrough model, we might not get a tool use response
    # Just verify we got a response
    if result.stopReason == "toolUse":
        # Extract tool uses
        tool_uses = []
        if isinstance(result.content, list):
            tool_uses = [c for c in result.content if isinstance(c, ToolUseContent)]
        elif isinstance(result.content, ToolUseContent):
            tool_uses = [result.content]

        if tool_uses:
            # Send follow-up with tool results
            tool_results = [
                ToolResultContent(
                    type="tool_result",
                    toolUseId=tu.id,
                    content=[TextContent(type="text", text="echo: hello")],
                )
                for tu in tool_uses
            ]

            # Second request with tool results
            final_result = await ctx.session.create_message(
                max_tokens=256,
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(
                            type="text", text="Use the echo tool to say hello"
                        ),
                    ),
                    SamplingMessage(role="assistant", content=result.content),
                    SamplingMessage(role="user", content=tool_results),
                ],
                tools=TEST_TOOLS,
            )

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Multi-turn completed: first={result.stopReason}, final={final_result.stopReason}",
                    )
                ]
            )

    # Single turn response
    return CallToolResult(
        content=[
            TextContent(type="text", text=f"Single turn: stopReason={result.stopReason}")
        ]
    )


if __name__ == "__main__":
    logger.info("Starting sampling tools test server...")
    mcp.run()
