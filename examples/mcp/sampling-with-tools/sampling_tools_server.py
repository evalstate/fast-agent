"""
MCP server demonstrating "sampling with tools" feature.

This server provides a tool that uses sampling with tools to perform
calculations. The server manages the tool loop - calling create_message()
with calculator tools, executing the tools when requested, and continuing
until the LLM provides a final answer.
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
logger = logging.getLogger("sampling_tools_server")

mcp = FastMCP("Sampling With Tools Demo", log_level="DEBUG")

# Calculator tools that we'll pass to the sampling request
CALCULATOR_TOOLS = [
    Tool(
        name="add",
        description="Add two numbers together",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="subtract",
        description="Subtract second number from first",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="multiply",
        description="Multiply two numbers together",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="divide",
        description="Divide first number by second",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Numerator"},
                "b": {"type": "number", "description": "Denominator (must not be zero)"},
            },
            "required": ["a", "b"],
        },
    ),
]


# Secret code tool for testing - the LLM must call this to get the secret
SECRET_CODE_TOOL = Tool(
    name="get_secret",
    description="Returns a secret code. You must call this tool to get the secret.",
    inputSchema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)

SECRET_CODE = "WHISKEY-TANGO-FOXTROT-42"


@mcp.tool()
async def fetch_secret(ctx: Context) -> CallToolResult:
    """
    Test tool that uses sampling with tools to fetch a secret code.

    This demonstrates the sampling-with-tools flow:
    1. Server sends sampling request with get_secret tool
    2. LLM should call the get_secret tool
    3. Server executes the tool and returns the secret
    4. LLM returns the secret in its response

    If sampling-with-tools is working, the response will contain the secret code.
    """
    logger.info("fetch_secret called - testing sampling with tools")

    # Ask the LLM to get the secret using the tool
    result = await ctx.session.create_message(
        max_tokens=256,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Call the get_secret tool to retrieve the secret code, then tell me what it is.",
                ),
            )
        ],
        tools=[SECRET_CODE_TOOL],
        tool_choice=ToolChoice(mode="required"),  # Force the LLM to use the tool
    )

    logger.info(f"Received response with stopReason: {result.stopReason}")

    # If LLM wants to use the tool
    if result.stopReason == "toolUse":
        tool_uses = []
        if isinstance(result.content, list):
            tool_uses = [c for c in result.content if isinstance(c, ToolUseContent)]
        elif isinstance(result.content, ToolUseContent):
            tool_uses = [result.content]

        if tool_uses:
            # Execute the get_secret tool - return the secret code
            tool_results = [
                ToolResultContent(
                    type="tool_result",
                    toolUseId=tu.id,
                    content=[TextContent(type="text", text=f"SECRET: {SECRET_CODE}")],
                )
                for tu in tool_uses
            ]

            # Send the tool result back and get final response
            final_result = await ctx.session.create_message(
                max_tokens=256,
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text="Call the get_secret tool to retrieve the secret code, then tell me what it is.",
                        ),
                    ),
                    SamplingMessage(role="assistant", content=result.content),
                    SamplingMessage(role="user", content=tool_results),
                ],
                tools=[SECRET_CODE_TOOL],
            )

            # Extract the final text
            if isinstance(final_result.content, list):
                text_parts = [c.text for c in final_result.content if isinstance(c, TextContent)]
                final_text = "\n".join(text_parts)
            elif isinstance(final_result.content, TextContent):
                final_text = final_result.content.text
            else:
                final_text = str(final_result.content)

            logger.info(f"Final response: {final_text}")
            return CallToolResult(content=[TextContent(type="text", text=final_text)])

    # Fallback - sampling with tools didn't work as expected
    logger.warning("Tool was not called - sampling with tools may not be working")
    return CallToolResult(
        content=[TextContent(type="text", text="ERROR: Tool was not called")],
        isError=True,
    )


if __name__ == "__main__":
    logger.info("Starting sampling with tools demo server...")
    mcp.run()
