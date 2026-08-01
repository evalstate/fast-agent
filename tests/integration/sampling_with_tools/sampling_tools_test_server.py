"""Modern MCPServer fixture for sampling-with-tools integration tests."""

from typing import Annotated

from mcp.server.mcpserver import MCPServer, Resolve, Sample
from mcp_types import (
    CreateMessageResult,
    CreateMessageResultWithTools,
    SamplingMessage,
    TextContent,
    Tool,
    ToolChoice,
    ToolResultContent,
    ToolUseContent,
)

TEST_TOOLS = [
    Tool(
        name="echo",
        description="Echo back the input",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to echo"}},
            "required": ["message"],
        },
    ),
]


def request_sampling_with_tools(message: str) -> Sample:
    return Sample(
        max_tokens=256,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=message))],
        tools=TEST_TOOLS,
        tool_choice=ToolChoice(mode="auto"),
    )


def request_sampling_without_tools(message: str) -> Sample:
    return Sample(
        max_tokens=256,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=message))],
    )


def request_tool_result_handling() -> Sample:
    return Sample(
        max_tokens=256,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text='***CALL_TOOL echo {"message": "hello"}',
                ),
            )
        ],
        tools=TEST_TOOLS,
        tool_choice=ToolChoice(mode="required"),
    )


def finish_tool_result_handling(
    first_result: Annotated[
        CreateMessageResultWithTools,
        Resolve(request_tool_result_handling),
    ],
) -> Sample | CreateMessageResultWithTools:
    content = first_result.content
    tool_uses = (
        [block for block in content if isinstance(block, ToolUseContent)]
        if isinstance(content, list)
        else [content]
        if isinstance(content, ToolUseContent)
        else []
    )
    if not tool_uses:
        return first_result

    tool_results = [
        ToolResultContent(
            type="tool_result",
            tool_use_id=tool_use.id,
            content=[TextContent(type="text", text="echo: hello")],
        )
        for tool_use in tool_uses
    ]
    return Sample(
        max_tokens=256,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text='***CALL_TOOL echo {"message": "hello"}',
                ),
            ),
            SamplingMessage(role="assistant", content=content),
            SamplingMessage(role="user", content=tool_results),
        ],
        tools=TEST_TOOLS,
    )


def text_content(result: CreateMessageResult | CreateMessageResultWithTools) -> str:
    content = result.content
    if isinstance(content, TextContent):
        return content.text
    if isinstance(content, list):
        return " ".join(block.text for block in content if isinstance(block, TextContent))
    return str(content)


server = MCPServer("Sampling Tools Test Server")


@server.tool()
def test_sampling_with_tools(
    message: str,
    result: Annotated[CreateMessageResultWithTools, Resolve(request_sampling_with_tools)],
) -> str:
    """Sample with a tool declaration through the modern resolver API."""
    del message
    return f"Sampling completed: stopReason={result.stop_reason}, model={result.model}"


@server.tool()
def test_sampling_without_tools(
    message: str,
    result: Annotated[CreateMessageResult, Resolve(request_sampling_without_tools)],
) -> str:
    """Sample without tools through the modern resolver API."""
    del message
    return f"Response: {text_content(result)}"


@server.tool()
def test_tool_result_handling(
    result: Annotated[
        CreateMessageResultWithTools,
        Resolve(finish_tool_result_handling),
    ],
) -> str:
    """Complete a second sampling turn when the model requests a tool."""
    return f"Multi-turn completed: stopReason={result.stop_reason}, response={text_content(result)}"


if __name__ == "__main__":
    server.run(transport="stdio")
