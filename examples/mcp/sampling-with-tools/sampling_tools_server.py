"""MCPServer example for sampling with client-provided tools."""

from typing import Annotated

from mcp.server.mcpserver import MCPServer, Resolve, Sample
from mcp_types import (
    CreateMessageResultWithTools,
    SamplingMessage,
    TextContent,
    Tool,
    ToolChoice,
    ToolResultContent,
    ToolUseContent,
)

CALCULATOR_TOOLS = [
    Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    )
    for name, description in (
        ("add", "Add two numbers together"),
        ("subtract", "Subtract second number from first"),
        ("multiply", "Multiply two numbers together"),
        ("divide", "Divide first number by second"),
    )
]

SECRET_CODE_TOOL = Tool(
    name="get_secret",
    description="Returns a secret code. You must call this tool to get the secret.",
    input_schema={"type": "object", "properties": {}, "required": []},
)
SECRET_CODE = "WHISKEY-TANGO-FOXTROT-42"
SECRET_PROMPT = "Call the get_secret tool to retrieve the secret code, then tell me what it is."


def request_secret() -> Sample:
    return Sample(
        max_tokens=256,
        messages=[
            SamplingMessage(role="user", content=TextContent(type="text", text=SECRET_PROMPT))
        ],
        tools=[SECRET_CODE_TOOL],
        tool_choice=ToolChoice(mode="required"),
    )


def finish_secret(
    first_result: Annotated[CreateMessageResultWithTools, Resolve(request_secret)],
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
            content=[TextContent(type="text", text=f"SECRET: {SECRET_CODE}")],
        )
        for tool_use in tool_uses
    ]
    return Sample(
        max_tokens=256,
        messages=[
            SamplingMessage(role="user", content=TextContent(type="text", text=SECRET_PROMPT)),
            SamplingMessage(role="assistant", content=content),
            SamplingMessage(role="user", content=tool_results),
        ],
        tools=[SECRET_CODE_TOOL],
    )


def response_text(result: CreateMessageResultWithTools) -> str:
    content = result.content
    if isinstance(content, TextContent):
        return content.text
    if isinstance(content, list):
        return "\n".join(block.text for block in content if isinstance(block, TextContent))
    return str(content)


server = MCPServer("Sampling With Tools Demo")


@server.tool()
def fetch_secret(
    result: Annotated[CreateMessageResultWithTools, Resolve(finish_secret)],
) -> str:
    """Ask the client model to call ``get_secret`` and return its final answer."""
    return response_text(result)


if __name__ == "__main__":
    server.run(transport="stdio")
