from typing import TYPE_CHECKING

from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from anthropic.types import MessageParam


def test_sanitizes_tool_use_ids_for_assistant_calls():
    dirty_id = "functions.fetch_magic_string:0"
    expected = "functions_fetch_magic_string_0"
    params = CallToolRequestParams(name="fetch_magic_string", arguments={})
    req = CallToolRequest(params=params)

    msg = PromptMessageExtended(role="assistant", content=[], tool_calls={dirty_id: req})

    converted: MessageParam = AnthropicConverter.convert_to_anthropic(msg)

    assert converted["role"] == "assistant"
    content_blocks = list(converted["content"])
    assert isinstance(content_blocks[0], dict)
    assert content_blocks[0]["id"] == expected


def test_sanitizes_tool_use_ids_for_tool_results():
    dirty_id = "functions.fetch_magic_string:0"
    expected = "functions_fetch_magic_string_0"
    result = CallToolResult(content=[TextContent(type="text", text="done")], isError=False)

    msg = PromptMessageExtended(role="user", content=[], tool_results={dirty_id: result})

    converted: MessageParam = AnthropicConverter.convert_to_anthropic(msg)

    assert converted["role"] == "user"
    content_blocks = list(converted["content"])
    assert isinstance(content_blocks[0], dict)
    assert content_blocks[0]["tool_use_id"] == expected
