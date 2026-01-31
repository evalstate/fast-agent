from mcp import Tool
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsResult

from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM
from fast_agent.llm.provider.bedrock.multipart_converter_bedrock import BedrockConverter
from fast_agent.types import PromptMessageExtended


def test_bedrock_converter_emits_tool_use_items():
    tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="demo-tool", arguments={"x": 1}),
    )
    msg = PromptMessageExtended(role="assistant", content=[], tool_calls={"call_1": tool_call})

    converted = BedrockConverter.convert_to_bedrock(msg)

    assert converted["role"] == "assistant"
    content = list(converted["content"])
    assert content[0]["type"] == "tool_use"
    assert content[0]["id"] == "call_1"
    assert content[0]["name"] == "demo-tool"
    assert content[0]["input"] == {"x": 1}


def test_bedrock_convert_messages_to_bedrock_includes_tool_use_block():
    tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="demo-tool", arguments={"x": 1}),
    )
    msg = PromptMessageExtended(role="assistant", content=[], tool_calls={"call_1": tool_call})

    converted = BedrockConverter.convert_to_bedrock(msg)
    llm = object.__new__(BedrockLLM)

    output = BedrockLLM._convert_messages_to_bedrock(llm, [converted])

    assert output
    content = output[0]["content"]
    assert any("toolUse" in block for block in content)
    tool_use = next(block["toolUse"] for block in content if "toolUse" in block)
    assert tool_use["toolUseId"] == "call_1"
    assert tool_use["name"] == "demo-tool"
    assert tool_use["input"] == {"x": 1}


def test_resolve_tool_use_name_uses_mapped_name():
    tool_list = ListToolsResult(
        tools=[Tool(name="my-tool", description="demo", inputSchema={"type": "object"})]
    )
    tool_name_mapping = {"my_tool": "my-tool"}

    resolved = BedrockLLM._resolve_tool_use_name(
        "call_1_my-tool", tool_list, tool_name_mapping
    )

    assert resolved == "my_tool"
