import pytest
from mcp import Tool
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsResult

from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM
from fast_agent.llm.provider.bedrock.multipart_converter_bedrock import BedrockConverter
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import PromptMessageExtended, RequestParams


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


@pytest.mark.asyncio
async def test_bedrock_structured_schema_path_preserves_tools(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    tool = Tool(
        name="lookup",
        description="Lookup data.",
        inputSchema={"type": "object", "properties": {}},
    )
    captured_tools = None

    llm = object.__new__(BedrockLLM)
    llm.default_request_params = RequestParams(model="amazon.nova-lite-v1:0")

    async def structured_schema(
        multipart_messages,
        schema_arg,
        request_params=None,
        tools=None,
    ):
        nonlocal captured_tools
        del multipart_messages, schema_arg, request_params
        captured_tools = tools
        return None, Prompt.assistant("ok")

    monkeypatch.setattr(
        llm,
        "_apply_prompt_provider_specific_structured_schema",
        structured_schema,
    )

    result = await BedrockLLM._apply_prompt_provider_specific(
        llm,
        [Prompt.user("call the tool")],
        RequestParams(structured_schema=schema),
        [tool],
    )

    assert result.last_text() == "ok"
    assert captured_tools == [tool]


@pytest.mark.asyncio
async def test_bedrock_structured_schema_prompt_preserves_history_and_tool_context(monkeypatch):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    tool = Tool(
        name="lookup",
        description="Lookup data.",
        inputSchema={"type": "object", "properties": {}},
    )
    tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="lookup", arguments={"query": "answer"}),
    )
    messages = [
        Prompt.user("Use the lookup tool."),
        PromptMessageExtended(role="assistant", content=[], tool_calls={"call_1": tool_call}),
        Prompt.user("Tool result: answer is 42."),
    ]
    captured_messages = None
    captured_params = None
    captured_tools = None

    llm = object.__new__(BedrockLLM)
    llm.default_request_params = RequestParams(model="amazon.nova-lite-v1:0")
    llm.capabilities = {}
    llm._reasoning_effort = None
    llm._reasoning_effort_spec = None

    async def apply_prompt(multipart_messages, request_params=None, tools=None):
        nonlocal captured_messages, captured_params, captured_tools
        captured_messages = multipart_messages
        captured_params = request_params
        captured_tools = tools
        return Prompt.assistant('{"value":"42"}')

    monkeypatch.setattr(llm, "_apply_prompt_provider_specific", apply_prompt)

    parsed, response = await BedrockLLM._apply_prompt_provider_specific_structured_schema(
        llm,
        messages,
        schema,
        RequestParams(structured_schema=schema),
        [tool],
    )

    assert parsed == {"value": "42"}
    assert response.last_text() == '{"value":"42"}'
    assert captured_messages is not None
    assert [message.role for message in captured_messages] == ["user", "assistant", "user"]
    assert captured_messages[1].tool_calls == {"call_1": tool_call}
    final_text = "\n".join(
        block.text for block in captured_messages[-1].content if block.type == "text"
    )
    assert "Tool result: answer is 42." in final_text
    assert "JSON Schema:" in final_text
    assert messages[-1].last_text() == "Tool result: answer is 42."
    assert captured_params is not None
    assert captured_params.structured_schema is None
    assert captured_tools == [tool]
