from typing import Literal

import pytest
from mcp import Tool
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import PromptMessageExtended, RequestParams


# Example model similar to what's used in the Router workflow
class StructuredResponseCategory(BaseModel):
    category: str
    confidence: Literal["high", "medium", "low"]
    reasoning: str | None


class StructuredResponse(BaseModel):
    categories: list[StructuredResponseCategory]


class StructuredValue(BaseModel):
    value: str


class _CompatibleStructuredHarness(OpenAICompatibleLLM):
    def __init__(self, model: str = "qwen/qwen3-32b") -> None:
        self.default_request_params = RequestParams(model=model)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages,
        request_params=None,
        tools=None,
        is_template: bool = False,
    ):
        del request_params, tools, is_template
        return multipart_messages[-1]

    def _structured_reasoning_mode(self) -> str | None:
        return None


class _GeneratePrepareHarness(PassthroughLLM):
    def __init__(self) -> None:
        super().__init__(name="generate-prepare")
        self.prepare_called = False
        self.applied_params: RequestParams | None = None
        self.applied_tools = None

    def _prepare_structured_request(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams,
        tools=None,
    ) -> tuple[list[PromptMessageExtended], RequestParams]:
        del tools
        self.prepare_called = True
        return messages, request_params.model_copy(
            update={"response_format": {"type": "json_object"}}
        )

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages,
        request_params=None,
        tools=None,
        is_template: bool = False,
    ):
        del multipart_messages, is_template
        self.applied_params = request_params
        self.applied_tools = tools
        return Prompt.assistant('{"value":"ok"}')


class _StructuredModelPathHarness(PassthroughLLM):
    def __init__(self) -> None:
        super().__init__(name="structured-model-path")
        self.model_path_called = False

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages,
        model,
        request_params=None,
    ):
        del multipart_messages, request_params
        self.model_path_called = True
        response = Prompt.assistant('{"value":"ok"}')
        return model(value="ok"), response


class _DefaultDeferOpenAIHarness(OpenAILLM):
    def _default_structured_tool_policy(self, model_name: str | None):
        del model_name
        return "defer"


@pytest.mark.asyncio
async def test_direct_pydantic_conversion():
    # JSON string that would typically come from an LLM
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support",
                "confidence": "high",
                "reasoning": "Query relates to system troubleshooting"
            },
            {
                "category": "documentation",
                "confidence": "medium",
                "reasoning": null
            }
        ]
    }
    """

    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM(name="structured")
    result, _ = await llm.structured([Prompt.user(json_str)], model=StructuredResponse)

    # Verify conversion worked correctly
    assert isinstance(result, StructuredResponse)
    assert len(result.categories) == 2
    assert result.categories[0].category == "tech_support"
    assert result.categories[0].confidence == "high"
    assert result.categories[1].category == "documentation"
    assert result.categories[1].confidence == "medium"
    assert result.categories[1].reasoning is None


@pytest.mark.asyncio
async def test_structured_uses_provider_model_path():
    llm = _StructuredModelPathHarness()

    result, response = await llm.structured([Prompt.user("return json")], StructuredValue)

    assert llm.model_path_called
    assert result is not None
    assert result.value == "ok"
    assert response.last_text() == '{"value":"ok"}'


@pytest.mark.asyncio
async def test_structured_with_bad_json():
    # JSON string that would typically come from an LLM
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support",
            },
            {
                "category": "documentation",
                "confidence": "medium",
                "reaso: null
            }
        ]
    }
    """

    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM(name="structured")
    result, _ = await llm.structured([Prompt.user(json_str)], model=StructuredResponse)

    assert None is result


@pytest.mark.asyncio
async def test_structured_schema_with_valid_json():
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support",
                "confidence": "high",
                "reasoning": "Query relates to system troubleshooting"
            }
        ]
    }
    """
    schema = StructuredResponse.model_json_schema()

    llm = PassthroughLLM(name="structured")
    result, _ = await llm.structured_schema([Prompt.user(json_str)], schema=schema)

    assert isinstance(result, dict)
    assert result["categories"][0]["category"] == "tech_support"


@pytest.mark.asyncio
async def test_structured_schema_with_bad_json():
    json_str = '{"categories": ['
    schema = StructuredResponse.model_json_schema()

    llm = PassthroughLLM(name="structured")
    result, _ = await llm.structured_schema([Prompt.user(json_str)], schema=schema)

    assert result is None


@pytest.mark.asyncio
async def test_structured_schema_with_schema_mismatch():
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support"
            }
        ]
    }
    """
    schema = StructuredResponse.model_json_schema()

    llm = PassthroughLLM(name="structured")
    result, _ = await llm.structured_schema([Prompt.user(json_str)], schema=schema)

    assert result is None


@pytest.mark.asyncio
async def test_structured_schema_delegates_through_generate_prepare_hook():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = _GeneratePrepareHarness()

    result, response = await llm.structured_schema([Prompt.user("ignored")], schema)

    assert result == {"value": "ok"}
    assert response.last_text() == '{"value":"ok"}'
    assert llm.prepare_called
    assert llm.applied_params is not None
    assert llm.applied_params.structured_schema == schema
    assert llm.applied_params.response_format == {"type": "json_object"}


@pytest.mark.asyncio
async def test_generate_strips_tools_for_deferred_structured_final_turn():
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
    llm = _GeneratePrepareHarness()
    messages = [
        Prompt.user("call the tool"),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="tool payload")],
                )
            },
        ),
    ]

    await llm.generate(
        messages,
        RequestParams(structured_schema=schema, structured_tool_policy="defer"),
        [tool],
    )

    assert llm.applied_params is not None
    assert llm.applied_params.response_format == {"type": "json_object"}
    assert llm.applied_tools is None


@pytest.mark.asyncio
async def test_generate_keeps_tools_for_deferred_first_turn():
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
    llm = _GeneratePrepareHarness()

    await llm.generate(
        [Prompt.user("call the tool")],
        RequestParams(structured_schema=schema, structured_tool_policy="defer"),
        [tool],
    )

    assert llm.applied_tools == [tool]


@pytest.mark.asyncio
async def test_generate_strips_tools_for_no_tools_structured_policy():
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
    llm = _GeneratePrepareHarness()

    await llm.generate(
        [Prompt.user("return json")],
        RequestParams(structured_schema=schema, structured_tool_policy="no_tools"),
        [tool],
    )

    assert llm.applied_tools is None


def test_openai_prepare_structured_request_sets_native_response_format():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = OpenAILLM(model="gpt-4.1")
    messages = [Prompt.user("return json")]
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    prepared_messages, prepared_params = llm._prepare_structured_request(messages, params)

    assert prepared_messages is messages
    assert params.response_format is None
    assert prepared_params.response_format == llm.schema_to_response_format(schema)


def test_openai_prepare_structured_request_defers_with_tools_until_tool_result():
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
    llm = OpenAILLM(model="gpt-4.1")
    messages = [Prompt.user("call the tool")]
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    prepared_messages, prepared_params = llm._prepare_structured_request(
        messages,
        params,
        [tool],
    )

    assert prepared_messages is messages
    assert prepared_params.structured_schema is None
    assert prepared_params.response_format is None
    assert params.structured_schema == schema


def test_structured_tool_policy_auto_uses_provider_default_and_allows_override():
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
    llm = _DefaultDeferOpenAIHarness(model="gpt-4.1")
    messages = [Prompt.user("call the tool")]

    _, auto_params = llm._prepare_structured_request(
        messages,
        RequestParams(structured_schema=schema),
        [tool],
    )
    _, override_params = llm._prepare_structured_request(
        messages,
        RequestParams(structured_schema=schema, structured_tool_policy="always"),
        [tool],
    )

    assert auto_params.structured_schema is None
    assert override_params.response_format == llm.schema_to_response_format(schema)


def test_anthropic_tool_use_structured_mode_defaults_to_no_tools_policy():
    llm = AnthropicLLM(model="claude-sonnet-4-6", structured_output_mode="tool_use")

    assert (
        llm.resolve_structured_tool_policy(RequestParams(structured_schema={"type": "object"}))
        == "no_tools"
    )
    assert (
        llm.resolve_structured_tool_policy(
            RequestParams(structured_schema={"type": "object"}, structured_tool_policy="defer")
        )
        == "defer"
    )


def test_openai_compatible_prepare_structured_request_prompts_without_mutating_history():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = _CompatibleStructuredHarness()
    original = Prompt.user("return json")
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    prepared_messages, prepared_params = llm._prepare_structured_request([original], params)

    assert original.all_text() == "return json"
    assert prepared_messages[0].all_text() != original.all_text()
    assert "YOU MUST RESPOND WITH A JSON OBJECT" in prepared_messages[0].all_text()
    assert params.response_format is None
    assert prepared_params.response_format == {"type": "json_object"}


def test_openai_compatible_prepare_structured_request_uses_schema_mode_without_prompt():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = _CompatibleStructuredHarness(model="moonshotai/Kimi-K2.6")
    original = Prompt.user("return json")
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    prepared_messages, prepared_params = llm._prepare_structured_request([original], params)

    assert prepared_messages[0].all_text() == "return json"
    assert prepared_params.response_format == llm.schema_to_response_format(schema)


def test_openai_compatible_prepare_structured_request_uses_request_model_override():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = _CompatibleStructuredHarness(model="qwen/qwen3-32b")
    original = Prompt.user("return json")
    params = RequestParams(
        model="moonshotai/Kimi-K2.6",
        structured_schema=schema,
        structured_tool_policy="defer",
    )

    prepared_messages, prepared_params = llm._prepare_structured_request([original], params)

    assert prepared_messages[0].all_text() == "return json"
    assert prepared_params.response_format == llm.schema_to_response_format(schema)


def test_openai_compatible_prepare_structured_request_request_model_override_to_object_mode():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = _CompatibleStructuredHarness(model="moonshotai/Kimi-K2.6")
    original = Prompt.user("return json")
    params = RequestParams(
        model="qwen/qwen3-32b",
        structured_schema=schema,
        structured_tool_policy="defer",
    )

    prepared_messages, prepared_params = llm._prepare_structured_request([original], params)

    assert "YOU MUST RESPOND WITH A JSON OBJECT" in prepared_messages[0].all_text()
    assert prepared_params.response_format == {"type": "json_object"}


def test_openai_compatible_prepare_structured_request_prompt_only_for_no_json_mode():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = _CompatibleStructuredHarness(model="claude-sonnet-4-0")
    original = Prompt.user("return json")
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    prepared_messages, prepared_params = llm._prepare_structured_request([original], params)

    assert "YOU MUST RESPOND WITH A JSON OBJECT" in prepared_messages[0].all_text()
    assert prepared_params.response_format is None


def test_openai_compatible_prepare_structured_request_defers_until_tool_result():
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
    llm = _CompatibleStructuredHarness()
    original = Prompt.user("call the tool")
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    prepared_messages, prepared_params = llm._prepare_structured_request(
        [original],
        params,
        [tool],
    )

    assert prepared_messages[0].all_text() == "call the tool"
    assert prepared_params.response_format is None
    assert prepared_params.structured_schema is None
    assert params.structured_schema == schema


@pytest.mark.asyncio
async def test_openai_compatible_structured_preserves_assistant_ended_messages():
    llm = _CompatibleStructuredHarness()
    assistant_message = Prompt.assistant('{"value":"ok"}')

    result, response = await llm._apply_prompt_provider_specific_structured(
        [assistant_message],
        StructuredValue,
    )

    assert result is not None
    assert result.value == "ok"
    assert response.last_text() == '{"value":"ok"}'


@pytest.mark.asyncio
async def test_openai_compatible_structured_schema_preserves_assistant_ended_messages():
    llm = _CompatibleStructuredHarness()
    assistant_message = Prompt.assistant('{"value":"ok"}')
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    result_or_response = await llm._apply_prompt_provider_specific_structured_schema(
        [assistant_message],
        schema,
    )

    assert isinstance(result_or_response, PromptMessageExtended)
    assert result_or_response.last_text() == '{"value":"ok"}'
    parsed, response = llm._structured_schema_from_multipart(
        result_or_response,
        schema,
    )
    assert parsed == {"value": "ok"}
    assert response.last_text() == '{"value":"ok"}'


@pytest.mark.asyncio
async def test_chat_turn_counting():
    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM()
    # no messages yet, so chat turn should be 1
    assert 1 == llm.chat_turn()
    await llm.generate([Prompt.user("test")])
    assert 2 == llm.chat_turn()

    # just increment for each assistant message
    await llm.generate([Prompt.user("foo")])
    await llm.generate([Prompt.user("bar")])

    assert 4 == llm.chat_turn()
