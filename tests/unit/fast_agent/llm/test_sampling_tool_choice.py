import pytest
from google.genai import types
from mcp_types import (
    CreateMessageRequestParams,
    ListToolsResult,
    SamplingMessage,
    TextContent,
    Tool,
    ToolChoice,
)

from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.bedrock.llm_bedrock import (
    BedrockLLM,
    ModelCapabilities,
    ToolSchemaType,
)
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.llm_generic import GenericLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.request_params import SamplingToolChoicePolicy
from fast_agent.llm.sampling_converter import SamplingConverter
from fast_agent.mcp.provider_management import (
    ProviderManagedMCPAttachment,
    ProviderManagedMCPState,
)
from fast_agent.types import RequestParams

TOOL = Tool(name="echo", description="Echo input", input_schema={"type": "object"})
BEDROCK_TOOL = {
    "toolSpec": {
        "name": "echo",
        "description": "Echo input",
        "inputSchema": {"json": {"type": "object"}},
    }
}


@pytest.mark.parametrize("mode", ("auto", "required", "none"))
def test_sampling_converter_carries_typed_tool_choice(mode: SamplingToolChoicePolicy) -> None:
    params = CreateMessageRequestParams(
        max_tokens=1,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="hi"))],
        tools=[TOOL],
        tool_choice=ToolChoice(mode=mode),
    )

    converted = SamplingConverter.extract_request_params(params)

    assert converted.sampling_tool_choice == mode


def test_sampling_converter_rejects_required_without_tools() -> None:
    params = CreateMessageRequestParams(
        max_tokens=1,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="hi"))],
        tool_choice=ToolChoice(mode="required"),
    )

    with pytest.raises(ValueError, match="requires at least one tool"):
        SamplingConverter.extract_request_params(params)


def test_sampling_converter_defaults_omitted_tool_choice_to_auto() -> None:
    params = CreateMessageRequestParams(
        max_tokens=1,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="hi"))],
        tools=[TOOL],
    )

    assert SamplingConverter.extract_request_params(params).sampling_tool_choice == "auto"


def test_generic_provider_argument_merge_excludes_mcp_sampling_fields() -> None:
    params = RequestParams(
        tools=[TOOL],
        tool_choice=ToolChoice(mode="required"),
        sampling_tool_choice="required",
    )

    args = OpenAILLM(model="unknown-model").prepare_provider_arguments({}, params)

    assert "tools" not in args
    assert "tool_choice" not in args
    assert "sampling_tool_choice" not in args


@pytest.mark.parametrize("llm_type", (OpenAILLM, GenericLLM))
@pytest.mark.parametrize("mode", ("auto", "required", "none"))
def test_openai_chat_tool_choice_mapping_excludes_mcp_fields(
    llm_type: type[OpenAILLM] | type[GenericLLM],
    mode: SamplingToolChoicePolicy,
) -> None:
    llm = llm_type(model="unknown-model")
    params = RequestParams(
        model="unknown-model",
        tools=[TOOL],
        tool_choice=ToolChoice(mode=mode),
        sampling_tool_choice=mode,
    )

    declared_tools = llm._openai_completion_tools([TOOL], "unknown-model")
    args = llm._prepare_api_request([], declared_tools, params)

    assert args["tool_choice"] == mode
    assert args["tools"] == declared_tools
    assert args["tools"][0]["function"]["name"] == TOOL.name


@pytest.mark.parametrize("mode", ("auto", "required", "none"))
def test_responses_and_codex_tool_choice_mapping(mode: SamplingToolChoicePolicy) -> None:
    params = RequestParams(
        model="gpt-5-mini",
        tools=[TOOL],
        tool_choice=ToolChoice(mode=mode),
        sampling_tool_choice=mode,
    )

    responses_llm = ResponsesLLM(model="gpt-5-mini", web_search=True)
    codex_llm = CodexResponsesLLM(model="gpt-5.3-codex", web_search=True)
    provider_managed_state = ProviderManagedMCPState(
        attachments=(
            ProviderManagedMCPAttachment(
                server_name="managed",
                server_description="Provider-managed MCP",
                server_url="https://mcp.example.com",
            ),
        )
    )
    responses_llm.set_provider_managed_mcp_state(provider_managed_state)
    codex_llm.set_provider_managed_mcp_state(provider_managed_state)
    response_args = responses_llm._build_response_args([], params, [TOOL])
    codex_args = codex_llm._build_response_args([], params, [TOOL])

    assert response_args["tool_choice"] == mode
    assert codex_args["tool_choice"] == mode
    assert response_args["tools"][0]["name"] == TOOL.name
    assert codex_args["tools"][0]["name"] == TOOL.name
    assert len(response_args["tools"]) == 1
    assert len(codex_args["tools"]) == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
    ),
)
@pytest.mark.asyncio
async def test_anthropic_tool_choice_mapping(
    mode: SamplingToolChoicePolicy, expected: dict[str, str]
) -> None:
    llm = AnthropicLLM(model="claude-sonnet-4-6")
    request_tools = await llm._prepare_tools("claude-sonnet-4-6", tools=[TOOL])

    args, _ = llm._build_anthropic_base_args(
        model="claude-sonnet-4-6",
        messages=[],
        params=RequestParams(sampling_tool_choice=mode),
        history=None,
        current_extended=None,
        request_tools=request_tools,
        structured_mode=None,
        structured_model=None,
    )

    assert args["tool_choice"] == expected
    assert args["tools"][0]["name"] == TOOL.name


@pytest.mark.asyncio
async def test_anthropic_structured_output_overrides_sampling_tool_choice() -> None:
    llm = AnthropicLLM(model="claude-sonnet-4-6")
    request_tools = await llm._prepare_tools("claude-sonnet-4-6", tools=[TOOL])

    args, _ = llm._build_anthropic_base_args(
        model="claude-sonnet-4-6",
        messages=[],
        params=RequestParams(sampling_tool_choice="none"),
        history=None,
        current_extended=None,
        request_tools=request_tools,
        structured_mode="tool_use",
        structured_model=None,
    )

    assert args["tool_choice"] == {"type": "tool", "name": "return_structured_output"}


@pytest.mark.asyncio
async def test_anthropic_sampling_excludes_provider_managed_and_web_tools() -> None:
    llm = AnthropicLLM(model="claude-sonnet-4-6", web_search=True)
    llm.set_provider_managed_mcp_state(
        ProviderManagedMCPState(
            attachments=(
                ProviderManagedMCPAttachment(
                    server_name="managed",
                    server_description="Provider-managed MCP",
                    server_url="https://mcp.example.com",
                ),
            )
        )
    )
    structured = llm._resolve_anthropic_structured_mode(
        "claude-sonnet-4-6",
        RequestParams(sampling_tool_choice="auto"),
        None,
        None,
    )

    request_tools, web_betas, provider_payload = await llm._anthropic_request_tools(
        "claude-sonnet-4-6",
        None,
        structured,
        [TOOL],
        include_provider_tools=False,
    )

    assert [tool["name"] for tool in request_tools] == [TOOL.name]
    assert web_betas == []
    assert provider_payload.servers == []
    assert provider_payload.tools == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("auto", types.FunctionCallingConfigMode.AUTO),
        ("required", types.FunctionCallingConfigMode.ANY),
        ("none", types.FunctionCallingConfigMode.NONE),
    ),
)
def test_google_tool_choice_mapping(
    mode: SamplingToolChoicePolicy,
    expected: types.FunctionCallingConfigMode,
) -> None:
    llm = GoogleNativeLLM(model="gemini-2.5-flash")
    llm.set_web_search_enabled(True)
    available_tools = llm._google_available_tools(
        [TOOL],
        suppress_tools=False,
        sampling_tool_choice=mode,
    )

    config = llm._google_generate_content_config(
        RequestParams(sampling_tool_choice=mode),
        tools=[TOOL],
        available_tools=available_tools,
        response_mime_type=None,
        response_schema=None,
        suppress_tools=False,
    )

    assert config.tool_config is not None
    assert config.tool_config.function_calling_config is not None
    assert config.tool_config.function_calling_config.mode == expected
    assert isinstance(config.tools, list)
    google_tools = [tool for tool in config.tools if isinstance(tool, types.Tool)]
    assert len(google_tools) == len(config.tools)
    function_declarations = google_tools[0].function_declarations
    assert function_declarations is not None
    assert function_declarations[0].name == TOOL.name
    assert all(tool.google_search is None for tool in google_tools)


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("auto", {"auto": {}}), ("required", {"any": {}})),
)
def test_bedrock_tool_choice_mapping(
    mode: SamplingToolChoicePolicy, expected: dict[str, dict[str, object]]
) -> None:
    llm = object.__new__(BedrockLLM)
    args: dict[str, object] = {}

    llm._apply_bedrock_tool_config(
        args,
        ToolSchemaType.DEFAULT,
        [BEDROCK_TOOL],
        has_tool_results=False,
        has_tool_use=False,
        sampling_tool_choice=mode,
    )

    assert args == {"toolConfig": {"tools": [BEDROCK_TOOL], "toolChoice": expected}}


def test_bedrock_none_suppresses_tools_on_fresh_request() -> None:
    llm = object.__new__(BedrockLLM)

    args: dict[str, object] = {}
    llm._apply_bedrock_tool_config(
        args,
        ToolSchemaType.DEFAULT,
        [BEDROCK_TOOL],
        has_tool_results=False,
        has_tool_use=False,
        sampling_tool_choice="none",
    )

    assert args == {}


def test_bedrock_none_rejects_tool_continuation() -> None:
    llm = object.__new__(BedrockLLM)

    with pytest.raises(ValueError, match="cannot continue.*existing tool use"):
        llm._apply_bedrock_tool_config(
            {},
            ToolSchemaType.DEFAULT,
            [BEDROCK_TOOL],
            has_tool_results=False,
            has_tool_use=True,
            sampling_tool_choice="none",
        )


def test_bedrock_none_does_not_emit_prompt_emulated_tools() -> None:
    model = "meta.llama3-8b-instruct-v1:0"
    llm = BedrockLLM(model=model)

    attempt = llm._prepare_bedrock_attempt(
        [{"role": "user", "content": [{"text": "hello"}]}],
        RequestParams(sampling_tool_choice="none"),
        model,
        None,
        ModelCapabilities(),
        ListToolsResult(tools=[TOOL]),
        ToolSchemaType.SYSTEM_PROMPT,
    )

    assert attempt.tools_payload is None
    assert "toolConfig" not in attempt.converse_args
    assert "system" not in attempt.converse_args
