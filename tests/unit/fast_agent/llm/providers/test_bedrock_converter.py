from copy import deepcopy
from enum import Enum

import pytest
from botocore.exceptions import ParamValidationError
from botocore.session import Session
from botocore.validate import ParamValidator
from mcp import Tool
from mcp_types import CallToolRequest, CallToolRequestParams, ListToolsResult, TextContent
from pydantic import BaseModel

from fast_agent.config import BedrockSettings
from fast_agent.llm.provider.bedrock.llm_bedrock import (
    BEDROCK_REASONING_SPEC,
    BedrockLLM,
    BedrockReasoningEffort,
    ModelCapabilities,
    ToolNamePolicy,
    ToolSchemaType,
    _is_reasoning_effort_error,
    _mentions_system_message,
)
from fast_agent.llm.provider.bedrock.multipart_converter_bedrock import BedrockConverter
from fast_agent.llm.provider.reasoning_config import reasoning_setting_from_config
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
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
        tools=[Tool(name="my-tool", description="demo", input_schema={"type": "object"})]
    )
    tool_name_mapping = {"my_tool": "my-tool"}

    resolved = BedrockLLM._resolve_tool_use_name("call_1_my-tool", tool_list, tool_name_mapping)

    assert resolved == "my_tool"


def test_reasoning_effort_error_detection_is_case_insensitive() -> None:
    assert _is_reasoning_effort_error(RuntimeError("OUTPUTCONFIG does not support EFFORT"))
    assert _is_reasoning_effort_error(RuntimeError("Reasoning is not supported"))
    assert not _is_reasoning_effort_error(RuntimeError("performance latency is invalid"))
    assert not _is_reasoning_effort_error(RuntimeError("validation failed"))


def test_system_message_error_detection_is_case_insensitive() -> None:
    assert _mentions_system_message("SYSTEM MESSAGES are not supported")
    assert _mentions_system_message("Invalid system message placement")
    assert not _mentions_system_message("message role is unsupported")


def test_clean_json_response_preserves_valid_single_response_key_object() -> None:
    llm = object.__new__(BedrockLLM)

    cleaned = BedrockLLM._clean_json_response(
        llm,
        'Here: {"SomeResponse": {"value": 1}}',
    )

    assert cleaned == '{"SomeResponse": {"value": 1}}'


def test_structured_from_multipart_unwraps_matching_model_wrapper_only_after_parse_fails() -> None:
    class SomeResponse(BaseModel):
        value: int

    llm = object.__new__(BedrockLLM)
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text='{"SomeResponse": {"value": 1}}')],
    )

    model_instance, parsed_message = BedrockLLM._structured_from_multipart(
        llm,
        message,
        SomeResponse,
    )

    assert model_instance == SomeResponse(value=1)
    assert parsed_message.content[0] == TextContent(
        type="text",
        text='{"value": 1}',
    )


def test_generate_simplified_schema_handles_modern_union_annotations() -> None:
    class Status(str, Enum):
        ready = "ready"
        blocked = "blocked"

    class Details(BaseModel):
        summary: str | None

    class Response(BaseModel):
        name: str | None
        count: int | None
        tags: list[str]
        statuses: list[Status]
        details: Details

    llm = object.__new__(BedrockLLM)

    schema_text = BedrockLLM._generate_simplified_schema(llm, Response)

    assert '"name": "string"' in schema_text
    assert '"count": "integer"' in schema_text
    assert '"tags": [\n    "string"\n  ]' in schema_text
    assert (
        '"statuses": [\n    "string (must be one of: \\"ready\\", \\"blocked\\")"\n  ]'
        in schema_text
    )
    assert '"details": {' in schema_text
    assert '"summary": "string"' in schema_text


def test_bedrock_reasoning_config_prefers_unified_reasoning() -> None:
    config = BedrockSettings(reasoning="high", reasoning_effort="low")

    raw_setting, should_warn = reasoning_setting_from_config(config)

    assert raw_setting == "high"
    assert should_warn is False


def test_bedrock_reasoning_config_warns_for_legacy_non_default_field() -> None:
    config = BedrockSettings(reasoning_effort="high")

    raw_setting, should_warn = reasoning_setting_from_config(config)

    assert raw_setting == "high"
    assert should_warn is True


def test_bedrock_reasoning_config_accepts_default_legacy_field_without_warning() -> None:
    config = BedrockSettings()

    raw_setting, should_warn = reasoning_setting_from_config(config)

    assert raw_setting == "minimal"
    assert should_warn is False


def test_bedrock_reasoning_spec_uses_native_effort_levels() -> None:
    assert BEDROCK_REASONING_SPEC.kind == "effort"
    assert BEDROCK_REASONING_SPEC.allowed_efforts == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert BEDROCK_REASONING_SPEC.allow_toggle_disable is True
    assert BEDROCK_REASONING_SPEC.allow_auto is True


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_bedrock_inference_config_emits_native_reasoning_effort(
    effort: BedrockReasoningEffort,
) -> None:
    llm = BedrockLLM(model="test.model")
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value=effort))
    converse_args: dict[str, object] = {}

    applied_effort = llm._apply_bedrock_inference_config(
        converse_args,
        RequestParams(temperature=0.2),
        "test.model",
        ModelCapabilities(),
    )

    assert applied_effort == effort
    assert converse_args == {"outputConfig": {"effort": effort}}


@pytest.mark.parametrize(
    ("setting", "expected_effort"),
    [
        (ReasoningEffortSetting(kind="toggle", value=True), "medium"),
        (ReasoningEffortSetting(kind="budget", value=1), "low"),
        (ReasoningEffortSetting(kind="budget", value=512), "low"),
        (ReasoningEffortSetting(kind="budget", value=513), "medium"),
        (ReasoningEffortSetting(kind="budget", value=1024), "medium"),
        (ReasoningEffortSetting(kind="budget", value=1025), "high"),
    ],
)
def test_bedrock_maps_legacy_reasoning_settings_to_effort(
    setting: ReasoningEffortSetting,
    expected_effort: BedrockReasoningEffort,
) -> None:
    llm = BedrockLLM(model="test.model")

    llm.set_reasoning_effort(setting)

    assert llm.reasoning_effort == ReasoningEffortSetting(
        kind="effort",
        value=expected_effort,
    )


@pytest.mark.parametrize(
    "setting",
    [
        ReasoningEffortSetting(kind="effort", value="minimal"),
        ReasoningEffortSetting(kind="effort", value="none"),
        ReasoningEffortSetting(kind="effort", value="auto"),
        ReasoningEffortSetting(kind="toggle", value=False),
        ReasoningEffortSetting(kind="budget", value=0),
    ],
)
def test_bedrock_disabled_or_auto_reasoning_omits_output_effort(
    setting: ReasoningEffortSetting,
) -> None:
    llm = BedrockLLM(model="test.model")
    llm.set_reasoning_effort(setting)
    converse_args: dict[str, object] = {}

    applied_effort = llm._apply_bedrock_inference_config(
        converse_args,
        RequestParams(temperature=0.2),
        "test.model",
        ModelCapabilities(),
    )

    assert applied_effort is None
    assert converse_args == {"inferenceConfig": {"temperature": 0.2}}


def test_bedrock_rejects_negative_legacy_reasoning_budget() -> None:
    llm = BedrockLLM(model="test.model")

    with pytest.raises(ValueError, match="Budget must be >= 0"):
        llm.set_reasoning_effort(ReasoningEffortSetting(kind="budget", value=-1))


def test_bedrock_nova_inference_config_normalizes_model_name() -> None:
    llm = object.__new__(BedrockLLM)
    llm._reasoning_effort = None
    llm._reasoning_effort_spec = None
    converse_args: dict[str, object] = {}

    reasoning_effort = BedrockLLM._apply_bedrock_inference_config(
        llm,
        converse_args,
        RequestParams(temperature=0.2),
        " AMAZON.NOVA-LITE-V1:0 ",
        ModelCapabilities(),
    )

    assert reasoning_effort is None
    assert converse_args["inferenceConfig"] == {
        "temperature": 0.2,
        "topP": 1.0,
    }
    assert converse_args["additionalModelRequestFields"] == {"inferenceConfig": {"topK": 1}}


def test_bedrock_reasoning_request_matches_botocore_service_model() -> None:
    llm = BedrockLLM(model="test.model")
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="max"))
    converse_args: dict[str, object] = {
        "modelId": "test.model",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
    }
    llm._apply_bedrock_inference_config(
        converse_args,
        RequestParams(),
        "test.model",
        ModelCapabilities(),
    )
    input_shape = (
        Session().get_service_model("bedrock-runtime").operation_model("Converse").input_shape
    )

    report = ParamValidator().validate(converse_args, input_shape)

    assert not report.has_errors(), report.generate_report()


class _ReasoningFallbackClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.requests.append(deepcopy(kwargs))
        if len(self.requests) == 1:
            raise ParamValidationError(
                report="Unknown parameter in outputConfig: effort is not supported"
            )
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 0, "totalTokens": 1},
        }


@pytest.mark.asyncio
async def test_bedrock_reasoning_fallback_keeps_capability_unsupported() -> None:
    model = "test.model"
    llm = BedrockLLM(model=model)
    caps = ModelCapabilities()
    client = _ReasoningFallbackClient()
    converse_args: dict[str, object] = {
        "modelId": model,
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "outputConfig": {"effort": "high"},
    }

    response, reasoning_applied = await llm._invoke_bedrock_with_reasoning_fallback(
        client,
        converse_args,
        model,
        ToolSchemaType.DEFAULT,
        RequestParams(temperature=0.2),
        caps,
        reasoning_effort="high",
        use_streaming=False,
    )
    llm._cache_successful_bedrock_attempt(
        model,
        caps,
        ToolSchemaType.DEFAULT,
        None,
        ToolNamePolicy.PRESERVE,
        has_tools=False,
        attempted_streaming=False,
        reasoning_applied=reasoning_applied,
    )

    assert response["content"] == [{"text": "ok"}]
    assert client.requests[0]["outputConfig"] == {"effort": "high"}
    assert "outputConfig" not in client.requests[1]
    assert client.requests[1]["inferenceConfig"] == {"temperature": 0.2}
    assert caps.reasoning_support is False


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
        input_schema={"type": "object", "properties": {}},
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
        input_schema={"type": "object", "properties": {}},
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
