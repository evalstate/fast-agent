"""Tests for Anthropic reasoning defaults and adaptive thinking behavior."""

import json

import pytest
from mcp import Tool
from pydantic import BaseModel

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.anthropic.beta_types import Message, ToolUseBlock, Usage
from fast_agent.llm.provider.anthropic.llm_anthropic import (
    FINE_GRAINED_TOOL_STREAMING_BETA,
    STRUCTURED_OUTPUT_BETA,
    STRUCTURED_OUTPUT_TOOL_NAME,
    AnthropicLLM,
)
from fast_agent.llm.provider.anthropic.llm_anthropic_vertex import AnthropicVertexLLM
from fast_agent.llm.reasoning_effort import is_auto_reasoning
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt
from fast_agent.types.llm_stop_reason import LlmStopReason


def _make_llm(
    model: str,
    reasoning: str | int | bool | None = None,
    *,
    long_context: bool = False,
) -> AnthropicLLM:
    settings = Settings()
    settings.anthropic = AnthropicSettings(api_key="test-key", reasoning=reasoning)
    context = Context(config=settings)
    return AnthropicLLM(
        context=context,
        model=model,
        name="test-agent",
        long_context=long_context,
    )


def _make_vertex_llm(
    model: str,
    reasoning: str | int | bool | None = None,
    *,
    long_context: bool = False,
) -> AnthropicVertexLLM:
    settings = Settings()
    settings.anthropic = AnthropicSettings(api_key="test-key", reasoning=reasoning)
    settings.anthropic.vertex_ai.enabled = True
    settings.anthropic.vertex_ai.project_id = "test-project"
    settings.anthropic.vertex_ai.location = "us-east5"
    context = Context(config=settings)
    return AnthropicVertexLLM(
        context=context,
        model=model,
        name="test-agent",
        long_context=long_context,
    )


class _StructuredResponse(BaseModel):
    answer: str


class _StructuredResponseWithMap(BaseModel):
    metadata: dict[str, str]


def test_opus_46_uses_adaptive_thinking_by_default():
    llm = _make_llm("claude-opus-4-6")

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-6",
        max_tokens=16000,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive"}
    # No explicit effort — the API uses its built-in automatic mode
    assert "output_config" not in args
    assert args["max_tokens"] == 16000


def test_opus_46_default_reasoning_effort_is_auto():
    """When no reasoning is configured, reasoning_effort should be 'auto'."""
    llm = _make_llm("claude-opus-4-6")
    assert is_auto_reasoning(llm.reasoning_effort)


def test_opus_46_supports_max_effort():
    llm = _make_llm("claude-opus-4-6", reasoning="max")

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-6",
        max_tokens=16000,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive"}
    assert args["output_config"] == {"effort": "max"}


def test_opus_47_requests_summarized_adaptive_thinking_by_default():
    llm = _make_llm("claude-opus-4-7")

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-7",
        max_tokens=16000,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "output_config" not in args


def test_opus_47_supports_xhigh_effort():
    llm = _make_llm("claude-opus-4-7", reasoning="xhigh")

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-7",
        max_tokens=16000,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert args["output_config"] == {"effort": "xhigh"}


def test_opus_47_task_budget_merges_into_output_config() -> None:
    llm = _make_llm("claude-opus-4-7")
    llm.set_task_budget_tokens(128_000)

    args, thinking_enabled = llm._build_anthropic_base_args(
        model="claude-opus-4-7",
        messages=[],
        params=RequestParams(maxTokens=1024),
        history=None,
        current_extended=None,
        request_tools=[],
        structured_mode=None,
        structured_model=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert args["output_config"]["task_budget"] == {"type": "tokens", "total": 128_000}


def test_opus_47_task_budget_adds_beta_flag() -> None:
    llm = _make_llm("claude-opus-4-7")
    llm.set_task_budget_tokens(64_000)

    beta_flags = llm._resolve_anthropic_beta_flags(
        model="claude-opus-4-7",
        structured_mode=None,
        thinking_enabled=True,
        request_tools=[],
        web_tool_betas=[],
    )

    assert "task-budgets-2026-03-13" in beta_flags


def test_vertex_opus_47_task_budget_merges_into_output_config() -> None:
    llm = _make_vertex_llm("claude-opus-4-7")
    llm.set_task_budget_tokens(128_000)

    args, thinking_enabled = llm._build_anthropic_base_args(
        model="claude-opus-4-7",
        messages=[],
        params=RequestParams(maxTokens=1024),
        history=None,
        current_extended=None,
        request_tools=[],
        structured_mode=None,
        structured_model=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert args["output_config"]["task_budget"] == {"type": "tokens", "total": 128_000}


def test_vertex_opus_47_task_budget_adds_beta_flag() -> None:
    llm = _make_vertex_llm("claude-opus-4-7")
    llm.set_task_budget_tokens(64_000)

    beta_flags = llm._resolve_anthropic_beta_flags(
        model="claude-opus-4-7",
        structured_mode=None,
        thinking_enabled=True,
        request_tools=[],
        web_tool_betas=[],
    )

    assert "task-budgets-2026-03-13" in beta_flags


def test_opus_46_supports_disable_toggle():
    llm = _make_llm("claude-opus-4-6", reasoning=False)

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-6",
        max_tokens=16000,
        structured_mode=None,
    )

    assert not thinking_enabled
    assert "thinking" not in args
    assert args["max_tokens"] == 16000


def test_opus_46_budget_falls_back_to_auto():
    llm = _make_llm("claude-opus-4-6", reasoning=4096)

    assert is_auto_reasoning(llm.reasoning_effort)

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-6",
        max_tokens=4096,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive"}
    assert "output_config" not in args
    assert args["max_tokens"] == 4096


def test_legacy_anthropic_models_still_use_budget_thinking_defaults():
    llm = _make_llm("claude-opus-4-5")

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-5",
        max_tokens=16000,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "output_config" not in args


def test_legacy_models_map_effort_to_budget():
    llm = _make_llm("claude-opus-4-5", reasoning="high")

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-5",
        max_tokens=16000,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "enabled", "budget_tokens": 32000}


def test_legacy_models_accept_explicit_budget():
    llm = _make_llm("claude-opus-4-5", reasoning=4096)

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-5",
        max_tokens=4096,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_tool_forced_structured_output_disables_thinking():
    llm = _make_llm("claude-opus-4-6")

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-6",
        max_tokens=16000,
        structured_mode="tool_use",
    )

    assert not thinking_enabled
    assert args == {"max_tokens": 16000}


def test_opus_46_explicit_auto_uses_adaptive_no_effort():
    """Explicitly passing 'auto' should behave same as default."""
    llm = _make_llm("claude-opus-4-6", reasoning="auto")

    assert is_auto_reasoning(llm.reasoning_effort)

    args, thinking_enabled = llm._resolve_thinking_arguments(
        model="claude-opus-4-6",
        max_tokens=16000,
        structured_mode=None,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive"}
    assert "output_config" not in args


def test_long_context_supported_models_source_from_model_database():
    """Anthropic long-context supported list should come from ModelDatabase."""
    llm = _make_llm("claude-opus-4-6")
    assert llm._list_supported_long_context_models() == ModelDatabase.list_long_context_models()


def test_46_models_ignore_explicit_long_context_flag():
    """Claude 4.6 models already expose 1M context without an opt-in flag."""
    llm = _make_llm("claude-opus-4-6", long_context=True)
    assert llm._long_context is False
    assert llm.model_info is not None
    assert llm.model_info.context_window == 1_000_000


def test_unsupported_model_keeps_long_context_disabled():
    """Models without long_context_window metadata should not enable long context."""
    llm = _make_llm("claude-haiku-4-5", long_context=True)
    assert llm._long_context is False
    assert llm.model_info is not None
    assert llm.model_info.context_window == 200_000


def test_json_structured_output_uses_output_config_format():
    llm = _make_llm("claude-opus-4-6", reasoning=False)

    args, thinking_enabled = llm._build_anthropic_base_args(
        model="claude-opus-4-6",
        messages=[],
        params=RequestParams(maxTokens=1024),
        history=None,
        current_extended=None,
        request_tools=[],
        structured_mode="json",
        structured_model=_StructuredResponse,
    )

    assert not thinking_enabled
    assert "output_format" not in args
    assert args["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in args["output_config"]["format"]


def test_json_structured_output_sanitizes_map_additional_properties():
    llm = _make_llm("claude-opus-4-6", reasoning=False)

    args, _ = llm._build_anthropic_base_args(
        model="claude-opus-4-6",
        messages=[],
        params=RequestParams(maxTokens=1024),
        history=None,
        current_extended=None,
        request_tools=[],
        structured_mode="json",
        structured_model=_StructuredResponseWithMap,
    )

    metadata_schema = args["output_config"]["format"]["schema"]["properties"]["metadata"]
    assert metadata_schema["additionalProperties"] is False


def test_auto_structured_output_mode_prefers_json_when_direct_beta_supported():
    llm = _make_llm("claude-opus-4-6", reasoning=False)

    structured_mode = llm._resolve_structured_output_mode(
        "claude-opus-4-6",
        _StructuredResponse,
    )

    assert structured_mode == "json"


def test_auto_structured_output_mode_falls_back_to_tool_use_for_legacy_model():
    llm = _make_llm("claude-sonnet-4-0", reasoning=False)

    structured_mode = llm._resolve_structured_output_mode(
        "claude-sonnet-4-0",
        _StructuredResponse,
    )

    assert structured_mode == "tool_use"


def test_auto_tool_use_structured_fallback_detects_legacy_model():
    llm = _make_llm("claude-sonnet-4-0", reasoning=False)

    assert llm._is_auto_tool_use_structured_fallback(
        "claude-sonnet-4-0",
        "tool_use",
        _StructuredResponse,
    )


def test_explicit_tool_use_mode_is_not_treated_as_auto_fallback():
    settings = Settings()
    settings.anthropic = AnthropicSettings(
        api_key="test-key",
        reasoning=False,
        structured_output_mode="tool_use",
    )
    context = Context(config=settings)
    llm = AnthropicLLM(
        context=context,
        model="claude-opus-4-6",
        name="test-agent",
    )

    assert not llm._is_auto_tool_use_structured_fallback(
        "claude-opus-4-6",
        "tool_use",
        _StructuredResponse,
    )


def test_json_structured_output_merges_with_adaptive_effort():
    llm = _make_llm("claude-opus-4-6", reasoning="max")

    args, thinking_enabled = llm._build_anthropic_base_args(
        model="claude-opus-4-6",
        messages=[],
        params=RequestParams(maxTokens=1024),
        history=None,
        current_extended=None,
        request_tools=[],
        structured_mode="json",
        structured_model=_StructuredResponse,
    )

    assert thinking_enabled
    assert args["thinking"] == {"type": "adaptive"}
    assert args["output_config"]["effort"] == "max"
    assert args["output_config"]["format"]["type"] == "json_schema"


def test_opus_47_drops_sampling_parameters_from_request_payload() -> None:
    llm = _make_llm("claude-opus-4-7")

    result = llm.prepare_provider_arguments(
        {
            "model": "claude-opus-4-7",
            "messages": [],
            "max_tokens": 1000,
        },
        RequestParams(temperature=0.7, top_p=0.9, top_k=10),
        llm.ANTHROPIC_EXCLUDE_FIELDS,
    )

    assert "temperature" not in result
    assert "top_p" not in result
    assert "top_k" not in result


def test_json_structured_output_uses_raw_schema_when_supplied() -> None:
    llm = _make_llm("claude-opus-4-6", reasoning=False)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    args, _ = llm._build_anthropic_base_args(
        model="claude-opus-4-6",
        messages=[],
        params=RequestParams(maxTokens=1024, structured_schema=schema),
        history=None,
        current_extended=None,
        request_tools=[],
        structured_mode="json",
        structured_model=None,
        structured_schema=schema,
    )

    assert args["output_config"]["format"]["schema"] == schema


def test_json_structured_output_transforms_raw_schema_with_anthropic_sdk() -> None:
    llm = _make_llm("claude-opus-4-6", reasoning=False)
    schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "minLength": 3,
            }
        },
        "required": ["answer"],
    }

    args, _ = llm._build_anthropic_base_args(
        model="claude-opus-4-6",
        messages=[],
        params=RequestParams(maxTokens=1024, structured_schema=schema),
        history=None,
        current_extended=None,
        request_tools=[],
        structured_mode="json",
        structured_model=None,
        structured_schema=schema,
    )

    transformed = args["output_config"]["format"]["schema"]
    answer_schema = transformed["properties"]["answer"]
    assert transformed["additionalProperties"] is False
    assert "minLength" not in answer_schema
    assert "minLength: 3" in answer_schema["description"]
    assert schema["properties"]["answer"]["minLength"] == 3


@pytest.mark.asyncio
async def test_json_structured_output_preserves_regular_tools() -> None:
    llm = _make_llm("claude-opus-4-6", reasoning=False)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    tool = Tool(
        name="lookup_probe_payload",
        description="Return the probe payload for validation.",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    tools = await llm._prepare_tools(
        "claude-opus-4-6",
        structured_model=None,
        structured_schema=schema,
        tools=[tool],
        structured_mode="json",
    )

    assert len(tools) == 1
    assert tools[0]["name"] == "lookup_probe_payload"
    input_schema = dict(tools[0]["input_schema"])
    assert input_schema.get("additionalProperties") is False


def test_structured_schema_with_tools_is_deferred_until_tool_result() -> None:
    llm = _make_llm("claude-sonnet-4-6", reasoning=False)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    tool = Tool(
        name="lookup_probe_payload",
        description="Return the probe payload for validation.",
        inputSchema={"type": "object", "properties": {}},
    )
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    _, prepared_params = llm._prepare_structured_request(
        [Prompt.user("call the tool, then return json")],
        params,
        [tool],
    )

    assert params.structured_schema == schema
    assert prepared_params.structured_schema is None


@pytest.mark.asyncio
async def test_tool_use_structured_output_uses_raw_schema_when_supplied() -> None:
    llm = _make_llm("claude-opus-4-6", reasoning=False)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    tools = await llm._prepare_tools(
        "claude-opus-4-6",
        structured_model=None,
        structured_schema=schema,
        tools=None,
        structured_mode="tool_use",
    )

    input_schema = tools[0]["input_schema"]
    assert isinstance(input_schema, dict)
    properties = input_schema["properties"]
    assert isinstance(properties, dict)
    normalized_properties = {str(key): value for key, value in properties.items()}
    answer_schema = normalized_properties.get("answer")
    assert isinstance(answer_schema, dict)
    normalized_answer_schema = {str(key): value for key, value in answer_schema.items()}
    assert normalized_answer_schema.get("type") == "string"


@pytest.mark.asyncio
async def test_tool_use_structured_output_sanitizes_raw_schema_for_anthropic() -> None:
    llm = _make_llm("claude-opus-4-6", reasoning=False)
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "context": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            },
        },
        "required": ["answer"],
    }

    tools = await llm._prepare_tools(
        "claude-opus-4-6",
        structured_model=None,
        structured_schema=schema,
        tools=None,
        structured_mode="tool_use",
    )

    input_schema = tools[0]["input_schema"]
    assert isinstance(input_schema, dict)
    normalized_input_schema = {str(key): value for key, value in input_schema.items()}
    assert normalized_input_schema["additionalProperties"] is False
    assert normalized_input_schema["required"] == ["answer"]
    properties = normalized_input_schema["properties"]
    assert isinstance(properties, dict)
    normalized_properties = {str(key): value for key, value in properties.items()}
    context_schema = normalized_properties["context"]
    assert isinstance(context_schema, dict)
    assert "default" not in context_schema
    assert schema["properties"]["context"]["default"] is None


@pytest.mark.asyncio
async def test_tool_use_structured_schema_response_is_finalized_without_model() -> None:
    llm = _make_llm("claude-sonnet-4-6", reasoning=False)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    response = Message(
        id="msg_structured",
        type="message",
        role="assistant",
        content=[
            ToolUseBlock(
                type="tool_use",
                id="toolu_structured",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                input={"answer": "ok"},
            )
        ],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    result = await llm._finalize_anthropic_response(
        response=response,
        model="claude-sonnet-4-6",
        messages=[],
        thinking_segments=[],
        streamed_text_segments=[],
        structured_mode="tool_use",
        structured_model=None,
        structured_schema=schema,
    )

    assert result.stop_reason == LlmStopReason.END_TURN
    assert result.tool_calls is None
    assert json.loads(result.last_text() or "{}") == {"answer": "ok"}


def test_structured_output_json_adds_structured_output_beta() -> None:
    llm = _make_llm("claude-opus-4-6")

    beta_flags = llm._resolve_anthropic_beta_flags(
        model="claude-opus-4-6",
        structured_mode="json",
        thinking_enabled=False,
        request_tools=[],
        web_tool_betas=[],
    )

    assert beta_flags == [STRUCTURED_OUTPUT_BETA]


def test_structured_output_tool_use_does_not_add_structured_output_beta() -> None:
    llm = _make_llm("claude-opus-4-6")

    beta_flags = llm._resolve_anthropic_beta_flags(
        model="claude-opus-4-6",
        structured_mode="tool_use",
        thinking_enabled=False,
        request_tools=[],
        web_tool_betas=[],
    )

    assert beta_flags == []


def test_structured_output_modes_still_preserve_other_beta_flags() -> None:
    llm = _make_llm("claude-opus-4-6")

    beta_flags = llm._resolve_anthropic_beta_flags(
        model="claude-opus-4-6",
        structured_mode="json",
        thinking_enabled=False,
        request_tools=[{"name": "demo", "description": "", "input_schema": {"type": "object"}}],
        web_tool_betas=["web-beta"],
    )

    assert FINE_GRAINED_TOOL_STREAMING_BETA in beta_flags
    assert STRUCTURED_OUTPUT_BETA in beta_flags
    assert "web-beta" in beta_flags
