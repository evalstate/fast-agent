"""Tests for Anthropic reasoning defaults and adaptive thinking behavior."""

from pydantic import BaseModel

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.anthropic.llm_anthropic import (
    FINE_GRAINED_TOOL_STREAMING_BETA,
    STRUCTURED_OUTPUT_BETA,
    AnthropicLLM,
)
from fast_agent.llm.reasoning_effort import is_auto_reasoning
from fast_agent.llm.request_params import RequestParams


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


class _StructuredResponse(BaseModel):
    answer: str


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


def test_auto_structured_output_mode_prefers_json_when_direct_beta_supported():
    llm = _make_llm("claude-opus-4-6", reasoning=False)

    structured_mode = llm._resolve_structured_output_mode(
        "claude-opus-4-6",
        _StructuredResponse,
    )

    assert structured_mode == "json"


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
