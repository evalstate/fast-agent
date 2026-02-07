"""Tests for Anthropic reasoning defaults and adaptive thinking behavior."""

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.reasoning_effort import is_auto_reasoning


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


def test_unsupported_model_keeps_long_context_disabled():
    """Models without long_context_window metadata should not enable long context."""
    llm = _make_llm("claude-haiku-4-5", long_context=True)
    assert llm._long_context is False
    assert llm.model_info is not None
    assert llm.model_info.context_window == 200_000
