"""Contract tests for Groq reasoning wiring and the Qwen3.6-27B model entry.

Groq exposes reasoning as a binary `reasoning_effort` toggle (`default`/`none`)
plus a `reasoning_format` extension. These tests pin the provider add-on that
shapes that wire contract and the model-database capability entry, without
restating internal implementation tables.
"""

from __future__ import annotations

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_groq import GroqLLM, _normalize_groq_reasoning_setting
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, available_reasoning_values
from fast_agent.llm.request_params import RequestParams


def _groq_llm(model: str) -> GroqLLM:
    return GroqLLM(context=Context(config=Settings()), model=model)


def test_qwen36_27b_model_database_entry() -> None:
    params = ModelDatabase.get_model_params("qwen/qwen3.6-27b")
    assert params is not None
    assert params.context_window == 131_072
    assert params.max_output_tokens == 32_768
    assert params.json_mode == "object"
    assert params.reasoning == "stream"
    assert params.default_provider == Provider.GROQ
    assert "image/jpeg" in params.tokenizes
    # Binary reasoning toggle exposed as on/off.
    spec = params.reasoning_effort_spec
    assert spec is not None and spec.kind == "toggle"
    assert available_reasoning_values(spec) == ["on", "off"]


def test_normalize_collsapses_effort_levels_to_toggle() -> None:
    assert _normalize_groq_reasoning_setting(
        ReasoningEffortSetting(kind="effort", value="high")
    ) == ReasoningEffortSetting(kind="toggle", value=True)
    assert _normalize_groq_reasoning_setting(
        ReasoningEffortSetting(kind="effort", value="auto")
    ) == ReasoningEffortSetting(kind="toggle", value=True)
    assert _normalize_groq_reasoning_setting(
        ReasoningEffortSetting(kind="effort", value="none")
    ) == ReasoningEffortSetting(kind="toggle", value=False)
    # Toggle input passes through unchanged.
    assert _normalize_groq_reasoning_setting(
        ReasoningEffortSetting(kind="toggle", value=False)
    ) == ReasoningEffortSetting(kind="toggle", value=False)


def test_resolve_reasoning_effort_maps_to_groq_wire_values() -> None:
    llm = _groq_llm("qwen/qwen3.6-27b")
    # Unset defaults to thinking on (Groq "default").
    assert llm._resolve_reasoning_effort() == "default"
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="high"))
    assert llm.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=True)
    assert llm._resolve_reasoning_effort() == "default"
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="none"))
    assert llm.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=False)
    assert llm._resolve_reasoning_effort() == "none"


def test_prepare_api_request_thinking_on_sends_parsed_format() -> None:
    llm = _groq_llm("qwen/qwen3.6-27b")
    args = llm._prepare_api_request([], None, RequestParams(maxTokens=512))
    assert args["reasoning_effort"] == "default"
    assert args["extra_body"]["reasoning_format"] == "parsed"


def test_prepare_api_request_thinking_off_omits_format() -> None:
    llm = _groq_llm("qwen/qwen3.6-27b")
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="toggle", value=False))
    args = llm._prepare_api_request([], None, RequestParams(maxTokens=512))
    assert args["reasoning_effort"] == "none"
    assert "reasoning_format" not in (args.get("extra_body") or {})


def test_tag_mode_groq_reasoner_is_unshaped() -> None:
    # The existing inline-tag reasoner must not receive the stream-mode wire
    # shaping (no reasoning_effort / reasoning_format injection).
    llm = _groq_llm("qwen/qwen3-32b")
    assert llm._reasoning_mode == "tags"
    args = llm._prepare_api_request([], None, RequestParams(maxTokens=512))
    assert "reasoning_effort" not in args
    assert "reasoning_format" not in (args.get("extra_body") or {})
