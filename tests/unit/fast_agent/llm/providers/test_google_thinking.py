"""Tests for Google thinking/reasoning support via the ThinkingConfig SDK feature."""

from __future__ import annotations

import pytest
from google.genai import types as google_types
from mcp.types import TextContent

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.google.google_converter import GoogleConverter
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM


def _build_llm(config: Settings | None = None, **kwargs) -> GoogleNativeLLM:
    config = config or Settings()
    return GoogleNativeLLM(context=Context(config=config), **kwargs)


# ---------- Model database wiring ----------


@pytest.mark.unit
def test_gemini_25_has_thinking_spec() -> None:
    """Gemini 2.5+ models should have a google_thinking reasoning spec."""
    for model in ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"):
        assert ModelDatabase.get_reasoning(model) == "google_thinking", model
        spec = ModelDatabase.get_reasoning_effort_spec(model)
        assert spec is not None, model
        assert spec.kind == "effort"
        assert spec.allow_auto is True


@pytest.mark.unit
def test_gemini_20_flash_no_thinking_spec() -> None:
    """Gemini 2.0 Flash should not have a reasoning spec."""
    assert ModelDatabase.get_reasoning("gemini-2.0-flash") is None
    assert ModelDatabase.get_reasoning_effort_spec("gemini-2.0-flash") is None


# ---------- _resolve_thinking_config ----------


@pytest.mark.unit
def test_resolve_thinking_config_auto() -> None:
    llm = _build_llm(model="gemini-2.5-flash")
    budget, level = llm._resolve_thinking_config()
    # Default is auto → budget=-1, no named level
    assert budget == -1
    assert level is None


@pytest.mark.unit
def test_resolve_thinking_config_disabled() -> None:
    llm = _build_llm(model="gemini-2.5-flash", reasoning_effort="off")
    budget, level = llm._resolve_thinking_config()
    assert budget == 0
    assert level is None


@pytest.mark.unit
def test_resolve_thinking_config_effort_levels() -> None:
    """Named effort levels should map to SDK ThinkingLevel names."""
    for effort, expected_level in [
        ("minimal", "MINIMAL"),
        ("low", "LOW"),
        ("medium", "MEDIUM"),
        ("high", "HIGH"),
    ]:
        llm = _build_llm(model="gemini-2.5-flash", reasoning_effort=effort)
        budget, level = llm._resolve_thinking_config()
        assert level == expected_level, f"effort={effort}"
        assert budget is None, f"effort={effort} should not set budget"


@pytest.mark.unit
def test_resolve_thinking_config_none_when_no_model_support() -> None:
    llm = _build_llm(model="gemini-2.0-flash")
    budget, level = llm._resolve_thinking_config()
    assert budget is None
    assert level is None


# ---------- Converter: thinking config in GenerateContentConfig ----------


@pytest.mark.unit
def test_converter_thinking_config_auto() -> None:
    from fast_agent.types import RequestParams

    converter = GoogleConverter()
    config = converter.convert_request_params_to_google_config(
        RequestParams(model="gemini-2.5-flash"), thinking_budget=-1
    )
    assert config.thinking_config is not None
    assert config.thinking_config.include_thoughts is True
    assert config.thinking_config.thinking_budget == -1
    assert config.thinking_config.thinking_level is None


@pytest.mark.unit
def test_converter_thinking_config_with_level() -> None:
    from fast_agent.types import RequestParams

    converter = GoogleConverter()
    config = converter.convert_request_params_to_google_config(
        RequestParams(model="gemini-2.5-flash"), thinking_level="HIGH"
    )
    assert config.thinking_config is not None
    assert config.thinking_config.include_thoughts is True
    assert config.thinking_config.thinking_level == "HIGH"
    assert config.thinking_config.thinking_budget is None


@pytest.mark.unit
def test_converter_thinking_config_disabled() -> None:
    from fast_agent.types import RequestParams

    converter = GoogleConverter()
    config = converter.convert_request_params_to_google_config(
        RequestParams(model="gemini-2.5-flash"), thinking_budget=0
    )
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0


@pytest.mark.unit
def test_converter_no_thinking_config_when_none() -> None:
    from fast_agent.types import RequestParams

    converter = GoogleConverter()
    config = converter.convert_request_params_to_google_config(
        RequestParams(model="gemini-2.0-flash")
    )
    assert config.thinking_config is None


@pytest.mark.unit
def test_converter_skips_thought_parts_in_non_stream_content() -> None:
    converter = GoogleConverter()
    content = google_types.Content.model_validate(
        {
            "role": "model",
            "parts": [
                {"text": "internal reasoning", "thought": True},
                {"text": '{"status":"ok"}'},
            ],
        }
    )

    parts = converter.convert_from_google_content(content)

    assert [part.text for part in parts if isinstance(part, TextContent)] == ['{"status":"ok"}']


@pytest.mark.unit
def test_converter_skips_thought_parts_when_building_message_history() -> None:
    converter = GoogleConverter()
    content = google_types.Content.model_validate(
        {
            "role": "model",
            "parts": [
                {"text": "internal reasoning", "thought": True},
                {"text": "final answer"},
            ],
        }
    )

    message = converter.convert_from_google_content_list([content])[0]

    assert [part.text for part in message.content if isinstance(part, TextContent)] == [
        "final answer"
    ]


# ---------- Finish reason mapping ----------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("IMAGE_PROHIBITED_CONTENT", "SAFETY"),
        ("IMAGE_RECITATION", "SAFETY"),
        ("NO_IMAGE", "ERROR"),
        ("IMAGE_OTHER", "ERROR"),
        ("STOP", "END_TURN"),
        ("MAX_TOKENS", "MAX_TOKENS"),
    ],
)
def test_map_finish_reason_new_values(reason: str, expected: str) -> None:
    llm = _build_llm(model="gemini-2.0-flash")
    result = llm._map_finish_reason(reason)
    assert result.name == expected, f"reason={reason}"


# ---------- Init reasoning from config ----------


@pytest.mark.unit
def test_reasoning_from_config_dict() -> None:
    """GoogleSettings(extra='allow') permits a 'reasoning' key."""
    config = Settings.model_validate({"google": {"reasoning": "high"}})
    llm = _build_llm(config=config, model="gemini-2.5-flash")
    assert llm.reasoning_effort is not None
    assert llm.reasoning_effort.kind == "effort"
    assert llm.reasoning_effort.value == "high"


@pytest.mark.unit
def test_reasoning_from_kwargs_overrides_config() -> None:
    config = Settings.model_validate({"google": {"reasoning": "high"}})
    llm = _build_llm(config=config, model="gemini-2.5-flash", reasoning_effort="low")
    assert llm.reasoning_effort is not None
    assert llm.reasoning_effort.value == "low"


# ---------- Available values for /model reasoning ----------


@pytest.mark.unit
def test_available_reasoning_values() -> None:
    from fast_agent.llm.reasoning_effort import available_reasoning_values

    spec = ModelDatabase.get_reasoning_effort_spec("gemini-2.5-flash")
    values = available_reasoning_values(spec)
    assert "auto" in values
    assert "minimal" in values
    assert "low" in values
    assert "medium" in values
    assert "high" in values
    assert "off" in values
