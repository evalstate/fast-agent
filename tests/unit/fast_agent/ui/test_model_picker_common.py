from __future__ import annotations

from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import (
    ModelOption,
    build_snapshot,
    model_capabilities,
    model_options_for_provider,
)


def test_generic_provider_uses_custom_local_model_option() -> None:
    snapshot = build_snapshot()

    options = model_options_for_provider(snapshot, Provider.GENERIC, source="curated")

    assert options == [
        ModelOption(
            spec="generic.__custom__",
            label="Enter local model string (e.g. llama3.2)",
        )
    ]


def test_openresponses_models_report_web_search_support() -> None:
    capabilities = model_capabilities("openresponses.gpt-5-mini")

    assert capabilities.provider == Provider.OPENRESPONSES
    assert capabilities.web_search_supported is True


def test_46_models_do_not_report_optional_long_context() -> None:
    capabilities = model_capabilities("claude-opus-4-6?context=1m")

    assert capabilities.provider == Provider.ANTHROPIC
    assert capabilities.supports_long_context is False
    assert capabilities.current_long_context is False
    assert capabilities.long_context_window is None
