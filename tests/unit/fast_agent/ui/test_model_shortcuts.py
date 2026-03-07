from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
from fast_agent.llm.text_verbosity import TextVerbositySpec
from fast_agent.ui.model_shortcuts import (
    ModelShortcutHint,
    build_model_shortcut_hints,
    cycle_reasoning_setting,
    cycle_text_verbosity,
)


def test_cycle_reasoning_setting_uses_available_values_order() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high"],
        allow_toggle_disable=True,
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    assert cycle_reasoning_setting(None, spec) == ReasoningEffortSetting(kind="effort", value="high")
    assert cycle_reasoning_setting(ReasoningEffortSetting(kind="effort", value="high"), spec) == ReasoningEffortSetting(kind="toggle", value=False)


def test_cycle_text_verbosity_uses_spec_default_first() -> None:
    spec = TextVerbositySpec(allowed=("low", "medium", "high"), default="medium")

    assert cycle_text_verbosity(None, spec) == "high"
    assert cycle_text_verbosity("high", spec) == "low"


class _ShortcutStub:
    service_tier_supported = True
    available_service_tiers = ("fast", "flex")
    web_search_supported = True
    web_fetch_supported = False
    reasoning_effort_spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high"],
        allow_toggle_disable=True,
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )
    text_verbosity_spec = TextVerbositySpec()


def test_build_model_shortcut_hints_only_lists_supported_controls() -> None:
    hints = build_model_shortcut_hints(_ShortcutStub())

    assert hints == [
        ModelShortcutHint("Shift+Tab", "Service tier", "standard, fast, flex"),
        ModelShortcutHint("F6", "Reasoning", "low, medium, high, off"),
        ModelShortcutHint("F7", "Verbosity", "low, medium, high"),
        ModelShortcutHint("F8", "Web search", "on, off"),
    ]

def test_build_model_shortcut_hints_codexresponses_omit_flex() -> None:
    class _CodexShortcutStub(_ShortcutStub):
        available_service_tiers = ("fast",)

    hints = build_model_shortcut_hints(_CodexShortcutStub())

    assert hints[0] == ModelShortcutHint("Shift+Tab", "Service tier", "standard, fast")


