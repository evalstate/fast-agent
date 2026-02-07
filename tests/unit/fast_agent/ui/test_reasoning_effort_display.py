from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
from fast_agent.ui.reasoning_effort_display import (
    AUTO_COLOR,
    FULL_BLOCK,
    INACTIVE_COLOR,
    render_reasoning_effort_gauge,
)


def test_toggle_reasoning_gauge_defaults_to_full_block():
    spec = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )

    gauge = render_reasoning_effort_gauge(None, spec)

    assert gauge == "<style bg='ansigreen'>" + FULL_BLOCK + "</style>"


def test_toggle_reasoning_gauge_disabled_is_inactive():
    spec = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )
    setting = ReasoningEffortSetting(kind="toggle", value=False)

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge == f"<style bg='{INACTIVE_COLOR}'>" + FULL_BLOCK + "</style>"


def test_effort_max_renders_highest_gauge():
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "max"],
        default=ReasoningEffortSetting(kind="effort", value="high"),
    )
    setting = ReasoningEffortSetting(kind="effort", value="max")

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge is not None
    assert "ansired" in gauge


def test_effort_auto_renders_blue():
    """The 'auto' effort setting should render as a blue full block."""
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "max"],
        default=ReasoningEffortSetting(kind="effort", value="high"),
    )
    setting = ReasoningEffortSetting(kind="effort", value="auto")

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge is not None
    assert AUTO_COLOR in gauge
    assert FULL_BLOCK in gauge


def test_effort_explicit_setting_not_blue():
    """When an explicit effort is supplied, the gauge should not be blue."""
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "max"],
        default=ReasoningEffortSetting(kind="effort", value="high"),
    )
    setting = ReasoningEffortSetting(kind="effort", value="high")

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge is not None
    assert AUTO_COLOR not in gauge
    assert "ansiyellow" in gauge


def test_toggle_auto_not_blue():
    """Toggle specs should never show blue even when setting is None."""
    spec = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )

    gauge = render_reasoning_effort_gauge(None, spec)

    assert gauge is not None
    assert AUTO_COLOR not in gauge
