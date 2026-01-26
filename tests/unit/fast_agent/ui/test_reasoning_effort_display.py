from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
from fast_agent.ui.reasoning_effort_display import (
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
