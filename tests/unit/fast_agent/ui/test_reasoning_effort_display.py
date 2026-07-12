from typing import Any, cast

from fast_agent.llm.reasoning_effort import (
    ReasoningEffortLevel,
    ReasoningEffortSetting,
    ReasoningEffortSpec,
)
from fast_agent.ui.gauge_glyph_palette import (
    PAIRED_REASONING_GAUGE_GLYPHS,
    STANDALONE_GAUGE_GLYPHS,
)
from fast_agent.ui.reasoning_effort_display import (
    AUTO_COLOR,
    INACTIVE_COLOR,
    render_reasoning_effort_gauge,
)

FULL_BLOCK = STANDALONE_GAUGE_GLYPHS.full_block


def test_toggle_reasoning_gauge_defaults_to_three_high_peak() -> None:
    spec = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )

    gauge = render_reasoning_effort_gauge(None, spec)

    assert gauge == "<style bg='ansigreen'>⣿</style>"


def test_toggle_reasoning_gauge_disabled_is_inactive() -> None:
    spec = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )
    setting = ReasoningEffortSetting(kind="toggle", value=False)

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge == f"<style bg='{INACTIVE_COLOR}'>" + FULL_BLOCK + "</style>"


def test_effort_max_renders_highest_gauge() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "max"],
        default=ReasoningEffortSetting(kind="effort", value="high"),
    )
    setting = ReasoningEffortSetting(kind="effort", value="max")

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge is not None
    assert "ansired" in gauge
    assert gauge == "<style bg='ansired'>⣿</style>"


def test_effort_xhigh_renders_red_when_it_is_the_highest_allowed_effort() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["minimal", "low", "medium", "high", "xhigh"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )
    setting = ReasoningEffortSetting(kind="effort", value="xhigh")

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge == "<style bg='ansired'>⣿</style>"


def test_effort_xhigh_uses_distinct_non_max_gauge_when_max_is_available() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "xhigh", "max"],
        default=ReasoningEffortSetting(kind="effort", value="high"),
    )
    setting = ReasoningEffortSetting(kind="effort", value="xhigh")

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge == "<style bg='ansiyellow'>⣿</style>"


def test_effort_auto_renders_blue() -> None:
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


def test_effort_explicit_setting_not_blue() -> None:
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
    assert "⣶" in gauge


def test_effort_none_low_medium_high_scale_uses_full_dynamic_range() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["none", "low", "medium", "high"],
        default=ReasoningEffortSetting(kind="effort", value="low"),
    )

    assert (
        render_reasoning_effort_gauge(ReasoningEffortSetting(kind="effort", value="low"), spec)
        == "<style bg='ansigreen'>⣤</style>"
    )
    assert (
        render_reasoning_effort_gauge(ReasoningEffortSetting(kind="effort", value="medium"), spec)
        == "<style bg='ansiyellow'>⣶</style>"
    )
    assert (
        render_reasoning_effort_gauge(ReasoningEffortSetting(kind="effort", value="high"), spec)
        == "<style bg='ansired'>⣿</style>"
    )


def test_effort_minimal_low_medium_high_scale_uses_full_dynamic_range() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["minimal", "low", "medium", "high"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    assert (
        render_reasoning_effort_gauge(ReasoningEffortSetting(kind="effort", value="minimal"), spec)
        == "<style bg='ansigreen'>⣀</style>"
    )
    assert (
        render_reasoning_effort_gauge(ReasoningEffortSetting(kind="effort", value="low"), spec)
        == "<style bg='ansigreen'>⣤</style>"
    )
    assert (
        render_reasoning_effort_gauge(ReasoningEffortSetting(kind="effort", value="medium"), spec)
        == "<style bg='ansiyellow'>⣶</style>"
    )
    assert (
        render_reasoning_effort_gauge(ReasoningEffortSetting(kind="effort", value="high"), spec)
        == "<style bg='ansired'>⣿</style>"
    )


def test_toggle_auto_not_blue() -> None:
    """Toggle specs should never show blue even when setting is None."""
    spec = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )

    gauge = render_reasoning_effort_gauge(None, spec)

    assert gauge is not None
    assert AUTO_COLOR not in gauge


def test_reasoning_gauge_can_render_paired_palette() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )
    setting = ReasoningEffortSetting(kind="effort", value="medium")

    gauge = render_reasoning_effort_gauge(
        setting,
        spec,
        glyph_palette=PAIRED_REASONING_GAUGE_GLYPHS,
    )

    assert gauge == "<style bg='ansigreen'>⢠</style>"


def test_gpt_56_reasoning_scale_renders_all_paired_status_bar_levels() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["none", "low", "medium", "high", "xhigh", "max"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )
    expected: dict[ReasoningEffortLevel, str] = {
        "none": "<style bg='ansibrightblack'>⢸</style>",
        "low": "<style bg='ansigreen'>⢀</style>",
        "medium": "<style bg='ansigreen'>⢠</style>",
        "high": "<style bg='ansiyellow'>⢰</style>",
        "xhigh": "<style bg='ansiyellow'>⢸</style>",
        "max": "<style bg='ansired'>⢸</style>",
    }

    for effort, gauge in expected.items():
        setting = ReasoningEffortSetting(kind="effort", value=effort)
        assert (
            render_reasoning_effort_gauge(
                setting,
                spec,
                glyph_palette=PAIRED_REASONING_GAUGE_GLYPHS,
            )
            == gauge
        )


def test_budget_gauge_renders_inactive_for_malformed_runtime_value() -> None:
    spec = ReasoningEffortSpec(
        kind="budget",
        min_budget_tokens=0,
        max_budget_tokens=10_000,
    )
    setting = ReasoningEffortSetting(kind="budget", value=cast("Any", "not-a-budget"))

    gauge = render_reasoning_effort_gauge(setting, spec)

    assert gauge == f"<style bg='{INACTIVE_COLOR}'>" + FULL_BLOCK + "</style>"


def test_budget_gauge_renders_inactive_for_non_positive_runtime_value() -> None:
    spec = ReasoningEffortSpec(
        kind="budget",
        min_budget_tokens=0,
        max_budget_tokens=10_000,
    )

    zero_gauge = render_reasoning_effort_gauge(
        ReasoningEffortSetting(kind="budget", value=0),
        spec,
    )
    negative_gauge = render_reasoning_effort_gauge(
        ReasoningEffortSetting(kind="budget", value=-1),
        spec,
    )

    inactive_gauge = f"<style bg='{INACTIVE_COLOR}'>" + FULL_BLOCK + "</style>"
    assert zero_gauge == inactive_gauge
    assert negative_gauge == inactive_gauge


def test_budget_gauge_normalizes_presets_for_level_and_color() -> None:
    spec = ReasoningEffortSpec(
        kind="budget",
        budget_presets=[20_000, 0, 1_000, 1_000, -5],
    )

    low_gauge = render_reasoning_effort_gauge(
        ReasoningEffortSetting(kind="budget", value=500),
        spec,
    )
    max_gauge = render_reasoning_effort_gauge(
        ReasoningEffortSetting(kind="budget", value=20_000),
        spec,
    )

    assert low_gauge == "<style bg='ansigreen'>⣀</style>"
    assert max_gauge == "<style bg='ansired'>⣿</style>"
