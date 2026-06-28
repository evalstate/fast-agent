"""Reasoning effort gauge rendering for the TUI toolbar."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeAlias

from fast_agent.llm.reasoning_effort import is_auto_reasoning
from fast_agent.ui.gauge_glyph_palette import (
    MAX_GAUGE_LEVEL,
    STANDALONE_GAUGE_GLYPHS,
    GaugeGlyphPalette,
    GaugeState,
    render_gauge_state,
)
from fast_agent.utils.numeric import positive_int_or_none, sorted_unique_positive_ints

if TYPE_CHECKING:
    from fast_agent.llm.reasoning_effort import (
        ReasoningEffortSetting,
        ReasoningEffortSpec,
    )

INACTIVE_COLOR = "ansibrightblack"
AUTO_COLOR = "ansiblue"
MAX_LEVEL = MAX_GAUGE_LEVEL


EFFORT_LEVEL_MAPPING = {
    "none": 0,
    "minimal": 1,
    "low": 1,
    "medium": 2,
    "high": 3,
    "xhigh": 4,
    "max": 4,
}

EFFORT_COLOR_MAPPING = {
    "none": INACTIVE_COLOR,
    "minimal": "ansigreen",
    "low": "ansigreen",
    "medium": "ansigreen",
    "high": "ansiyellow",
    "xhigh": "ansiyellow",
    "max": "ansired",
}

EFFORT_ORDER_MAPPING = {
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
}

EffortScaleKey: TypeAlias = tuple[str, str, str, str]
_EFFORT_SCALE_LEVELS = {
    ("none", "low", "medium", "high"): {
        "none": 0,
        "low": 2,
        "medium": 3,
        "high": 4,
    },
    ("minimal", "low", "medium", "high"): {
        "minimal": 1,
        "low": 2,
        "medium": 3,
        "high": 4,
    },
}
_EFFORT_SCALE_COLORS = {
    ("none", "low", "medium", "high"): {
        "none": INACTIVE_COLOR,
        "low": "ansigreen",
        "medium": "ansiyellow",
        "high": "ansired",
    },
    ("minimal", "low", "medium", "high"): {
        "minimal": "ansigreen",
        "low": "ansigreen",
        "medium": "ansiyellow",
        "high": "ansired",
    },
}


def _effort_scale_key(spec: ReasoningEffortSpec) -> EffortScaleKey | None:
    allowed_efforts = spec.allowed_efforts or []
    if len(allowed_efforts) != 4:
        return None
    key: EffortScaleKey = (
        allowed_efforts[0],
        allowed_efforts[1],
        allowed_efforts[2],
        allowed_efforts[3],
    )
    return key if key in _EFFORT_SCALE_LEVELS else None


def _effort_to_level(value: str, spec: ReasoningEffortSpec | None = None) -> int:
    if spec is not None and (scale_key := _effort_scale_key(spec)):
        return _EFFORT_SCALE_LEVELS[scale_key].get(value, 0)
    return EFFORT_LEVEL_MAPPING.get(value, 0)


def _is_spec_highest_effort(value: str, spec: ReasoningEffortSpec) -> bool:
    allowed_efforts = spec.allowed_efforts
    if not allowed_efforts:
        return False

    highest_effort = max(allowed_efforts, key=lambda effort: EFFORT_ORDER_MAPPING.get(effort, -1))
    return value == highest_effort and _effort_to_level(value, spec) == MAX_LEVEL


def _effort_color(value: str, spec: ReasoningEffortSpec) -> str:
    if scale_key := _effort_scale_key(spec):
        return _EFFORT_SCALE_COLORS[scale_key].get(value, "ansiyellow")
    if value == "xhigh" and _is_spec_highest_effort(value, spec):
        return "ansired"
    return EFFORT_COLOR_MAPPING.get(value, "ansiyellow")


def _positive_budget_presets(spec: ReasoningEffortSpec) -> list[int]:
    return sorted_unique_positive_ints(spec.budget_presets or [])


def _budget_to_level(value: int, spec: ReasoningEffortSpec) -> int:
    if value <= 0:
        return 0
    presets = _positive_budget_presets(spec)
    if presets:
        low_threshold = presets[0]
        high_threshold = presets[-1]
        if value < low_threshold:
            return 1
        if high_threshold <= low_threshold:
            return MAX_LEVEL
        ratio = (value - low_threshold) / (high_threshold - low_threshold)
        ratio = min(max(ratio, 0.0), 1.0)
        return min(
            MAX_LEVEL,
            2 + round(ratio * (MAX_LEVEL - 2)),
        )
    min_budget = spec.min_budget_tokens
    max_budget = spec.max_budget_tokens
    if min_budget is None or max_budget is None or max_budget <= min_budget:
        return 1

    ratio = (value - min_budget) / (max_budget - min_budget)
    ratio = min(max(ratio, 0.0), 1.0)
    return max(1, min(MAX_LEVEL, 1 + math.floor(ratio * (MAX_LEVEL - 1))))


def _budget_color(value: int, spec: ReasoningEffortSpec, level: int) -> str:
    presets = _positive_budget_presets(spec)
    if presets and value >= presets[-1]:
        return "ansired"
    if level <= 3:
        return "ansigreen"
    return "ansiyellow"


def _budget_value(setting: ReasoningEffortSetting) -> int | None:
    return positive_int_or_none(setting.value)


def _inactive_state() -> GaugeState:
    return GaugeState(level=0, color=INACTIVE_COLOR)


def _toggle_reasoning_gauge_state(setting: ReasoningEffortSetting) -> GaugeState:
    enabled = bool(setting.value)
    color = "ansigreen" if enabled else INACTIVE_COLOR
    return GaugeState(level=MAX_LEVEL if enabled else 0, color=color)


def _effort_reasoning_gauge_state(
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> GaugeState:
    effort_value = str(setting.value)
    level = _effort_to_level(effort_value, spec)
    if level <= 0:
        return _inactive_state()
    return GaugeState(level=level, color=_effort_color(effort_value, spec))


def _budget_reasoning_gauge_state(
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> GaugeState:
    budget_value = _budget_value(setting)
    if budget_value is None:
        return _inactive_state()
    level = _budget_to_level(budget_value, spec)
    if level <= 0:
        return _inactive_state()
    return GaugeState(
        level=level,
        color=_budget_color(budget_value, spec, level),
    )


def _reasoning_gauge_state(
    setting: ReasoningEffortSetting | None,
    spec: ReasoningEffortSpec,
) -> GaugeState:
    effective = setting or spec.default
    # "auto" means the provider chooses: show as blue full block.
    if is_auto_reasoning(setting) or is_auto_reasoning(effective):
        return GaugeState(level=MAX_LEVEL, color=AUTO_COLOR)
    if effective is None:
        return _inactive_state()

    match effective.kind:
        case "toggle":
            return _toggle_reasoning_gauge_state(effective)
        case "effort":
            return _effort_reasoning_gauge_state(effective, spec)
        case "budget":
            return _budget_reasoning_gauge_state(effective, spec)

    return _inactive_state()


def render_reasoning_effort_gauge(
    setting: ReasoningEffortSetting | None,
    spec: ReasoningEffortSpec | None,
    *,
    glyph_palette: GaugeGlyphPalette = STANDALONE_GAUGE_GLYPHS,
) -> str | None:
    if spec is None:
        return None

    state = _reasoning_gauge_state(setting, spec)
    return render_gauge_state(
        state,
        glyph_palette=glyph_palette,
        inactive_color=INACTIVE_COLOR,
    )
