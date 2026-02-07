"""Reasoning effort gauge rendering for the TUI toolbar."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.llm.reasoning_effort import (
        ReasoningEffortSetting,
        ReasoningEffortSpec,
    )

BRAILLE_FILL = {0: "⣿", 1: "⣀", 2: "⣤", 3: "⣶", 4: "⣿"}
FULL_BLOCK = "⣿"
INACTIVE_COLOR = "ansibrightblack"
AUTO_COLOR = "ansiblue"
MAX_LEVEL = 4

EFFORT_LEVEL_MAPPING = {
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 4,
    "max": 4,
}

EFFORT_COLOR_MAPPING = {
    "none": INACTIVE_COLOR,
    "minimal": "ansigreen",
    "low": "ansigreen",
    "medium": "ansiyellow",
    "high": "ansiyellow",
    "xhigh": "ansired",
    "max": "ansired",
}


def _effort_to_level(value: str) -> int:
    return EFFORT_LEVEL_MAPPING.get(value, 0)


def _effort_color(value: str) -> str:
    return EFFORT_COLOR_MAPPING.get(value, "ansiyellow")


def _budget_to_level(value: int, spec: ReasoningEffortSpec) -> int:
    if value <= 0:
        return 0
    presets = sorted({preset for preset in (spec.budget_presets or []) if preset > 0})
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
            2 + int(round(ratio * (MAX_LEVEL - 2))),
        )
    min_budget = spec.min_budget_tokens
    max_budget = spec.max_budget_tokens
    if min_budget is None or max_budget is None or max_budget <= min_budget:
        return 1

    ratio = (value - min_budget) / (max_budget - min_budget)
    ratio = min(max(ratio, 0.0), 1.0)
    return max(1, min(MAX_LEVEL, 1 + int(math.floor(ratio * (MAX_LEVEL - 1)))))


def _budget_color(value: int, spec: ReasoningEffortSpec, level: int) -> str:
    presets = sorted({preset for preset in (spec.budget_presets or []) if preset > 0})
    if presets and value >= presets[-1]:
        return "ansired"
    if level <= 3:
        return "ansigreen"
    return "ansiyellow"


def render_reasoning_effort_gauge(
    setting: ReasoningEffortSetting | None,
    spec: ReasoningEffortSpec | None,
) -> str | None:
    from fast_agent.llm.reasoning_effort import is_auto_reasoning

    if spec is None:
        return None

    effective = setting or spec.default
    # "auto" means the provider chooses — show as blue full block.
    if is_auto_reasoning(setting) or is_auto_reasoning(effective):
        return f"<style bg='{AUTO_COLOR}'>{FULL_BLOCK}</style>"
    if effective is None:
        level = 0
    elif effective.kind == "toggle":
        level = 0 if not effective.value else MAX_LEVEL
    elif effective.kind == "effort":
        effort_value = str(effective.value)
        level = _effort_to_level(effort_value)
    elif effective.kind == "budget":
        level = _budget_to_level(int(effective.value), spec)
    else:
        level = 0

    if level <= 0:
        return f"<style bg='{INACTIVE_COLOR}'>{FULL_BLOCK}</style>"

    char = BRAILLE_FILL.get(min(level, MAX_LEVEL), BRAILLE_FILL[MAX_LEVEL])
    if effective is None:
        color = INACTIVE_COLOR
    elif effective.kind == "effort":
        color = _effort_color(effort_value)
    elif effective.kind == "toggle":
        color = "ansigreen" if effective.value else INACTIVE_COLOR
    else:
        color = _budget_color(int(effective.value), spec, level)
    return f"<style bg='{color}'>{char}</style>"
