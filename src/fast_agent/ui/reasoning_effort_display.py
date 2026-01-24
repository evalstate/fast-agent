"""Reasoning effort gauge rendering for the TUI toolbar."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec

BRAILLE_FILL = {0: "⣿", 1: "⣀", 2: "⣤", 3: "⣶", 4: "⣿"}
FULL_BLOCK = "⣿"
INACTIVE_COLOR = "ansibrightblack"
MAX_LEVEL = 4


def _effort_to_level(value: str) -> int:
    mapping = {
        "minimal": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "xhigh": 4,
    }
    return mapping.get(value, 0)


def _budget_to_level(value: int, spec: ReasoningEffortSpec) -> int:
    if value <= 0:
        return 0
    min_budget = spec.min_budget_tokens
    max_budget = spec.max_budget_tokens
    if min_budget is None or max_budget is None or max_budget <= min_budget:
        return 1

    ratio = (value - min_budget) / (max_budget - min_budget)
    ratio = min(max(ratio, 0.0), 1.0)
    return max(1, min(MAX_LEVEL, 1 + int(math.floor(ratio * (MAX_LEVEL - 1)))))


def render_reasoning_effort_gauge(
    setting: ReasoningEffortSetting | None,
    spec: ReasoningEffortSpec | None,
) -> str | None:
    if spec is None:
        return None

    effective = setting or spec.default
    if effective is None:
        level = 0
    elif effective.kind == "toggle":
        level = 0 if not effective.value else 1
    elif effective.kind == "effort":
        level = _effort_to_level(str(effective.value))
    elif effective.kind == "budget":
        level = _budget_to_level(int(effective.value), spec)
    else:
        level = 0

    if level <= 0:
        return f"<style bg='{INACTIVE_COLOR}'>{FULL_BLOCK}</style>"

    char = BRAILLE_FILL.get(min(level, MAX_LEVEL), BRAILLE_FILL[MAX_LEVEL])
    if level <= 1:
        color = "ansigreen"
    elif level <= 3:
        color = "ansiyellow"
    else:
        color = "ansired"
    return f"<style bg='{color}'>{char}</style>"
