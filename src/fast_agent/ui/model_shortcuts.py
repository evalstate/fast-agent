"""Model-control shortcut helpers for the interactive prompt."""

from __future__ import annotations

from dataclasses import dataclass

from fast_agent.llm.reasoning_effort import (
    ReasoningEffortSetting,
    ReasoningEffortSpec,
    available_reasoning_values,
    parse_reasoning_setting,
    validate_reasoning_setting,
)
from fast_agent.llm.text_verbosity import (
    TextVerbosityLevel,
    TextVerbositySpec,
    available_text_verbosity_values,
    parse_text_verbosity,
)


@dataclass(frozen=True, slots=True)
class ModelShortcutHint:
    key: str
    label: str
    values_text: str


def _dedupe_preserve_order[T](values: list[T]) -> list[T]:
    deduped: list[T] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped


def _shortcut_reasoning_values(spec: ReasoningEffortSpec) -> list[str]:
    return [
        token
        for token in available_reasoning_values(spec)
        if token not in {"auto", "default"}
        and not (
            token == "off"
            and spec.kind == "effort"
            and "none" in (spec.allowed_efforts or [])
        )
    ]






def cycle_reasoning_setting(
    current: ReasoningEffortSetting | None,
    spec: ReasoningEffortSpec | None,
) -> ReasoningEffortSetting | None:
    if spec is None:
        return None

    candidates: list[ReasoningEffortSetting] = []
    for token in _shortcut_reasoning_values(spec):
        parsed = parse_reasoning_setting(token)
        if parsed is None:
            continue
        try:
            candidates.append(validate_reasoning_setting(parsed, spec))
        except ValueError:
            continue

    candidates = _dedupe_preserve_order(candidates)
    if not candidates:
        return None

    effective_current = current or spec.default
    if effective_current is None:
        return candidates[0]

    try:
        current_index = candidates.index(effective_current)
    except ValueError:
        return candidates[0]
    return candidates[(current_index + 1) % len(candidates)]



def cycle_text_verbosity(
    current: TextVerbosityLevel | None,
    spec: TextVerbositySpec | None,
) -> TextVerbosityLevel | None:
    if spec is None:
        return None

    candidates = [
        value
        for token in available_text_verbosity_values(spec)
        if (value := parse_text_verbosity(token)) is not None
    ]
    candidates = _dedupe_preserve_order(candidates)
    if not candidates:
        return None

    effective_current = current or spec.default
    try:
        current_index = candidates.index(effective_current)
    except ValueError:
        return candidates[0]
    return candidates[(current_index + 1) % len(candidates)]





def _service_tier_hint_values(llm: object) -> str:
    raw_values = getattr(llm, "available_service_tiers", ())
    available_values = [value for value in raw_values if value in {"fast", "flex"}]
    if not available_values and bool(getattr(llm, "service_tier_supported", False)):
        available_values = ["fast", "flex"]
    values = ["standard", *available_values]
    return ", ".join(_dedupe_preserve_order(values))


def build_model_shortcut_hints(llm: object | None) -> list[ModelShortcutHint]:
    if llm is None:
        return []

    hints: list[ModelShortcutHint] = []

    if bool(getattr(llm, "service_tier_supported", False)):
        hints.append(
            ModelShortcutHint(
                key="Shift+Tab",
                label="Service tier",
                values_text=_service_tier_hint_values(llm),
            )
        )

    reasoning_spec = getattr(llm, "reasoning_effort_spec", None)
    if isinstance(reasoning_spec, ReasoningEffortSpec):
        values = ", ".join(_dedupe_preserve_order(_shortcut_reasoning_values(reasoning_spec)))
        hints.append(ModelShortcutHint(key="F6", label="Reasoning", values_text=values))

    verbosity_spec = getattr(llm, "text_verbosity_spec", None)
    if isinstance(verbosity_spec, TextVerbositySpec):
        values = ", ".join(_dedupe_preserve_order(available_text_verbosity_values(verbosity_spec)))
        hints.append(ModelShortcutHint(key="F7", label="Verbosity", values_text=values))

    if bool(getattr(llm, "web_search_supported", False)):
        hints.append(
            ModelShortcutHint(
                key="F8",
                label="Web search",
                values_text="on, off",
            )
        )

    if bool(getattr(llm, "web_fetch_supported", False)):
        hints.append(
            ModelShortcutHint(
                key="F9",
                label="Web fetch",
                values_text="on, off",
            )
        )

    return hints
