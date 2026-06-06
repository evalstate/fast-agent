"""Model-control shortcut helpers for the interactive prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.commands.model_capabilities import (
    available_service_tier_values,
    resolve_reasoning_effort_spec,
    resolve_service_tier_supported,
    resolve_text_verbosity_spec,
)
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
from fast_agent.ui.model_binary_toggles import MODEL_BINARY_TOGGLES
from fast_agent.utils.collections import cycle_next, unique_preserve_order

if TYPE_CHECKING:
    from fast_agent.interfaces import FastAgentLLMProtocol


@dataclass(frozen=True, slots=True)
class ModelShortcutHint:
    key: str
    label: str
    values_text: str


_SHORTCUT_REASONING_VALUE_LABELS = {
    "auto": "adaptive",
}


def _shortcut_reasoning_values(spec: ReasoningEffortSpec) -> list[str]:
    values = [
        token
        for token in available_reasoning_values(spec)
        if token != "default"
        and not (
            token == "off" and spec.kind == "effort" and "none" in (spec.allowed_efforts or [])
        )
    ]
    if "auto" in values:
        values = [token for token in values if token != "auto"] + ["auto"]
    return values


def _display_shortcut_reasoning_value(token: str) -> str:
    return _SHORTCUT_REASONING_VALUE_LABELS.get(token, token)


def _shortcut_values_text(values: list[str]) -> str:
    return ", ".join(unique_preserve_order(values))


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

    candidates = unique_preserve_order(candidates)
    return cycle_next(current, candidates, default=spec.default)


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
    candidates = unique_preserve_order(candidates)
    return cycle_next(current, candidates, default=spec.default)


def _service_tier_hint_values(llm: "FastAgentLLMProtocol") -> str:
    available_values = available_service_tier_values(llm)
    values = ["standard", *available_values]
    return ", ".join(unique_preserve_order(values))


def build_model_shortcut_hints(llm: "FastAgentLLMProtocol | None") -> list[ModelShortcutHint]:
    if llm is None:
        return []

    hints: list[ModelShortcutHint] = []

    if resolve_service_tier_supported(llm):
        hints.append(
            ModelShortcutHint(
                key="Shift+Tab",
                label="Service tier",
                values_text=_service_tier_hint_values(llm),
            )
        )

    reasoning_spec = resolve_reasoning_effort_spec(llm)
    if isinstance(reasoning_spec, ReasoningEffortSpec):
        values = _shortcut_values_text(
            [
                _display_shortcut_reasoning_value(token)
                for token in _shortcut_reasoning_values(reasoning_spec)
            ]
        )
        hints.append(ModelShortcutHint(key="F6", label="Reasoning", values_text=values))

    verbosity_spec = resolve_text_verbosity_spec(llm)
    if isinstance(verbosity_spec, TextVerbositySpec):
        values = _shortcut_values_text(available_text_verbosity_values(verbosity_spec))
        hints.append(ModelShortcutHint(key="F7", label="Verbosity", values_text=values))

    for toggle in MODEL_BINARY_TOGGLES:
        if toggle.resolve_supported(llm):
            hints.append(
                ModelShortcutHint(
                    key=toggle.shortcut_key,
                    label=toggle.label,
                    values_text="on, off",
                )
            )

    return hints
