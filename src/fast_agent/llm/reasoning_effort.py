"""Shared reasoning effort types and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

from fast_agent.utils.action_normalization import (
    FALSE_ACTION_ALIASES,
    TRUE_ACTION_ALIASES,
    enabled_disabled_label,
)
from fast_agent.utils.action_normalization import (
    parse_boolean_alias as _parse_boolean_alias,
)
from fast_agent.utils.numeric import sorted_unique_positive_ints
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Callable

ReasoningEffortKind = Literal["effort", "toggle", "budget"]
ReasoningEffortLevel = Literal[
    "auto",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]
ReasoningEffortValue = ReasoningEffortLevel | bool | int

EFFORT_LEVELS: Final[tuple[ReasoningEffortLevel, ...]] = (
    "auto",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

TRUE_VALUES: Final[frozenset[str]] = TRUE_ACTION_ALIASES
FALSE_VALUES: Final[frozenset[str]] = FALSE_ACTION_ALIASES
_EFFORT_LEVEL_SET: Final[frozenset[str]] = frozenset(EFFORT_LEVELS)
_EFFORT_NORMALIZATION_ALIASES: Final[dict[ReasoningEffortLevel, ReasoningEffortLevel]] = {
    "minimal": "low",
    "xhigh": "max",
    "max": "xhigh",
}
_BUDGET_EFFORT_INDEX: Final[dict[ReasoningEffortLevel, int]] = {
    "minimal": 0,
    "low": 0,
    "medium": 1,
}

# Sentinel setting that means "use the provider's automatic/default reasoning".
AUTO_REASONING = "auto"


@dataclass(frozen=True, slots=True)
class ReasoningEffortSetting:
    """User-configurable reasoning effort selection."""

    kind: ReasoningEffortKind
    value: ReasoningEffortValue


@dataclass(frozen=True, slots=True)
class ReasoningEffortSpec:
    """Capability info describing how a model accepts reasoning effort."""

    kind: ReasoningEffortKind
    allowed_efforts: list[ReasoningEffortLevel] | None = None
    min_budget_tokens: int | None = None
    max_budget_tokens: int | None = None
    budget_presets: list[int] | None = None
    allow_toggle_disable: bool = False
    allow_auto: bool = False
    default: ReasoningEffortSetting | None = None


_STRING_SETTING_ALIASES: Final[dict[str, ReasoningEffortSetting]] = {
    **{value: ReasoningEffortSetting(kind="toggle", value=True) for value in TRUE_VALUES},
    **{value: ReasoningEffortSetting(kind="toggle", value=False) for value in FALSE_VALUES},
}


ReasoningEffortInput: TypeAlias = ReasoningEffortSetting | str | bool | int | None
parse_boolean_alias = _parse_boolean_alias


def _parse_string_reasoning_setting(value: str) -> ReasoningEffortSetting | None:
    cleaned = strip_casefold(value)
    if not cleaned:
        return None
    if cleaned == "adaptive":
        cleaned = AUTO_REASONING
    if cleaned in _EFFORT_LEVEL_SET:
        return ReasoningEffortSetting(
            kind="effort",
            value=cast("ReasoningEffortLevel", cleaned),
        )
    if alias := _STRING_SETTING_ALIASES.get(cleaned):
        return alias
    try:
        return ReasoningEffortSetting(kind="budget", value=int(cleaned))
    except ValueError:
        return None


def parse_reasoning_setting(value: ReasoningEffortInput) -> ReasoningEffortSetting | None:
    """Parse a reasoning setting from raw input."""
    if value is None:
        return None
    if isinstance(value, ReasoningEffortSetting):
        return value
    if isinstance(value, bool):
        return ReasoningEffortSetting(kind="toggle", value=value)
    if isinstance(value, int):
        return ReasoningEffortSetting(kind="budget", value=value)
    if isinstance(value, str):
        return _parse_string_reasoning_setting(value)
    return None


def normalize_effort_for_spec(
    value: ReasoningEffortLevel, allowed: list[ReasoningEffortLevel] | None
) -> ReasoningEffortLevel | None:
    if allowed is None:
        return value
    if value in allowed:
        return value
    alias = _EFFORT_NORMALIZATION_ALIASES.get(value)
    if alias in allowed:
        return alias
    return None


def _budget_presets_for_spec(spec: ReasoningEffortSpec) -> list[int]:
    budgets: list[int] = []
    if spec.budget_presets:
        budgets.extend(spec.budget_presets)
    if not budgets:
        if spec.min_budget_tokens is not None:
            budgets.append(spec.min_budget_tokens)
        if spec.max_budget_tokens is not None:
            budgets.append(spec.max_budget_tokens)
    return sorted_unique_positive_ints(budgets)


def map_effort_to_budget(
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> ReasoningEffortSetting | None:
    """Map effort levels to budget presets when a model only supports budgets."""
    if setting.kind != "effort" or spec.kind != "budget":
        return None
    if not isinstance(setting.value, str):
        return None
    effort = setting.value
    if effort in ("auto", "none"):
        return None

    budgets = _budget_presets_for_spec(spec)
    if not budgets:
        return None

    index = _BUDGET_EFFORT_INDEX.get(effort)
    budget = budgets[-1] if index is None else budgets[index * (len(budgets) // 2)]
    return ReasoningEffortSetting(kind="budget", value=budget)


def is_auto_reasoning(setting: ReasoningEffortSetting | None) -> bool:
    """Return True when the setting represents automatic/provider-default reasoning."""
    return setting is not None and setting.kind == "effort" and setting.value == AUTO_REASONING


def _allowed_efforts_text(spec: ReasoningEffortSpec) -> str:
    return ", ".join(spec.allowed_efforts or []) or "any"


def _validate_auto_reasoning(
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> ReasoningEffortSetting:
    if spec.kind != "effort":
        raise ValueError(f"Expected reasoning kind '{spec.kind}', got '{setting.kind}'.")
    if not spec.allow_auto:
        raise ValueError(
            f"Effort '{setting.value}' not allowed (allowed: {_allowed_efforts_text(spec)})."
        )
    return setting


def _validate_effort_setting(
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> ReasoningEffortSetting:
    value = setting.value
    if not isinstance(value, str):
        raise ValueError("Effort value must be a string effort level.")
    normalized = normalize_effort_for_spec(value, spec.allowed_efforts)
    if normalized is None:
        raise ValueError(f"Effort '{value}' not allowed (allowed: {_allowed_efforts_text(spec)}).")
    if normalized == value:
        return setting
    return ReasoningEffortSetting(kind="effort", value=normalized)


def _validate_budget_setting(setting: ReasoningEffortSetting, spec: ReasoningEffortSpec) -> None:
    value = setting.value
    if not isinstance(value, int):
        raise ValueError("Budget value must be an integer token count.")
    min_budget = spec.min_budget_tokens
    max_budget = spec.max_budget_tokens
    if min_budget is not None and value < min_budget:
        raise ValueError(f"Budget must be >= {min_budget} tokens.")
    if max_budget is not None and value > max_budget:
        raise ValueError(f"Budget must be <= {max_budget} tokens.")


def _validate_toggle_setting(setting: ReasoningEffortSetting) -> None:
    if not isinstance(setting.value, bool):
        raise ValueError("Toggle value must be a boolean.")


def _spec_allows_toggle_disable(spec: ReasoningEffortSpec) -> bool:
    return (
        spec.kind != "effort"
        or "none" in (spec.allowed_efforts or [])
        or spec.allow_toggle_disable
    )


def validate_reasoning_setting(
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> ReasoningEffortSetting:
    """Validate a reasoning setting against a model spec."""
    if setting.kind == "toggle" and setting.value is False:
        if not _spec_allows_toggle_disable(spec):
            raise ValueError("Reasoning disable is not supported for this model.")
        return setting

    # "auto" is only valid when the spec allows provider-default reasoning.
    if is_auto_reasoning(setting):
        return _validate_auto_reasoning(setting, spec)

    if spec.kind == "budget" and setting.kind == "effort":
        mapped = map_effort_to_budget(setting, spec)
        if mapped is None:
            raise ValueError("Effort values are not supported for budget-based reasoning.")
        setting = mapped

    if setting.kind != spec.kind:
        raise ValueError(f"Expected reasoning kind '{spec.kind}', got '{setting.kind}'.")

    if setting.kind == "effort":
        return _validate_effort_setting(setting, spec)

    if setting.kind == "budget":
        _validate_budget_setting(setting, spec)
        return setting

    if setting.kind == "toggle":
        _validate_toggle_setting(setting)
        return setting

    raise ValueError("Unsupported reasoning setting.")


def format_reasoning_setting(setting: ReasoningEffortSetting | None) -> str:
    if setting is None:
        return "unset"
    if is_auto_reasoning(setting):
        return "auto"
    if setting.kind == "effort":
        return f"effort={setting.value}"
    if setting.kind == "budget":
        return f"budget={setting.value}"
    if setting.kind == "toggle":
        return enabled_disabled_label(cast("bool", setting.value))
    return "unknown"


def _available_effort_values(spec: ReasoningEffortSpec) -> list[str]:
    values: list[str] = list(spec.allowed_efforts or EFFORT_LEVELS)
    if spec.allow_auto:
        return ["auto"] + [value for value in values if value != "auto"]
    return [value for value in values if value != "auto"]


def _available_budget_values(spec: ReasoningEffortSpec) -> list[str]:
    if spec.budget_presets:
        values = [str(value) for value in spec.budget_presets]
    else:
        values = [
            str(value)
            for value in (spec.min_budget_tokens, spec.max_budget_tokens)
            if value is not None
        ]
    aliases = [alias for alias in ("low", "medium", "high", "max") if alias not in values]
    return aliases + values


def _values_with_off(values: list[str], spec: ReasoningEffortSpec) -> list[str]:
    if _spec_allows_toggle_disable(spec) and "off" not in values:
        return [*values, "off"]
    return values


_AVAILABLE_VALUES_BY_KIND: Final[
    dict[ReasoningEffortKind, Callable[[ReasoningEffortSpec], list[str]]]
] = {
    "effort": _available_effort_values,
    "budget": _available_budget_values,
    "toggle": lambda _spec: ["on", "off"],
}


def available_reasoning_values(spec: ReasoningEffortSpec | None) -> list[str]:
    if spec is None:
        return []
    values = _AVAILABLE_VALUES_BY_KIND[spec.kind](spec)
    return _values_with_off(values, spec)
