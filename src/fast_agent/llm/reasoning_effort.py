"""Shared reasoning effort types and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

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

TRUE_VALUES: Final[set[str]] = {"true", "on", "1", "yes", "enable", "enabled"}
FALSE_VALUES: Final[set[str]] = {"false", "off", "0", "no", "disable", "disabled"}

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


ReasoningEffortInput: TypeAlias = ReasoningEffortSetting | str | bool | int | None


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
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if cleaned in EFFORT_LEVELS:
            return ReasoningEffortSetting(
                kind="effort",
                value=cast("ReasoningEffortLevel", cleaned),
            )
        if cleaned in TRUE_VALUES:
            return ReasoningEffortSetting(kind="toggle", value=True)
        if cleaned in FALSE_VALUES:
            return ReasoningEffortSetting(kind="toggle", value=False)
        try:
            return ReasoningEffortSetting(kind="budget", value=int(cleaned))
        except ValueError:
            return None
    return None


def normalize_effort_for_spec(
    value: ReasoningEffortLevel, allowed: list[ReasoningEffortLevel] | None
) -> ReasoningEffortLevel | None:
    if allowed is None:
        return value
    if value in allowed:
        return value
    if value == "minimal" and "low" in allowed:
        return "low"
    if value == "xhigh" and "max" in allowed:
        return "max"
    if value == "max" and "xhigh" in allowed:
        return "xhigh"
    return None


def _budget_presets_for_spec(spec: ReasoningEffortSpec) -> list[int]:
    budgets: list[int] = []
    if spec.budget_presets:
        budgets.extend(value for value in spec.budget_presets if value > 0)
    if not budgets:
        if spec.min_budget_tokens is not None:
            budgets.append(spec.min_budget_tokens)
        if spec.max_budget_tokens is not None:
            budgets.append(spec.max_budget_tokens)
    return sorted({value for value in budgets if value > 0})


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

    if effort in ("minimal", "low"):
        budget = budgets[0]
    elif effort == "medium":
        budget = budgets[len(budgets) // 2]
    else:
        budget = budgets[-1]
    return ReasoningEffortSetting(kind="budget", value=budget)


def is_auto_reasoning(setting: ReasoningEffortSetting | None) -> bool:
    """Return True when the setting represents automatic/provider-default reasoning."""
    return setting is not None and setting.kind == "effort" and setting.value == AUTO_REASONING


def validate_reasoning_setting(
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> ReasoningEffortSetting:
    """Validate a reasoning setting against a model spec."""
    if setting.kind == "toggle" and setting.value is False:
        return setting

    # "auto" is only valid when the spec allows provider-default reasoning.
    if is_auto_reasoning(setting):
        if spec.kind != "effort":
            raise ValueError(f"Expected reasoning kind '{spec.kind}', got '{setting.kind}'.")
        if not spec.allow_auto:
            allowed = ", ".join(spec.allowed_efforts or []) or "any"
            raise ValueError(f"Effort '{setting.value}' not allowed (allowed: {allowed}).")
        return setting

    if spec.kind == "budget" and setting.kind == "effort":
        mapped = map_effort_to_budget(setting, spec)
        if mapped is None:
            raise ValueError("Effort values are not supported for budget-based reasoning.")
        setting = mapped

    if setting.kind != spec.kind:
        raise ValueError(f"Expected reasoning kind '{spec.kind}', got '{setting.kind}'.")

    if setting.kind == "effort":
        value = setting.value
        if not isinstance(value, str):
            raise ValueError("Effort value must be a string effort level.")
        normalized = normalize_effort_for_spec(value, spec.allowed_efforts)
        if normalized is None:
            allowed = ", ".join(spec.allowed_efforts or []) or "any"
            raise ValueError(f"Effort '{value}' not allowed (allowed: {allowed}).")
        if normalized != value:
            return ReasoningEffortSetting(kind="effort", value=normalized)
        return setting

    if setting.kind == "budget":
        value = setting.value
        if not isinstance(value, int):
            raise ValueError("Budget value must be an integer token count.")
        min_budget = spec.min_budget_tokens
        max_budget = spec.max_budget_tokens
        if min_budget is not None and value < min_budget:
            raise ValueError(f"Budget must be >= {min_budget} tokens.")
        if max_budget is not None and value > max_budget:
            raise ValueError(f"Budget must be <= {max_budget} tokens.")
        return setting

    if setting.kind == "toggle":
        if not isinstance(setting.value, bool):
            raise ValueError("Toggle value must be a boolean.")
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
        return "enabled" if setting.value else "disabled"
    return "unknown"


def available_reasoning_values(spec: ReasoningEffortSpec | None) -> list[str]:
    if spec is None:
        return []
    if spec.kind == "effort":
        values = list(spec.allowed_efforts or EFFORT_LEVELS)
        if spec.allow_auto:
            if "auto" in values:
                values = ["auto"] + [value for value in values if value != "auto"]
            else:
                values.insert(0, "auto")
        else:
            values = [value for value in values if value != "auto"]
    elif spec.kind == "budget":
        values = []
        presets = spec.budget_presets
        if presets:
            values.extend(str(value) for value in presets)
        else:
            if spec.min_budget_tokens is not None:
                values.append(str(spec.min_budget_tokens))
            if spec.max_budget_tokens is not None:
                values.append(str(spec.max_budget_tokens))
        aliases = [alias for alias in ("low", "medium", "high", "max") if alias not in values]
        values = aliases + values
    else:
        values = ["on", "off"]

    if spec.kind != "effort" or "none" in values or spec.allow_toggle_disable:
        if "off" not in values:
            values.append("off")
    return values
