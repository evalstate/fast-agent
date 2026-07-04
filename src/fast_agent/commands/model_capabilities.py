"""Runtime capability helpers for model command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
    from fast_agent.llm.resolved_model import ResolvedModelSpec
    from fast_agent.llm.text_verbosity import TextVerbosityLevel, TextVerbositySpec


ServiceTierValue = Literal["fast", "flex"]
SERVICE_TIER_VALUES: tuple[ServiceTierValue, ...] = ("fast", "flex")


def _llm_attr(llm: object | None, name: str) -> object | None:
    if llm is None:
        return None
    try:
        return object.__getattribute__(llm, name)
    except AttributeError as exc:
        if any(name in vars(cls) for cls in type(llm).__mro__):
            raise exc
        return None


def resolve_web_search_enabled(llm: "FastAgentLLMProtocol | None") -> bool:
    return bool(_llm_attr(llm, "web_search_enabled"))


def resolve_x_search_enabled(llm: "FastAgentLLMProtocol | None") -> bool:
    return bool(_llm_attr(llm, "x_search_enabled"))


def resolve_web_fetch_enabled(llm: "FastAgentLLMProtocol | None") -> bool:
    return bool(_llm_attr(llm, "web_fetch_enabled"))


def resolve_web_search_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return bool(_llm_attr(llm, "web_search_supported"))


def resolve_x_search_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return bool(_llm_attr(llm, "x_search_supported"))


def resolve_web_fetch_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return bool(_llm_attr(llm, "web_fetch_supported"))


def resolve_resolved_model(
    llm: "FastAgentLLMProtocol | None",
) -> "ResolvedModelSpec | None":
    return cast("ResolvedModelSpec | None", _llm_attr(llm, "resolved_model"))


def resolve_model_name(llm: "FastAgentLLMProtocol | None") -> str | None:
    value = _llm_attr(llm, "model_name")
    return value if isinstance(value, str) else None


def resolve_model_info(llm: "FastAgentLLMProtocol | None") -> "ModelInfo | None":
    return cast("ModelInfo | None", _llm_attr(llm, "model_info"))


def resolve_reasoning_effort(
    llm: "FastAgentLLMProtocol | None",
) -> "ReasoningEffortSetting | None":
    configured = cast("ReasoningEffortSetting | None", _llm_attr(llm, "reasoning_effort"))
    if configured is not None:
        return configured
    spec = resolve_reasoning_effort_spec(llm)
    return spec.default if spec is not None else None


def resolve_reasoning_effort_spec(
    llm: "FastAgentLLMProtocol | None",
) -> "ReasoningEffortSpec | None":
    return cast("ReasoningEffortSpec | None", _llm_attr(llm, "reasoning_effort_spec"))


def set_reasoning_effort(
    llm: "FastAgentLLMProtocol",
    value: "ReasoningEffortSetting | None",
) -> None:
    llm.set_reasoning_effort(value)


def resolve_text_verbosity(
    llm: "FastAgentLLMProtocol | None",
) -> "TextVerbosityLevel | None":
    return cast("TextVerbosityLevel | None", _llm_attr(llm, "text_verbosity"))


def resolve_text_verbosity_spec(
    llm: "FastAgentLLMProtocol | None",
) -> "TextVerbositySpec | None":
    return cast("TextVerbositySpec | None", _llm_attr(llm, "text_verbosity_spec"))


def set_text_verbosity(
    llm: "FastAgentLLMProtocol",
    value: "TextVerbosityLevel | None",
) -> None:
    llm.set_text_verbosity(value)


def set_web_search_enabled(llm: "FastAgentLLMProtocol", value: bool | None) -> None:
    llm.set_web_search_enabled(value)


def set_x_search_enabled(llm: "FastAgentLLMProtocol", value: bool | None) -> None:
    llm.set_x_search_enabled(value)


def set_web_fetch_enabled(llm: "FastAgentLLMProtocol", value: bool | None) -> None:
    llm.set_web_fetch_enabled(value)


def resolve_task_budget_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return _llm_attr(llm, "task_budget_supported") is True


def resolve_task_budget_tokens(llm: "FastAgentLLMProtocol | None") -> int | None:
    value = _llm_attr(llm, "task_budget_tokens")
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def set_task_budget_tokens(llm: "FastAgentLLMProtocol", value: int | None) -> None:
    llm.set_task_budget_tokens(value)


def resolve_service_tier_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return _llm_attr(llm, "service_tier_supported") is True


def available_service_tier_values(
    llm: "FastAgentLLMProtocol | None",
) -> tuple[ServiceTierValue, ...]:
    raw_values = _llm_attr(llm, "available_service_tiers")
    if not isinstance(raw_values, tuple):
        raw_values = ()
    service_tiers = (
        cast("ServiceTierValue", value) for value in raw_values if value in SERVICE_TIER_VALUES
    )
    values = tuple(unique_preserve_order(service_tiers))
    if values:
        return values
    if resolve_service_tier_supported(llm):
        return SERVICE_TIER_VALUES
    return ()


def service_tier_command_values(llm: "FastAgentLLMProtocol | None") -> tuple[str, ...]:
    values = ["on", "off"]
    if "flex" in available_service_tier_values(llm):
        values.append("flex")
    values.append("status")
    return tuple(values)


def resolve_service_tier(llm: "FastAgentLLMProtocol | None") -> ServiceTierValue | None:
    value = _llm_attr(llm, "service_tier")
    return cast("ServiceTierValue", value) if value in SERVICE_TIER_VALUES else None


def set_service_tier(
    llm: "FastAgentLLMProtocol",
    value: ServiceTierValue | None,
) -> None:
    llm.set_service_tier(value)


def describe_service_tier_state(llm: "FastAgentLLMProtocol | None") -> str:
    current_tier = resolve_service_tier(llm)
    return current_tier or "default"


def model_supports_web_search(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports web_search runtime configuration."""
    return resolve_web_search_supported(llm)


def model_supports_x_search(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports x_search runtime configuration."""
    return resolve_x_search_supported(llm)


def model_supports_web_fetch(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports web_fetch runtime configuration."""
    return resolve_web_fetch_supported(llm)


def model_supports_service_tier(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports service tier runtime configuration."""
    return resolve_service_tier_supported(llm)


def model_supports_task_budget(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports task budget runtime configuration."""
    return resolve_task_budget_supported(llm)


def model_supports_text_verbosity(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model exposes text verbosity controls."""
    return resolve_text_verbosity_spec(llm) is not None
