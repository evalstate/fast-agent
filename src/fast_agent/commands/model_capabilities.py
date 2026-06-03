"""Runtime capability helpers for model command handlers."""

from __future__ import annotations

from collections.abc import Sequence
from inspect import getattr_static
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from fast_agent.llm.capabilities import read_bool_capability, read_capability
from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
    from fast_agent.llm.resolved_model import ResolvedModelSpec
    from fast_agent.llm.text_verbosity import TextVerbosityLevel, TextVerbositySpec


T = TypeVar("T")
ServiceTierValue = Literal["fast", "flex"]
SERVICE_TIER_VALUES: tuple[ServiceTierValue, ...] = ("fast", "flex")


def _set_capability(
    llm: FastAgentLLMProtocol | object,
    attribute_name: str,
    setter: "Callable[[FastAgentLLMProtocol], Callable[[T], None]]",
    value: T,
    *,
    unsupported_message: str,
) -> None:
    candidate = cast("FastAgentLLMProtocol", llm)
    try:
        getattr_static(candidate, attribute_name)
    except AttributeError as exc:
        raise ValueError(unsupported_message) from exc
    try:
        apply = setter(candidate)
    except AttributeError as exc:
        raise ValueError(unsupported_message) from exc
    apply(value)


def resolve_web_search_enabled(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "web_search_enabled", lambda candidate: candidate.web_search_enabled
    )


def resolve_x_search_enabled(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "x_search_enabled", lambda candidate: candidate.x_search_enabled
    )


def resolve_web_fetch_enabled(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "web_fetch_enabled", lambda candidate: candidate.web_fetch_enabled
    )


def resolve_web_search_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "web_search_supported", lambda candidate: candidate.web_search_supported
    )


def resolve_x_search_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "x_search_supported", lambda candidate: candidate.x_search_supported
    )


def resolve_web_fetch_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "web_fetch_supported", lambda candidate: candidate.web_fetch_supported
    )


def resolve_resolved_model(
    llm: FastAgentLLMProtocol | object | None,
) -> "ResolvedModelSpec | None":
    return read_capability(
        llm,
        "resolved_model",
        lambda candidate: candidate.resolved_model,
        default=None,
    )


def resolve_model_name(llm: FastAgentLLMProtocol | object | None) -> str | None:
    return read_capability(llm, "model_name", lambda candidate: candidate.model_name, default=None)


def resolve_model_info(llm: FastAgentLLMProtocol | object | None) -> "ModelInfo | None":
    return read_capability(
        llm,
        "model_info",
        lambda candidate: candidate.model_info,
        default=None,
    )


def resolve_reasoning_effort(
    llm: FastAgentLLMProtocol | object | None,
) -> "ReasoningEffortSetting | None":
    return read_capability(
        llm, "reasoning_effort", lambda candidate: candidate.reasoning_effort, default=None
    )


def resolve_reasoning_effort_spec(
    llm: FastAgentLLMProtocol | object | None,
) -> "ReasoningEffortSpec | None":
    return read_capability(
        llm,
        "reasoning_effort_spec",
        lambda candidate: candidate.reasoning_effort_spec,
        default=None,
    )


def set_reasoning_effort(
    llm: FastAgentLLMProtocol | object,
    value: "ReasoningEffortSetting | None",
) -> None:
    _set_capability(
        llm,
        "set_reasoning_effort",
        lambda candidate: candidate.set_reasoning_effort,
        value,
        unsupported_message="Current model does not support reasoning effort configuration.",
    )


def resolve_text_verbosity(
    llm: FastAgentLLMProtocol | object | None,
) -> "TextVerbosityLevel | None":
    return read_capability(
        llm, "text_verbosity", lambda candidate: candidate.text_verbosity, default=None
    )


def resolve_text_verbosity_spec(
    llm: FastAgentLLMProtocol | object | None,
) -> "TextVerbositySpec | None":
    return read_capability(
        llm, "text_verbosity_spec", lambda candidate: candidate.text_verbosity_spec, default=None
    )


def set_text_verbosity(
    llm: FastAgentLLMProtocol | object,
    value: "TextVerbosityLevel | None",
) -> None:
    _set_capability(
        llm,
        "set_text_verbosity",
        lambda candidate: candidate.set_text_verbosity,
        value,
        unsupported_message="Current model does not support text verbosity configuration.",
    )


def set_web_search_enabled(llm: FastAgentLLMProtocol | object, value: bool | None) -> None:
    _set_capability(
        llm,
        "set_web_search_enabled",
        lambda candidate: candidate.set_web_search_enabled,
        value,
        unsupported_message="Current model does not support web search configuration.",
    )


def set_x_search_enabled(llm: FastAgentLLMProtocol | object, value: bool | None) -> None:
    _set_capability(
        llm,
        "set_x_search_enabled",
        lambda candidate: candidate.set_x_search_enabled,
        value,
        unsupported_message="Current model does not support X Search configuration.",
    )


def set_web_fetch_enabled(llm: FastAgentLLMProtocol | object, value: bool | None) -> None:
    _set_capability(
        llm,
        "set_web_fetch_enabled",
        lambda candidate: candidate.set_web_fetch_enabled,
        value,
        unsupported_message="Current model does not support web fetch configuration.",
    )


def resolve_task_budget_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "task_budget_supported", lambda candidate: candidate.task_budget_supported
    )


def resolve_task_budget_tokens(llm: FastAgentLLMProtocol | object | None) -> int | None:
    value = read_capability(
        llm, "task_budget_tokens", lambda candidate: candidate.task_budget_tokens, default=None
    )
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def set_task_budget_tokens(llm: FastAgentLLMProtocol | object, value: int | None) -> None:
    _set_capability(
        llm,
        "set_task_budget_tokens",
        lambda candidate: candidate.set_task_budget_tokens,
        value,
        unsupported_message="Current model does not support task budget configuration.",
    )


def resolve_service_tier_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return read_bool_capability(
        llm, "service_tier_supported", lambda candidate: candidate.service_tier_supported
    )


def available_service_tier_values(
    llm: FastAgentLLMProtocol | object | None,
) -> tuple[ServiceTierValue, ...]:
    raw_values = read_capability(
        llm,
        "available_service_tiers",
        lambda candidate: candidate.available_service_tiers,
        default=(),
    )
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
        raw_values = ()
    values = tuple(
        unique_preserve_order(value for value in raw_values if value in SERVICE_TIER_VALUES)
    )
    if values:
        return values
    if resolve_service_tier_supported(llm):
        return SERVICE_TIER_VALUES
    return ()


def service_tier_command_values(llm: FastAgentLLMProtocol | object | None) -> tuple[str, ...]:
    values = ["on", "off"]
    if "flex" in available_service_tier_values(llm):
        values.append("flex")
    values.append("status")
    return tuple(values)


def resolve_service_tier(llm: FastAgentLLMProtocol | object | None) -> ServiceTierValue | None:
    value = read_capability(
        llm, "service_tier", lambda candidate: candidate.service_tier, default=None
    )
    return value if value in SERVICE_TIER_VALUES else None


def set_service_tier(
    llm: FastAgentLLMProtocol | object,
    value: ServiceTierValue | None,
) -> None:
    _set_capability(
        llm,
        "set_service_tier",
        lambda candidate: candidate.set_service_tier,
        value,
        unsupported_message="Current model does not support service tier configuration.",
    )


def describe_service_tier_state(llm: FastAgentLLMProtocol | object | None) -> str:
    current_tier = resolve_service_tier(llm)
    return current_tier or "default"


def model_supports_web_search(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model/provider supports web_search runtime configuration."""
    return resolve_web_search_supported(llm)


def model_supports_x_search(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model/provider supports x_search runtime configuration."""
    return resolve_x_search_supported(llm)


def model_supports_web_fetch(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model/provider supports web_fetch runtime configuration."""
    return resolve_web_fetch_supported(llm)


def model_supports_service_tier(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model/provider supports service tier runtime configuration."""
    return resolve_service_tier_supported(llm)


def model_supports_task_budget(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model/provider supports task budget runtime configuration."""
    return resolve_task_budget_supported(llm)


def model_supports_text_verbosity(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model exposes text verbosity controls."""
    return resolve_text_verbosity_spec(llm) is not None
