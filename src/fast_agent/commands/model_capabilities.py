"""Runtime capability helpers for model command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.interfaces import FastAgentLLMProtocol


T = TypeVar("T")


def _read_capability(
    llm: FastAgentLLMProtocol | object | None,
    getter: "Callable[[FastAgentLLMProtocol], T]",
    *,
    default: T,
) -> T:
    if llm is None:
        return default

    candidate = cast("FastAgentLLMProtocol", llm)
    try:
        return getter(candidate)
    except AttributeError:
        return default


def _set_capability(
    llm: FastAgentLLMProtocol | object,
    setter: "Callable[[FastAgentLLMProtocol], Callable[[T], None]]",
    value: T,
    *,
    unsupported_message: str,
) -> None:
    candidate = cast("FastAgentLLMProtocol", llm)
    try:
        apply = setter(candidate)
    except AttributeError as exc:
        raise ValueError(unsupported_message) from exc
    apply(value)


def resolve_web_search_enabled(llm: FastAgentLLMProtocol | object | None) -> bool:
    return bool(_read_capability(llm, lambda candidate: candidate.web_search_enabled, default=False))


def resolve_web_fetch_enabled(llm: FastAgentLLMProtocol | object | None) -> bool:
    return bool(_read_capability(llm, lambda candidate: candidate.web_fetch_enabled, default=False))


def resolve_web_search_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return bool(
        _read_capability(llm, lambda candidate: candidate.web_search_supported, default=False)
    )


def resolve_web_fetch_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return bool(
        _read_capability(llm, lambda candidate: candidate.web_fetch_supported, default=False)
    )


def set_web_search_enabled(llm: FastAgentLLMProtocol | object, value: bool | None) -> None:
    _set_capability(
        llm,
        lambda candidate: candidate.set_web_search_enabled,
        value,
        unsupported_message="Current model does not support web search configuration.",
    )


def set_web_fetch_enabled(llm: FastAgentLLMProtocol | object, value: bool | None) -> None:
    _set_capability(
        llm,
        lambda candidate: candidate.set_web_fetch_enabled,
        value,
        unsupported_message="Current model does not support web fetch configuration.",
    )


def resolve_task_budget_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return bool(
        _read_capability(llm, lambda candidate: candidate.task_budget_supported, default=False)
    )


def resolve_task_budget_tokens(llm: FastAgentLLMProtocol | object | None) -> int | None:
    return _read_capability(llm, lambda candidate: candidate.task_budget_tokens, default=None)


def set_task_budget_tokens(llm: FastAgentLLMProtocol | object, value: int | None) -> None:
    _set_capability(
        llm,
        lambda candidate: candidate.set_task_budget_tokens,
        value,
        unsupported_message="Current model does not support task budget configuration.",
    )


def resolve_service_tier_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    return bool(
        _read_capability(llm, lambda candidate: candidate.service_tier_supported, default=False)
    )


def available_service_tier_values(llm: FastAgentLLMProtocol | object | None) -> tuple[str, ...]:
    values = tuple(
        value
        for value in _read_capability(
            llm,
            lambda candidate: candidate.available_service_tiers,
            default=(),
        )
        if value in {"fast", "flex"}
    )
    if values:
        return values
    if resolve_service_tier_supported(llm):
        return ("fast", "flex")
    return ()


def service_tier_command_values(llm: FastAgentLLMProtocol | object | None) -> tuple[str, ...]:
    values = ["on", "off"]
    if "flex" in available_service_tier_values(llm):
        values.append("flex")
    values.append("status")
    return tuple(values)


def resolve_service_tier(llm: FastAgentLLMProtocol | object | None) -> str | None:
    value = _read_capability(llm, lambda candidate: candidate.service_tier, default=None)
    return value if value in {"fast", "flex"} else None


def set_service_tier(
    llm: FastAgentLLMProtocol | object,
    value: Literal["fast", "flex"] | None,
) -> None:
    _set_capability(
        llm,
        lambda candidate: candidate.set_service_tier,
        value,
        unsupported_message="Current model does not support service tier configuration.",
    )


def describe_service_tier_state(llm: FastAgentLLMProtocol | object | None) -> str:
    current_tier = resolve_service_tier(llm)
    if current_tier == "fast":
        return "fast"
    if current_tier == "flex":
        return "flex"
    return "default"


def model_supports_web_search(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model/provider supports web_search runtime configuration."""
    return resolve_web_search_supported(llm)


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
    return _read_capability(llm, lambda candidate: candidate.text_verbosity_spec, default=None) is not None
