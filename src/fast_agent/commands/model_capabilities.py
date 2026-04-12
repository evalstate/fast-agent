"""Runtime capability helpers for model command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from fast_agent.interfaces import FastAgentLLMProtocol


def resolve_web_search_enabled(llm: FastAgentLLMProtocol | object | None) -> bool:
    if llm is None:
        return False
    llm = cast("FastAgentLLMProtocol", llm)
    return bool(llm.web_search_enabled)


def resolve_web_fetch_enabled(llm: FastAgentLLMProtocol | object | None) -> bool:
    if llm is None:
        return False
    llm = cast("FastAgentLLMProtocol", llm)
    return bool(llm.web_fetch_enabled)


def resolve_web_search_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    if llm is None:
        return False
    llm = cast("FastAgentLLMProtocol", llm)
    return bool(llm.web_search_supported)


def resolve_web_fetch_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    if llm is None:
        return False
    llm = cast("FastAgentLLMProtocol", llm)
    return bool(llm.web_fetch_supported)


def set_web_search_enabled(llm: FastAgentLLMProtocol | object, value: bool | None) -> None:
    llm = cast("FastAgentLLMProtocol", llm)
    llm.set_web_search_enabled(value)


def set_web_fetch_enabled(llm: FastAgentLLMProtocol | object, value: bool | None) -> None:
    llm = cast("FastAgentLLMProtocol", llm)
    llm.set_web_fetch_enabled(value)


def resolve_service_tier_supported(llm: FastAgentLLMProtocol | object | None) -> bool:
    if llm is None:
        return False
    llm = cast("FastAgentLLMProtocol", llm)
    return bool(llm.service_tier_supported)


def available_service_tier_values(llm: FastAgentLLMProtocol | object | None) -> tuple[str, ...]:
    if llm is None:
        return ()
    llm = cast("FastAgentLLMProtocol", llm)
    values = tuple(value for value in llm.available_service_tiers if value in {"fast", "flex"})
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
    if llm is None:
        return None
    llm = cast("FastAgentLLMProtocol", llm)
    value = llm.service_tier
    return value if value in {"fast", "flex"} else None


def set_service_tier(
    llm: FastAgentLLMProtocol | object,
    value: Literal["fast", "flex"] | None,
) -> None:
    llm = cast("FastAgentLLMProtocol", llm)
    llm.set_service_tier(value)


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


def model_supports_text_verbosity(llm: FastAgentLLMProtocol | object | None) -> bool:
    """Return True when model exposes text verbosity controls."""
    if llm is None:
        return False
    llm = cast("FastAgentLLMProtocol", llm)
    return llm.text_verbosity_spec is not None
