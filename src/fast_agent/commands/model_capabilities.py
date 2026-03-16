"""Runtime capability helpers for model command handlers."""

from __future__ import annotations


def resolve_web_search_enabled(llm: object) -> bool:
    web_tools_enabled = getattr(llm, "web_tools_enabled", None)
    if isinstance(web_tools_enabled, tuple) and len(web_tools_enabled) >= 1:
        return bool(web_tools_enabled[0])

    enabled = getattr(llm, "web_search_enabled", None)
    if isinstance(enabled, bool):
        return enabled
    return False


def resolve_web_fetch_enabled(llm: object) -> bool:
    web_tools_enabled = getattr(llm, "web_tools_enabled", None)
    if isinstance(web_tools_enabled, tuple) and len(web_tools_enabled) >= 2:
        return bool(web_tools_enabled[1])

    enabled = getattr(llm, "web_fetch_enabled", None)
    if isinstance(enabled, bool):
        return enabled
    return False


def resolve_web_search_supported(llm: object) -> bool:
    supported = getattr(llm, "web_search_supported", None)
    return bool(supported) if isinstance(supported, bool) else False


def resolve_web_fetch_supported(llm: object) -> bool:
    supported = getattr(llm, "web_fetch_supported", None)
    return bool(supported) if isinstance(supported, bool) else False


def set_web_search_enabled(llm: object, value: bool | None) -> None:
    setter = getattr(llm, "set_web_search_enabled", None)
    if callable(setter):
        setter(value)
        return
    raise ValueError("Current model does not support web search configuration.")


def set_web_fetch_enabled(llm: object, value: bool | None) -> None:
    setter = getattr(llm, "set_web_fetch_enabled", None)
    if callable(setter):
        setter(value)
        return
    raise ValueError("Current model does not support web fetch configuration.")


def resolve_service_tier_supported(llm: object) -> bool:
    supported = getattr(llm, "service_tier_supported", None)
    return bool(supported) if isinstance(supported, bool) else False


def available_service_tier_values(llm: object) -> tuple[str, ...]:
    raw_values = getattr(llm, "available_service_tiers", None)
    if isinstance(raw_values, tuple | list):
        values = tuple(value for value in raw_values if value in {"fast", "flex"})
        if values:
            return values
    if resolve_service_tier_supported(llm):
        return ("fast", "flex")
    return ()


def service_tier_command_values(llm: object) -> tuple[str, ...]:
    values = ["on", "off"]
    if "flex" in available_service_tier_values(llm):
        values.append("flex")
    values.append("status")
    return tuple(values)


def resolve_service_tier(llm: object) -> str | None:
    value = getattr(llm, "service_tier", None)
    return value if value in {"fast", "flex"} else None


def set_service_tier(llm: object, value: str | None) -> None:
    setter = getattr(llm, "set_service_tier", None)
    if callable(setter):
        setter(value)
        return
    raise ValueError("Current model does not support service tier configuration.")


def describe_service_tier_state(llm: object) -> str:
    current_tier = resolve_service_tier(llm)
    if current_tier == "fast":
        return "fast"
    if current_tier == "flex":
        return "flex"
    return "default"


def model_supports_web_search(llm: object) -> bool:
    """Return True when model/provider supports web_search runtime configuration."""
    return resolve_web_search_supported(llm)


def model_supports_web_fetch(llm: object) -> bool:
    """Return True when model/provider supports web_fetch runtime configuration."""
    return resolve_web_fetch_supported(llm)


def model_supports_service_tier(llm: object) -> bool:
    """Return True when model/provider supports service tier runtime configuration."""
    return resolve_service_tier_supported(llm)


def model_supports_text_verbosity(llm: object) -> bool:
    """Return True when model exposes text verbosity controls."""
    return getattr(llm, "text_verbosity_spec", None) is not None
