from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fast_agent.constants import DEFAULT_MAX_ITERATIONS
from fast_agent.core.model_resolution import get_context_model_references, resolve_model_reference
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.request_params import RequestParams

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def deep_merge(dict1: dict[Any, Any], dict2: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively merge ``dict2`` into ``dict1`` in place."""
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            deep_merge(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1


def get_provider_config(
    *,
    context_config: object | None,
    provider_value: str | None,
    config_section: str | None = None,
    fallback_sections: Sequence[str] = (),
) -> Any | None:
    """Return the first configured provider section that exists and is non-null."""
    if context_config is None:
        return None

    checked_sections: set[str] = set()
    section_names = (config_section or provider_value, *fallback_sections)
    for section_name in section_names:
        if not section_name or section_name in checked_sections:
            continue
        checked_sections.add(section_name)

        if not hasattr(context_config, section_name):
            continue

        provider_config = getattr(context_config, section_name)
        if provider_config is not None:
            return provider_config

    return None


def normalize_model_name(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized or None


def resolve_config_default_model(
    *,
    context_config: object | None,
    provider_value: str | None,
    config_section: str | None = None,
    fallback_sections: Sequence[str] = (),
) -> str | None:
    """Resolve an optional provider-level default model from config."""
    provider_config = get_provider_config(
        context_config=context_config,
        provider_value=provider_value,
        config_section=config_section,
        fallback_sections=fallback_sections,
    )
    if provider_config is None:
        return None

    value = getattr(provider_config, "default_model", None)
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized or None


def resolve_model_references(*, context: object, value: str) -> str:
    aliases = get_context_model_references(context)
    return resolve_model_reference(value, aliases)


def initialize_base_default_params(
    *,
    instruction: str | None,
    kwargs: Mapping[str, Any],
) -> RequestParams:
    """Build provider-agnostic default request params."""
    model = kwargs.get("model")
    max_tokens = ModelDatabase.get_default_max_tokens(model) if model else 16384

    return RequestParams(
        model=model,
        maxTokens=max_tokens,
        systemPrompt=instruction,
        parallel_tool_calls=True,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        use_history=True,
    )
def merge_request_params(
    default_params: RequestParams,
    provided_params: RequestParams,
) -> RequestParams:
    """Merge default and provided request parameters."""
    merged = deep_merge(
        default_params.model_dump(),
        provided_params.model_dump(exclude_unset=True),
    )
    return RequestParams(**merged)
