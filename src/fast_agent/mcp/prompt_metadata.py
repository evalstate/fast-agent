from typing import Any

from mcp.types import GetPromptResult

_NAMESPACED_NAME_KEY = "fast_agent.namespaced_name"
_ARGUMENTS_KEY = "fast_agent.arguments"


def _prompt_meta(result: GetPromptResult) -> dict[str, Any]:
    meta = result.meta
    return dict(meta) if isinstance(meta, dict) else {}


def with_prompt_metadata(
    result: GetPromptResult,
    *,
    namespaced_name: str,
    arguments: dict[str, str] | None = None,
) -> GetPromptResult:
    meta = _prompt_meta(result)
    meta[_NAMESPACED_NAME_KEY] = namespaced_name
    if arguments:
        meta[_ARGUMENTS_KEY] = arguments
    return result.model_copy(update={"meta": meta})


def prompt_display_name(result: GetPromptResult, fallback: str) -> str:
    value = _prompt_meta(result).get(_NAMESPACED_NAME_KEY)
    return value if isinstance(value, str) else fallback


def prompt_arguments(result: GetPromptResult) -> dict[str, str] | None:
    value = _prompt_meta(result).get(_ARGUMENTS_KEY)
    if not isinstance(value, dict):
        return None
    arguments: dict[str, str] = {}
    for key, argument_value in value.items():
        if not isinstance(key, str) or not isinstance(argument_value, str):
            return None
        arguments[key] = argument_value
    return arguments or None
