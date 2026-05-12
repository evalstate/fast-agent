from typing import Any

from mcp.types import GetPromptResult

_NAMESPACED_NAME_KEY = "fast_agent.namespaced_name"
_ARGUMENTS_KEY = "fast_agent.arguments"


def with_prompt_metadata(
    result: GetPromptResult,
    *,
    namespaced_name: str,
    arguments: dict[str, str] | None = None,
) -> GetPromptResult:
    meta: dict[str, Any] = dict(result.meta or {})
    meta[_NAMESPACED_NAME_KEY] = namespaced_name
    if arguments:
        meta[_ARGUMENTS_KEY] = arguments
    return result.model_copy(update={"meta": meta})


def prompt_display_name(result: GetPromptResult, fallback: str) -> str:
    value = (result.meta or {}).get(_NAMESPACED_NAME_KEY)
    return value if isinstance(value, str) else fallback
