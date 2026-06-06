from typing import Literal

from fast_agent.utils.text import strip_casefold

StructuredOutputMode = Literal["json", "tool_use"]

_STRUCTURED_OUTPUT_MODE_ALIASES: dict[str, StructuredOutputMode] = {
    "json": "json",
    "tool_use": "tool_use",
    "tool-use": "tool_use",
}


def parse_structured_output_mode(value: object) -> StructuredOutputMode | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None

    normalized = strip_casefold(value)
    return _STRUCTURED_OUTPUT_MODE_ALIASES.get(normalized)
