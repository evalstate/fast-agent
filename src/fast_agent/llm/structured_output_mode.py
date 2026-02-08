from typing import Literal

StructuredOutputMode = Literal["json", "tool_use"]


def parse_structured_output_mode(value: str | None) -> StructuredOutputMode | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized == "json":
        return "json"
    if normalized in {"tool_use", "tool-use"}:
        return "tool_use"
    return None
