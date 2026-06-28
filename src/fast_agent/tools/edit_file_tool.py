from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from mcp.types import Tool

from fast_agent.tools.filesystem_tool_args import (
    coerce_required_string_argument,
    coerce_tool_arguments,
)

EDIT_FILE_TOOL_NAME: Final = "edit_file"
EDIT_FILE_TOOL_DESCRIPTION: Final = (
    "Edit a text file by replacing an exact string match with new text. "
    "Returns a structured result with match details and a unified diff."
)


@dataclass(frozen=True, slots=True)
class EditFileInput:
    path: str
    old_string: str
    new_string: str
    replace_all: bool


def build_edit_file_tool() -> Tool:
    """Return the shared ``edit_file`` tool definition."""
    return Tool(
        name=EDIT_FILE_TOOL_NAME,
        description=EDIT_FILE_TOOL_DESCRIPTION,
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to search for. Must be non-empty.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text. Use an empty string for deletion.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "When true, replace all non-overlapping occurrences in a single pass. "
                        "When false, replace only one occurrence and fail on ambiguity."
                    ),
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    )


def extract_edit_file_input(arguments: dict[str, Any] | None) -> EditFileInput | None:
    try:
        payload = coerce_tool_arguments(arguments)
        path = coerce_required_string_argument(payload.get("path"), "path", strip=True)
        old_string = coerce_required_string_argument(
            payload.get("old_string"),
            "old_string",
            allow_empty=True,
        )
        new_string = coerce_required_string_argument(
            payload.get("new_string"),
            "new_string",
            allow_empty=True,
        )
    except ValueError:
        return None

    replace_all = payload.get("replace_all", False)

    if not isinstance(replace_all, bool):
        return None

    return EditFileInput(
        path=path,
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
    )
