from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from mcp_types import Tool

from fast_agent.tools.filesystem_tool_args import (
    coerce_required_string_argument,
    coerce_tool_arguments,
)

EDIT_FILE_TOOL_NAME: Final = "edit_file"
EDIT_FILE_TOOL_DESCRIPTION: Final = (
    "Create a missing text file or edit an existing one by replacing an exact string "
    "match with new text. Omit old_string or use an empty string only when creating. "
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
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path.",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "Exact text to replace in an existing file. Omit or use an empty "
                        "string to create a missing file; creation fails if the path exists."
                    ),
                    "default": "",
                },
                "new_string": {
                    "type": "string",
                    "description": (
                        "Complete contents for a new file, or replacement text for an edit. "
                        "Use an empty string to create an empty file or delete matched text."
                    ),
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
            "required": ["path", "new_string"],
            "additionalProperties": False,
        },
    )


def extract_edit_file_input(arguments: dict[str, Any] | None) -> EditFileInput | None:
    try:
        payload = coerce_tool_arguments(arguments)
        path = coerce_required_string_argument(payload.get("path"), "path", strip=True)
        old_string = coerce_required_string_argument(
            payload.get("old_string", ""),
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
