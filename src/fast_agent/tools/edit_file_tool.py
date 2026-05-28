from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from mcp.types import Tool

from fast_agent.tools.apply_patch_tool import normalize_tool_name

if TYPE_CHECKING:
    from collections.abc import Mapping

EDIT_FILE_TOOL_NAME: Final = "edit_file"
EDIT_FILE_TOOL_DESCRIPTION: Final = (
    "Edit a text file by replacing an exact string match with new text. "
    "Returns a structured result with match details and a unified diff."
)


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


def is_edit_file_tool_name(tool_name: str | None) -> bool:
    normalized = normalize_tool_name(tool_name)
    return normalized == EDIT_FILE_TOOL_NAME or normalized.endswith("__edit_file")


def extract_edit_file_input(arguments: Mapping[str, Any] | None) -> tuple[str, str, str, bool] | None:
    if arguments is None:
        return None

    path = arguments.get("path")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    replace_all = arguments.get("replace_all", False)

    if not isinstance(path, str):
        return None
    if not isinstance(old_string, str):
        return None
    if not isinstance(new_string, str):
        return None
    if not isinstance(replace_all, bool):
        return None

    stripped_path = path.strip()
    if not stripped_path:
        return None

    return stripped_path, old_string, new_string, replace_all
