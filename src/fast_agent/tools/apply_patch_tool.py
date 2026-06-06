from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from mcp.types import Tool

from fast_agent.utils.tool_names import matches_tool_name

APPLY_PATCH_TOOL_NAME: Final = "apply_patch"
APPLY_PATCH_INPUT_FIELD: Final = "input"
APPLY_PATCH_TOOL_DESCRIPTION: Final = (
    "Use the `apply_patch` tool to edit files. "
    "This is a FREEFORM tool, so do not wrap the patch in JSON."
)
OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY: Final = "fast-agent/openai.responses_custom_tool"
APPLY_PATCH_LARK_GRAMMAR: Final = """start: begin_patch hunk+ end_patch
begin_patch: \"*** Begin Patch\" LF
end_patch: \"*** End Patch\" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: \"*** Add File: \" filename LF add_line+
delete_hunk: \"*** Delete File: \" filename LF
update_hunk: \"*** Update File: \" filename LF change_move? change?

filename: /(.+)/
add_line: \"+\" /(.*)/ LF -> line

change_move: \"*** Move to: \" filename LF
change: (change_context | change_line)+ eof_line?
change_context: (\"@@\" | \"@@ \" /(.+)/) LF
change_line: (\"+\" | \"-\" | \" \" ) /(.*)/ LF
eof_line: \"*** End of File\" LF

%import common.LF
"""


def build_apply_patch_tool() -> Tool:
    """Return the shared ``apply_patch`` tool definition."""
    return Tool(
        name=APPLY_PATCH_TOOL_NAME,
        description=APPLY_PATCH_TOOL_DESCRIPTION,
        inputSchema={
            "type": "object",
            "properties": {
                APPLY_PATCH_INPUT_FIELD: {
                    "type": "string",
                    "description": (
                        "Patch text in apply_patch format beginning with "
                        "'*** Begin Patch' and ending with '*** End Patch'."
                    ),
                }
            },
            "required": [APPLY_PATCH_INPUT_FIELD],
            "additionalProperties": False,
        },
        _meta={
            OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY: {
                "type": "custom",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": APPLY_PATCH_LARK_GRAMMAR,
                },
            }
        },
    )


def is_apply_patch_tool_name(tool_name: str | None) -> bool:
    return matches_tool_name(tool_name, APPLY_PATCH_TOOL_NAME)


def extract_apply_patch_input(arguments: Mapping[str, Any] | None) -> str | None:
    if not isinstance(arguments, Mapping):
        return None
    raw_input = arguments.get(APPLY_PATCH_INPUT_FIELD)
    if not isinstance(raw_input, str):
        return None
    stripped = raw_input.strip()
    return stripped or None


def _string_mapping_value(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if isinstance(value, str):
        return value
    return None


def get_openai_responses_custom_tool_payload(tool: Tool) -> dict[str, Any] | None:
    meta_source = tool.meta
    if not isinstance(meta_source, Mapping):
        return None

    raw_spec = meta_source.get(OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY)
    if not isinstance(raw_spec, Mapping):
        return None

    tool_type = raw_spec.get("type")
    format_payload = raw_spec.get("format")
    if tool_type != "custom" or not isinstance(format_payload, Mapping):
        return None

    format_type = _string_mapping_value(format_payload, "type")
    syntax = _string_mapping_value(format_payload, "syntax")
    definition = _string_mapping_value(format_payload, "definition")
    if format_type is None or syntax is None or definition is None:
        return None

    return {
        "type": "custom",
        "name": tool.name,
        "description": tool.description or "",
        "format": {
            "type": format_type,
            "syntax": syntax,
            "definition": definition,
        },
    }
