from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from mcp.types import Tool

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
        meta={
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



def normalize_tool_name(tool_name: str | None) -> str:
    if not tool_name:
        return ""
    normalized = tool_name.lower()
    for sep in ("/", ".", ":"):
        if sep in normalized:
            normalized = normalized.rsplit(sep, 1)[-1]
    return normalized



def is_apply_patch_tool_name(tool_name: str | None) -> bool:
    normalized = normalize_tool_name(tool_name)
    return normalized == APPLY_PATCH_TOOL_NAME or normalized.endswith("__apply_patch")



def extract_apply_patch_input(arguments: Mapping[str, Any] | None) -> str | None:
    if arguments is None:
        return None
    raw_input = arguments.get(APPLY_PATCH_INPUT_FIELD)
    if not isinstance(raw_input, str):
        return None
    stripped = raw_input.strip()
    return stripped or None



def get_openai_responses_custom_tool_payload(tool: Tool) -> dict[str, Any] | None:
    meta_source = tool.model_dump(mode="json", by_alias=True, exclude_none=True).get("meta")
    if not isinstance(meta_source, Mapping):
        model_extra = getattr(tool, "model_extra", None)
        if isinstance(model_extra, Mapping):
            meta_source = model_extra.get("meta")
    if not isinstance(meta_source, Mapping):
        return None

    raw_spec = meta_source.get(OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY)
    if not isinstance(raw_spec, Mapping):
        return None

    tool_type = raw_spec.get("type")
    format_payload = raw_spec.get("format")
    if tool_type != "custom" or not isinstance(format_payload, Mapping):
        return None

    format_type = format_payload.get("type")
    syntax = format_payload.get("syntax")
    definition = format_payload.get("definition")
    if not isinstance(format_type, str):
        return None
    if not isinstance(syntax, str):
        return None
    if not isinstance(definition, str):
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
