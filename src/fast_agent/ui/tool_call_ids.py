"""Shared formatting helpers for user-facing tool call identifiers."""

TOOL_CALL_ID_PREFIX_LENGTH = 5
TOOL_CALL_ID_SUFFIX_LENGTH = 6
TOOL_CALL_ID_ELLIPSIS = "…"
TOOL_CALL_ID_MAX_LENGTH = (
    TOOL_CALL_ID_PREFIX_LENGTH
    + len(TOOL_CALL_ID_ELLIPSIS)
    + TOOL_CALL_ID_SUFFIX_LENGTH
)


def format_tool_call_id(tool_call_id: str | None) -> str | None:
    """Return a compact, correlatable tool call id.

    Keep both the beginning and end of long ids so live progress rows and final
    tool-call/result blocks can be visually matched.
    """
    if not tool_call_id:
        return None
    if len(tool_call_id) <= TOOL_CALL_ID_MAX_LENGTH:
        return tool_call_id
    return (
        f"{tool_call_id[:TOOL_CALL_ID_PREFIX_LENGTH]}"
        f"{TOOL_CALL_ID_ELLIPSIS}"
        f"{tool_call_id[-TOOL_CALL_ID_SUFFIX_LENGTH:]}"
    )
