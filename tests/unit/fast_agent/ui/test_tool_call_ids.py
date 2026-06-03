from fast_agent.ui.tool_call_ids import (
    TOOL_CALL_ID_MAX_LENGTH,
    format_tool_call_id,
)


def test_format_tool_call_id_omits_missing_values() -> None:
    assert format_tool_call_id(None) is None
    assert format_tool_call_id("") is None


def test_format_tool_call_id_keeps_short_and_boundary_values() -> None:
    assert format_tool_call_id("call_1") == "call_1"
    assert format_tool_call_id("x" * TOOL_CALL_ID_MAX_LENGTH) == (
        "x" * TOOL_CALL_ID_MAX_LENGTH
    )


def test_format_tool_call_id_keeps_prefix_and_suffix_for_long_values() -> None:
    assert format_tool_call_id("call_abcdef0123456789") == "call_…456789"
