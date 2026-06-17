import pytest

from fast_agent.types.llm_stop_reason import LlmStopReason


def test_llm_stop_reason_compares_with_raw_string() -> None:
    assert LlmStopReason.END_TURN == "endTurn"
    assert "endTurn" == LlmStopReason.END_TURN


def test_llm_stop_reason_is_hashable() -> None:
    reasons = {LlmStopReason.END_TURN: "done"}

    assert reasons[LlmStopReason.END_TURN] == "done"
    assert LlmStopReason.TOOL_USE in {LlmStopReason.TOOL_USE}


def test_llm_stop_reason_from_string_accepts_raw_value() -> None:
    assert LlmStopReason.from_string("toolUse") is LlmStopReason.TOOL_USE


def test_llm_stop_reason_from_string_accepts_member() -> None:
    assert LlmStopReason.from_string(LlmStopReason.END_TURN) is LlmStopReason.END_TURN


def test_llm_stop_reason_from_string_reports_valid_values() -> None:
    with pytest.raises(ValueError, match="Valid values are:"):
        LlmStopReason.from_string("done")


def test_llm_stop_reason_is_valid_uses_enum_values() -> None:
    assert LlmStopReason.is_valid("endTurn") is True
    assert LlmStopReason.is_valid("END_TURN") is False
