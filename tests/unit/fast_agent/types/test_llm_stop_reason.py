from fast_agent.types.llm_stop_reason import LlmStopReason


def test_llm_stop_reason_compares_with_raw_string() -> None:
    assert LlmStopReason.END_TURN == "endTurn"
    assert "endTurn" == LlmStopReason.END_TURN


def test_llm_stop_reason_is_hashable() -> None:
    reasons = {LlmStopReason.END_TURN: "done"}

    assert reasons[LlmStopReason.END_TURN] == "done"
    assert LlmStopReason.TOOL_USE in {LlmStopReason.TOOL_USE}


def test_llm_stop_reason_constructs_from_raw_value() -> None:
    assert LlmStopReason("toolUse") is LlmStopReason.TOOL_USE
