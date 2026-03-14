from fast_agent.ui.command_payloads import (
    HistoryReviewCommand,
    HistoryRewindCommand,
    HistoryShowCommand,
    HistoryWebClearCommand,
    LoadHistoryCommand,
    ShowHistoryCommand,
)
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_history_webclear_without_agent() -> None:
    result = parse_special_input("/history webclear")
    assert isinstance(result, HistoryWebClearCommand)
    assert result.agent is None


def test_parse_history_webclear_with_agent() -> None:
    result = parse_special_input("/history webclear analyst")
    assert isinstance(result, HistoryWebClearCommand)
    assert result.agent == "analyst"


def test_parse_history_detail_with_turn() -> None:
    result = parse_special_input("/history detail 3")
    assert isinstance(result, HistoryReviewCommand)
    assert result.turn_index == 3
    assert result.error is None


def test_parse_history_detail_requires_turn() -> None:
    result = parse_special_input("/history detail")
    assert isinstance(result, HistoryReviewCommand)
    assert result.turn_index is None
    assert result.error == "Turn number required for /history detail"


def test_parse_history_rewind_requires_turn() -> None:
    result = parse_special_input("/history rewind")
    assert isinstance(result, HistoryRewindCommand)
    assert result.turn_index is None
    assert result.error == "Turn number required for /history rewind"


def test_parse_history_quoted_reserved_detail_agent_name_uses_history_overview() -> None:
    result = parse_special_input('/history "detail"')
    assert isinstance(result, ShowHistoryCommand)
    assert result.agent == "detail"


def test_parse_history_quoted_arguments_match_previous_tui_behavior() -> None:
    load_result = parse_special_input('/history load "my history.json"')
    assert load_result == LoadHistoryCommand(filename="my history.json", error=None)

    show_result = parse_special_input('/history show "agent name"')
    assert show_result == HistoryShowCommand(agent="agent name")

    detail_result = parse_special_input('/history detail "5"')
    assert detail_result == HistoryReviewCommand(turn_index=5, error=None)
