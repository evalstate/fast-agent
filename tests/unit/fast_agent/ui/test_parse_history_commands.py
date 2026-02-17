from fast_agent.ui.command_payloads import HistoryWebClearCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_history_webclear_without_agent() -> None:
    result = parse_special_input("/history webclear")
    assert isinstance(result, HistoryWebClearCommand)
    assert result.agent is None


def test_parse_history_webclear_with_agent() -> None:
    result = parse_special_input("/history webclear analyst")
    assert isinstance(result, HistoryWebClearCommand)
    assert result.agent == "analyst"
