from fast_agent.ui.command_payloads import CardsCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_cards_defaults_to_list() -> None:
    result = parse_special_input("/cards")
    assert isinstance(result, CardsCommand)
    assert result.action == "list"
    assert result.argument is None


def test_parse_cards_with_action_and_argument() -> None:
    result = parse_special_input("/cards update all --force")
    assert isinstance(result, CardsCommand)
    assert result.action == "update"
    assert result.argument == "all --force"
