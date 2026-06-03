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


def test_parse_cards_preserves_raw_argument_text() -> None:
    result = parse_special_input('/cards add "team pack" --registry local')
    assert isinstance(result, CardsCommand)
    assert result.action == "add"
    assert result.argument == '"team pack" --registry local'


def test_parse_cards_readme_with_argument() -> None:
    result = parse_special_input("/cards readme alpha")
    assert isinstance(result, CardsCommand)
    assert result.action == "readme"
    assert result.argument == "alpha"


def test_parse_cards_accepts_quoted_action_and_preserves_raw_argument() -> None:
    result = parse_special_input('/cards "readme" "team pack"')
    assert isinstance(result, CardsCommand)
    assert result.action == "readme"
    assert result.argument == '"team pack"'
