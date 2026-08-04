from fast_agent.ui.command_payloads import PacksCommand, UnknownCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_packs_defaults_to_list() -> None:
    result = parse_special_input("/packs")
    assert isinstance(result, PacksCommand)
    assert result.action == "list"
    assert result.argument is None


def test_parse_packs_with_action_and_argument() -> None:
    result = parse_special_input("/packs update all --force")
    assert isinstance(result, PacksCommand)
    assert result.action == "update"
    assert result.argument == "all --force"


def test_parse_packs_readme_with_argument() -> None:
    result = parse_special_input("/packs readme alpha")
    assert isinstance(result, PacksCommand)
    assert result.action == "readme"
    assert result.argument == "alpha"


def test_cards_command_is_removed() -> None:
    result = parse_special_input("/cards")
    assert result == UnknownCommand(command="/cards")
