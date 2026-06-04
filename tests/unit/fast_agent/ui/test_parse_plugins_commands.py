from fast_agent.ui.command_payloads import PluginsCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_plugins_defaults_to_list() -> None:
    result = parse_special_input("/plugins")
    assert isinstance(result, PluginsCommand)
    assert result.action == "list"
    assert result.argument is None


def test_parse_plugins_with_action_and_argument() -> None:
    result = parse_special_input("/plugins update all --force")
    assert isinstance(result, PluginsCommand)
    assert result.action == "update"
    assert result.argument == "all --force"
