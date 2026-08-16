from fast_agent.ui.command_payloads import ModelManagerCommand, UnknownCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_model_catalog_command() -> None:
    result = parse_special_input("/model catalog anthropic --all")
    assert isinstance(result, ModelManagerCommand)
    assert result.action == "catalog"
    assert result.argument == "anthropic --all"


def test_parse_model_references_set_argument_passthrough() -> None:
    result = parse_special_input(
        "/model references set $system.fast claude-haiku-4-5 --target env --dry-run"
    )
    assert isinstance(result, ModelManagerCommand)
    assert result.action == "references"
    assert result.argument == "set $system.fast claude-haiku-4-5 --target env --dry-run"


def test_models_command_is_removed() -> None:
    result = parse_special_input("/models doctor")
    assert result == UnknownCommand(command="/models doctor")
