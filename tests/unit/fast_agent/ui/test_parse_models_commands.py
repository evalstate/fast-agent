from fast_agent.ui.command_payloads import ModelsCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_models_defaults_to_doctor() -> None:
    result = parse_special_input("/models")
    assert isinstance(result, ModelsCommand)
    assert result.action == "doctor"
    assert result.argument is None


def test_parse_models_with_action_and_argument() -> None:
    result = parse_special_input("/models catalog anthropic --all")
    assert isinstance(result, ModelsCommand)
    assert result.action == "catalog"
    assert result.argument == "anthropic --all"


def test_parse_models_aliases_set_argument_passthrough() -> None:
    result = parse_special_input(
        "/models aliases set $system.fast claude-haiku-4-5 --target env --dry-run"
    )
    assert isinstance(result, ModelsCommand)
    assert result.action == "aliases"
    assert result.argument == "set $system.fast claude-haiku-4-5 --target env --dry-run"
