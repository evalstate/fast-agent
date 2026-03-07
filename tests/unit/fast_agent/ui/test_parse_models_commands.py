from fast_agent.ui.command_payloads import ModelsCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_model_catalog_command() -> None:
    result = parse_special_input("/model catalog anthropic --all")
    assert isinstance(result, ModelsCommand)
    assert result.action == "catalog"
    assert result.argument == "anthropic --all"


def test_parse_model_aliases_set_argument_passthrough() -> None:
    result = parse_special_input(
        "/model aliases set $system.fast claude-haiku-4-5 --target env --dry-run"
    )
    assert isinstance(result, ModelsCommand)
    assert result.action == "aliases"
    assert result.argument == "set $system.fast claude-haiku-4-5 --target env --dry-run"
