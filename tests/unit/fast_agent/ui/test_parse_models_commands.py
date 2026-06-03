from fast_agent.ui.command_payloads import CommandError, ModelsCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_model_catalog_command() -> None:
    result = parse_special_input("/model catalog anthropic --all")
    assert isinstance(result, ModelsCommand)
    assert result.action == "catalog"
    assert result.argument == "anthropic --all"


def test_parse_models_command_defaults_to_doctor() -> None:
    result = parse_special_input("/models")
    assert isinstance(result, ModelsCommand)
    assert result.action == "doctor"
    assert result.argument is None
    assert result.command_name == "models"


def test_parse_models_command_accepts_manager_action() -> None:
    result = parse_special_input("/models doctor")
    assert isinstance(result, ModelsCommand)
    assert result.action == "doctor"
    assert result.argument is None
    assert result.command_name == "models"


def test_parse_models_catalog_command_preserves_argument_text() -> None:
    result = parse_special_input("/models catalog openai --all")
    assert isinstance(result, ModelsCommand)
    assert result.action == "catalog"
    assert result.argument == "openai --all"
    assert result.command_name == "models"


def test_parse_models_command_rejects_model_value_action() -> None:
    result = parse_special_input("/models fast on")
    assert isinstance(result, CommandError)
    assert result.message == (
        "Invalid /models action 'fast'. Use /model for runtime model settings."
    )


def test_parse_models_command_reports_unknown_action_token() -> None:
    result = parse_special_input("/models doctro")
    assert isinstance(result, CommandError)
    assert result.message == (
        "Invalid /models action 'doctro'. Use /model for runtime model settings."
    )


def test_parse_models_command_reports_unclosed_quotes() -> None:
    result = parse_special_input('/models catalog "openai')
    assert isinstance(result, CommandError)
    assert result.message == "Invalid /models arguments: No closing quotation"


def test_parse_model_references_set_argument_passthrough() -> None:
    result = parse_special_input(
        "/model references set $system.fast claude-haiku-4-5 --target env --dry-run"
    )
    assert isinstance(result, ModelsCommand)
    assert result.action == "references"
    assert result.argument == "set $system.fast claude-haiku-4-5 --target env --dry-run"
