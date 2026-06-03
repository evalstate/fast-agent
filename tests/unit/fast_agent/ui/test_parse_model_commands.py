from fast_agent.commands.shared_command_intents import (
    MODEL_MANAGER_COMMAND_ACTIONS,
    MODEL_VALUE_COMMAND_ACTIONS,
)
from fast_agent.ui.command_payloads import (
    CommandError,
    ModelFastCommand,
    ModelReasoningCommand,
    ModelsCommand,
    ModelSwitchCommand,
    ModelTaskBudgetCommand,
    ModelVerbosityCommand,
    ModelWebFetchCommand,
    ModelWebSearchCommand,
    ModelXSearchCommand,
)
from fast_agent.ui.enhanced_prompt import parse_special_input
from fast_agent.ui.prompt import parser as prompt_parser


def test_prompt_model_value_factories_match_shared_action_classification() -> None:
    assert frozenset(prompt_parser._MODEL_VALUE_COMMAND_FACTORIES) == MODEL_VALUE_COMMAND_ACTIONS


def test_prompt_model_manager_actions_match_shared_action_classification() -> None:
    for action in MODEL_MANAGER_COMMAND_ACTIONS:
        result = parse_special_input(f"/model {action}")
        assert isinstance(result, ModelsCommand)
        assert result.action == action
        assert result.command_name == "model"


def test_parse_model_reasoning_command() -> None:
    result = parse_special_input("/model reasoning high")
    assert isinstance(result, ModelReasoningCommand)
    assert result.value == "high"


def test_parse_model_verbosity_command() -> None:
    result = parse_special_input("/model verbosity low")
    assert isinstance(result, ModelVerbosityCommand)
    assert result.value == "low"


def test_parse_model_task_budget_command() -> None:
    result = parse_special_input("/model task_budget 128k")
    assert isinstance(result, ModelTaskBudgetCommand)
    assert result.value == "128k"


def test_parse_model_fast_command() -> None:
    result = parse_special_input("/model fast on")
    assert isinstance(result, ModelFastCommand)
    assert result.value == "on"




def test_parse_model_fast_flex_command() -> None:
    result = parse_special_input("/model fast flex")
    assert isinstance(result, ModelFastCommand)
    assert result.value == "flex"

def test_parse_hidden_fast_alias_command() -> None:
    result = parse_special_input("/fast status")
    assert isinstance(result, ModelFastCommand)
    assert result.value == "status"


def test_parse_model_web_search_command() -> None:
    result = parse_special_input("/model web_search on")
    assert isinstance(result, ModelWebSearchCommand)
    assert result.value == "on"


def test_parse_model_x_search_command() -> None:
    result = parse_special_input("/model x_search on")
    assert isinstance(result, ModelXSearchCommand)
    assert result.value == "on"


def test_parse_model_web_fetch_command() -> None:
    result = parse_special_input("/model web_fetch default")
    assert isinstance(result, ModelWebFetchCommand)
    assert result.value == "default"


def test_parse_model_switch_command() -> None:
    result = parse_special_input("/model switch gpt-5-mini")
    assert isinstance(result, ModelSwitchCommand)
    assert result.value == "gpt-5-mini"


def test_parse_model_switch_matches_case_insensitively() -> None:
    result = parse_special_input("/MODEL SWITCH gpt-5-mini")
    assert isinstance(result, ModelSwitchCommand)
    assert result.value == "gpt-5-mini"


def test_parse_model_switch_command_unquotes_single_argument() -> None:
    result = parse_special_input('/model switch "gpt-5-mini"')
    assert isinstance(result, ModelSwitchCommand)
    assert result.value == "gpt-5-mini"


def test_parse_model_switch_command_accepts_quoted_subcommand() -> None:
    result = parse_special_input('/model "switch" gpt-5-mini')
    assert isinstance(result, ModelSwitchCommand)
    assert result.value == "gpt-5-mini"


def test_parse_model_command_reports_unclosed_quotes() -> None:
    result = parse_special_input('/model switch "gpt-5')

    assert isinstance(result, CommandError)
    assert result.message == "Invalid /model arguments: No closing quotation"


def test_parse_model_doctor_command() -> None:
    result = parse_special_input("/model doctor")
    assert isinstance(result, ModelsCommand)
    assert result.action == "doctor"
    assert result.argument is None
