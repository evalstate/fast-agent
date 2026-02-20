from fast_agent.ui.command_payloads import (
    ModelReasoningCommand,
    ModelVerbosityCommand,
    ModelWebFetchCommand,
    ModelWebSearchCommand,
)
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_model_reasoning_command() -> None:
    result = parse_special_input("/model reasoning high")
    assert isinstance(result, ModelReasoningCommand)
    assert result.value == "high"


def test_parse_model_verbosity_command() -> None:
    result = parse_special_input("/model verbosity low")
    assert isinstance(result, ModelVerbosityCommand)
    assert result.value == "low"


def test_parse_model_web_search_command() -> None:
    result = parse_special_input("/model web_search on")
    assert isinstance(result, ModelWebSearchCommand)
    assert result.value == "on"


def test_parse_model_web_fetch_command() -> None:
    result = parse_special_input("/model web_fetch default")
    assert isinstance(result, ModelWebFetchCommand)
    assert result.value == "default"
