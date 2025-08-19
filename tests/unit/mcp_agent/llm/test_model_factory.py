import pytest

from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.llm.model_factory import (
    ModelFactory,
    Provider,
    ReasoningEffort,
)
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_generic import GenericAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM


def test_simple_model_names():
    """Test parsing of simple model names"""
    cases = [
        ("o1-mini", Provider.OPENAI),
        ("claude-3-haiku-20240307", Provider.ANTHROPIC),
        ("claude-3-5-sonnet-20240620", Provider.ANTHROPIC),
    ]

    for model_name, expected_provider in cases:
        config = ModelFactory.parse_model_string(model_name)
        assert config.provider == expected_provider
        assert config.model_name == model_name
        assert config.reasoning_effort is None


def test_full_model_strings():
    """Test parsing of full model strings with providers"""
    cases = [
        (
            "anthropic.claude-3-haiku-20240307",
            Provider.ANTHROPIC,
            "claude-3-haiku-20240307",
            None,
        ),
        ("openai.gpt-4.1", Provider.OPENAI, "gpt-4.1", None),
        ("openai.o1.high", Provider.OPENAI, "o1", ReasoningEffort.HIGH),
    ]

    for model_str, exp_provider, exp_model, exp_effort in cases:
        config = ModelFactory.parse_model_string(model_str)
        assert config.provider == exp_provider
        assert config.model_name == exp_model
        assert config.reasoning_effort == exp_effort


def test_invalid_inputs():
    """Test handling of invalid inputs"""
    invalid_cases = [
        "unknown-model",  # Unknown simple model
        "invalid.gpt-4",  # Invalid provider
    ]

    for invalid_str in invalid_cases:
        with pytest.raises(ModelConfigError):
            ModelFactory.parse_model_string(invalid_str)

def test_llm_class_creation(mocker):
    """Test creation of LLM classes"""
    # Mock the client initialization for all relevant LLM providers
    mocker.patch(
        'mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._initialize_client',
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        'mcp_agent.llm.providers.augmented_llm_openai.OpenAIAugmentedLLM._initialize_client',
        return_value=mocker.MagicMock()
    )

    cases = [
        ("gpt-4.1", OpenAIAugmentedLLM),
        ("claude-3-haiku-20240307", AnthropicAugmentedLLM),
        ("openai.gpt-4.1", OpenAIAugmentedLLM),
    ]

    for model_str, expected_class in cases:
        factory = ModelFactory.create_factory(model_str)
        assert callable(factory)

        # This will now succeed without needing an API key
        instance = factory(agent=None)
        assert isinstance(instance, expected_class)

def test_allows_generic_model(mocker):
    """Test that generic model names are allowed"""
    # Mock the client and the base_url method for a more robust test
    mocker.patch(
        'mcp_agent.llm.providers.augmented_llm_generic.GenericAugmentedLLM._initialize_client',
        return_value=mocker.MagicMock()
    )
    mock_base_url = mocker.patch(
        'mcp_agent.llm.providers.augmented_llm_generic.GenericAugmentedLLM._base_url',
        return_value="http://localhost:11434/v1"
    )

    generic_model = "generic.llama3.2:latest"
    factory = ModelFactory.create_factory(generic_model)
    instance = factory(agent=None)

    assert isinstance(instance, GenericAugmentedLLM)
    # Assert against the value returned by the mocked method
    assert instance._base_url() == "http://localhost:11434/v1"
    mock_base_url.assert_called_once()