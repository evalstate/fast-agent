import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_factory import ModelFactory, Provider
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.openai.llm_generic import GenericLLM
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting

# Test aliases - decoupled from production MODEL_ALIASES
# These provide stable test data that won't break when production aliases change
TEST_ALIASES = {
    "kimi": "hf.moonshotai/Kimi-K2-Instruct-0905",  # No default provider
    "glm": "hf.zai-org/GLM-4.6:cerebras",  # Has default provider
    "qwen3": "hf.Qwen/Qwen3-Next-80B-A3B-Instruct:together",
    "minimax": "hf.MiniMaxAI/MiniMax-M2",  # No default provider
}


def test_simple_model_names():
    """Test parsing of simple model names"""
    cases = [
        ("o1-mini", Provider.RESPONSES),
        ("claude-3-haiku-20240307", Provider.ANTHROPIC),
        ("claude-3-5-sonnet-20240620", Provider.ANTHROPIC),
        ("claude-opus-4-6", Provider.ANTHROPIC),
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
        ("openai/gpt-4.1", Provider.OPENAI, "gpt-4.1", None),
        (
            "openai.o1.high",
            Provider.OPENAI,
            "o1",
            ReasoningEffortSetting(kind="effort", value="high"),
        ),
        (
            "openai/o1.high",
            Provider.OPENAI,
            "o1",
            ReasoningEffortSetting(kind="effort", value="high"),
        ),
    ]

    for model_str, exp_provider, exp_model, exp_effort in cases:
        config = ModelFactory.parse_model_string(model_str)
        assert config.provider == exp_provider
        assert config.model_name == exp_model
        assert config.reasoning_effort == exp_effort


def test_model_query_reasoning_effort():
    config = ModelFactory.parse_model_string("openai.o1?reasoning=low")
    assert config.provider == Provider.OPENAI
    assert config.model_name == "o1"
    assert config.reasoning_effort == ReasoningEffortSetting(kind="effort", value="low")


def test_model_query_reasoning_budget():
    config = ModelFactory.parse_model_string("openai.o1?reasoning=2048")
    assert config.provider == Provider.OPENAI
    assert config.reasoning_effort == ReasoningEffortSetting(kind="budget", value=2048)


def test_model_query_reasoning_toggle():
    config = ModelFactory.parse_model_string("hf.zai-org/GLM-4.7?reasoning=off")
    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "zai-org/GLM-4.7"
    assert config.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=False)


def test_model_query_instant_mode_toggle():
    config = ModelFactory.parse_model_string("hf.moonshotai/Kimi-K2.5?instant=on")
    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "moonshotai/Kimi-K2.5"
    assert config.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=False)

    config = ModelFactory.parse_model_string("hf.moonshotai/Kimi-K2.5?instant=off")
    assert config.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=True)


def test_model_query_structured_json():
    config = ModelFactory.parse_model_string("claude-sonnet-4-5?structured=json")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-5"
    assert config.structured_output_mode == "json"


def test_model_query_structured_tool_use():
    config = ModelFactory.parse_model_string("claude-sonnet-4-5?structured=tool_use")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-5"
    assert config.structured_output_mode == "tool_use"


def test_model_query_text_verbosity():
    config = ModelFactory.parse_model_string("gpt-5?verbosity=med&reasoning=high")
    assert config.provider == Provider.RESPONSES
    assert config.model_name == "gpt-5"
    assert config.text_verbosity == "medium"


def test_invalid_inputs():
    """Test handling of invalid inputs"""
    invalid_cases = [
        "unknown-model",  # Unknown simple model
        "invalid.gpt-4",  # Invalid provider
    ]

    for invalid_str in invalid_cases:
        with pytest.raises(ModelConfigError):
            ModelFactory.parse_model_string(invalid_str)


def test_invalid_structured_query():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("claude-sonnet-4-5?structured=maybe")


def test_invalid_instant_query():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("hf.zai-org/GLM-4.7?instant=on")


def test_invalid_verbosity_query():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("gpt-5?verbosity=verbose")


def test_llm_class_creation():
    """Test creation of LLM classes"""
    cases = [
        ("gpt-4.1", OpenAILLM),
        ("claude-3-haiku-20240307", AnthropicLLM),
        ("openai.gpt-4.1", OpenAILLM),
    ]

    for model_str, expected_class in cases:
        factory = ModelFactory.create_factory(model_str)
        # Check that we get a callable factory function
        assert callable(factory)

        # Instantiate with minimal params to check it creates the correct class
        # Note: You may need to adjust params based on what the factory requires
        instance = factory(LlmAgent(AgentConfig(name="Test Agent")))
        assert isinstance(instance, expected_class)


def test_allows_generic_model():
    """Test that generic model names are allowed"""
    generic_model = "generic.llama3.2:latest"
    factory = ModelFactory.create_factory(generic_model)
    instance = factory(LlmAgent(AgentConfig(name="test")))
    assert isinstance(instance, GenericLLM)
    assert instance._base_url() == "http://localhost:11434/v1"


def test_huggingface_alias_without_provider():
    """Test HuggingFace alias without explicit provider"""
    config = ModelFactory.parse_model_string("kimi", aliases=TEST_ALIASES)
    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "moonshotai/Kimi-K2-Instruct-0905"


def test_opus_aliases_resolve_to_opus_46():
    config = ModelFactory.parse_model_string("opus")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-opus-4-6"

    config = ModelFactory.parse_model_string("opus46")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-opus-4-6"


def test_codexplan_aliases_use_codex_oauth_provider():
    config = ModelFactory.parse_model_string("codexplan")
    assert config.provider == Provider.CODEX_RESPONSES
    assert config.model_name == "gpt-5.3-codex"

    config = ModelFactory.parse_model_string("codexplan52")
    assert config.provider == Provider.CODEX_RESPONSES
    assert config.model_name == "gpt-5.2-codex"


def test_huggingface_alias_with_default_provider():
    """Test HuggingFace alias that includes a default provider in the alias"""
    # glm alias has :cerebras as default provider
    config = ModelFactory.parse_model_string("glm", aliases=TEST_ALIASES)
    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "zai-org/GLM-4.6:cerebras"


def test_huggingface_alias_provider_override():
    """Test that user-specified provider overrides the alias default"""
    # glm alias is "hf.zai-org/GLM-4.6:cerebras" - user specifies :groq
    config = ModelFactory.parse_model_string("glm:groq", aliases=TEST_ALIASES)
    assert config.provider == Provider.HUGGINGFACE
    # User's :groq should replace the alias's :cerebras
    assert config.model_name == "zai-org/GLM-4.6:groq"


def test_huggingface_alias_without_default_provider_gets_user_provider():
    """Test that an alias without a default provider can receive a user provider"""
    # kimi alias is "hf.moonshotai/Kimi-K2-Instruct-0905" (no default provider)
    config = ModelFactory.parse_model_string("kimi:groq", aliases=TEST_ALIASES)
    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "moonshotai/Kimi-K2-Instruct-0905:groq"


def test_huggingface_alias_provider_override_together():
    """Test provider override with together"""
    # qwen3 alias is "hf.Qwen/Qwen3-Next-80B-A3B-Instruct:together"
    config = ModelFactory.parse_model_string("qwen3:nebius", aliases=TEST_ALIASES)
    assert config.provider == Provider.HUGGINGFACE
    # User's :nebius should replace the alias's :together
    assert config.model_name == "Qwen/Qwen3-Next-80B-A3B-Instruct:nebius"


def test_huggingface_display_info_with_provider():
    """Test HuggingFaceLLM displays correct model and provider info"""
    # Create HuggingFace LLM with explicit provider
    factory = ModelFactory.create_factory("glm", aliases=TEST_ALIASES)  # glm has :cerebras default
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)

    assert isinstance(llm, HuggingFaceLLM)
    assert hasattr(llm, "get_hf_display_info")

    info = llm.get_hf_display_info()
    assert info["model"] == "zai-org/GLM-4.6"
    assert info["provider"] == "cerebras"


def test_huggingface_display_info_auto_routing():
    """Test HuggingFaceLLM displays auto-routing when no provider specified"""
    # Create HuggingFace LLM without provider suffix
    factory = ModelFactory.create_factory(
        "minimax", aliases=TEST_ALIASES
    )  # minimax has no default provider
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)

    assert isinstance(llm, HuggingFaceLLM)
    info = llm.get_hf_display_info()
    assert info["model"] == "MiniMaxAI/MiniMax-M2"
    assert info["provider"] == "auto-routing"


def test_huggingface_display_info_user_override():
    """Test HuggingFaceLLM displays user-specified provider correctly"""
    # User overrides glm's :cerebras with :groq
    factory = ModelFactory.create_factory("glm:groq", aliases=TEST_ALIASES)
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)

    assert isinstance(llm, HuggingFaceLLM)
    info = llm.get_hf_display_info()
    assert info["model"] == "zai-org/GLM-4.6"
    assert info["provider"] == "groq"
