import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory, Provider
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.openai.llm_generic import GenericLLM
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
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


def test_model_query_temperature():
    config = ModelFactory.parse_model_string("gpt-5?temperature=0.35")
    assert config.provider == Provider.RESPONSES
    assert config.model_name == "gpt-5"
    assert config.temperature == 0.35


def test_model_query_temp_alias():
    config = ModelFactory.parse_model_string("gpt-5?temp=0.2")
    assert config.temperature == 0.2


def test_model_query_sampling_parameters():
    config = ModelFactory.parse_model_string(
        "hf.Qwen/Qwen3.5-397B-A17B:novita"
        "?temperature=0.6&top_p=0.95&top_k=20&min_p=0.0"
        "&presence_penalty=0.0&repetition_penalty=1.0"
    )

    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "Qwen/Qwen3.5-397B-A17B:novita"
    assert config.temperature == 0.6
    assert config.top_p == 0.95
    assert config.top_k == 20
    assert config.min_p == 0.0
    assert config.presence_penalty == 0.0
    assert config.repetition_penalty == 1.0


def test_alias_sampling_defaults_allow_user_query_overrides() -> None:
    config = ModelFactory.parse_model_string("qwen35?temperature=0.9&top_p=0.7")

    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "Qwen/Qwen3.5-397B-A17B:novita"
    assert config.temperature == 0.9
    assert config.top_p == 0.7
    assert config.top_k == 20
    assert config.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=True)


def test_alias_sampling_defaults_preserve_user_provider_suffix_override() -> None:
    config = ModelFactory.parse_model_string("qwen35:nebius")

    assert config.provider == Provider.HUGGINGFACE
    assert config.model_name == "Qwen/Qwen3.5-397B-A17B:nebius"
    assert config.temperature == 0.6
    assert config.top_p == 0.95
    assert config.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=True)


def test_model_query_transport_websocket_alias():
    config = ModelFactory.parse_model_string("codexplan?transport=ws")
    assert config.provider == Provider.CODEX_RESPONSES
    assert config.model_name == "gpt-5.3-codex"
    assert config.transport == "websocket"


def test_model_query_transport_auto():
    config = ModelFactory.parse_model_string("codexplan52?transport=auto")
    assert config.transport == "auto"


def test_model_query_transport_sse():
    config = ModelFactory.parse_model_string("codexplan?transport=sse")
    assert config.transport == "sse"


def test_model_query_web_tool_flags():
    config = ModelFactory.parse_model_string("claude-sonnet-4-6?web_search=on&web_fetch=off")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-6"
    assert config.web_search is True
    assert config.web_fetch is False


def test_model_query_web_tool_flags_boolean_aliases():
    config = ModelFactory.parse_model_string("sonnet?web_search=true&web_fetch=0")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-6"
    assert config.web_search is True
    assert config.web_fetch is False


def test_model_query_web_search_flag_for_responses_provider():
    config = ModelFactory.parse_model_string("responses.gpt-5-mini?web_search=on")
    assert config.provider == Provider.RESPONSES
    assert config.model_name == "gpt-5-mini"
    assert config.web_search is True


def test_invalid_web_tool_query_values():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("claude-sonnet-4-6?web_search=maybe")

    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("claude-sonnet-4-6?web_fetch=maybe")


def test_invalid_transport_query():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("codexplan?transport=websock")


def test_transport_query_allows_responses_default_model():
    config = ModelFactory.parse_model_string("gpt-5?transport=ws")
    assert config.provider == Provider.RESPONSES
    assert config.model_name == "gpt-5"
    assert config.transport == "websocket"


def test_transport_query_allows_responses_gpt_5_2() -> None:
    config = ModelFactory.parse_model_string("responses.gpt-5.2?transport=ws")
    assert config.provider == Provider.RESPONSES
    assert config.model_name == "gpt-5.2"
    assert config.transport == "websocket"


def test_transport_query_allows_responses_codex_model():
    config = ModelFactory.parse_model_string("responses.gpt-5.3-codex?transport=ws")
    assert config.provider == Provider.RESPONSES
    assert config.model_name == "gpt-5.3-codex"
    assert config.transport == "websocket"


def test_transport_query_rejects_responses_provider_for_codex_spark():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("responses.gpt-5.3-codex-spark?transport=ws")


def test_transport_query_allows_codexresponses_provider_for_codex_spark():
    config = ModelFactory.parse_model_string("codexresponses.gpt-5.3-codex-spark?transport=ws")
    assert config.provider == Provider.CODEX_RESPONSES
    assert config.model_name == "gpt-5.3-codex-spark"
    assert config.transport == "websocket"


def test_transport_query_rejects_openai_provider_even_with_responses_model():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("openai.gpt-5?transport=ws")


def test_transport_query_composes_with_reasoning_and_verbosity():
    config = ModelFactory.parse_model_string("codexplan?transport=ws&reasoning=high&verbosity=low")
    assert config.transport == "websocket"
    assert config.reasoning_effort == ReasoningEffortSetting(kind="effort", value="high")
    assert config.text_verbosity == "low"


def test_factory_passes_transport_to_responses_llm():
    factory = ModelFactory.create_factory("codexplan?transport=ws")
    llm = factory(LlmAgent(AgentConfig(name="Test Agent")))
    assert isinstance(llm, ResponsesLLM)
    assert llm._transport == "websocket"


def test_factory_passes_transport_to_responses_llm_for_openai_responses_model() -> None:
    factory = ModelFactory.create_factory("responses.gpt-5?transport=ws")
    llm = factory(LlmAgent(AgentConfig(name="Test Agent")))
    assert isinstance(llm, ResponsesLLM)
    assert llm.provider == Provider.RESPONSES
    assert llm._transport == "websocket"


def test_factory_passes_web_tool_overrides_to_anthropic_llm():
    factory = ModelFactory.create_factory("claude-sonnet-4-6?web_search=on&web_fetch=off")
    llm = factory(LlmAgent(AgentConfig(name="Test Agent")))
    assert isinstance(llm, AnthropicLLM)
    assert llm._web_search_override is True
    assert llm._web_fetch_override is False


def test_factory_passes_web_search_override_to_responses_llm():
    factory = ModelFactory.create_factory("responses.gpt-5-mini?web_search=on")
    llm = factory(LlmAgent(AgentConfig(name="Test Agent")))
    assert isinstance(llm, ResponsesLLM)
    assert llm._web_search_override is True


def test_factory_passes_web_search_override_to_codex_responses_llm():
    factory = ModelFactory.create_factory("codexplan?web_search=on")
    llm = factory(LlmAgent(AgentConfig(name="Test Agent")))
    assert isinstance(llm, ResponsesLLM)
    assert llm.provider == Provider.CODEX_RESPONSES
    assert llm._web_search_override is True


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


def test_invalid_temperature_query():
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("gpt-5?temperature=hot")


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


def test_claude_alias_resolves_to_sonnet_46():
    config = ModelFactory.parse_model_string("claude")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-6"

    config = ModelFactory.parse_model_string("opus46")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-opus-4-6"


def test_gemini31_alias_resolves_to_google_31_preview():
    config = ModelFactory.parse_model_string("gemini3.1")
    assert config.provider == Provider.GOOGLE
    assert config.model_name == "gemini-3.1-pro-preview"


def test_curated_catalog_aliases_are_parseable():
    for entry in ModelSelectionCatalog.list_current_entries():
        if "?" in entry.model:
            continue

        alias_config = ModelFactory.parse_model_string(entry.alias)
        model_config = ModelFactory.parse_model_string(entry.model)

        assert alias_config.provider == model_config.provider
        assert ModelDatabase.normalize_model_name(alias_config.model_name) == ModelDatabase.normalize_model_name(
            model_config.model_name
        )


def test_codexplan_aliases_use_codex_oauth_provider():
    config = ModelFactory.parse_model_string("codexplan")
    assert config.provider == Provider.CODEX_RESPONSES
    assert config.model_name == "gpt-5.3-codex"

    config = ModelFactory.parse_model_string("codexplan52")
    assert config.provider == Provider.CODEX_RESPONSES
    assert config.model_name == "gpt-5.2-codex"

    config = ModelFactory.parse_model_string("codexspark")
    assert config.provider == Provider.CODEX_RESPONSES
    assert config.model_name == "gpt-5.3-codex-spark"


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


# --- Long context (context=1m) tests ---


def test_model_query_context_1m():
    """Test parsing context=1m for a supported Anthropic model."""
    config = ModelFactory.parse_model_string("claude-opus-4-6?context=1m")
    assert config.provider == Provider.ANTHROPIC
    assert config.model_name == "claude-opus-4-6"
    assert config.long_context is True


def test_model_query_context_1m_with_reasoning():
    """Test context=1m composes with other query parameters."""
    config = ModelFactory.parse_model_string("claude-sonnet-4-5?context=1m&reasoning=4096")
    assert config.long_context is True
    assert config.reasoning_effort == ReasoningEffortSetting(kind="budget", value=4096)


def test_model_query_context_1m_case_insensitive():
    """The context value should be case-insensitive."""
    config = ModelFactory.parse_model_string("claude-sonnet-4-0?context=1M")
    assert config.long_context is True


def test_model_query_context_invalid_value():
    """Only '1m' is accepted; anything else raises."""
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("claude-opus-4-6?context=2m")


def test_model_query_context_empty_is_ignored():
    """Empty context= is dropped by parse_qs, treated as absent."""
    config = ModelFactory.parse_model_string("claude-opus-4-6?context=")
    assert config.long_context is False


def test_model_query_context_absent_means_false():
    """Without context=, long_context defaults to False."""
    config = ModelFactory.parse_model_string("claude-opus-4-6")
    assert config.long_context is False


def test_model_query_context_non_anthropic_parses():
    """Parsing context=1m succeeds even for non-Anthropic models.

    Provider-level validation happens later, not at parse time.
    """
    config = ModelFactory.parse_model_string("gpt-5?context=1m")
    assert config.long_context is True
    assert config.provider == Provider.RESPONSES


# --- Long context: LLM instantiation tests ---


def test_anthropic_long_context_creates_llm_with_override():
    """Test that creating an Anthropic LLM with long_context sets the override."""
    factory = ModelFactory.create_factory("claude-opus-4-6?context=1m")
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)
    assert isinstance(llm, AnthropicLLM)
    assert llm._long_context is True
    assert llm._context_window_override == 1_000_000
    assert llm._usage_accumulator.context_window_size == 1_000_000
    # model_info should reflect the override
    info = llm.model_info
    assert info is not None
    assert info.context_window == 1_000_000


def test_anthropic_long_context_default_is_200k():
    """Without context=1m, context window should be 200K."""
    factory = ModelFactory.create_factory("claude-opus-4-6")
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)
    assert isinstance(llm, AnthropicLLM)
    assert llm._long_context is False
    info = llm.model_info
    assert info is not None
    assert info.context_window == 200_000


def test_factory_passes_temperature_query_to_request_params():
    factory = ModelFactory.create_factory("gpt-5?temperature=0.42")
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)
    assert llm.default_request_params.temperature == 0.42


def test_factory_passes_sampling_query_to_request_params() -> None:
    factory = ModelFactory.create_factory("qwen35")
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)

    assert llm.default_request_params.model == "Qwen/Qwen3.5-397B-A17B"
    assert llm.default_request_params.temperature == 0.6
    assert llm.default_request_params.top_p == 0.95
    assert llm.default_request_params.top_k == 20
    assert llm.default_request_params.min_p == 0.0
    assert llm.default_request_params.presence_penalty == 0.0
    assert llm.default_request_params.repetition_penalty == 1.0
    assert llm.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=True)


def test_hf_sampling_overrides_route_non_openai_fields_to_extra_body() -> None:
    factory = ModelFactory.create_factory("qwen35")
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)

    assert isinstance(llm, HuggingFaceLLM)

    args = llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm.default_request_params,
    )

    assert args["temperature"] == 0.6
    assert args["top_p"] == 0.95
    assert args["presence_penalty"] == 0.0
    assert "top_k" not in args
    assert "min_p" not in args
    assert "repetition_penalty" not in args

    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["top_k"] == 20
    assert extra_body["min_p"] == 0.0
    assert extra_body["repetition_penalty"] == 1.0
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": True}


def test_hf_qwen35_instruct_alias_disables_thinking_via_chat_template_kwargs() -> None:
    factory = ModelFactory.create_factory("qwen35instruct")
    agent = LlmAgent(AgentConfig(name="test"))
    llm = factory(agent)

    assert isinstance(llm, HuggingFaceLLM)

    args = llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm.default_request_params,
    )

    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_runtime_model_provider_registration():
    model_name = "runtime-fast-model"
    ModelFactory.register_runtime_model_provider(model_name, Provider.FAST_AGENT)
    try:
        config = ModelFactory.parse_model_string(model_name)
        assert config.provider == Provider.FAST_AGENT
        assert config.model_name == model_name
    finally:
        ModelFactory.unregister_runtime_model_provider(model_name)
