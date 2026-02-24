from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.constants import DEFAULT_MAX_ITERATIONS
from fast_agent.context import Context
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider


def test_model_database_context_windows():
    """Test that ModelDatabase returns expected context windows"""
    # Test known models
    assert ModelDatabase.get_context_window("claude-sonnet-4-0") == 200000
    assert ModelDatabase.get_context_window("gpt-4o") == 128000
    assert ModelDatabase.get_context_window("gemini-2.0-flash") == 1048576

    # Test unknown model
    assert ModelDatabase.get_context_window("unknown-model") is None


def test_model_database_long_context_windows():
    """Explicit long-context capability should be tracked in ModelDatabase."""
    assert ModelDatabase.get_long_context_window("claude-opus-4-6") == 1_000_000
    assert ModelDatabase.get_long_context_window("claude-sonnet-4-0") == 1_000_000
    assert ModelDatabase.get_long_context_window("claude-haiku-4-5") is None
    assert ModelDatabase.get_long_context_window("unknown-model") is None


def test_model_database_long_context_model_listing():
    """Long-context model listing should come from ModelDatabase metadata."""
    models = ModelDatabase.list_long_context_models()
    assert "claude-opus-4-6" in models
    assert "claude-sonnet-4-5" in models
    assert "claude-sonnet-4-5-20250929" in models
    assert "claude-sonnet-4-0" in models
    assert "claude-sonnet-4-20250514" in models
    assert "claude-haiku-4-5" not in models


def test_model_database_fast_model_flags():
    assert ModelDatabase.is_fast_model("gpt-4.1-mini")
    assert ModelDatabase.is_fast_model("openai.gpt-4.1-mini")
    assert not ModelDatabase.is_fast_model("gpt-5")


def test_model_database_fast_model_listing():
    fast_models = ModelDatabase.list_fast_models()
    assert "gpt-4.1-mini" in fast_models
    assert "gpt-5-mini" in fast_models
    assert "gpt-5" not in fast_models


def test_model_database_default_provider_lookup():
    assert ModelDatabase.get_default_provider("gpt-4.1") == Provider.OPENAI
    assert ModelDatabase.get_default_provider("claude-sonnet-4-6") == Provider.ANTHROPIC
    assert ModelDatabase.get_default_provider("openai.gpt-4.1") == Provider.OPENAI
    assert ModelDatabase.get_default_provider("gpt-5?reasoning=low") == Provider.RESPONSES
    assert ModelDatabase.get_default_provider("unknown-model") is None


def test_model_database_anthropic_web_tool_versions_for_46_models():
    assert ModelDatabase.get_anthropic_web_search_version("claude-opus-4-6") == "web_search_20260209"
    assert ModelDatabase.get_anthropic_web_fetch_version("claude-opus-4-6") == "web_fetch_20260209"
    assert ModelDatabase.get_anthropic_required_betas("claude-opus-4-6") == (
        "code-execution-web-tools-2026-02-09",
    )

    assert (
        ModelDatabase.get_anthropic_web_search_version("claude-sonnet-4-6")
        == "web_search_20260209"
    )
    assert (
        ModelDatabase.get_anthropic_web_fetch_version("claude-sonnet-4-6")
        == "web_fetch_20260209"
    )


def test_model_database_anthropic_web_tool_versions_for_non_46_models():
    assert (
        ModelDatabase.get_anthropic_web_search_version("claude-sonnet-4-5")
        == "web_search_20250305"
    )
    assert (
        ModelDatabase.get_anthropic_web_fetch_version("claude-sonnet-4-5")
        == "web_fetch_20250910"
    )
    assert ModelDatabase.get_anthropic_required_betas("claude-sonnet-4-5") is None


def test_model_database_anthropic_web_tool_versions_unknown_model():
    assert ModelDatabase.get_anthropic_web_search_version("unknown-model") is None
    assert ModelDatabase.get_anthropic_web_fetch_version("unknown-model") is None
    assert ModelDatabase.get_anthropic_required_betas("unknown-model") is None


def test_model_database_max_tokens():
    """Test that ModelDatabase returns expected max tokens"""
    # Test known models with different max_output_tokens (no cap)
    assert ModelDatabase.get_default_max_tokens("claude-sonnet-4-0") == 64000  # ANTHROPIC_SONNET
    assert ModelDatabase.get_default_max_tokens("gpt-4o") == 16384  # OPENAI_STANDARD
    assert ModelDatabase.get_default_max_tokens("o1") == 100000  # High max_output_tokens

    # Test fallbacks
    assert ModelDatabase.get_default_max_tokens("unknown-model") == 2048
    assert ModelDatabase.get_default_max_tokens("") == 2048


def test_model_database_default_temperature():
    assert ModelDatabase.get_default_temperature("passthrough") == 0.0
    assert ModelDatabase.get_default_temperature("unknown-model") is None
    assert ModelDatabase.get_default_temperature(None) is None


def test_model_database_tokenizes():
    """Test that ModelDatabase returns expected tokenization types"""
    # Test multimodal model
    claude_tokenizes = ModelDatabase.get_tokenizes("claude-sonnet-4-0")
    assert claude_tokenizes is not None
    assert "text/plain" in claude_tokenizes
    assert "image/jpeg" in claude_tokenizes
    assert "application/pdf" in claude_tokenizes

    # Test unknown model
    assert ModelDatabase.get_tokenizes("unknown-model") is None


def test_model_database_supports_mime_basic():
    """Test MIME support lookups with normalization and aliases."""
    # Known multimodal model supports images and pdf
    assert ModelDatabase.supports_mime("claude-sonnet-4-0", "image/png")
    assert ModelDatabase.supports_mime(
        "claude-sonnet-4-0", "document/pdf"
    )  # alias -> application/pdf

    # Text-only models should not support images
    assert not ModelDatabase.supports_mime("deepseek-chat", "image/png")
    assert not ModelDatabase.supports_mime("deepseek-chat", "pdf")

    # Wildcard checks
    assert ModelDatabase.supports_mime("gpt-4o", "image/*")
    # Bare extensions
    assert ModelDatabase.supports_mime("gpt-4o", "png")


def test_model_database_google_video_audio_mime_types():
    """Test that Google models support expanded video/audio MIME types."""
    # Video formats (MP4, AVI, FLV, MOV, MPEG, MPG, WebM)
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "video/mp4")
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "video/x-msvideo")  # AVI
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "video/x-flv")  # FLV
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "video/quicktime")  # MOV
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "video/mpeg")  # MPEG, MPG
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "video/webm")

    # Audio formats
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "audio/wav")
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "audio/mpeg")
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "audio/mp3")
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "audio/aac")
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "audio/ogg")
    assert ModelDatabase.supports_mime("gemini-2.0-flash", "audio/flac")

    # Non-Google models should NOT support video/audio
    assert not ModelDatabase.supports_mime("claude-sonnet-4-0", "video/mp4")
    assert not ModelDatabase.supports_mime("claude-sonnet-4-0", "audio/wav")
    assert not ModelDatabase.supports_mime("gpt-4o", "video/mp4")
    assert not ModelDatabase.supports_mime("gpt-4o", "audio/mpeg")


def test_llm_uses_model_database_for_max_tokens():
    """Test that LLM instances use ModelDatabase for maxTokens defaults"""

    agent = LlmAgent(AgentConfig(name="Test Agent"))
    # Test with a model that has 8192 max_output_tokens (should get full amount)
    factory = ModelFactory.create_factory("claude-sonnet-4-0")
    llm = factory(agent=agent)
    assert isinstance(llm, FastAgentLLM)
    assert llm.default_request_params.maxTokens == 64000

    # Test with a model that has high max_output_tokens (should get full amount)
    factory2 = ModelFactory.create_factory("o1")
    llm2 = factory2(agent=agent)
    assert isinstance(llm2, FastAgentLLM)
    assert llm2.default_request_params.maxTokens == 100000

    # Test with passthrough model (should get its configured max tokens)
    factory3 = ModelFactory.create_factory("passthrough")
    llm3 = factory3(agent=agent)
    assert isinstance(llm3, FastAgentLLM)
    expected_max_tokens = ModelDatabase.get_default_max_tokens("passthrough")
    assert llm3.default_request_params.maxTokens == expected_max_tokens
    assert llm3.default_request_params.temperature is None


def test_llm_usage_tracking_uses_model_database():
    """Test that usage tracking uses ModelDatabase for context windows"""
    factory = ModelFactory.create_factory("passthrough")
    agent = LlmAgent(AgentConfig(name="Test Agent"))
    llm = factory(agent=agent, model="claude-sonnet-4-0")
    assert isinstance(llm, FastAgentLLM)

    # The usage_accumulator should be able to get context window from ModelDatabase
    # when it has a model set (this happens when turns are added)
    usage_accumulator = llm.usage_accumulator
    assert usage_accumulator is not None
    usage_accumulator.model = "claude-sonnet-4-0"
    assert usage_accumulator.context_window_size == 200000
    assert llm.default_request_params.maxTokens == 64000  # Should match ModelDatabase default

    # Test with unknown model
    usage_accumulator.model = "unknown-model"
    assert usage_accumulator.context_window_size is None


def test_openai_provider_preserves_all_settings():
    """Test that OpenAI provider doesn't lose any original settings"""
    factory = ModelFactory.create_factory("gpt-4o")
    agent = LlmAgent(AgentConfig(name="Test Agent"))

    llm = factory(agent=agent, instruction="You are a helpful assistant")
    assert isinstance(llm, FastAgentLLM)

    # Verify all the original OpenAI settings are preserved
    params = llm.default_request_params
    assert params.model == "gpt-4o"
    assert params.parallel_tool_calls  # Should come from base
    assert (
        params.max_iterations == DEFAULT_MAX_ITERATIONS
    )  # Should come from default setting    assert params.use_history  # Should come from base
    assert (
        params.systemPrompt == "You are a helpful assistant"
    )  # Should come from base (self.instruction)
    assert params.maxTokens == 16384  # Model-aware from ModelDatabase (gpt-4o)


def test_model_database_stream_modes():
    """Ensure models can opt into manual streaming mode."""
    assert ModelDatabase.get_stream_mode("gpt-4o") == "openai"
    assert ModelDatabase.get_stream_mode("minimaxai/minimax-m2") == "manual"
    assert ModelDatabase.get_stream_mode("unknown-model") == "openai"


def test_model_database_response_transports():
    """Codex models should expose websocket transport metadata."""
    assert ModelDatabase.get_response_transports("gpt-5.3-codex") == ("sse", "websocket")
    assert ModelDatabase.get_response_transports("gpt-5.3-codex-spark") == ("sse", "websocket")
    assert ModelDatabase.get_response_transports("gpt-4o") is None
    assert ModelDatabase.supports_response_transport("gpt-5.3-codex", "websocket") is True
    assert ModelDatabase.supports_response_transport("gpt-4o", "websocket") is None


def test_model_database_response_websocket_provider_support() -> None:
    assert (
        ModelDatabase.supports_response_websocket_provider(
            "gpt-5.3-codex-spark", Provider.CODEX_RESPONSES
        )
        is True
    )
    assert (
        ModelDatabase.supports_response_websocket_provider(
            "gpt-5.3-codex-spark", Provider.RESPONSES
        )
        is False
    )
    assert ModelDatabase.supports_response_websocket_provider("gpt-4o", Provider.RESPONSES) is None


def test_model_database_reasoning_modes():
    """Ensure reasoning types are tracked per model."""
    assert ModelDatabase.get_reasoning("o1") == "openai"
    assert ModelDatabase.get_reasoning("o3-mini") == "openai"
    assert ModelDatabase.get_reasoning("gpt-5") == "openai"
    assert ModelDatabase.get_reasoning("gpt-5.3-codex-spark") is None
    assert ModelDatabase.get_reasoning("claude-opus-4-6") == "anthropic_thinking"
    assert ModelDatabase.get_reasoning("zai-org/glm-4.6") == "reasoning_content"
    assert ModelDatabase.get_reasoning("gpt-4o") is None


def test_model_database_codex_spark_is_text_only() -> None:
    assert ModelDatabase.supports_mime("gpt-5.3-codex-spark", "text/plain")
    assert not ModelDatabase.supports_mime("gpt-5.3-codex-spark", "application/pdf")
    assert not ModelDatabase.supports_mime("gpt-5.3-codex-spark", "image/png")


def test_model_database_opus_46_reasoning_spec():
    """Opus 4.6 should expose adaptive effort settings."""
    spec = ModelDatabase.get_reasoning_effort_spec("claude-opus-4-6")
    assert spec is not None
    assert spec.kind == "effort"
    assert spec.allowed_efforts == ["low", "medium", "high", "max"]
    assert spec.allow_toggle_disable


def test_model_database_text_verbosity_spec():
    """Ensure text verbosity support is tracked for GPT-5 models."""
    spec = ModelDatabase.get_text_verbosity_spec("gpt-5")
    assert spec is not None
    assert "low" in spec.allowed
    assert ModelDatabase.get_text_verbosity_spec("gpt-4o") is None


def test_openai_llm_normalizes_repeated_roles():
    """Verify role normalization collapses repeated role strings."""
    agent = LlmAgent(AgentConfig(name="Test Agent"))
    factory = ModelFactory.create_factory("gpt-4o")
    llm = factory(agent=agent)
    assert isinstance(llm, OpenAILLM)

    assert llm._normalize_role("assistantassistant") == "assistant"
    assert llm._normalize_role("assistantASSISTANTassistant") == "assistant"
    assert llm._normalize_role("user") == "user"
    assert llm._normalize_role(None) == "assistant"


def test_openai_llm_uses_model_database_reasoning_flag():
    """Ensure reasoning detection honors ModelDatabase capabilities."""
    agent = LlmAgent(AgentConfig(name="Test Agent"))

    reasoning_llm = ModelFactory.create_factory("o1")(agent=agent)
    assert isinstance(reasoning_llm, ResponsesLLM)
    assert reasoning_llm._reasoning
    assert getattr(reasoning_llm, "_reasoning_mode", None) == "openai"

    standard_llm = ModelFactory.create_factory("gpt-4o")(agent=agent)
    assert isinstance(standard_llm, OpenAILLM)
    assert not standard_llm._reasoning
    assert getattr(standard_llm, "_reasoning_mode", None) is None


def _hf_request_args(llm: HuggingFaceLLM):
    messages = [{"role": "user", "content": "hi"}]
    return llm._prepare_api_request(messages, None, llm.default_request_params)


def _make_hf_llm(model: str, hf_settings: HuggingFaceSettings | None = None) -> HuggingFaceLLM:
    settings = Settings(hf=hf_settings or HuggingFaceSettings())
    context = Context(config=settings)
    return HuggingFaceLLM(context=context, model=model, name="test-agent")


def _make_hf_llm_with_reasoning(
    model: str,
    reasoning: bool | str | int | None,
) -> HuggingFaceLLM:
    settings = Settings(hf=HuggingFaceSettings())
    context = Context(config=settings)
    return HuggingFaceLLM(
        context=context,
        model=model,
        name="test-agent",
        reasoning_effort=reasoning,
    )


def test_huggingface_appends_default_provider_from_config():
    llm = _make_hf_llm(
        "moonshotai/kimi-k2-instruct", HuggingFaceSettings(default_provider="fireworks-ai")
    )

    assert llm.default_request_params.model == "moonshotai/kimi-k2-instruct"

    args = _hf_request_args(llm)
    assert args["model"] == "moonshotai/kimi-k2-instruct:fireworks-ai"


def test_huggingface_env_default_provider(monkeypatch):
    monkeypatch.setenv("HF_DEFAULT_PROVIDER", "router")
    llm = _make_hf_llm("moonshotai/kimi-k2-instruct")

    args = _hf_request_args(llm)
    assert args["model"] == "moonshotai/kimi-k2-instruct:router"


def test_huggingface_explicit_provider_overrides_default():
    llm = _make_hf_llm(
        "moonshotai/kimi-k2-instruct:custom", HuggingFaceSettings(default_provider="router")
    )

    assert llm.default_request_params.model == "moonshotai/kimi-k2-instruct"
    args = _hf_request_args(llm)
    assert args["model"] == "moonshotai/kimi-k2-instruct:custom"


def test_huggingface_glm_disable_reasoning_toggle():
    llm = _make_hf_llm_with_reasoning("zai-org/glm-4.7", reasoning=False)

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["disable_reasoning"] is True


def test_huggingface_kimi25_disable_reasoning_toggle():
    llm = _make_hf_llm_with_reasoning("moonshotai/kimi-k2.5", reasoning=False)

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["thinking"] == {"type": "disabled"}


def test_huggingface_kimi25_default_reasoning_toggle_enabled():
    llm = _make_hf_llm("moonshotai/kimi-k2.5")

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["thinking"] == {"type": "enabled"}


def test_model_database_runtime_model_params_registration():
    model_name = "vendor/runtime-model"
    ModelDatabase.unregister_runtime_model_params(model_name)

    params = ModelParameters(
        context_window=12345,
        max_output_tokens=678,
        tokenizes=["text/plain"],
        default_provider=Provider.OPENROUTER,
    )

    ModelDatabase.register_runtime_model_params(model_name, params)

    retrieved = ModelDatabase.get_model_params(model_name)
    assert retrieved is not None
    assert retrieved.context_window == 12345
    assert retrieved.max_output_tokens == 678
    assert ModelDatabase.get_default_provider(model_name) == Provider.OPENROUTER
    assert model_name in ModelDatabase.list_runtime_models(Provider.OPENROUTER)

    ModelDatabase.unregister_runtime_model_params(model_name)
    assert ModelDatabase.get_model_params(model_name) is None
