"""
Testing notes:

- This module owns capability lookup and request-shaping contracts that flow
  from ModelDatabase metadata into runtime behavior.
- Prefer behavior and parity assertions (for example, replacement models sharing
  capabilities with prior models) over repeating raw table entries.
- HuggingFace tests here should focus on provider-aware lookups, reasoning
  toggles, stream mode, and request shaping that depends on ModelDatabase data.
- Provider-config fallback when the user omits a model belongs in
  providers/test_provider_default_models.py; ACP max-token regressions belong in
  test_max_tokens_acp_regression.py.
"""

from mcp import Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, OpenAISettings, Settings
from fast_agent.constants import DEFAULT_MAX_ITERATIONS
from fast_agent.context import Context
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import RequestParams
from fast_agent.utils.reasoning_chunk_join import ReasoningTextAccumulator


def test_model_database_context_window_lookup_contract() -> None:
    context_window = ModelDatabase.get_context_window("gpt-5.6")

    assert isinstance(context_window, int)
    assert context_window > 0
    assert ModelDatabase.get_context_window("unknown-model") is None


def test_gpt_56_context_windows_follow_provider_limits() -> None:
    responses_window = ModelDatabase.get_context_window(
        "gpt-5.6",
        provider=Provider.RESPONSES,
    )
    codex_window = ModelDatabase.get_context_window(
        "gpt-5.6",
        provider=Provider.CODEX_RESPONSES,
    )

    assert isinstance(responses_window, int)
    assert isinstance(codex_window, int)
    assert codex_window < responses_window


def test_managed_process_poll_folding_is_enabled_for_gpt5_and_grok45() -> None:
    grok = ModelDatabase.get_model_params(
        "grok-4.5",
        provider=Provider.XAI,
    )
    assert grok is not None
    assert grok.managed_process_poll_folding is True

    gpt5_models = [
        model for model in ModelDatabase.MODELS if model.startswith("gpt-5")
    ]
    assert gpt5_models
    for provider in (Provider.RESPONSES, Provider.CODEX_RESPONSES):
        for model in gpt5_models:
            params = ModelDatabase.get_model_params(
                model,
                provider=provider,
            )
            assert params is not None
            assert params.managed_process_poll_folding is True


def test_glm52_hf_provider_suffix_resolves_without_provider_prefix():
    parsed = ModelFactory.parse_model_spec("zai-org/GLM-5.2:zai-org")

    assert parsed.provider == Provider.HUGGINGFACE
    assert parsed.model_name == "zai-org/GLM-5.2:zai-org"


def test_model_database_long_context_listing_matches_lookup() -> None:
    models = ModelDatabase.list_long_context_models()

    assert models
    assert all(
        ModelDatabase.get_long_context_window(model) is not None
        for model in models
    )
    assert ModelDatabase.get_long_context_window("unknown-model") is None


def test_model_database_fast_listing_matches_lookup() -> None:
    fast_models = ModelDatabase.list_fast_models()

    assert fast_models
    assert all(ModelDatabase.is_fast_model(model) for model in fast_models)
    assert not ModelDatabase.is_fast_model("unknown-model")


def test_model_database_default_provider_lookup_handles_queries_and_unknowns():
    assert ModelDatabase.get_default_provider("gpt-5?reasoning=low") == Provider.RESPONSES
    assert ModelDatabase.get_default_provider("unknown-model") is None


def test_model_database_default_provider_lookup_uses_alias_normalization() -> None:
    assert ModelDatabase.get_default_provider("sonnet") == Provider.ANTHROPIC
    assert ModelDatabase.get_default_provider("gpt-oss") == Provider.HUGGINGFACE
    assert ModelDatabase.get_default_provider("codexspark") == Provider.CODEX_RESPONSES


def test_model_database_default_provider_prefers_exact_slash_catalog_key() -> None:
    params = ModelDatabase.get_model_params("openai/gpt-oss-120b")
    assert params is not None
    assert ModelDatabase.get_default_provider("openai/gpt-oss-120b") == params.default_provider


def test_model_database_provider_qualified_aliases_keep_capabilities() -> None:
    assert ModelDatabase.get_max_output_tokens(
        "anthropic-vertex.sonnet"
    ) == ModelDatabase.get_max_output_tokens("sonnet")


def test_anthropic_catalog_keeps_current_and_vertex_legacy_models() -> None:
    assert ModelDatabase.get_model_params("claude-opus-4-7") is not None
    assert ModelDatabase.get_model_params("claude-opus-4-6") is not None
    assert ModelDatabase.get_model_params("claude-sonnet-4-6") is not None
    assert ModelDatabase.get_model_params("claude-haiku-4-5") is not None

    assert ModelDatabase.get_model_params("claude-3-5-haiku-latest") is not None
    assert ModelDatabase.get_model_params("claude-sonnet-4-20250514") is not None
    assert ModelDatabase.get_model_params("claude-opus-4-20250514") is not None

    assert ModelDatabase.get_model_params("claude-3-haiku-20240307") is None
    assert ModelDatabase.get_model_params("claude-3-5-sonnet-20241022") is None
    assert ModelDatabase.get_model_params("claude-3-7-sonnet-20250219") is None


def _google_native_catalog_entries() -> list[tuple[str, ModelParameters]]:
    entries: list[tuple[str, ModelParameters]] = []
    for model in ModelDatabase.list_models():
        if not model.startswith("gemini-"):
            continue
        if ModelDatabase.get_default_provider(model) != Provider.GOOGLE:
            continue
        params = ModelDatabase.get_model_params(model, provider=Provider.GOOGLE)
        assert params is not None
        entries.append((model, params))
    return entries


def test_google_native_catalog_uses_schema_mode() -> None:
    gemini_entries = _google_native_catalog_entries()

    assert gemini_entries
    assert {params.json_mode for _, params in gemini_entries} == {"schema"}


def test_google_native_catalog_has_no_gemini_25_preview_entries() -> None:
    gemini_models = {model for model, _ in _google_native_catalog_entries()}

    assert {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.5-flash"} <= gemini_models
    assert not {
        model for model in gemini_models if model.startswith("gemini-2.5-") and "preview" in model
    }


def test_google_native_schema_tool_policy_matches_catalog_entries() -> None:
    policies = {params.structured_tool_policy for _, params in _google_native_catalog_entries()}

    assert policies == {None, "no_tools"}


def test_gemini35_flash_specs_match_api_guide() -> None:
    params = ModelDatabase.get_model_params("gemini-3.5-flash")

    assert params is not None
    assert params.context_window == 1_048_576
    assert params.max_output_tokens == 65_536
    assert params.fast is True
    assert params.structured_tool_policy is None
    assert params.reasoning == "google_thinking"
    assert params.reasoning_effort_spec is not None
    assert params.reasoning_effort_spec.default is not None
    assert params.reasoning_effort_spec.default.kind == "effort"
    assert params.reasoning_effort_spec.default.value == "medium"


def test_gemini31_pro_allows_tools_with_structured_output() -> None:
    params = ModelDatabase.get_model_params("gemini-3.1-pro-preview")

    assert params is not None
    assert params.structured_tool_policy is None


def test_huggingface_qwen35_structured_output_uses_prompted_json_object_mode() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    tool = Tool(
        name="lookup",
        description="Lookup data.",
        inputSchema={"type": "object", "properties": {}},
    )
    llm = _make_hf_llm("Qwen/Qwen3.5-397B-A17B")

    prepared_messages, prepared_params = llm._prepare_structured_request(
        [Prompt.user("return json")],
        RequestParams(structured_schema=schema),
        [tool],
    )

    assert llm.resolve_structured_tool_policy(RequestParams(structured_schema=schema)) == "no_tools"
    assert prepared_params.response_format == {"type": "json_object"}
    prepared_text = prepared_messages[-1].last_text()
    assert prepared_text is not None
    assert "YOU MUST RESPOND WITH A JSON OBJECT" in prepared_text


def test_huggingface_qwen36_structured_output_uses_prompt_only() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    llm = _make_hf_llm("Qwen/Qwen3.6-35B-A3B")

    prepared_messages, prepared_params = llm._prepare_structured_request(
        [Prompt.user("return json")],
        RequestParams(structured_schema=schema),
    )

    assert llm.resolve_structured_tool_policy(RequestParams(structured_schema=schema)) == "no_tools"
    assert prepared_params.response_format is None
    prepared_text = prepared_messages[-1].last_text()
    assert prepared_text is not None
    assert "YOU MUST RESPOND WITH A JSON OBJECT" in prepared_text


def test_huggingface_kimi25_uses_schema_mode() -> None:
    params = ModelDatabase.get_model_params("moonshotai/Kimi-K2.5")

    assert params is not None
    assert params.json_mode == "schema"
    assert params.structured_tool_policy is None


def test_huggingface_gemma4_31b_metadata() -> None:
    params = ModelDatabase.get_model_params("google/gemma-4-31B-it:cerebras")

    assert params is not None
    assert params.context_window == 131_000
    assert params.max_output_tokens == 40_000
    assert params.json_mode == "schema"
    assert params.structured_tool_policy == "no_tools"
    assert params.reasoning == "reasoning_content"
    assert params.reasoning_effort_spec is not None
    assert params.reasoning_effort_spec.default is not None
    assert params.reasoning_effort_spec.default.value == "none"
    assert ModelDatabase.supports_mime("google/gemma-4-31B-it", "image/png")
    assert not ModelDatabase.supports_mime("google/gemma-4-31B-it", "audio/mpeg")


def test_model_database_anthropic_web_tool_versions_for_46_models():
    assert (
        ModelDatabase.get_anthropic_web_search_version("claude-opus-4-6") == "web_search_20260209"
    )
    assert ModelDatabase.get_anthropic_web_fetch_version("claude-opus-4-6") == "web_fetch_20260209"
    assert ModelDatabase.get_anthropic_required_betas("claude-opus-4-6") == (
        "code-execution-web-tools-2026-02-09",
    )

    assert (
        ModelDatabase.get_anthropic_web_search_version("claude-sonnet-4-6") == "web_search_20260209"
    )
    assert (
        ModelDatabase.get_anthropic_web_fetch_version("claude-sonnet-4-6") == "web_fetch_20260209"
    )


def test_model_database_anthropic_web_tool_versions_for_non_46_models():
    assert (
        ModelDatabase.get_anthropic_web_search_version("claude-sonnet-4-5") == "web_search_20250305"
    )
    assert (
        ModelDatabase.get_anthropic_web_fetch_version("claude-sonnet-4-5") == "web_fetch_20250910"
    )
    assert ModelDatabase.get_anthropic_required_betas("claude-sonnet-4-5") is None


def test_model_database_anthropic_web_tool_versions_unknown_model():
    assert ModelDatabase.get_anthropic_web_search_version("unknown-model") is None
    assert ModelDatabase.get_anthropic_web_fetch_version("unknown-model") is None
    assert ModelDatabase.get_anthropic_required_betas("unknown-model") is None


def test_model_database_anthropic_task_budget_support() -> None:
    assert ModelDatabase.supports_anthropic_task_budget("claude-opus-4-7") is True
    assert ModelDatabase.supports_anthropic_task_budget("claude-opus-4-6") is False


def test_model_database_anthropic_vertex_caps_are_provider_aware() -> None:
    assert (
        ModelDatabase.get_anthropic_web_search_version(
            "claude-sonnet-4-6",
            provider=Provider.ANTHROPIC_VERTEX,
        )
        == "web_search_20260209"
    )
    assert (
        ModelDatabase.get_anthropic_web_fetch_version(
            "claude-sonnet-4-6",
            provider=Provider.ANTHROPIC_VERTEX,
        )
        is None
    )
    assert ModelDatabase.get_anthropic_required_betas(
        "claude-sonnet-4-6",
        provider=Provider.ANTHROPIC_VERTEX,
    ) == ("code-execution-web-tools-2026-02-09",)
    assert (
        ModelDatabase.get_long_context_window(
            "claude-sonnet-4-5",
            provider=Provider.ANTHROPIC_VERTEX,
        )
        == 1_000_000
    )
    assert not ModelDatabase.supports_mime(
        "claude-sonnet-4-6",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        provider=Provider.ANTHROPIC_VERTEX,
    )
    assert ModelDatabase.supports_mime(
        "claude-sonnet-4-6",
        "application/pdf",
        provider=Provider.ANTHROPIC_VERTEX,
    )


def test_model_database_anthropic_linked_office_docs_are_not_supported() -> None:
    assert not ModelDatabase.supports_mime(
        "claude-sonnet-4-5",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        provider=Provider.ANTHROPIC,
        resource_source="link",
    )
    assert ModelDatabase.supports_mime(
        "claude-sonnet-4-5",
        "image/png",
        provider=Provider.ANTHROPIC,
        resource_source="link",
    )


def test_model_database_max_tokens_lookup_contract() -> None:
    max_tokens = ModelDatabase.get_default_max_tokens("gpt-5.6")

    assert isinstance(max_tokens, int)
    assert max_tokens > 0
    assert ModelDatabase.get_default_max_tokens("unknown-model") is None
    assert ModelDatabase.get_default_max_tokens("") is None


def test_model_database_default_temperature():
    assert ModelDatabase.get_default_temperature("passthrough") == 0.0
    assert ModelDatabase.get_default_temperature("unknown-model") is None
    assert ModelDatabase.get_default_temperature(None) is None


def test_model_database_tokenizes_returns_copy() -> None:
    tokenizes = ModelDatabase.get_tokenizes("claude-sonnet-4-0")
    assert tokenizes is not None

    tokenizes.append("application/x-test")

    assert "application/x-test" not in (ModelDatabase.get_tokenizes("claude-sonnet-4-0") or [])


def test_model_database_supports_mime_basic():
    """Test MIME support lookups with normalization and aliases."""
    # Known multimodal model supports images and pdf
    assert ModelDatabase.supports_mime("claude-sonnet-4-0", "image/png")
    assert ModelDatabase.supports_mime(
        "claude-sonnet-4-0", "document/pdf"
    )  # alias -> application/pdf
    assert ModelDatabase.supports_mime(
        "claude-sonnet-4-0",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert ModelDatabase.supports_mime(
        "gpt-4o",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert not ModelDatabase.supports_mime(
        "gpt-4o",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        provider=Provider.OPENAI,
    )
    assert ModelDatabase.supports_mime("gpt-4o", "document/pdf", provider=Provider.OPENAI)
    assert not ModelDatabase.supports_mime(
        "deepseek-chat",
        "document/pdf",
        provider=Provider.OPENAI,
    )

    # Text-only models should not support images
    assert not ModelDatabase.supports_mime("deepseek-chat", "image/png")
    assert not ModelDatabase.supports_mime("deepseek-chat", "pdf")
    assert not ModelDatabase.supports_mime(
        "deepseek-chat",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Wildcard checks
    assert ModelDatabase.supports_mime("gpt-4o", "image/*")
    # Bare extensions
    assert ModelDatabase.supports_mime("gpt-4o", "png")


def test_model_database_xai_grok_aliases_and_responses_transport():
    assert ModelDatabase.get_default_provider("grok") == Provider.XAI
    assert ModelDatabase.get_default_provider("grok-4.3") == Provider.XAI
    assert ModelDatabase.get_default_provider("grok-4.5") == Provider.XAI

    assert ModelDatabase.get_context_window("grok") == 500_000
    assert ModelDatabase.get_context_window("grok-4.3") == 1_000_000
    assert ModelDatabase.get_context_window("grok-4.5") == 500_000
    assert ModelDatabase.get_model_params("grok-4.3-latest") is None
    assert ModelDatabase.get_model_params("grok-4-fast-reasoning") is None
    assert ModelDatabase.get_model_params("grok-3") is None
    assert ModelDatabase.get_response_transports("grok-4.3") == ("sse", "websocket")
    assert ModelDatabase.supports_response_websocket_provider("grok-4.3", Provider.XAI)


def test_model_database_xai_image_input_mime_types_match_docs():
    vision_model = "grok-4.5"

    assert ModelDatabase.supports_mime(vision_model, "image/jpeg")
    assert ModelDatabase.supports_mime(vision_model, "jpg")
    assert ModelDatabase.supports_mime(vision_model, "image/png")
    assert not ModelDatabase.supports_mime(vision_model, "image/webp")
    assert ModelDatabase.supports_mime("grok-4.3", "image/png")
    assert not ModelDatabase.supports_mime("grok-4.3", "image/webp")


def test_model_database_metaai_muse_spark_metadata():
    assert ModelDatabase.get_default_provider("muse-spark-1.1") == Provider.META_AI
    assert ModelDatabase.get_context_window("muse-spark-1.1") == 1_048_576
    assert ModelDatabase.get_response_transports("muse-spark-1.1") == ("sse",)
    assert not ModelDatabase.supports_response_websocket_provider(
        "muse-spark-1.1",
        Provider.META_AI,
    )
    assert ModelDatabase.supports_mime("muse-spark-1.1", "image/png")
    assert ModelDatabase.supports_mime("muse-spark-1.1", "application/pdf")
    assert ModelDatabase.supports_mime("muse-spark-1.1", "video/mp4")


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
    assert ModelDatabase.supports_response_transport("gpt-5.3-codex", "websocket") is True
    assert ModelDatabase.supports_response_transport("gpt-4o", "websocket") is None


def test_model_database_response_service_tiers() -> None:
    assert ModelDatabase.supports_response_service_tier("gpt-5.3-chat-latest", "flex") is False
    assert ModelDatabase.supports_response_service_tier("gpt-5.4", "flex") is True
    assert ModelDatabase.supports_response_service_tier("gpt-4o", "flex") is None


def test_model_database_response_websocket_provider_support() -> None:
    assert ModelDatabase.supports_response_websocket_provider("gpt-5.4", Provider.RESPONSES) is True
    assert (
        ModelDatabase.supports_response_websocket_provider(
            "gpt-5.3-codex-spark", Provider.RESPONSES
        )
        is False
    )
    assert ModelDatabase.supports_response_websocket_provider("gpt-4o", Provider.RESPONSES) is None


def test_model_database_grok_43_reasoning_spec() -> None:
    spec = ModelDatabase.get_reasoning_effort_spec("grok-4.3")

    assert spec is not None
    assert spec.kind == "effort"
    assert spec.allowed_efforts == ["none", "low", "medium", "high"]
    assert spec.default is not None
    assert spec.default.kind == "effort"
    assert spec.default.value == "low"


def test_glm_51_matches_glm_5_capabilities() -> None:
    old = ModelDatabase.get_model_params("zai-org/glm-5")
    new = ModelDatabase.get_model_params("zai-org/glm-5.1")

    assert old is not None
    assert new is not None
    old_dump = old.model_dump()
    new_dump = new.model_dump()
    old_dump.pop("structured_tool_policy", None)
    new_dump.pop("structured_tool_policy", None)
    assert new_dump == old_dump
    assert new.structured_tool_policy == "no_tools"


def test_model_database_codex_spark_is_text_only() -> None:
    assert ModelDatabase.supports_mime("gpt-5.3-codex-spark", "text/plain")
    assert not ModelDatabase.supports_mime("gpt-5.3-codex-spark", "application/pdf")
    assert not ModelDatabase.supports_mime("gpt-5.3-codex-spark", "image/png")


def test_model_database_opus_47_reasoning_spec():
    """Opus 4.7 should expose adaptive effort settings including xhigh."""
    spec = ModelDatabase.get_reasoning_effort_spec("claude-opus-4-7")
    assert spec is not None
    assert spec.kind == "effort"
    assert spec.allowed_efforts == ["low", "medium", "high", "xhigh", "max"]
    assert spec.allow_toggle_disable


def test_model_database_fable_5_reasoning_spec_is_always_on():
    """Fable 5 adaptive thinking is always on and needs no thinking field."""
    params = ModelDatabase.get_model_params("claude-fable-5")
    spec = ModelDatabase.get_reasoning_effort_spec("claude-fable-5")

    assert params is not None
    assert params.anthropic_thinking_field_required is False
    assert spec is not None
    assert spec.kind == "effort"
    assert spec.allowed_efforts == ["low", "medium", "high", "xhigh"]
    assert spec.allow_auto is True
    assert spec.allow_toggle_disable is False
    assert spec.default is not None
    assert spec.default.value == "auto"


def test_model_database_sonnet_5_reasoning_spec():
    """Sonnet 5 defaults to adaptive thinking, supports disable, and has 128k output."""
    params = ModelDatabase.get_model_params("claude-sonnet-5")
    spec = ModelDatabase.get_reasoning_effort_spec("claude-sonnet-5")

    assert params is not None
    assert params.context_window == 1_000_000
    assert params.max_output_tokens == 128_000
    assert params.anthropic_thinking_field_required is False
    assert params.anthropic_thinking_disable_supported is True
    assert spec is not None
    assert spec.kind == "effort"
    assert spec.allowed_efforts == ["low", "medium", "high", "xhigh", "max"]
    assert spec.allow_auto is True
    assert spec.allow_toggle_disable is True


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
    assert llm._normalize_role(" assistantASSISTANTassistant ") == "assistant"
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


def test_huggingface_glm52_default_preserves_thinking():
    llm = _make_hf_llm("zai-org/glm-5.2:zai-org")

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert args["model"] == "zai-org/glm-5.2:zai-org"
    assert args["reasoning_effort"] == "max"
    assert extra_body["thinking"] == {"type": "enabled", "clear_thinking": False}


def test_huggingface_glm52_default_ignores_openai_reasoning_default():
    settings = Settings(
        hf=HuggingFaceSettings(),
        openai=OpenAISettings(reasoning="medium"),
    )
    context = Context(config=settings)
    llm = HuggingFaceLLM(context=context, model="zai-org/glm-5.2:zai-org", name="test-agent")

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "max"


def test_huggingface_glm52_routes_use_json_object_structured_mode():
    for provider in ("zai-org", "together", "deepinfra", "novita", "fireworks-ai"):
        llm = _make_hf_llm(f"zai-org/glm-5.2:{provider}")

        assert llm._structured_json_mode(llm.default_request_params) == "object"


def test_huggingface_glm52_deepinfra_default_uses_xhigh_reasoning_effort():
    llm = _make_hf_llm("zai-org/glm-5.2:deepinfra")

    args = _hf_request_args(llm)
    assert args["model"] == "zai-org/glm-5.2:deepinfra"
    assert args["reasoning_effort"] == "xhigh"
    assert "extra_body" not in args


def test_huggingface_glm52_deepinfra_xhigh_reasoning_effort():
    llm = _make_hf_llm_with_reasoning("zai-org/glm-5.2:deepinfra", reasoning="xhigh")

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "xhigh"
    assert "extra_body" not in args


def test_huggingface_glm52_deepinfra_disable_reasoning_uses_none_effort():
    llm = _make_hf_llm_with_reasoning("zai-org/glm-5.2:deepinfra", reasoning=False)

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "none"
    assert "extra_body" not in args


def test_huggingface_glm52_fireworks_default_uses_reasoning_effort_only():
    llm = _make_hf_llm("zai-org/glm-5.2:fireworks-ai")

    args = _hf_request_args(llm)
    assert args["model"] == "zai-org/glm-5.2:fireworks-ai"
    assert args["reasoning_effort"] == "max"
    assert "extra_body" not in args


def test_huggingface_glm52_fireworks_passes_requested_reasoning_effort():
    llm = _make_hf_llm_with_reasoning("zai-org/glm-5.2:fireworks-ai", reasoning="low")

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "low"
    assert "extra_body" not in args


def test_huggingface_glm52_fireworks_disable_reasoning_uses_none_effort():
    llm = _make_hf_llm_with_reasoning("zai-org/glm-5.2:fireworks-ai", reasoning=False)

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "none"
    assert "extra_body" not in args


def test_huggingface_glm52_reasoning_effort():
    llm = _make_hf_llm_with_reasoning("zai-org/glm-5.2:zai-org", reasoning="high")

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert args["reasoning_effort"] == "high"
    assert extra_body["thinking"] == {"type": "enabled", "clear_thinking": False}


def test_huggingface_glm52_disable_reasoning():
    llm = _make_hf_llm_with_reasoning("zai-org/glm-5.2:zai-org", reasoning=False)

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert "reasoning_effort" not in args
    assert extra_body["thinking"] == {"type": "disabled"}


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
    if isinstance(extra_body, dict):
        assert "thinking" not in extra_body
    else:
        assert extra_body is None


def test_huggingface_kimi26_disable_reasoning_toggle():
    llm = _make_hf_llm_with_reasoning("moonshotai/kimi-k2.6", reasoning=False)

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"thinking": False}


def test_huggingface_kimi26_default_reasoning_toggle_enabled():
    llm = _make_hf_llm("moonshotai/kimi-k2.6")

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    if isinstance(extra_body, dict):
        assert "chat_template_kwargs" not in extra_body
    else:
        assert extra_body is None


def test_huggingface_qwen35_reasoning_toggle_uses_chat_template_kwargs_disabled():
    llm = _make_hf_llm_with_reasoning("Qwen/Qwen3.5-397B-A17B", reasoning=False)

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_huggingface_qwen35_reasoning_toggle_uses_chat_template_kwargs_enabled():
    llm = _make_hf_llm_with_reasoning("Qwen/Qwen3.5-397B-A17B", reasoning=True)

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": True}


def test_huggingface_gemma4_cerebras_reasoning_effort_enabled():
    llm = _make_hf_llm_with_reasoning("google/gemma-4-31B-it:cerebras", reasoning="medium")

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "medium"
    assert args["model"] == "google/gemma-4-31B-it:cerebras"
    assert "extra_body" not in args


def test_huggingface_gemma4_cerebras_reasoning_effort_disabled():
    llm = _make_hf_llm_with_reasoning("google/gemma-4-31B-it:cerebras", reasoning="none")

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "none"


def test_huggingface_gemma4_cerebras_default_reasoning_is_disabled():
    llm = _make_hf_llm("google/gemma-4-31B-it:cerebras")

    args = _hf_request_args(llm)
    assert args["reasoning_effort"] == "none"


def test_huggingface_chat_template_kwargs_helper_preserves_existing_values() -> None:
    extra_body: dict[str, object] = {"chat_template_kwargs": {"temperature": 0.2}}

    HuggingFaceLLM._set_chat_template_kwarg(extra_body, "enable_thinking", False)

    assert extra_body["chat_template_kwargs"] == {
        "temperature": 0.2,
        "enable_thinking": False,
    }


def test_huggingface_qwen35_default_reasoning_emits_chat_template_kwargs_enabled():
    llm = _make_hf_llm("Qwen/Qwen3.5-397B-A17B")

    args = _hf_request_args(llm)
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": True}


def test_huggingface_deepseek_v4_pro_uses_reasoning_content_streaming_metadata():
    llm = _make_hf_llm("deepseek-ai/DeepSeek-V4-Pro:fireworks-ai")

    assert llm.default_request_params.model == "deepseek-ai/DeepSeek-V4-Pro"
    assert llm.default_request_params.maxTokens == 393_216
    assert getattr(llm, "_reasoning_mode", None) == "reasoning_content"

    args = _hf_request_args(llm)
    assert args["model"] == "deepseek-ai/DeepSeek-V4-Pro:fireworks-ai"
    assert args["max_tokens"] == 393_216


def test_huggingface_qwen35_reasoning_stream_hidden_when_disabled():
    llm = _make_hf_llm_with_reasoning("Qwen/Qwen3.5-397B-A17B", reasoning=False)

    segments = ReasoningTextAccumulator()
    active = llm._handle_reasoning_delta(
        reasoning_mode="reasoning_content",
        reasoning_text="hidden reasoning",
        reasoning_active=False,
        reasoning_segments=segments,
    )

    assert active is False
    assert segments.parts() == []


def test_huggingface_qwen35_reasoning_stream_visible_when_enabled():
    llm = _make_hf_llm_with_reasoning("Qwen/Qwen3.5-397B-A17B", reasoning=True)

    segments = ReasoningTextAccumulator()
    active = llm._handle_reasoning_delta(
        reasoning_mode="reasoning_content",
        reasoning_text="visible reasoning",
        reasoning_active=False,
        reasoning_segments=segments,
    )

    assert active is False
    assert segments.parts() == ["visible reasoning"]


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


def test_model_specific_defaults_for_gpt_53_plus_family():
    expected = "Before making tool calls, send a brief preamble to the user explaining what you’re about to do."

    for model_name in (
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.3-chat-latest",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
    ):
        assert ModelDatabase.get_model_specific(model_name) == expected

    assert ModelDatabase.get_model_specific("gpt-5.2") == ""


def test_gpt_54_mini_default_reasoning_effort_is_medium():
    spec = ModelDatabase.get_reasoning_effort_spec("gpt-5.4-mini")

    assert spec is not None
    assert spec.default == ReasoningEffortSetting(kind="effort", value="medium")


def test_gpt_56_supports_api_reasoning_efforts_through_max():
    for model_name in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        spec = ModelDatabase.get_reasoning_effort_spec(model_name)

        assert spec is not None
        assert spec.allowed_efforts == ["none", "low", "medium", "high", "xhigh", "max"]
        assert spec.default == ReasoningEffortSetting(kind="effort", value="medium")
        assert ModelDatabase.supports_mime(model_name, "application/pdf")
        assert ModelDatabase.supports_mime(model_name, "image/png")

    assert ModelDatabase.uses_codex_responses_lite("gpt-5.6-luna")
    assert not ModelDatabase.uses_codex_responses_lite("gpt-5.6")
    assert not ModelDatabase.uses_codex_responses_lite("gpt-5.6-sol")
    assert not ModelDatabase.uses_codex_responses_lite("gpt-5.6-terra")
    assert not ModelDatabase.uses_codex_responses_lite("gpt-5.5")


def test_gemini_model_specific_mentions_youtube_capability():
    for model_name in (
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-3.5-flash",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ):
        model_specific = ModelDatabase.get_model_specific(model_name)
        assert "YouTube" in model_specific
        assert "capable" in model_specific
        assert "free" not in model_specific.lower()
        assert ModelDatabase.supports_mime(model_name, "video/mp4")
