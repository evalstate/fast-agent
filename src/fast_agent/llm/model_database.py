"""
Model database for LLM parameters.

This module provides a centralized lookup for model parameters including
context windows, max output tokens, and supported tokenization types.
"""

from typing import Literal

from pydantic import BaseModel

from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import (
    AUTO_REASONING,
    ReasoningEffortSetting,
    ReasoningEffortSpec,
)
from fast_agent.llm.text_verbosity import TextVerbositySpec


class ModelParameters(BaseModel):
    """Configuration parameters for a specific model"""

    context_window: int
    """Maximum context window size in tokens"""

    max_output_tokens: int
    """Maximum output tokens the model can generate"""

    tokenizes: list[str]
    """List of supported content types for tokenization"""

    json_mode: None | str = "schema"
    """Structured output style. 'schema', 'object' or None for unsupported """

    reasoning: None | str = None
    """Reasoning output style. 'tags' if enclosed in <thinking> tags, 'none' if not used"""

    reasoning_effort_spec: ReasoningEffortSpec | None = None
    """Reasoning effort input configuration supported by the model, if any."""

    text_verbosity_spec: TextVerbositySpec | None = None
    """Text verbosity configuration supported by the model, if any."""

    stream_mode: Literal["openai", "manual"] = "openai"
    """Determines how streaming deltas should be processed."""

    system_role: None | str = "system"
    """Role to use for the System Prompt"""

    cache_ttl: Literal["5m", "1h"] | None = None
    """Cache TTL for providers that support caching. None if not supported."""

    long_context_window: int | None = None
    """Optional extended context window when explicitly requested by query params."""

    response_transports: tuple[Literal["sse", "websocket"], ...] | None = None
    """Supported transports for Responses APIs, if the model exposes alternatives."""

    response_websocket_providers: tuple[Provider, ...] | None = None
    """Providers allowed to use websocket transport for this Responses model."""

    anthropic_web_search_version: str | None = None
    """Anthropic built-in web_search tool version, if supported by the model."""

    anthropic_web_fetch_version: str | None = None
    """Anthropic built-in web_fetch tool version, if supported by the model."""

    anthropic_required_betas: tuple[str, ...] | None = None
    """Anthropic beta headers required for model-specific server tool support."""

    default_temperature: float | None = None
    """Optional default sampling temperature for this model."""

    default_provider: Provider | None = None
    """Default provider used when model is referenced without an explicit prefix."""

    fast: bool = False
    """Whether this model is recommended for fast/simple tasks."""


def _with_fast(params: ModelParameters) -> ModelParameters:
    """Return a model variant marked as fast."""
    return params.model_copy(update={"fast": True})


def _with_long_context(params: ModelParameters, window: int) -> ModelParameters:
    """Return a model variant with an explicit long-context window override."""
    return params.model_copy(update={"long_context_window": window})


class ModelDatabase:
    """Centralized model configuration database"""

    _RUNTIME_MODEL_DEFAULT_PROVIDERS: dict[str, Provider] = {}
    _RUNTIME_MODEL_PARAMS: dict[str, ModelParameters] = {}

    # Common parameter sets
    OPENAI_MULTIMODAL = ["text/plain", "image/jpeg", "image/png", "image/webp", "application/pdf"]
    OPENAI_VISION = ["text/plain", "image/jpeg", "image/png", "image/webp"]
    ANTHROPIC_MULTIMODAL = [
        "text/plain",
        "image/jpeg",
        "image/png",
        "image/webp",
        "application/pdf",
    ]
    GOOGLE_MULTIMODAL = [
        "text/plain",
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "application/pdf",
        # Audio formats
        "audio/wav",
        "audio/mpeg",  # Official MP3 MIME type
        "audio/mp3",  # Common alias
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/webm",
        # Video formats (MP4, AVI, FLV, MOV, MPEG, MPG, WebM)
        "video/mp4",
        "video/x-msvideo",  # AVI
        "video/x-flv",  # FLV
        "video/quicktime",  # MOV
        "video/mpeg",  # MPEG, MPG
        "video/webm",
    ]
    QWEN_MULTIMODAL = ["text/plain", "image/jpeg", "image/png", "image/webp"]
    XAI_VISION = ["text/plain", "image/jpeg", "image/png", "image/webp"]
    TEXT_ONLY = ["text/plain"]

    OPENAI_O_CLASS_REASONING = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    OPENAI_GPT_5_CLASS_REASONING = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["minimal", "low", "medium", "high"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    OPENAI_GPT_51_CLASS_REASONING = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["none", "low", "medium", "high", "xhigh"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    OPENAI_GPT_5_CODEX_CLASS_REASONING = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "xhigh"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    OPENAI_REASONING_EFFORT_SPEC = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["minimal", "low", "medium", "high", "xhigh"],
        default=ReasoningEffortSetting(kind="effort", value="medium"),
    )

    OPENAI_TEXT_VERBOSITY_SPEC = TextVerbositySpec()

    GLM_REASONING_TOGGLE_SPEC = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )

    KIMI_REASONING_TOGGLE_SPEC = ReasoningEffortSpec(
        kind="toggle",
        default=ReasoningEffortSetting(kind="toggle", value=True),
    )

    ANTHROPIC_THINKING_EFFORT_SPEC = ReasoningEffortSpec(
        kind="budget",
        min_budget_tokens=1024,
        max_budget_tokens=128000,
        budget_presets=[0, 1024, 16000, 32000],
        default=ReasoningEffortSetting(kind="budget", value=1024),
    )

    ANTHROPIC_ADAPTIVE_THINKING_EFFORT_SPEC = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "max"],
        allow_toggle_disable=True,
        allow_auto=True,
        default=ReasoningEffortSetting(kind="effort", value=AUTO_REASONING),
    )

    ANTHROPIC_WEB_SEARCH_LEGACY = "web_search_20250305"
    ANTHROPIC_WEB_FETCH_LEGACY = "web_fetch_20250910"
    ANTHROPIC_WEB_SEARCH_46 = "web_search_20260209"
    ANTHROPIC_WEB_FETCH_46 = "web_fetch_20260209"
    ANTHROPIC_WEB_TOOLS_BETA_46 = "code-execution-web-tools-2026-02-09"

    # Common parameter configurations
    OPENAI_STANDARD = ModelParameters(
        context_window=128000,
        max_output_tokens=16384,
        tokenizes=OPENAI_MULTIMODAL,
        default_provider=Provider.OPENAI,
    )

    OPENAI_4_1_STANDARD = ModelParameters(
        context_window=1047576,
        max_output_tokens=32768,
        tokenizes=OPENAI_MULTIMODAL,
        default_provider=Provider.OPENAI,
    )

    OPENAI_O_SERIES = ModelParameters(
        context_window=200000,
        max_output_tokens=100000,
        tokenizes=OPENAI_VISION,
        reasoning="openai",
        reasoning_effort_spec=OPENAI_REASONING_EFFORT_SPEC,
        default_provider=Provider.RESPONSES,
    )

    ANTHROPIC_LEGACY = ModelParameters(
        context_window=200000,
        max_output_tokens=4096,
        tokenizes=ANTHROPIC_MULTIMODAL,
        json_mode=None,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )

    ANTHROPIC_35_SERIES = ModelParameters(
        context_window=200000,
        max_output_tokens=8192,
        tokenizes=ANTHROPIC_MULTIMODAL,
        json_mode=None,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )

    # TODO--- TO USE 64,000 NEED TO SUPPORT STREAMING
    ANTHROPIC_37_SERIES = ModelParameters(
        context_window=200000,
        max_output_tokens=16384,
        tokenizes=ANTHROPIC_MULTIMODAL,
        json_mode=None,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )

    QWEN_STANDARD = ModelParameters(
        context_window=32000,
        max_output_tokens=8192,
        tokenizes=QWEN_MULTIMODAL,
        json_mode="object",
        default_provider=Provider.ALIYUN,
    )
    QWEN3_REASONER = ModelParameters(
        context_window=131072,
        max_output_tokens=16384,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="tags",
    )

    FAST_AGENT_STANDARD = ModelParameters(
        context_window=1000000,
        max_output_tokens=100000,
        tokenizes=TEXT_ONLY,
        default_temperature=0.0,
        default_provider=Provider.FAST_AGENT,
    )

    OPENAI_4_1_SERIES = ModelParameters(
        context_window=1047576,
        max_output_tokens=32768,
        tokenizes=OPENAI_MULTIMODAL,
        default_provider=Provider.OPENAI,
    )

    OPENAI_4O_SERIES = ModelParameters(
        context_window=128000,
        max_output_tokens=16384,
        tokenizes=OPENAI_MULTIMODAL,
        default_provider=Provider.OPENAI,
    )

    OPENAI_O3_SERIES = ModelParameters(
        context_window=200000,
        max_output_tokens=100000,
        tokenizes=OPENAI_MULTIMODAL,
        reasoning="openai",
        reasoning_effort_spec=OPENAI_O_CLASS_REASONING,
        default_provider=Provider.RESPONSES,
    )

    OPENAI_O3_MINI_SERIES = ModelParameters(
        context_window=200000,
        max_output_tokens=100000,
        tokenizes=TEXT_ONLY,
        reasoning="openai",
        reasoning_effort_spec=OPENAI_O_CLASS_REASONING,
        default_provider=Provider.RESPONSES,
    )
    OPENAI_GPT_OSS_SERIES = ModelParameters(
        context_window=131072,
        max_output_tokens=32766,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="gpt_oss",
    )
    OPENAI_GPT_5 = ModelParameters(
        context_window=400000,
        max_output_tokens=128000,
        tokenizes=OPENAI_MULTIMODAL,
        reasoning="openai",
        reasoning_effort_spec=OPENAI_GPT_5_CLASS_REASONING,
        text_verbosity_spec=OPENAI_TEXT_VERBOSITY_SPEC,
        default_provider=Provider.RESPONSES,
    )

    OPENAI_GPT_5_2 = ModelParameters(
        context_window=400000,
        max_output_tokens=128000,
        tokenizes=OPENAI_MULTIMODAL,
        reasoning="openai",
        reasoning_effort_spec=OPENAI_GPT_51_CLASS_REASONING,
        text_verbosity_spec=OPENAI_TEXT_VERBOSITY_SPEC,
        default_provider=Provider.RESPONSES,
    )

    OPENAI_GPT_CODEX = ModelParameters(
        context_window=400000,
        max_output_tokens=128000,
        tokenizes=OPENAI_MULTIMODAL,
        reasoning="openai",
        reasoning_effort_spec=OPENAI_GPT_5_CODEX_CLASS_REASONING,
        text_verbosity_spec=OPENAI_TEXT_VERBOSITY_SPEC,
        response_transports=("sse", "websocket"),
        response_websocket_providers=(Provider.RESPONSES, Provider.CODEX_RESPONSES),
        default_provider=Provider.RESPONSES,
    )

    OPENAI_GPT_CODEX_SPARK = ModelParameters(
        context_window=128000,
        max_output_tokens=128000,
        tokenizes=TEXT_ONLY,
        # Spark does not support reasoning effort or text verbosity controls.
        response_transports=("sse", "websocket"),
        response_websocket_providers=(Provider.CODEX_RESPONSES,),
        default_provider=Provider.CODEX_RESPONSES,
    )

    ANTHROPIC_OPUS_4_VERSIONED = ModelParameters(
        context_window=200000,
        max_output_tokens=32000,
        tokenizes=ANTHROPIC_MULTIMODAL,
        reasoning="anthropic_thinking",
        reasoning_effort_spec=ANTHROPIC_THINKING_EFFORT_SPEC,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )
    ANTHROPIC_OPUS_46 = ModelParameters(
        context_window=200000,
        max_output_tokens=128000,
        tokenizes=ANTHROPIC_MULTIMODAL,
        reasoning="anthropic_thinking",
        reasoning_effort_spec=ANTHROPIC_ADAPTIVE_THINKING_EFFORT_SPEC,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_46,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_46,
        anthropic_required_betas=(ANTHROPIC_WEB_TOOLS_BETA_46,),
        default_provider=Provider.ANTHROPIC,
    )
    ANTHROPIC_OPUS_4_LEGACY = ModelParameters(
        context_window=200000,
        max_output_tokens=32000,
        tokenizes=ANTHROPIC_MULTIMODAL,
        reasoning="anthropic_thinking",
        reasoning_effort_spec=ANTHROPIC_THINKING_EFFORT_SPEC,
        json_mode=None,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )
    ANTHROPIC_SONNET_4_VERSIONED = ModelParameters(
        context_window=200000,
        max_output_tokens=64000,
        tokenizes=ANTHROPIC_MULTIMODAL,
        reasoning="anthropic_thinking",
        reasoning_effort_spec=ANTHROPIC_THINKING_EFFORT_SPEC,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )
    ANTHROPIC_SONNET_46 = ModelParameters(
        context_window=200000,
        max_output_tokens=64000,
        tokenizes=ANTHROPIC_MULTIMODAL,
        reasoning="anthropic_thinking",
        reasoning_effort_spec=ANTHROPIC_ADAPTIVE_THINKING_EFFORT_SPEC,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_46,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_46,
        anthropic_required_betas=(ANTHROPIC_WEB_TOOLS_BETA_46,),
        default_provider=Provider.ANTHROPIC,
    )

    ANTHROPIC_SONNET_4_LEGACY = ModelParameters(
        context_window=200000,
        max_output_tokens=64000,
        tokenizes=ANTHROPIC_MULTIMODAL,
        reasoning="anthropic_thinking",
        reasoning_effort_spec=ANTHROPIC_THINKING_EFFORT_SPEC,
        json_mode=None,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )
    # Claude 3.7 Sonnet supports extended thinking (deprecated but still available)
    ANTHROPIC_37_SERIES_THINKING = ModelParameters(
        context_window=200000,
        max_output_tokens=16384,
        tokenizes=ANTHROPIC_MULTIMODAL,
        reasoning="anthropic_thinking",
        reasoning_effort_spec=ANTHROPIC_THINKING_EFFORT_SPEC,
        json_mode=None,
        cache_ttl="5m",
        anthropic_web_search_version=ANTHROPIC_WEB_SEARCH_LEGACY,
        anthropic_web_fetch_version=ANTHROPIC_WEB_FETCH_LEGACY,
        default_provider=Provider.ANTHROPIC,
    )

    DEEPSEEK_CHAT_STANDARD = ModelParameters(
        context_window=65536,
        max_output_tokens=8192,
        tokenizes=TEXT_ONLY,
        default_provider=Provider.DEEPSEEK,
    )

    DEEPSEEK_REASONER = ModelParameters(
        context_window=65536, max_output_tokens=32768, tokenizes=TEXT_ONLY
    )

    DEEPSEEK_V_32 = ModelParameters(
        context_window=65536,
        max_output_tokens=32768,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="gpt-oss",
        system_role="developer",
    )

    DEEPSEEK_DISTILL = ModelParameters(
        context_window=131072,
        max_output_tokens=131072,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="tags",
    )

    GEMINI_STANDARD = ModelParameters(
        context_window=1_048_576,
        max_output_tokens=65_536,
        tokenizes=GOOGLE_MULTIMODAL,
        default_provider=Provider.GOOGLE,
    )

    GEMINI_2_FLASH = ModelParameters(
        context_window=1_048_576,
        max_output_tokens=8192,
        tokenizes=GOOGLE_MULTIMODAL,
        default_provider=Provider.GOOGLE,
    )

    # 31/08/25 switched to object mode (even though groq says schema supported and used to work..)
    KIMI_MOONSHOT = ModelParameters(
        context_window=262144,
        max_output_tokens=16384,
        tokenizes=TEXT_ONLY,
        json_mode="object",
    )
    KIMI_MOONSHOT_THINKING = ModelParameters(
        context_window=262144,
        max_output_tokens=16384,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="reasoning_content",
    )
    KIMI_MOONSHOT_25 = ModelParameters(
        context_window=262144,
        max_output_tokens=16384,
        tokenizes=OPENAI_VISION,
        json_mode="schema",
        reasoning="reasoning_content",
        reasoning_effort_spec=KIMI_REASONING_TOGGLE_SPEC,
    )
    # FIXME: xAI has not documented the max output tokens for Grok 4. Using Grok 3 as a placeholder. Will need to update when available (if ever)
    GROK_4 = ModelParameters(
        context_window=256000,
        max_output_tokens=16385,
        tokenizes=TEXT_ONLY,
        default_provider=Provider.XAI,
    )

    GROK_4_VLM = ModelParameters(
        context_window=2000000,
        max_output_tokens=16385,
        tokenizes=XAI_VISION,
        default_provider=Provider.XAI,
    )

    # Source for Grok 3 max output: https://www.reddit.com/r/grok/comments/1j7209p/exploring_grok_3_beta_output_capacity_a_simple/
    # xAI does not document Grok 3 max output tokens, using the above source as a reference.
    GROK_3 = ModelParameters(
        context_window=131072,
        max_output_tokens=16385,
        tokenizes=TEXT_ONLY,
        default_provider=Provider.XAI,
    )

    # H U G G I N G F A C E - max output tokens are not documented, using 16k as a reasonable default
    GLM_46 = ModelParameters(
        context_window=202752,
        max_output_tokens=8192,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="reasoning_content",
        stream_mode="manual",
    )

    GLM_47 = ModelParameters(
        context_window=202752,
        max_output_tokens=65536,  # default from https://docs.z.ai/guides/overview/concept-param#token-usage-calculation - max is 131072
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="reasoning_content",
        reasoning_effort_spec=GLM_REASONING_TOGGLE_SPEC,
        stream_mode="manual",
    )

    GLM_5 = ModelParameters(
        context_window=202800,
        max_output_tokens=131072,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="reasoning_content",
        reasoning_effort_spec=GLM_REASONING_TOGGLE_SPEC,
        stream_mode="manual",
    )

    MINIMAX_21 = ModelParameters(
        context_window=202752,
        max_output_tokens=131072,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="reasoning_content",
        stream_mode="manual",
    )

    HF_PROVIDER_DEEPSEEK31 = ModelParameters(
        context_window=163_800, max_output_tokens=8192, tokenizes=TEXT_ONLY
    )

    HF_PROVIDER_DEEPSEEK32 = ModelParameters(
        context_window=163_800,
        max_output_tokens=8192,
        tokenizes=TEXT_ONLY,
        reasoning="gpt_oss",
    )

    HF_PROVIDER_QWEN3_NEXT = ModelParameters(
        context_window=262_000, max_output_tokens=8192, tokenizes=TEXT_ONLY
    )

    HF_PROVIDER_QWEN35 = ModelParameters(
        context_window=262_144,
        max_output_tokens=65_536,
        tokenizes=QWEN_MULTIMODAL,
        reasoning="reasoning_content",
        reasoning_effort_spec=GLM_REASONING_TOGGLE_SPEC,
        default_provider=Provider.HUGGINGFACE,
    )

    ALIYUN_QWEN3_MODERN = ModelParameters(
        context_window=256_000,
        max_output_tokens=64_000,
        tokenizes=TEXT_ONLY,
        default_provider=Provider.ALIYUN,
    )

    ANTHROPIC_LONG_CONTEXT_WINDOW = 1_000_000

    # Model configuration database
    # KEEP ALL LOWER CASE KEYS
    MODELS: dict[str, ModelParameters] = {
        # internal models
        "passthrough": FAST_AGENT_STANDARD,
        "silent": FAST_AGENT_STANDARD,
        "playback": FAST_AGENT_STANDARD,
        "slow": FAST_AGENT_STANDARD,
        # aliyun models
        "qwen-turbo": _with_fast(QWEN_STANDARD),
        "qwen-plus": QWEN_STANDARD,
        "qwen-max": QWEN_STANDARD,
        "qwen-long": ModelParameters(
            context_window=10000000,
            max_output_tokens=8192,
            tokenizes=TEXT_ONLY,
            default_provider=Provider.ALIYUN,
        ),
        # OpenAI Models (vanilla aliases and versioned)
        "gpt-4.1": OPENAI_4_1_SERIES,
        "gpt-4.1-mini": _with_fast(OPENAI_4_1_SERIES),
        "gpt-4.1-nano": _with_fast(OPENAI_4_1_SERIES),
        "gpt-4.1-2025-04-14": OPENAI_4_1_SERIES,
        "gpt-4.1-mini-2025-04-14": OPENAI_4_1_SERIES,
        "gpt-4.1-nano-2025-04-14": OPENAI_4_1_SERIES,
        "gpt-4o": OPENAI_4O_SERIES,
        "gpt-4o-mini": OPENAI_4O_SERIES,
        "gpt-4o-2024-11-20": OPENAI_4O_SERIES,
        "gpt-4o-mini-2024-07-18": OPENAI_4O_SERIES,
        "o1": OPENAI_O_SERIES,
        "o1-mini": OPENAI_O_SERIES,
        "o1-preview": OPENAI_O_SERIES,
        "o1-2024-12-17": OPENAI_O_SERIES,
        "o3": OPENAI_O3_SERIES,
        "o3-pro": ModelParameters(
            context_window=200_000, max_output_tokens=100_000, tokenizes=TEXT_ONLY
        ),
        "o3-mini": OPENAI_O3_MINI_SERIES,
        "o4-mini": OPENAI_O3_SERIES,
        "o3-2025-04-16": OPENAI_O3_SERIES,
        "o3-mini-2025-01-31": OPENAI_O3_MINI_SERIES,
        "o4-mini-2025-04-16": OPENAI_O3_SERIES,
        "gpt-5": OPENAI_GPT_5,
        "gpt-5-mini": _with_fast(OPENAI_GPT_5),
        "gpt-5-nano": _with_fast(OPENAI_GPT_5),
        "gpt-5-nano-2025-08-07": _with_fast(OPENAI_GPT_5),
        "gpt-5.1": OPENAI_GPT_5_2,
        "gpt-5.1-codex": OPENAI_GPT_CODEX,
        "gpt-5.2-codex": OPENAI_GPT_CODEX,
        "gpt-5.3-codex": OPENAI_GPT_CODEX,
        "gpt-5.3-codex-spark": _with_fast(OPENAI_GPT_CODEX_SPARK),
        "gpt-5.2": OPENAI_GPT_5_2.model_copy(
            update={
                "response_transports": ("sse", "websocket"),
                "response_websocket_providers": (Provider.RESPONSES,),
            }
        ),
        # Anthropic Models
        "claude-3-haiku": ANTHROPIC_35_SERIES,
        "claude-3-haiku-20240307": ANTHROPIC_LEGACY,
        "claude-3-sonnet": ANTHROPIC_LEGACY,
        "claude-3-opus": ANTHROPIC_LEGACY,
        "claude-3-opus-20240229": ANTHROPIC_LEGACY,
        "claude-3-opus-latest": ANTHROPIC_LEGACY,
        "claude-3-5-haiku": ANTHROPIC_35_SERIES,
        "claude-3-5-haiku-20241022": ANTHROPIC_35_SERIES,
        "claude-3-5-haiku-latest": _with_fast(ANTHROPIC_35_SERIES),
        "claude-3-sonnet-20240229": ANTHROPIC_LEGACY,
        "claude-3-5-sonnet": ANTHROPIC_35_SERIES,
        "claude-3-5-sonnet-20240620": ANTHROPIC_35_SERIES,
        "claude-3-5-sonnet-20241022": ANTHROPIC_35_SERIES,
        "claude-3-5-sonnet-latest": ANTHROPIC_35_SERIES,
        "claude-3-7-sonnet": ANTHROPIC_37_SERIES_THINKING,
        "claude-3-7-sonnet-20250219": ANTHROPIC_37_SERIES_THINKING,
        "claude-3-7-sonnet-latest": ANTHROPIC_37_SERIES_THINKING,
        "claude-sonnet-4-0": _with_long_context(
            ANTHROPIC_SONNET_4_LEGACY, ANTHROPIC_LONG_CONTEXT_WINDOW
        ),
        "claude-sonnet-4-20250514": _with_long_context(
            ANTHROPIC_SONNET_4_LEGACY, ANTHROPIC_LONG_CONTEXT_WINDOW
        ),
        "claude-sonnet-4-5": _with_long_context(
            ANTHROPIC_SONNET_4_VERSIONED, ANTHROPIC_LONG_CONTEXT_WINDOW
        ),
        "claude-sonnet-4-5-20250929": _with_long_context(
            ANTHROPIC_SONNET_4_VERSIONED, ANTHROPIC_LONG_CONTEXT_WINDOW
        ),
        "claude-sonnet-4-6": _with_long_context(ANTHROPIC_SONNET_46, ANTHROPIC_LONG_CONTEXT_WINDOW),
        "claude-opus-4-0": ANTHROPIC_OPUS_4_LEGACY,
        "claude-opus-4-1": ANTHROPIC_OPUS_4_VERSIONED,
        "claude-opus-4-5": ANTHROPIC_OPUS_4_VERSIONED,
        "claude-opus-4-6": _with_long_context(ANTHROPIC_OPUS_46, ANTHROPIC_LONG_CONTEXT_WINDOW),
        "claude-opus-4-20250514": ANTHROPIC_OPUS_4_LEGACY,
        "claude-haiku-4-5-20251001": ANTHROPIC_SONNET_4_VERSIONED,
        "claude-haiku-4-5": _with_fast(ANTHROPIC_SONNET_4_VERSIONED),
        # DeepSeek Models
        "deepseek-chat": _with_fast(DEEPSEEK_CHAT_STANDARD),
        # Google Gemini Models (vanilla aliases and versioned)
        "gemini-2.0-flash": _with_fast(GEMINI_2_FLASH),
        "gemini-2.5-flash-preview": GEMINI_STANDARD,
        "gemini-2.5-pro-preview": GEMINI_STANDARD,
        "gemini-2.5-flash-preview-05-20": GEMINI_STANDARD,
        "gemini-2.5-pro-preview-05-06": GEMINI_STANDARD,
        "gemini-2.5-pro": GEMINI_STANDARD,
        "gemini-2.5-flash-preview-09-2025": _with_fast(GEMINI_STANDARD),
        "gemini-2.5-flash": _with_fast(GEMINI_STANDARD),
        "gemini-3-pro-preview": GEMINI_STANDARD,
        "gemini-3-flash-preview": GEMINI_STANDARD,
        "gemini-3.1-pro-preview": GEMINI_STANDARD,
        # xAI Grok Models
        "grok-4-1-fast-reasoning": GROK_4_VLM,
        "grok-4-1-fast-non-reasoning": GROK_4_VLM,
        "grok-4-fast-reasoning": GROK_4_VLM,
        "grok-4-fast-non-reasoning": GROK_4_VLM,
        "grok-4": GROK_4,
        "grok-4-0709": GROK_4,
        "grok-3": GROK_3,
        "grok-3-mini": GROK_3,
        "grok-3-fast": GROK_3,
        "grok-3-mini-fast": _with_fast(GROK_3),
        "moonshotai/kimi-k2": KIMI_MOONSHOT,
        "moonshotai/kimi-k2-instruct-0905": _with_fast(KIMI_MOONSHOT),
        "moonshotai/kimi-k2-thinking": KIMI_MOONSHOT_THINKING,
        "moonshotai/kimi-k2-thinking-0905": KIMI_MOONSHOT_THINKING,
        "moonshotai/kimi-k2.5": KIMI_MOONSHOT_25,
        "qwen/qwen3-32b": QWEN3_REASONER,
        "deepseek-r1-distill-llama-70b": DEEPSEEK_DISTILL,
        "openai/gpt-oss-120b": OPENAI_GPT_OSS_SERIES,  # https://cookbook.openai.com/articles/openai-harmony
        "openai/gpt-oss-20b": OPENAI_GPT_OSS_SERIES,  # tool/reasoning interleave guidance
        "zai-org/glm-4.6": GLM_46,
        "zai-org/glm-4.7": GLM_47,
        "zai-org/glm-5": _with_fast(GLM_5),
        "minimaxai/minimax-m2": GLM_46,
        "minimaxai/minimax-m2.1": MINIMAX_21,
        "qwen/qwen3-next-80b-a3b-instruct": HF_PROVIDER_QWEN3_NEXT,
        "qwen/qwen3.5-397b-a17b": HF_PROVIDER_QWEN35,
        "deepseek-ai/deepseek-v3.1": HF_PROVIDER_DEEPSEEK31,
        "deepseek-ai/deepseek-v3.2": HF_PROVIDER_DEEPSEEK32,
        # aliyun modern
        "qwen3-max": ALIYUN_QWEN3_MODERN,
    }

    @classmethod
    def get_model_params(cls, model: str) -> ModelParameters | None:
        """Get model parameters for a given model name"""
        if not model:
            return None

        normalized = cls.normalize_model_name(model)
        params = cls.MODELS.get(normalized)
        if params is not None:
            return params
        return cls._RUNTIME_MODEL_PARAMS.get(normalized)

    @classmethod
    def normalize_model_name(cls, model: str) -> str:
        """Normalize model specs (provider/effort/aliases) to a ModelDatabase key.

        This intentionally delegates to ModelFactory parsing where possible rather than
        re-implementing model string semantics in the database layer.
        """
        from fast_agent.core.exceptions import ModelConfigError
        from fast_agent.llm.model_factory import ModelFactory
        from fast_agent.llm.provider_types import Provider

        model_spec = (model or "").strip()
        if not model_spec:
            return ""

        if "?" in model_spec:
            model_spec = model_spec.split("?", 1)[0].strip()

        # If it's already a known key, keep it as-is (after casing/whitespace normalization).
        direct_key = model_spec.lower()
        if direct_key in cls.MODELS:
            return direct_key

        # Apply aliases first (case-insensitive).
        aliased = ModelFactory.MODEL_ALIASES.get(model_spec)
        if not aliased:
            aliased = ModelFactory.MODEL_ALIASES.get(model_spec.lower())
        if aliased:
            model_spec = aliased
            direct_key = model_spec.strip().lower()
            if direct_key in cls.MODELS:
                return direct_key

        # Parse known spec formats to strip provider prefixes and reasoning effort.
        try:
            parsed = ModelFactory.parse_model_string(model_spec)
            model_spec = parsed.model_name

            # HF uses `model:provider` for routing; the suffix is not part of the model id.
            if parsed.provider == Provider.HUGGINGFACE and ":" in model_spec:
                model_spec = model_spec.rsplit(":", 1)[0]
        except ModelConfigError:
            # Best-effort fallback: keep original spec if it can't be parsed.
            pass

        # If parsing failed, still support common "model:route" forms by stripping the suffix
        # only when the base resolves to a known database key.
        if ":" in model_spec:
            base = model_spec.rsplit(":", 1)[0].strip().lower()
            if base in cls.MODELS:
                return base

        return model_spec.strip().lower()

    @classmethod
    def get_context_window(cls, model: str) -> int | None:
        """Get context window size for a model"""
        params = cls.get_model_params(model)
        return params.context_window if params else None

    @classmethod
    def get_max_output_tokens(cls, model: str) -> int | None:
        """Get maximum output tokens for a model"""
        params = cls.get_model_params(model)
        return params.max_output_tokens if params else None

    @classmethod
    def get_tokenizes(cls, model: str) -> list[str] | None:
        """Get supported tokenization types for a model"""
        params = cls.get_model_params(model)
        return params.tokenizes if params else None

    @classmethod
    def supports_mime(cls, model: str, mime_type: str) -> bool:
        """
        Return True if the given model supports the provided MIME type.

        Normalizes common aliases (e.g., image/jpg->image/jpeg, document/pdf->application/pdf)
        and also accepts bare extensions like "pdf" or "png".
        """
        from fast_agent.mcp.mime_utils import normalize_mime_type

        tokenizes = cls.get_tokenizes(model) or []

        # Normalize the candidate and the database entries to lowercase
        normalized_supported = [t.lower() for t in tokenizes]

        # Handle wildcard inputs like "image/*" quickly
        mt = (mime_type or "").strip().lower()
        if mt.endswith("/*") and "/" in mt:
            prefix = mt.split("/", 1)[0] + "/"
            return any(s.startswith(prefix) for s in normalized_supported)

        normalized = normalize_mime_type(mime_type)
        if not normalized:
            return False

        return normalized.lower() in normalized_supported

    @classmethod
    def supports_any_mime(cls, model: str, mime_types: list[str]) -> bool:
        """Return True if the model supports any of the provided MIME types."""
        return any(cls.supports_mime(model, m) for m in mime_types)

    @classmethod
    def get_json_mode(cls, model: str) -> str | None:
        """Get supported json mode (structured output) for a model"""
        params = cls.get_model_params(model)
        return params.json_mode if params else None

    @classmethod
    def get_reasoning(cls, model: str) -> str | None:
        """Get supported reasoning output style for a model"""
        params = cls.get_model_params(model)
        return params.reasoning if params else None

    @classmethod
    def get_reasoning_effort_spec(cls, model: str) -> ReasoningEffortSpec | None:
        """Get reasoning effort capabilities for a model, if defined."""
        params = cls.get_model_params(model)
        return params.reasoning_effort_spec if params else None

    @classmethod
    def get_text_verbosity_spec(cls, model: str) -> TextVerbositySpec | None:
        """Get text verbosity capabilities for a model, if defined."""
        params = cls.get_model_params(model)
        return params.text_verbosity_spec if params else None

    @classmethod
    def get_stream_mode(cls, model: str | None) -> Literal["openai", "manual"]:
        """Return preferred streaming accumulation strategy for a model."""
        if not model:
            return "openai"

        params = cls.get_model_params(model)
        return params.stream_mode if params else "openai"

    @classmethod
    def get_default_max_tokens(cls, model: str) -> int:
        """Get default max_tokens for RequestParams based on model"""
        if not model:
            return 2048  # Fallback when no model specified

        params = cls.get_model_params(model)
        if params:
            return params.max_output_tokens
        return 2048  # Fallback for unknown models

    @classmethod
    def get_default_temperature(cls, model: str | None) -> float | None:
        """Get default temperature for RequestParams based on model metadata."""
        if not model:
            return None

        params = cls.get_model_params(model)
        return params.default_temperature if params else None

    @classmethod
    def get_cache_ttl(cls, model: str) -> Literal["5m", "1h"] | None:
        """Get cache TTL for a model, or None if not supported"""
        params = cls.get_model_params(model)
        return params.cache_ttl if params else None

    @classmethod
    def get_long_context_window(cls, model: str) -> int | None:
        """Get optional long-context override window for a model."""
        params = cls.get_model_params(model)
        return params.long_context_window if params else None

    @classmethod
    def get_response_transports(cls, model: str) -> tuple[Literal["sse", "websocket"], ...] | None:
        """Get supported Responses transports for a model, if explicitly defined."""
        params = cls.get_model_params(model)
        return params.response_transports if params else None

    @classmethod
    def get_response_websocket_providers(cls, model: str) -> tuple[Provider, ...] | None:
        """Get providers that may use websocket transport for this model."""
        params = cls.get_model_params(model)
        return params.response_websocket_providers if params else None

    @classmethod
    def supports_response_transport(
        cls, model: str, transport: Literal["sse", "websocket"]
    ) -> bool | None:
        """Return transport support for a model, or None when unconstrained.

        A `None` return means the model has no explicit transport metadata and callers
        may apply provider-level defaults.
        """
        transports = cls.get_response_transports(model)
        if transports is None:
            return None
        return transport in transports

    @classmethod
    def supports_response_websocket_provider(cls, model: str, provider: Provider) -> bool | None:
        """Return websocket provider support for a model, or None when unconstrained."""
        providers = cls.get_response_websocket_providers(model)
        if providers is None:
            return None
        return provider in providers

    @classmethod
    def get_anthropic_web_search_version(cls, model: str) -> str | None:
        """Get Anthropic web_search tool version for a model, if available."""
        params = cls.get_model_params(model)
        return params.anthropic_web_search_version if params else None

    @classmethod
    def get_anthropic_web_fetch_version(cls, model: str) -> str | None:
        """Get Anthropic web_fetch tool version for a model, if available."""
        params = cls.get_model_params(model)
        return params.anthropic_web_fetch_version if params else None

    @classmethod
    def get_anthropic_required_betas(cls, model: str) -> tuple[str, ...] | None:
        """Get Anthropic beta headers required for model-specific capabilities."""
        params = cls.get_model_params(model)
        return params.anthropic_required_betas if params else None

    @classmethod
    def list_long_context_models(cls) -> list[str]:
        """List model names that support explicit long-context overrides."""
        return sorted(
            name for name, params in cls.MODELS.items() if params.long_context_window is not None
        )

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available model names"""
        models = list(cls.MODELS.keys())
        if not cls._RUNTIME_MODEL_PARAMS:
            return models

        for runtime_key in sorted(cls._RUNTIME_MODEL_PARAMS.keys()):
            if runtime_key not in cls.MODELS:
                models.append(runtime_key)
        return models

    @classmethod
    def _normalize_provider_lookup_name(cls, model: str | None) -> str:
        model_spec = (model or "").strip()
        if not model_spec:
            return ""
        if "?" in model_spec:
            model_spec = model_spec.split("?", 1)[0]
        return model_spec.lower()

    @classmethod
    def _provider_from_explicit_prefix(cls, model_spec: str) -> Provider | None:
        if "/" in model_spec:
            prefix, rest = model_spec.split("/", 1)
            if rest and any(prefix == provider.value for provider in Provider):
                return Provider(prefix)

        if "." in model_spec:
            prefix, _ = model_spec.split(".", 1)
            if any(prefix == provider.value for provider in Provider):
                return Provider(prefix)

        return None

    @classmethod
    def get_default_provider(cls, model: str | None) -> Provider | None:
        """Get default provider for a model name."""
        model_key = cls._normalize_provider_lookup_name(model)
        if not model_key:
            return None

        explicit_provider = cls._provider_from_explicit_prefix(model_key)
        if explicit_provider is not None:
            return explicit_provider

        runtime_provider = cls._RUNTIME_MODEL_DEFAULT_PROVIDERS.get(model_key)
        if runtime_provider is not None:
            return runtime_provider

        params = cls.MODELS.get(model_key)
        return params.default_provider if params else None

    @classmethod
    def register_runtime_default_provider(cls, model: str, provider: Provider) -> None:
        """Register or override a runtime default provider for a model name."""
        model_key = cls._normalize_provider_lookup_name(model)
        if not model_key:
            return
        cls._RUNTIME_MODEL_DEFAULT_PROVIDERS[model_key] = provider

    @classmethod
    def register_runtime_model_params(cls, model: str, params: ModelParameters) -> None:
        """Register runtime model parameters for dynamic providers."""
        model_key = cls.normalize_model_name(model)
        if not model_key:
            return
        cls._RUNTIME_MODEL_PARAMS[model_key] = params

        if params.default_provider is not None:
            cls._RUNTIME_MODEL_DEFAULT_PROVIDERS[model_key] = params.default_provider

    @classmethod
    def unregister_runtime_model_params(cls, model: str) -> None:
        """Remove runtime model parameter metadata for a model."""
        model_key = cls.normalize_model_name(model)
        if not model_key:
            return
        cls._RUNTIME_MODEL_PARAMS.pop(model_key, None)
        cls._RUNTIME_MODEL_DEFAULT_PROVIDERS.pop(model_key, None)

    @classmethod
    def clear_runtime_model_params(cls, provider: Provider | None = None) -> None:
        """Clear runtime model parameter metadata.

        Args:
            provider: Optional provider filter. If omitted, all runtime metadata is cleared.
        """
        if provider is None:
            cls._RUNTIME_MODEL_PARAMS.clear()
            cls._RUNTIME_MODEL_DEFAULT_PROVIDERS.clear()
            return

        for model_key, params in list(cls._RUNTIME_MODEL_PARAMS.items()):
            if params.default_provider == provider:
                cls._RUNTIME_MODEL_PARAMS.pop(model_key, None)
                cls._RUNTIME_MODEL_DEFAULT_PROVIDERS.pop(model_key, None)

    @classmethod
    def list_runtime_models(cls, provider: Provider | None = None) -> list[str]:
        """List runtime-registered models, optionally filtered by provider."""
        if provider is None:
            return sorted(cls._RUNTIME_MODEL_PARAMS.keys())

        return sorted(
            model_key
            for model_key, params in cls._RUNTIME_MODEL_PARAMS.items()
            if params.default_provider == provider
        )

    @classmethod
    def unregister_runtime_default_provider(cls, model: str) -> None:
        """Remove a runtime default provider override for a model name."""
        model_key = cls._normalize_provider_lookup_name(model)
        if not model_key:
            return
        cls._RUNTIME_MODEL_DEFAULT_PROVIDERS.pop(model_key, None)

    @classmethod
    def is_fast_model(cls, model: str) -> bool:
        """Return True when model metadata marks the model as fast."""
        params = cls.get_model_params(model)
        return bool(params.fast) if params else False

    @classmethod
    def list_fast_models(cls) -> list[str]:
        """List model names marked as fast in metadata."""
        return sorted(name for name, params in cls.MODELS.items() if params.fast)
