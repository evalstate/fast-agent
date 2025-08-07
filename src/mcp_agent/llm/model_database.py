"""
Model database for LLM parameters.

This module provides a centralized lookup for model parameters including
context windows, max output tokens, and supported tokenization types.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class ModelParameters(BaseModel):
    """Configuration parameters for a specific model"""

    context_window: int
    """Maximum context window size in tokens"""

    max_output_tokens: int
    """Maximum output tokens the model can generate"""

    tokenizes: List[str]
    """List of supported content types for tokenization"""

    json_mode: None | str = "schema"
    """Structured output style. 'schema', 'object' or None for unsupported """

    reasoning: None | str = None
    """Reasoning output style. 'tags' if enclosed in <thinking> tags, 'none' if not used"""


class ModelDatabase:
    """Centralized model configuration database"""

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
        "application/pdf",
        "audio/wav",
        "audio/mp3",
        "video/mp4",
    ]
    QWEN_MULTIMODAL = ["text/plain", "image/jpeg", "image/png", "image/webp"]
    XAI_VISION = ["text/plain", "image/jpeg", "image/png", "image/webp"]
    TEXT_ONLY = ["text/plain"]

    # Common parameter configurations
    OPENAI_STANDARD = ModelParameters(
        context_window=128000, max_output_tokens=16384, tokenizes=OPENAI_MULTIMODAL
    )

    OPENAI_4_1_STANDARD = ModelParameters(
        context_window=1047576, max_output_tokens=32768, tokenizes=OPENAI_MULTIMODAL
    )

    OPENAI_O_SERIES = ModelParameters(
        context_window=200000, max_output_tokens=100000, tokenizes=OPENAI_VISION
    )

    ANTHROPIC_LEGACY = ModelParameters(
        context_window=200000, max_output_tokens=4096, tokenizes=ANTHROPIC_MULTIMODAL
    )

    ANTHROPIC_35_SERIES = ModelParameters(
        context_window=200000, max_output_tokens=8192, tokenizes=ANTHROPIC_MULTIMODAL
    )

    # TODO--- TO USE 64,000 NEED TO SUPPORT STREAMING
    ANTHROPIC_37_SERIES = ModelParameters(
        context_window=200000, max_output_tokens=16384, tokenizes=ANTHROPIC_MULTIMODAL
    )

    GEMINI_FLASH = ModelParameters(
        context_window=1048576, max_output_tokens=8192, tokenizes=GOOGLE_MULTIMODAL
    )

    GEMINI_PRO = ModelParameters(
        context_window=2097152, max_output_tokens=8192, tokenizes=GOOGLE_MULTIMODAL
    )

    QWEN_STANDARD = ModelParameters(
        context_window=32000, max_output_tokens=8192, tokenizes=QWEN_MULTIMODAL
    )
    QWEN3_REASONER = ModelParameters(
        context_window=131072,
        max_output_tokens=16384,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="tags",
    )

    FAST_AGENT_STANDARD = ModelParameters(
        context_window=1000000, max_output_tokens=100000, tokenizes=TEXT_ONLY
    )

    OPENAI_4_1_SERIES = ModelParameters(
        context_window=1047576, max_output_tokens=32768, tokenizes=OPENAI_MULTIMODAL
    )

    OPENAI_4O_SERIES = ModelParameters(
        context_window=128000, max_output_tokens=16384, tokenizes=OPENAI_VISION
    )

    OPENAI_O3_SERIES = ModelParameters(
        context_window=200000, max_output_tokens=100000, tokenizes=OPENAI_MULTIMODAL
    )

    OPENAI_O3_MINI_SERIES = ModelParameters(
        context_window=200000, max_output_tokens=100000, tokenizes=TEXT_ONLY
    )
    OPENAI_GPT_OSS_SERIES = ModelParameters(
        context_window=131072, max_output_tokens=32766, tokenizes=TEXT_ONLY, json_mode="object"
    )
    OPENAI_GPT_5 = ModelParameters(
        context_window=400000, max_output_tokens=128000, tokenizes=OPENAI_MULTIMODAL
    )

    # TODO update to 32000
    ANTHROPIC_OPUS_4_VERSIONED = ModelParameters(
        context_window=200000, max_output_tokens=32000, tokenizes=ANTHROPIC_MULTIMODAL
    )
    # TODO update to 64000
    ANTHROPIC_SONNET_4_VERSIONED = ModelParameters(
        context_window=200000, max_output_tokens=64000, tokenizes=ANTHROPIC_MULTIMODAL
    )

    DEEPSEEK_CHAT_STANDARD = ModelParameters(
        context_window=65536, max_output_tokens=8192, tokenizes=TEXT_ONLY
    )

    DEEPSEEK_REASONER = ModelParameters(
        context_window=65536, max_output_tokens=32768, tokenizes=TEXT_ONLY
    )

    DEEPSEEK_DISTILL = ModelParameters(
        context_window=131072,
        max_output_tokens=131072,
        tokenizes=TEXT_ONLY,
        json_mode="object",
        reasoning="tags",
    )
    GEMINI_2_5_PRO = ModelParameters(
        context_window=2097152, max_output_tokens=8192, tokenizes=GOOGLE_MULTIMODAL
    )

    KIMI_MOONSHOT = ModelParameters(
        context_window=131072, max_output_tokens=16384, tokenizes=TEXT_ONLY
    )

    # FIXME: xAI has not documented the max output tokens for Grok 4. Using Grok 3 as a placeholder. Will need to update when available (if ever)
    GROK_4 = ModelParameters(context_window=256000, max_output_tokens=16385, tokenizes=XAI_VISION)

    # Source for Grok 3 max output: https://www.reddit.com/r/grok/comments/1j7209p/exploring_grok_3_beta_output_capacity_a_simple/
    # xAI does not document Grok 3 max output tokens, using the above source as a reference.
    GROK_3 = ModelParameters(context_window=131072, max_output_tokens=16385, tokenizes=TEXT_ONLY)

    # Model configuration database
    MODELS: Dict[str, ModelParameters] = {
        # internal models
        "passthrough": FAST_AGENT_STANDARD,
        "silent": FAST_AGENT_STANDARD,
        "playback": FAST_AGENT_STANDARD,
        "slow": FAST_AGENT_STANDARD,
        # aliyun models
        "qwen-turbo": QWEN_STANDARD,
        "qwen-plus": QWEN_STANDARD,
        "qwen-max": QWEN_STANDARD,
        "qwen-long": ModelParameters(
            context_window=10000000, max_output_tokens=8192, tokenizes=TEXT_ONLY
        ),
        # OpenAI Models (vanilla aliases and versioned)
        "gpt-4.1": OPENAI_4_1_SERIES,
        "gpt-4.1-mini": OPENAI_4_1_SERIES,
        "gpt-4.1-nano": OPENAI_4_1_SERIES,
        "gpt-4.1-2025-04-14": OPENAI_4_1_SERIES,
        "gpt-4.1-mini-2025-04-14": OPENAI_4_1_SERIES,
        "gpt-4.1-nano-2025-04-14": OPENAI_4_1_SERIES,
        "gpt-4o": OPENAI_4O_SERIES,
        "gpt-4o-2024-11-20": OPENAI_4O_SERIES,
        "gpt-4o-mini-2024-07-18": OPENAI_4O_SERIES,
        "o1": OPENAI_O_SERIES,
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
        "gpt-5-mini": OPENAI_GPT_5,
        "gpt-5-nano": OPENAI_GPT_5,
        # Anthropic Models
        "claude-3-haiku": ANTHROPIC_35_SERIES,
        "claude-3-haiku-20240307": ANTHROPIC_LEGACY,
        "claude-3-sonnet": ANTHROPIC_LEGACY,
        "claude-3-opus": ANTHROPIC_LEGACY,
        "claude-3-opus-20240229": ANTHROPIC_LEGACY,
        "claude-3-opus-latest": ANTHROPIC_LEGACY,
        "claude-3-5-haiku": ANTHROPIC_35_SERIES,
        "claude-3-5-haiku-20241022": ANTHROPIC_35_SERIES,
        "claude-3-5-haiku-latest": ANTHROPIC_35_SERIES,
        "claude-3-sonnet-20240229": ANTHROPIC_LEGACY,
        "claude-3-5-sonnet": ANTHROPIC_35_SERIES,
        "claude-3-5-sonnet-20240620": ANTHROPIC_35_SERIES,
        "claude-3-5-sonnet-20241022": ANTHROPIC_35_SERIES,
        "claude-3-5-sonnet-latest": ANTHROPIC_35_SERIES,
        "claude-3-7-sonnet": ANTHROPIC_37_SERIES,
        "claude-3-7-sonnet-20250219": ANTHROPIC_37_SERIES,
        "claude-3-7-sonnet-latest": ANTHROPIC_37_SERIES,
        "claude-sonnet-4-0": ANTHROPIC_SONNET_4_VERSIONED,
        "claude-sonnet-4-20250514": ANTHROPIC_SONNET_4_VERSIONED,
        "claude-opus-4-0": ANTHROPIC_OPUS_4_VERSIONED,
        "claude-opus-4-1": ANTHROPIC_OPUS_4_VERSIONED,
        "claude-opus-4-20250514": ANTHROPIC_OPUS_4_VERSIONED,
        # DeepSeek Models
        "deepseek-chat": DEEPSEEK_CHAT_STANDARD,
        # Google Gemini Models (vanilla aliases and versioned)
        "gemini-2.0-flash": GEMINI_FLASH,
        "gemini-2.5-flash-preview": GEMINI_FLASH,
        "gemini-2.5-pro-preview": GEMINI_2_5_PRO,
        "gemini-2.5-flash-preview-05-20": GEMINI_FLASH,
        "gemini-2.5-pro-preview-05-06": GEMINI_PRO,
        # xAI Grok Models
        "grok-4": GROK_4,
        "grok-4-0709": GROK_4,
        "grok-3": GROK_3,
        "grok-3-mini": GROK_3,
        "grok-3-fast": GROK_3,
        "grok-3-mini-fast": GROK_3,
        "moonshotai/kimi-k2-instruct": KIMI_MOONSHOT,
        "qwen/qwen3-32b": QWEN3_REASONER,
        "deepseek-r1-distill-llama-70b": DEEPSEEK_DISTILL,
        "openai/gpt-oss-120b": OPENAI_GPT_OSS_SERIES,
        "openai/gpt-oss-20b": OPENAI_GPT_OSS_SERIES,
    }

    @classmethod
    def get_model_params(cls, model: str) -> Optional[ModelParameters]:
        """Get model parameters for a given model name"""
        return cls.MODELS.get(model)

    @classmethod
    def get_context_window(cls, model: str) -> Optional[int]:
        """Get context window size for a model"""
        params = cls.get_model_params(model)
        return params.context_window if params else None

    @classmethod
    def get_max_output_tokens(cls, model: str) -> Optional[int]:
        """Get maximum output tokens for a model"""
        params = cls.get_model_params(model)
        return params.max_output_tokens if params else None

    @classmethod
    def get_tokenizes(cls, model: str) -> Optional[List[str]]:
        """Get supported tokenization types for a model"""
        params = cls.get_model_params(model)
        return params.tokenizes if params else None

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
    def get_default_max_tokens(cls, model: str) -> int:
        """Get default max_tokens for RequestParams based on model"""
        if not model:
            return 2048  # Fallback when no model specified

        params = cls.get_model_params(model)
        if params:
            return params.max_output_tokens
        return 2048  # Fallback for unknown models

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model names"""
        return list(cls.MODELS.keys())
