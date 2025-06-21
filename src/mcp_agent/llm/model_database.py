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


class ModelDatabase:
    """Centralized model configuration database"""

    # Common parameter sets
    OPENAI_MULTIMODAL = ["text/plain", "image/jpeg", "image/png", "image/webp", "application/pdf"]
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
    TEXT_ONLY = ["text/plain"]

    # Common parameter configurations
    OPENAI_STANDARD = ModelParameters(
        context_window=128000, max_output_tokens=16384, tokenizes=OPENAI_MULTIMODAL
    )

    ANTHROPIC_SONNET = ModelParameters(
        context_window=200000, max_output_tokens=8192, tokenizes=ANTHROPIC_MULTIMODAL
    )

    ANTHROPIC_HAIKU = ModelParameters(
        context_window=200000, max_output_tokens=8192, tokenizes=ANTHROPIC_MULTIMODAL
    )

    ANTHROPIC_OPUS_3 = ModelParameters(
        context_window=200000, max_output_tokens=4096, tokenizes=ANTHROPIC_MULTIMODAL
    )

    ANTHROPIC_OPUS_4 = ModelParameters(
        context_window=200000, max_output_tokens=8192, tokenizes=ANTHROPIC_MULTIMODAL
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

    FAST_AGENT_STANDARD = ModelParameters(
        context_window=1000000, max_output_tokens=100000, tokenizes=TEXT_ONLY
    )

    # Model configuration database
    MODELS: Dict[str, ModelParameters] = {
        # OpenAI Models
        "gpt-4o": OPENAI_STANDARD,
        "gpt-4o-mini": OPENAI_STANDARD,
        "gpt-4.1": OPENAI_STANDARD,
        "gpt-4.1-mini": OPENAI_STANDARD,
        "gpt-4.1-nano": OPENAI_STANDARD,
        "o1-mini": ModelParameters(
            context_window=128000, max_output_tokens=65536, tokenizes=TEXT_ONLY
        ),
        "o1": ModelParameters(context_window=200000, max_output_tokens=100000, tokenizes=TEXT_ONLY),
        "o1-preview": ModelParameters(
            context_window=128000, max_output_tokens=32768, tokenizes=TEXT_ONLY
        ),
        "o3": ModelParameters(context_window=200000, max_output_tokens=100000, tokenizes=TEXT_ONLY),
        "o3-mini": ModelParameters(
            context_window=128000, max_output_tokens=65536, tokenizes=TEXT_ONLY
        ),
        # Anthropic Models
        "claude-3-haiku-20240307": ModelParameters(
            context_window=200000, max_output_tokens=4096, tokenizes=ANTHROPIC_MULTIMODAL
        ),
        "claude-3-5-haiku-20241022": ANTHROPIC_HAIKU,
        "claude-3-5-haiku-latest": ANTHROPIC_HAIKU,
        "claude-3-5-sonnet-20240620": ANTHROPIC_SONNET,
        "claude-3-5-sonnet-20241022": ANTHROPIC_SONNET,
        "claude-3-5-sonnet-latest": ANTHROPIC_SONNET,
        "claude-3-7-sonnet-20250219": ANTHROPIC_SONNET,
        "claude-3-7-sonnet-latest": ANTHROPIC_SONNET,
        "claude-3-opus-20240229": ANTHROPIC_OPUS_3,
        "claude-3-opus-latest": ANTHROPIC_OPUS_3,
        "claude-opus-4-0": ANTHROPIC_OPUS_4,
        "claude-opus-4-20250514": ANTHROPIC_OPUS_4,
        "claude-sonnet-4-20250514": ANTHROPIC_SONNET,
        "claude-sonnet-4-0": ANTHROPIC_SONNET,
        # Google Models
        "gemini-2.0-flash": GEMINI_FLASH,
        "gemini-2.5-flash-preview-05-20": GEMINI_FLASH,
        "gemini-2.5-pro-preview-05-06": GEMINI_PRO,
        # DeepSeek Models
        "deepseek-chat": ModelParameters(
            context_window=64000, max_output_tokens=8192, tokenizes=TEXT_ONLY
        ),
        # Aliyun Models
        "qwen-turbo": QWEN_STANDARD,
        "qwen-plus": QWEN_STANDARD,
        "qwen-max": QWEN_STANDARD,
        "qwen-long": ModelParameters(
            context_window=10000000, max_output_tokens=8192, tokenizes=TEXT_ONLY
        ),
        # Fast-agent providers
        "passthrough": FAST_AGENT_STANDARD,
        "playback": FAST_AGENT_STANDARD,
        "slow": FAST_AGENT_STANDARD,
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
