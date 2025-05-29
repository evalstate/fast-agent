"""
Type definitions for LLM providers.
"""

from enum import Enum


class Provider(Enum):
    """Supported LLM providers"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    FAST_AGENT = "fast-agent"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    GENERIC = "generic"
    OPENROUTER = "openrouter"
    TENSORZERO = "tensorzero"  # For TensorZero Gateway
    AZURE = "azure"  # Azure OpenAI Service
