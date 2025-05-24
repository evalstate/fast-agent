"""
Type definitions for LLM providers.
"""

from enum import Enum


class Provider(Enum):
    """Supported LLM providers"""

    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    FAST_AGENT = "fast-agent"
    GENERIC = "generic"
    GOOGLE = "google"  # For Google through OpenAI libraries
    GOOGLE_NATIVE = "google.native"  # For Google GenAI native library
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    TENSORZERO = "tensorzero"  # For TensorZero Gateway
    AZURE = "azure"  # Azure OpenAI Service
