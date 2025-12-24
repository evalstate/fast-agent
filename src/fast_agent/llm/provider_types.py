"""
Type definitions for LLM providers.
"""

from enum import Enum


class Provider(Enum):
    """Supported LLM providers"""

    display_name: str

    def __new__(cls, config_name, display_name=None):
        obj = object.__new__(cls)
        obj._value_ = config_name
        obj.display_name = display_name or config_name.title()
        return obj

    @property
    def config_name(self) -> str:
        """Return the provider's config name (typed accessor for _value_)."""
        return self._value_  # type: ignore[return-value]

    ANTHROPIC = ("anthropic", "Anthropic")
    DEEPSEEK = ("deepseek", "Deepseek")
    FAST_AGENT = ("fast-agent", "fast-agent-internal")
    GENERIC = ("generic", "Generic")
    GOOGLE_OAI = ("googleoai", "GoogleOAI")  # For Google through OpenAI libraries
    GOOGLE = ("google", "Google")  # For Google GenAI native library
    OPENAI = ("openai", "OpenAI")
    OPENROUTER = ("openrouter", "OpenRouter")
    TENSORZERO = ("tensorzero", "TensorZero")  # For TensorZero Gateway
    AZURE = ("azure", "Azure")  # Azure OpenAI Service
    ALIYUN = ("aliyun", "Aliyun")  # Aliyun Bailian OpenAI Service
    HUGGINGFACE = ("hf", "HuggingFace")  # For HuggingFace MCP connections
    XAI = ("xai", "XAI")  # For xAI Grok models
    BEDROCK = ("bedrock", "Bedrock")
    GROQ = ("groq", "Groq")
    RESPONSES = ("responses", "responses")
