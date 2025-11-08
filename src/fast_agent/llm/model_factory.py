from enum import Enum
from typing import Dict, Optional, Type, Union

from pydantic import BaseModel

from fast_agent.core.exceptions import ModelConfigError
from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LLMFactoryProtocol
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.internal.playback import PlaybackLLM
from fast_agent.llm.internal.silent import SilentLLM
from fast_agent.llm.internal.slow import SlowLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

# Type alias for LLM classes
LLMClass = Union[Type[PassthroughLLM], Type[PlaybackLLM], Type[SilentLLM], Type[SlowLLM], type]


class ReasoningEffort(Enum):
    """Optional reasoning effort levels"""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    provider: Provider
    model_name: str
    reasoning_effort: ReasoningEffort | None = None


class ModelFactory:
    """Factory for creating LLM instances based on model specifications"""

    # Mapping of effort strings to enum values
    # TODO -- move this to the model database
    EFFORT_MAP = {
        "minimal": ReasoningEffort.MINIMAL,  # Alias for low effort
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }

    """
    TODO -- add audio supporting got-4o-audio-preview
    TODO -- bring model parameter configuration here
    Mapping of model names to their default providers
    """
    DEFAULT_PROVIDERS = {
        "passthrough": Provider.FAST_AGENT,
        "silent": Provider.FAST_AGENT,
        "playback": Provider.FAST_AGENT,
        "slow": Provider.FAST_AGENT,
        "gpt-4o": Provider.OPENAI,
        "gpt-4o-mini": Provider.OPENAI,
        "gpt-4.1": Provider.OPENAI,
        "gpt-4.1-mini": Provider.OPENAI,
        "gpt-4.1-nano": Provider.OPENAI,
        "gpt-5": Provider.OPENAI,
        "gpt-5-mini": Provider.OPENAI,
        "gpt-5-nano": Provider.OPENAI,
        "o1-mini": Provider.OPENAI,
        "o1": Provider.OPENAI,
        "o1-preview": Provider.OPENAI,
        "o3": Provider.OPENAI,
        "o3-mini": Provider.OPENAI,
        "o4-mini": Provider.OPENAI,
        "claude-3-haiku-20240307": Provider.ANTHROPIC,
        "claude-3-5-haiku-20241022": Provider.ANTHROPIC,
        "claude-3-5-haiku-latest": Provider.ANTHROPIC,
        "claude-3-5-sonnet-20240620": Provider.ANTHROPIC,
        "claude-3-5-sonnet-20241022": Provider.ANTHROPIC,
        "claude-3-5-sonnet-latest": Provider.ANTHROPIC,
        "claude-3-7-sonnet-20250219": Provider.ANTHROPIC,
        "claude-3-7-sonnet-latest": Provider.ANTHROPIC,
        "claude-3-opus-20240229": Provider.ANTHROPIC,
        "claude-3-opus-latest": Provider.ANTHROPIC,
        "claude-opus-4-0": Provider.ANTHROPIC,
        "claude-opus-4-1": Provider.ANTHROPIC,
        "claude-opus-4-20250514": Provider.ANTHROPIC,
        "claude-sonnet-4-20250514": Provider.ANTHROPIC,
        "claude-sonnet-4-0": Provider.ANTHROPIC,
        "claude-sonnet-4-5-20250929": Provider.ANTHROPIC,
        "claude-sonnet-4-5": Provider.ANTHROPIC,
        "claude-haiku-4-5": Provider.ANTHROPIC,
        "deepseek-chat": Provider.DEEPSEEK,
        "gemini-2.0-flash": Provider.GOOGLE,
        "gemini-2.5-flash-preview-05-20": Provider.GOOGLE,
        "gemini-2.5-flash-preview-09-2025": Provider.GOOGLE,
        "gemini-2.5-pro-preview-05-06": Provider.GOOGLE,
        "gemini-2.5-pro": Provider.GOOGLE,
        "grok-4": Provider.XAI,
        "grok-4-0709": Provider.XAI,
        "grok-3": Provider.XAI,
        "grok-3-mini": Provider.XAI,
        "grok-3-fast": Provider.XAI,
        "grok-3-mini-fast": Provider.XAI,
        "qwen-turbo": Provider.ALIYUN,
        "qwen-plus": Provider.ALIYUN,
        "qwen-max": Provider.ALIYUN,
        "qwen-long": Provider.ALIYUN,
    }

    MODEL_ALIASES = {
        "sonnet": "claude-sonnet-4-5",
        "sonnet4": "claude-sonnet-4-0",
        "sonnet45": "claude-sonnet-4-5",
        "sonnet35": "claude-3-5-sonnet-latest",
        "sonnet37": "claude-3-7-sonnet-latest",
        "claude": "claude-sonnet-4-0",
        "haiku": "claude-haiku-4-5",
        "haiku3": "claude-3-haiku-20240307",
        "haiku35": "claude-3-5-haiku-latest",
        "hauku45": "claude-haiku-4-5",
        "opus": "claude-opus-4-1",
        "opus4": "claude-opus-4-1",
        "opus3": "claude-3-opus-latest",
        "deepseekv3": "deepseek-chat",
        "deepseek": "deepseek-chat",
        "gemini2": "gemini-2.0-flash",
        "gemini25": "gemini-2.5-flash-preview-09-2025",
        "gemini25pro": "gemini-2.5-pro",
        "grok-4-fast": "xai.grok-4-fast-non-reasoning",
        "grok-4-fast-reasoning": "xai.grok-4-fast-reasoning",
        "kimigroq": "groq.moonshotai/kimi-k2-instruct-0905",
        "minimax": "hf.MiniMaxAI/MiniMax-M2",
        "kimi": "hf.moonshotai/Kimi-K2-Instruct-0905",
        "gpt-oss": "hf.openai/gpt-oss-120b",
        "gpt-oss-20b": "hf.openai/gpt-oss-20b",
        "glm": "hf.zai-org/GLM-4.6",
        "qwen3": "hf.Qwen/Qwen3-Next-80B-A3B-Instruct",
        "deepseek31": "hf.deepseek-ai/DeepSeek-V3.1",
    }

    @staticmethod
    def _bedrock_pattern_matches(model_name: str) -> bool:
        """Return True if model_name matches Bedrock's expected pattern, else False.

        Uses provider's helper if available; otherwise, returns False.
        """
        try:
            from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM  # type: ignore

            return BedrockLLM.matches_model_pattern(model_name)
        except Exception:
            return False

    # Mapping of providers to their LLM classes
    PROVIDER_CLASSES: Dict[Provider, LLMClass] = {}

    # Mapping of special model names to their specific LLM classes
    # This overrides the provider-based class selection
    MODEL_SPECIFIC_CLASSES: Dict[str, LLMClass] = {
        "playback": PlaybackLLM,
        "silent": SilentLLM,
        "slow": SlowLLM,
    }

    @classmethod
    def parse_model_string(cls, model_string: str) -> ModelConfig:
        """Parse a model string into a ModelConfig object"""
        suffix: str | None = None
        if ":" in model_string:
            base, suffix = model_string.rsplit(":", 1)
            if base:
                model_string = base

        model_string = cls.MODEL_ALIASES.get(model_string, model_string)
        parts = model_string.split(".")

        model_name_str = model_string  # Default full string as model name initially
        provider = None
        reasoning_effort = None
        parts_for_provider_model = []

        # Check for reasoning effort first (last part)
        if len(parts) > 1 and parts[-1].lower() in cls.EFFORT_MAP:
            reasoning_effort = cls.EFFORT_MAP[parts[-1].lower()]
            # Remove effort from parts list for provider/model name determination
            parts_for_provider_model = parts[:-1]
        else:
            parts_for_provider_model = parts[:]

        # Try to match longest possible provider string
        identified_provider_parts = 0  # How many parts belong to the provider string

        if len(parts_for_provider_model) >= 2:
            potential_provider_str = f"{parts_for_provider_model[0]}.{parts_for_provider_model[1]}"
            if any(p.value == potential_provider_str for p in Provider):
                provider = Provider(potential_provider_str)
                identified_provider_parts = 2

        if provider is None and len(parts_for_provider_model) >= 1:
            potential_provider_str = parts_for_provider_model[0]
            if any(p.value == potential_provider_str for p in Provider):
                provider = Provider(potential_provider_str)
                identified_provider_parts = 1

        # Construct model_name from remaining parts
        if identified_provider_parts > 0:
            model_name_str = ".".join(parts_for_provider_model[identified_provider_parts:])
        else:
            # If no provider prefix was matched, the whole string (after effort removal) is the model name
            model_name_str = ".".join(parts_for_provider_model)

        # If provider still None, try to get from DEFAULT_PROVIDERS using the model_name_str
        if provider is None:
            provider = cls.DEFAULT_PROVIDERS.get(model_name_str)

            # If still None, try pattern matching for Bedrock models
            if provider is None and cls._bedrock_pattern_matches(model_name_str):
                provider = Provider.BEDROCK

            if provider is None:
                raise ModelConfigError(
                    f"Unknown model or provider for: {model_string}. Model name parsed as '{model_name_str}'"
                )

        if provider == Provider.TENSORZERO and not model_name_str:
            raise ModelConfigError(
                f"TensorZero provider requires a function name after the provider "
                f"(e.g., tensorzero.my-function), got: {model_string}"
            )

        if suffix:
            model_name_str = f"{model_name_str}:{suffix}"

        return ModelConfig(
            provider=provider, model_name=model_name_str, reasoning_effort=reasoning_effort
        )

    @classmethod
    def create_factory(cls, model_string: str) -> LLMFactoryProtocol:
        """
        Creates a factory function that follows the attach_llm protocol.

        Args:
            model_string: The model specification string (e.g. "gpt-4.1")

        Returns:
            A callable that takes an agent parameter and returns an LLM instance
        """
        config = cls.parse_model_string(model_string)

        # Ensure provider is valid before trying to access PROVIDER_CLASSES with it
        # Lazily ensure provider class map is populated and supports this provider
        if config.model_name not in cls.MODEL_SPECIFIC_CLASSES:
            llm_class = cls._load_provider_class(config.provider)
            # Stash for next time
            cls.PROVIDER_CLASSES[config.provider] = llm_class

        if config.model_name in cls.MODEL_SPECIFIC_CLASSES:
            llm_class = cls.MODEL_SPECIFIC_CLASSES[config.model_name]
        else:
            llm_class = cls.PROVIDER_CLASSES[config.provider]

        def factory(
            agent: AgentProtocol, request_params: Optional[RequestParams] = None, **kwargs
        ) -> FastAgentLLMProtocol:
            base_params = RequestParams()
            base_params.model = config.model_name
            if config.reasoning_effort:
                kwargs["reasoning_effort"] = config.reasoning_effort.value
            llm_args = {
                "model": config.model_name,
                "request_params": request_params,
                "name": getattr(agent, "name", "fast-agent"),
                "instructions": getattr(agent, "instruction", None),
                **kwargs,
            }
            llm: FastAgentLLMProtocol = llm_class(**llm_args)
            return llm

        return factory

    @classmethod
    def _load_provider_class(cls, provider: Provider) -> type:
        """Import provider-specific LLM classes lazily to avoid heavy deps at import time."""
        try:
            if provider == Provider.FAST_AGENT:
                return PassthroughLLM
            if provider == Provider.ANTHROPIC:
                from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM

                return AnthropicLLM
            if provider == Provider.OPENAI:
                from fast_agent.llm.provider.openai.llm_openai import OpenAILLM

                return OpenAILLM
            if provider == Provider.DEEPSEEK:
                from fast_agent.llm.provider.openai.llm_deepseek import DeepSeekLLM

                return DeepSeekLLM
            if provider == Provider.GENERIC:
                from fast_agent.llm.provider.openai.llm_generic import GenericLLM

                return GenericLLM
            if provider == Provider.GOOGLE_OAI:
                from fast_agent.llm.provider.openai.llm_google_oai import GoogleOaiLLM

                return GoogleOaiLLM
            if provider == Provider.GOOGLE:
                from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM

                return GoogleNativeLLM

            if provider == Provider.HUGGINGFACE:
                from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM

                return HuggingFaceLLM
            if provider == Provider.XAI:
                from fast_agent.llm.provider.openai.llm_xai import XAILLM

                return XAILLM
            if provider == Provider.OPENROUTER:
                from fast_agent.llm.provider.openai.llm_openrouter import OpenRouterLLM

                return OpenRouterLLM
            if provider == Provider.TENSORZERO:
                from fast_agent.llm.provider.openai.llm_tensorzero_openai import TensorZeroOpenAILLM

                return TensorZeroOpenAILLM
            if provider == Provider.AZURE:
                from fast_agent.llm.provider.openai.llm_azure import AzureOpenAILLM

                return AzureOpenAILLM
            if provider == Provider.ALIYUN:
                from fast_agent.llm.provider.openai.llm_aliyun import AliyunLLM

                return AliyunLLM
            if provider == Provider.BEDROCK:
                from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM

                return BedrockLLM
            if provider == Provider.GROQ:
                from fast_agent.llm.provider.openai.llm_groq import GroqLLM

                return GroqLLM
            if provider == Provider.RESPONSES:
                from fast_agent.llm.provider.openai.responses import ResponsesLLM

                return ResponsesLLM

        except Exception as e:
            raise ModelConfigError(
                f"Provider '{provider.value}' is unavailable or missing dependencies: {e}"
            )
        raise ModelConfigError(f"Unsupported provider: {provider}")
