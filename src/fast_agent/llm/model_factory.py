from enum import Enum
from typing import Dict, Optional, Type, Union

from pydantic import BaseModel

from fast_agent.interfaces import FastAgentLLMProtocol, LLMFactoryProtocol
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.internal.playback import PlaybackLLM
from fast_agent.llm.internal.silent import SilentLLM
from fast_agent.llm.internal.slow import SlowLLM
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider.openai.llm_aliyun import AliyunLLM
from fast_agent.llm.provider.openai.llm_azure import AzureOpenAILLM
from fast_agent.llm.provider.openai.llm_deepseek import DeepSeekLLM
from fast_agent.llm.provider.openai.llm_generic import GenericLLM
from fast_agent.llm.provider.openai.llm_google_oai import GoogleOaiLLM
from fast_agent.llm.provider.openai.llm_groq import GroqLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.llm_openrouter import OpenRouterLLM
from fast_agent.llm.provider.openai.llm_tensorzero_openai import TensorZeroOpenAILLM
from fast_agent.llm.provider.openai.llm_xai import XAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams
from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ModelConfigError

# from mcp_agent.workflows.llm.augmented_llm_deepseek import DeekSeekAugmentedLLM


# Type alias for LLM classes
LLMClass = Union[
    Type[AnthropicLLM],
    Type[OpenAILLM],
    Type[PassthroughLLM],
    Type[PlaybackLLM],
    Type[SilentLLM],
    Type[SlowLLM],
    Type[DeepSeekLLM],
    Type[OpenRouterLLM],
    Type[TensorZeroOpenAILLM],
    Type[GoogleNativeLLM],
    Type[GenericLLM],
    Type[AzureOpenAILLM],
    Type[BedrockLLM],
    Type[GroqLLM],
]


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
    reasoning_effort: Optional[ReasoningEffort] = None


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
        "deepseek-chat": Provider.DEEPSEEK,
        "gemini-2.0-flash": Provider.GOOGLE,
        "gemini-2.5-flash-preview-05-20": Provider.GOOGLE,
        "gemini-2.5-pro-preview-05-06": Provider.GOOGLE,
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
        "sonnet": "claude-sonnet-4-0",
        "sonnet4": "claude-sonnet-4-0",
        "sonnet35": "claude-3-5-sonnet-latest",
        "sonnet37": "claude-3-7-sonnet-latest",
        "claude": "claude-sonnet-4-0",
        "haiku": "claude-3-5-haiku-latest",
        "haiku3": "claude-3-haiku-20240307",
        "haiku35": "claude-3-5-haiku-latest",
        "opus": "claude-opus-4-1",
        "opus4": "claude-opus-4-1",
        "opus3": "claude-3-opus-latest",
        "deepseekv3": "deepseek-chat",
        "deepseek": "deepseek-chat",
        "gemini2": "gemini-2.0-flash",
        "gemini25": "gemini-2.5-flash-preview-05-20",
        "gemini25pro": "gemini-2.5-pro-preview-05-06",
        "kimi": "groq.moonshotai/kimi-k2-instruct",
        "gpt-oss": "groq.openai/gpt-oss-120b",
        "gpt-oss-20b": "groq.openai/gpt-oss-20b",
    }

    # Mapping of providers to their LLM classes
    PROVIDER_CLASSES: Dict[Provider, LLMClass] = {
        Provider.ANTHROPIC: AnthropicLLM,
        Provider.OPENAI: OpenAILLM,
        Provider.FAST_AGENT: PassthroughLLM,
        Provider.DEEPSEEK: DeepSeekLLM,
        Provider.GENERIC: GenericLLM,
        Provider.GOOGLE_OAI: GoogleOaiLLM,
        Provider.GOOGLE: GoogleNativeLLM,
        Provider.XAI: XAILLM,
        Provider.OPENROUTER: OpenRouterLLM,
        Provider.TENSORZERO: TensorZeroOpenAILLM,
        Provider.AZURE: AzureOpenAILLM,
        Provider.ALIYUN: AliyunLLM,
        Provider.BEDROCK: BedrockLLM,
        Provider.GROQ: GroqLLM,
    }

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
            if provider is None and BedrockLLM.matches_model_pattern(model_name_str):
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
        if (
            config.provider not in cls.PROVIDER_CLASSES
            and config.model_name not in cls.MODEL_SPECIFIC_CLASSES
        ):
            # This check is important if a provider (like old GOOGLE) is commented out from PROVIDER_CLASSES
            raise ModelConfigError(
                f"Provider '{config.provider}' not configured in PROVIDER_CLASSES and model '{config.model_name}' not in MODEL_SPECIFIC_CLASSES."
            )

        if config.model_name in cls.MODEL_SPECIFIC_CLASSES:
            llm_class = cls.MODEL_SPECIFIC_CLASSES[config.model_name]
        else:
            # This line is now safer due to the check above
            llm_class = cls.PROVIDER_CLASSES[config.provider]

        def factory(
            agent: Agent, request_params: Optional[RequestParams] = None, **kwargs
        ) -> FastAgentLLMProtocol:
            base_params = RequestParams()
            base_params.model = config.model_name
            if config.reasoning_effort:
                kwargs["reasoning_effort"] = config.reasoning_effort.value
            llm_args = {
                "model": config.model_name,
                "request_params": request_params,
                "name": agent.name,
                "instructions": agent.instruction,
                **kwargs,
            }
            llm: FastAgentLLMProtocol = llm_class(**llm_args)
            return llm

        return factory
