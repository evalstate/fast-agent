from enum import Enum
from typing import Callable, Dict, Optional, Type, Union

from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM
from mcp_agent.llm.augmented_llm_playback import PlaybackLLM
from mcp_agent.llm.augmented_llm_slow import SlowLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_aliyun import AliyunAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_azure import AzureOpenAIAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_deepseek import DeepSeekAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_generic import GenericAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_google_native import GoogleNativeAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_google_oai import GoogleOaiAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_openrouter import OpenRouterAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_tensorzero import TensorZeroAugmentedLLM
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol

# from mcp_agent.workflows.llm.augmented_llm_deepseek import DeekSeekAugmentedLLM


# Type alias for LLM classes
LLMClass = Union[
    Type[AnthropicAugmentedLLM],
    Type[OpenAIAugmentedLLM],
    Type[PassthroughLLM],
    Type[PlaybackLLM],
    Type[SlowLLM],
    Type[DeepSeekAugmentedLLM],
    Type[OpenRouterAugmentedLLM],
    Type[TensorZeroAugmentedLLM],
    Type[GoogleNativeAugmentedLLM],
    Type[GenericAugmentedLLM],
    Type[AzureOpenAIAugmentedLLM],
]


class ReasoningEffort(Enum):
    """Optional reasoning effort levels"""

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
    EFFORT_MAP = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }

    """
    TODO -- add context window size information for display/management
    TODO -- add audio supporting got-4o-audio-preview
    TODO -- bring model parameter configuration here
    Mapping of model names to their default providers
    """
    DEFAULT_PROVIDERS = {
        "passthrough": Provider.FAST_AGENT,
        "playback": Provider.FAST_AGENT,
        "slow": Provider.FAST_AGENT,
        "gpt-4o": Provider.OPENAI,
        "gpt-4o-mini": Provider.OPENAI,
        "gpt-4.1": Provider.OPENAI,
        "gpt-4.1-mini": Provider.OPENAI,
        "gpt-4.1-nano": Provider.OPENAI,
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
        "claude-opus-4-20250514": Provider.ANTHROPIC,
        "claude-sonnet-4-20250514": Provider.ANTHROPIC,
        "claude-sonnet-4-0": Provider.ANTHROPIC,
        "deepseek-chat": Provider.DEEPSEEK,
        "gemini-2.0-flash": Provider.GOOGLE,
        "gemini-2.5-flash-preview-05-20": Provider.GOOGLE,
        "gemini-2.5-pro-preview-05-06": Provider.GOOGLE,
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
        "opus": "claude-opus-4-0",
        "opus4": "claude-opus-4-0",
        "opus3": "claude-3-opus-latest",
        "deepseekv3": "deepseek-chat",
        "deepseek": "deepseek-chat",
        "gemini2": "gemini-2.0-flash",
        "gemini25": "gemini-2.5-flash-preview-05-20",
        "gemini25pro": "gemini-2.5-pro-preview-05-06",
    }

    # Mapping of providers to their LLM classes
    PROVIDER_CLASSES: Dict[Provider, LLMClass] = {
        Provider.ANTHROPIC: AnthropicAugmentedLLM,
        Provider.OPENAI: OpenAIAugmentedLLM,
        Provider.FAST_AGENT: PassthroughLLM,
        Provider.DEEPSEEK: DeepSeekAugmentedLLM,
        Provider.GENERIC: GenericAugmentedLLM,
        Provider.GOOGLE_OAI: GoogleOaiAugmentedLLM,
        Provider.GOOGLE: GoogleNativeAugmentedLLM,
        Provider.OPENROUTER: OpenRouterAugmentedLLM,
        Provider.TENSORZERO: TensorZeroAugmentedLLM,
        Provider.AZURE: AzureOpenAIAugmentedLLM,
        Provider.ALIYUN: AliyunAugmentedLLM,
    }

    # Mapping of special model names to their specific LLM classes
    # This overrides the provider-based class selection
    MODEL_SPECIFIC_CLASSES: Dict[str, LLMClass] = {
        "playback": PlaybackLLM,
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
    def create_factory(
        cls, model_string: str, request_params: Optional[RequestParams] = None
    ) -> Callable[..., AugmentedLLMProtocol]:
        """
        Creates a factory function that follows the attach_llm protocol.

        Args:
            model_string: The model specification string (e.g. "gpt-4.1")
            request_params: Optional parameters to configure LLM behavior

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
        ) -> AugmentedLLMProtocol:
            base_params = RequestParams()
            base_params.model = config.model_name
            if config.reasoning_effort:
                kwargs["reasoning_effort"] = config.reasoning_effort.value
            llm_args = {
                "agent": agent,
                "model": config.model_name,
                "request_params": request_params,
                **kwargs,
            }
            llm: AugmentedLLMProtocol = llm_class(**llm_args)
            return llm

        return factory
