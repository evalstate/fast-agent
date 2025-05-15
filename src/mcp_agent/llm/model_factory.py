from enum import Enum
from typing import Callable, Dict, Optional, Type, Union

from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM
from mcp_agent.llm.augmented_llm_playback import PlaybackLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_azure import AzureOpenAIAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_deepseek import DeepSeekAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_generic import GenericAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_google import GoogleAugmentedLLM
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
    Type[DeepSeekAugmentedLLM],
    Type[OpenRouterAugmentedLLM],
    Type[TensorZeroAugmentedLLM],
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

    # TODO -- add context window size information for display/management
    # TODO -- add audio supporting got-4o-audio-preview
    # TODO -- bring model parameter configuration here
    # Mapping of model names to their default providers
    DEFAULT_PROVIDERS = {
        "passthrough": Provider.FAST_AGENT,
        "playback": Provider.FAST_AGENT,
        "gpt-4o": Provider.OPENAI,
        "gpt-4o-mini": Provider.OPENAI,
        "gpt-4.1": Provider.OPENAI,
        "gpt-4.1-mini": Provider.OPENAI,
        "gpt-4.1-nano": Provider.OPENAI,
        "o1-mini": Provider.OPENAI,
        "o1": Provider.OPENAI,
        "o1-preview": Provider.OPENAI,
        "o3-mini": Provider.OPENAI,
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
        "deepseek-chat": Provider.DEEPSEEK,
        #        "deepseek-reasoner": Provider.DEEPSEEK, reinstate on release
    }

    MODEL_ALIASES = {
        "sonnet": "claude-3-7-sonnet-latest",
        "sonnet35": "claude-3-5-sonnet-latest",
        "sonnet37": "claude-3-7-sonnet-latest",
        "claude": "claude-3-7-sonnet-latest",
        "haiku": "claude-3-5-haiku-latest",
        "haiku3": "claude-3-haiku-20240307",
        "haiku35": "claude-3-5-haiku-latest",
        "opus": "claude-3-opus-latest",
        "opus3": "claude-3-opus-latest",
        "deepseekv3": "deepseek-chat",
        "deepseek": "deepseek-chat",
    }

    # Mapping of providers to their LLM classes
    PROVIDER_CLASSES: Dict[Provider, LLMClass] = {
        Provider.ANTHROPIC: AnthropicAugmentedLLM,
        Provider.OPENAI: OpenAIAugmentedLLM,
        Provider.FAST_AGENT: PassthroughLLM,
        Provider.DEEPSEEK: DeepSeekAugmentedLLM,
        Provider.GENERIC: GenericAugmentedLLM,
        Provider.GOOGLE: GoogleAugmentedLLM,  # type: ignore
        Provider.OPENROUTER: OpenRouterAugmentedLLM,
        Provider.TENSORZERO: TensorZeroAugmentedLLM,
        Provider.AZURE: AzureOpenAIAugmentedLLM,
    }

    # Mapping of special model names to their specific LLM classes
    # This overrides the provider-based class selection
    MODEL_SPECIFIC_CLASSES: Dict[str, LLMClass] = {
        "playback": PlaybackLLM,
    }

    @classmethod
    def parse_model_string(cls, model_string: str) -> ModelConfig:
        """Parse a model string into a ModelConfig object"""
        # Check if model string is an alias
        model_string = cls.MODEL_ALIASES.get(model_string, model_string)
        parts = model_string.split(".")

        # Start with all parts as the model name
        model_parts = parts.copy()
        provider = None
        reasoning_effort = None

        # Check last part for reasoning effort
        if len(parts) > 1 and parts[-1].lower() in cls.EFFORT_MAP:
            reasoning_effort = cls.EFFORT_MAP[parts[-1].lower()]
            model_parts = model_parts[:-1]

        # Check first part for provider
        if len(model_parts) > 1:
            potential_provider = model_parts[0]
            if any(provider.value == potential_provider for provider in Provider):
                provider = Provider(potential_provider)
                model_parts = model_parts[1:]

        if provider == Provider.TENSORZERO and not model_parts:
            raise ModelConfigError(
                f"TensorZero provider requires a function name after the provider "
                f"(e.g., tensorzero.my-function), got: {model_string}"
            )
        # Join remaining parts as model name
        model_name = ".".join(model_parts)

        # If no provider was found in the string, look it up in defaults
        if provider is None:
            provider = cls.DEFAULT_PROVIDERS.get(model_name)
            if provider is None:
                raise ModelConfigError(f"Unknown model: {model_name}")

        return ModelConfig(
            provider=provider, model_name=model_name, reasoning_effort=reasoning_effort
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
        # Parse configuration up front
        config = cls.parse_model_string(model_string)
        if config.model_name in cls.MODEL_SPECIFIC_CLASSES:
            llm_class = cls.MODEL_SPECIFIC_CLASSES[config.model_name]
        else:
            llm_class = cls.PROVIDER_CLASSES[config.provider]

        # Create a factory function matching the updated attach_llm protocol
        def factory(
            agent: Agent, request_params: Optional[RequestParams] = None, **kwargs
        ) -> AugmentedLLMProtocol:
            # Create base params with parsed model name
            base_params = RequestParams()
            base_params.model = config.model_name  # Use the parsed model name, not the alias

            # Add reasoning effort if available
            if config.reasoning_effort:
                kwargs["reasoning_effort"] = config.reasoning_effort.value

            # Forward all arguments to LLM constructor
            llm_args = {
                "agent": agent,
                "model": config.model_name,
                "request_params": request_params,
                **kwargs,
            }

            llm: AugmentedLLMProtocol = llm_class(**llm_args)
            return llm

        return factory
