from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Optional, Type, Union

from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM
from mcp_agent.llm.augmented_llm_playback import PlaybackLLM
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_deepseek import DeepSeekAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_generic import GenericAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_openrouter import AugmentedOpenRouterLLM
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol

# from mcp_agent.workflows.llm.augmented_llm_deepseek import DeekSeekAugmentedLLM


# Type alias for LLM classes
LLMClass = Union[
    Type[AnthropicAugmentedLLM],
    Type[OpenAIAugmentedLLM],
    Type[AugmentedOpenRouterLLM],
    Type[PassthroughLLM],
    Type[PlaybackLLM],
    Type[DeepSeekAugmentedLLM],
]


class Provider(Enum):
    """Supported LLM providers"""

    ANTHROPIC = auto()
    OPENAI = auto()
    OPENROUTER = auto()
    FAST_AGENT = auto()
    DEEPSEEK = auto()
    GENERIC = auto()


class ReasoningEffort(Enum):
    """Optional reasoning effort levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    provider: Provider
    model_name: str
    reasoning_effort: Optional[ReasoningEffort] = None
    original_specifier: Optional[str] = None


class ModelFactory:
    """Factory for creating LLM instances based on model specifications"""

    # Mapping of provider strings to enum values
    PROVIDER_MAP = {
        "anthropic": Provider.ANTHROPIC,
        "openai": Provider.OPENAI,
        "openrouter": Provider.OPENROUTER,
        "fast-agent": Provider.FAST_AGENT,
        "deepseek": Provider.DEEPSEEK,
        "generic": Provider.GENERIC,
    }

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
        "claude": "claude-3-5-sonnet-latest",
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
        Provider.OPENROUTER: AugmentedOpenRouterLLM,
        Provider.FAST_AGENT: PassthroughLLM,
        Provider.DEEPSEEK: DeepSeekAugmentedLLM,
        Provider.GENERIC: GenericAugmentedLLM,
    }

    # Mapping of special model names to their specific LLM classes
    # This overrides the provider-based class selection
    MODEL_SPECIFIC_CLASSES: Dict[str, LLMClass] = {
        "playback": PlaybackLLM,
    }

    @classmethod
    def parse_model_string(cls, model_string: str) -> ModelConfig:
        """Parse a model string into a ModelConfig object"""
        original_specifier = model_string
        # Check if model string is an alias (before provider check)
        model_string = cls.MODEL_ALIASES.get(model_string, model_string)

        provider = None
        model_name = model_string
        reasoning_effort = None

        # Check for explicit provider prefix (e.g., "openrouter:google/gemini...")
        if ":" in model_string:
            potential_provider, potential_model_name = model_string.split(":", 1)
            if potential_provider in cls.PROVIDER_MAP:
                provider = cls.PROVIDER_MAP[potential_provider]
                model_name = potential_model_name
                model_string = model_name # Continue parsing the rest for effort
            # else: Keep original model_string if prefix doesn't match known provider

        # Parse reasoning effort (if any)
        parts = model_string.split(".")
        model_parts = parts.copy()

        if len(parts) > 1 and parts[-1].lower() in cls.EFFORT_MAP:
            reasoning_effort = cls.EFFORT_MAP[parts[-1].lower()]
            model_parts = model_parts[:-1]
            model_name = ".".join(model_parts) # Update model name without effort suffix


        # If no provider was found yet, look it up in defaults
        if provider is None:
            provider = cls.DEFAULT_PROVIDERS.get(model_name)
            if provider is None:
                # Special case: if it wasn't found and contains '/', assume it's an OpenRouter model
                # This allows using names like 'google/gemini-flash' without the prefix
                # IF it's not in DEFAULT_PROVIDERS.
                if '/' in model_name:
                    provider = Provider.OPENROUTER
                    # original_specifier = f"openrouter:{model_name}" # Optional: reconstruct for clarity
                else:
                    raise ModelConfigError(f"Unknown model or provider for: {original_specifier}")

        return ModelConfig(
            provider=provider,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            original_specifier=original_specifier # Store original string
        )

    @classmethod
    def create_factory(
        cls, model_string: str, request_params: Optional[RequestParams] = None
    ) -> Callable[..., AugmentedLLMProtocol]:
        """
        Creates a factory function that follows the attach_llm protocol.

        Args:
            model_string: The model specification string (e.g. "gpt-4o.high")
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
            agent: Agent,
            request_params_override: Optional[RequestParams] = None, # Renamed for clarity
            **kwargs
        ) -> AugmentedLLMProtocol:

            # Merge request params: start with factory defaults, then override with function call params
            final_request_params = request_params.model_copy(deep=True) if request_params else RequestParams()
            if request_params_override:
                final_request_params = final_request_params.model_copy(update=request_params_override.model_dump(exclude_unset=True))

            # Ensure the model name from the parsed config is set in the final params
            # This is crucial for OpenRouter where model_name is the specific OR model ID
            final_request_params.model = config.model_name

            # Add reasoning effort if available
            if config.reasoning_effort:
                kwargs["reasoning_effort"] = config.reasoning_effort.value

            # Forward all arguments to LLM constructor
            llm_args = {
                "agent": agent,
                "request_params": final_request_params, # Pass merged params
                "model": config.model_name, # Pass the specific model name again for constructor
                **kwargs
            }

            llm: AugmentedLLMProtocol = llm_class(**llm_args)
            return llm

        return factory
