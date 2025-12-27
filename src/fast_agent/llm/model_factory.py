from enum import Enum
from typing import Any, Type, Union
from urllib.parse import parse_qs

from pydantic import BaseModel, ConfigDict, Field

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


class ModelOptions(BaseModel):
    """
    Parsed model options from query-string style parameters.

    Options can be specified in the model string as:
        model-name?option1=value1&option2=value2

    Known options:
        - reasoning: low|medium|high|minimal (OpenAI-style reasoning effort)
        - thinking: on|off (Anthropic extended thinking toggle)
        - thinking_budget: int (Anthropic extended thinking token budget)

    Unknown options are stored in `extra` and passed through to the provider.
    """

    model_config = ConfigDict(extra="allow")

    # Reasoning effort for OpenAI o-series models
    reasoning: str | None = None

    # Anthropic extended thinking options
    thinking: str | None = None
    thinking_budget: int | None = None

    # Flag to control strict validation of options
    # When True (default), unknown options that don't match known_options will error
    # When False, unknown options are passed through without validation
    strict: bool = True

    # Store any extra/unknown options
    extra: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_query_string(cls, query: str, strict: bool = True) -> "ModelOptions":
        """Parse options from a query string (without leading '?')."""
        if not query:
            return cls(strict=strict)

        parsed = parse_qs(query, keep_blank_values=True)
        opts: dict[str, Any] = {"strict": strict, "extra": {}}

        for key, values in parsed.items():
            # Take the last value if multiple are specified
            value = values[-1] if values else ""
            key_lower = key.lower()

            if key_lower == "reasoning":
                opts["reasoning"] = value.lower()
            elif key_lower == "thinking":
                opts["thinking"] = value.lower()
            elif key_lower == "thinking_budget":
                try:
                    opts["thinking_budget"] = int(value)
                except ValueError:
                    raise ModelConfigError(
                        f"Invalid thinking_budget value: {value}. Must be an integer."
                    )
            elif key_lower == "strict":
                opts["strict"] = value.lower() in ("true", "1", "yes", "on")
            else:
                # Store as extra option
                opts["extra"][key] = value

        return cls(**opts)

    def get_reasoning_effort(self) -> "ReasoningEffort | None":
        """Convert reasoning string to ReasoningEffort enum."""
        if not self.reasoning:
            return None
        effort_map = {
            "minimal": ReasoningEffort.MINIMAL,
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }
        effort = effort_map.get(self.reasoning.lower())
        if effort is None and self.strict:
            raise ModelConfigError(
                f"Unknown reasoning effort: {self.reasoning}. "
                f"Valid values: {', '.join(effort_map.keys())}"
            )
        return effort

    def is_thinking_enabled(self) -> bool | None:
        """Check if thinking is explicitly enabled/disabled."""
        if self.thinking is None:
            return None
        return self.thinking.lower() in ("on", "true", "1", "yes", "enabled")


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    provider: Provider
    model_name: str
    reasoning_effort: ReasoningEffort | None = None
    options: ModelOptions = Field(default_factory=ModelOptions)


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
        "gpt-5.1": Provider.OPENAI,
        "gpt-5.1-mini": Provider.OPENAI,
        "gpt-5.1-nano": Provider.OPENAI,
        "gpt-5-mini": Provider.OPENAI,
        "gpt-5-nano": Provider.OPENAI,
        "gpt-5.2": Provider.OPENAI,
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
        "claude-opus-4-5": Provider.ANTHROPIC,
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
        "gemini-3-pro-preview": Provider.GOOGLE,
        "gemini-3-flash-preview": Provider.GOOGLE,
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
        "gpt51": "openai.gpt-5.1",
        "gpt52": "openai.gpt-5.2",
        "sonnet": "claude-sonnet-4-5",
        "sonnet4": "claude-sonnet-4-0",
        "sonnet45": "claude-sonnet-4-5",
        "sonnet35": "claude-3-5-sonnet-latest",
        "sonnet37": "claude-3-7-sonnet-latest",
        "claude": "claude-sonnet-4-5",
        "haiku": "claude-haiku-4-5",
        "haiku3": "claude-3-haiku-20240307",
        "haiku35": "claude-3-5-haiku-latest",
        "haiku45": "claude-haiku-4-5",
        "opus": "claude-opus-4-5",
        "opus4": "claude-opus-4-1",
        "opus45": "claude-opus-4-5",
        "opus3": "claude-3-opus-latest",
        "deepseekv3": "deepseek-chat",
        "deepseek3": "deepseek-chat",
        "deepseek": "deepseek-chat",
        "gemini2": "gemini-2.0-flash",
        "gemini25": "gemini-2.5-flash-preview-09-2025",
        "gemini25pro": "gemini-2.5-pro",
        "gemini3": "gemini-3-pro-preview",
        "gemini3flash": "gemini-3-flash-preview",
        "grok-4-fast": "xai.grok-4-fast-non-reasoning",
        "grok-4-fast-reasoning": "xai.grok-4-fast-reasoning",
        "kimigroq": "groq.moonshotai/kimi-k2-instruct-0905",
        "minimax": "hf.MiniMaxAI/MiniMax-M2.1:novita",
        "kimi": "hf.moonshotai/Kimi-K2-Instruct-0905:groq",
        "gpt-oss": "hf.openai/gpt-oss-120b:cerebras",
        "gpt-oss-20b": "hf.openai/gpt-oss-20b",
        "glm": "hf.zai-org/GLM-4.7:zai-org:novita",
        "qwen3": "hf.Qwen/Qwen3-Next-80B-A3B-Instruct:together",
        "deepseek31": "hf.deepseek-ai/DeepSeek-V3.1",
        "kimithink": "hf.moonshotai/Kimi-K2-Thinking:together",
        "deepseek32": "deepseek-ai/DeepSeek-V3.2-Exp:novita",
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
    PROVIDER_CLASSES: dict[Provider, LLMClass] = {}

    # Mapping of special model names to their specific LLM classes
    # This overrides the provider-based class selection
    MODEL_SPECIFIC_CLASSES: dict[str, LLMClass] = {
        "playback": PlaybackLLM,
        "silent": SilentLLM,
        "slow": SlowLLM,
    }

    @classmethod
    def parse_model_string(
        cls, model_string: str, aliases: dict[str, str] | None = None
    ) -> ModelConfig:
        """Parse a model string into a ModelConfig object

        Supports two formats:
        1. Legacy format: [provider].[model-name].[reasoning][:downstream-provider]
        2. New format: [provider].[model-name][:downstream-provider][?option1=foo&option2=bar]

        Args:
            model_string: The model specification string (e.g. "gpt-4.1", "kimi:groq",
                         "openai.o1?reasoning=high", "anthropic.claude-opus-4-5?thinking=on")
            aliases: Optional custom aliases map. Defaults to MODEL_ALIASES.
        """
        if aliases is None:
            aliases = cls.MODEL_ALIASES

        # Step 1: Extract query-string options (new format)
        options_str: str = ""
        if "?" in model_string:
            model_string, options_str = model_string.split("?", 1)

        # Step 2: Extract downstream provider suffix (e.g., :groq)
        suffix: str | None = None
        if ":" in model_string:
            base, suffix = model_string.rsplit(":", 1)
            if base:
                model_string = base

        # Step 3: Apply aliases
        model_string = aliases.get(model_string, model_string)

        # If user provided a suffix (e.g., kimi:groq), strip any existing suffix
        # from the resolved alias (e.g., hf.model:cerebras -> hf.model)
        if suffix and ":" in model_string:
            model_string = model_string.rsplit(":", 1)[0]

        # Step 4: Handle slash-notation provider prefix (e.g., openai/gpt-4.1)
        provider_override: Provider | None = None
        if "/" in model_string:
            prefix, rest = model_string.split("/", 1)
            if prefix and rest and any(p.value == prefix for p in Provider):
                provider_override = Provider(prefix)
                model_string = rest

        parts = model_string.split(".")

        model_name_str = model_string  # Default full string as model name initially
        provider: Provider | None = provider_override
        reasoning_effort: ReasoningEffort | None = None
        parts_for_provider_model = []

        # Step 5: Check for legacy reasoning effort suffix (last part, e.g., "o1.high")
        # This is for backward compatibility - new format uses ?reasoning=high
        legacy_reasoning_detected = False
        if len(parts) > 1 and parts[-1].lower() in cls.EFFORT_MAP:
            reasoning_effort = cls.EFFORT_MAP[parts[-1].lower()]
            legacy_reasoning_detected = True
            # Remove effort from parts list for provider/model name determination
            parts_for_provider_model = parts[:-1]
        else:
            parts_for_provider_model = parts[:]

        # Step 6: Try to match longest possible provider string
        identified_provider_parts = 0  # How many parts belong to the provider string

        if provider is None and len(parts_for_provider_model) >= 2:
            potential_provider_str = f"{parts_for_provider_model[0]}.{parts_for_provider_model[1]}"
            if any(p.value == potential_provider_str for p in Provider):
                provider = Provider(potential_provider_str)
                identified_provider_parts = 2

        if provider is None and len(parts_for_provider_model) >= 1:
            potential_provider_str = parts_for_provider_model[0]
            if any(p.value == potential_provider_str for p in Provider):
                provider = Provider(potential_provider_str)
                identified_provider_parts = 1

        # Step 7: Construct model_name from remaining parts
        if identified_provider_parts > 0:
            model_name_str = ".".join(parts_for_provider_model[identified_provider_parts:])
        else:
            # If no provider prefix was matched, the whole string (after effort removal) is the model name
            model_name_str = ".".join(parts_for_provider_model)

        # Step 8: If provider still None, try to get from DEFAULT_PROVIDERS
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

        # Step 9: Re-attach downstream provider suffix
        if suffix:
            model_name_str = f"{model_name_str}:{suffix}"

        # Step 10: Parse query-string options
        options = ModelOptions.from_query_string(options_str)

        # Step 11: Merge reasoning from options if not set via legacy format
        # Options take precedence over legacy format if both are specified
        if options.reasoning:
            option_effort = options.get_reasoning_effort()
            if option_effort:
                reasoning_effort = option_effort
        elif legacy_reasoning_detected and not options.reasoning:
            # Populate options.reasoning from legacy format for consistency
            if reasoning_effort:
                options.reasoning = reasoning_effort.value

        return ModelConfig(
            provider=provider,
            model_name=model_name_str,
            reasoning_effort=reasoning_effort,
            options=options,
        )

    @classmethod
    def create_factory(
        cls, model_string: str, aliases: dict[str, str] | None = None
    ) -> LLMFactoryProtocol:
        """
        Creates a factory function that follows the attach_llm protocol.

        Args:
            model_string: The model specification string (e.g. "gpt-4.1",
                         "openai.o1?reasoning=high", "anthropic.claude-opus-4-5?thinking=on")
            aliases: Optional custom aliases map. Defaults to MODEL_ALIASES.

        Returns:
            A callable that takes an agent parameter and returns an LLM instance
        """
        config = cls.parse_model_string(model_string, aliases=aliases)

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
            agent: AgentProtocol, request_params: RequestParams | None = None, **kwargs
        ) -> FastAgentLLMProtocol:
            base_params = RequestParams()
            base_params.model = config.model_name
            if config.reasoning_effort:
                kwargs["reasoning_effort"] = config.reasoning_effort.value

            # Pass model options to LLM for provider-specific handling
            kwargs["model_options"] = config.options

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
