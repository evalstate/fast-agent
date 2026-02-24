import math
from typing import Literal, Type, Union
from urllib.parse import parse_qs

from pydantic import BaseModel

from fast_agent.core.exceptions import ModelConfigError
from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LLMFactoryProtocol
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.internal.playback import PlaybackLLM
from fast_agent.llm.internal.silent import SilentLLM
from fast_agent.llm.internal.slow import SlowLLM
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, parse_reasoning_setting
from fast_agent.llm.structured_output_mode import (
    StructuredOutputMode,
    parse_structured_output_mode,
)
from fast_agent.llm.text_verbosity import TextVerbosityLevel, parse_text_verbosity
from fast_agent.types import RequestParams

# Type alias for LLM classes
LLMClass = Union[Type[PassthroughLLM], Type[PlaybackLLM], Type[SilentLLM], Type[SlowLLM], type]


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    provider: Provider
    model_name: str
    reasoning_effort: ReasoningEffortSetting | None = None
    text_verbosity: TextVerbosityLevel | None = None
    structured_output_mode: StructuredOutputMode | None = None
    long_context: bool = False
    transport: Literal["sse", "websocket", "auto"] | None = None
    web_search: bool | None = None
    web_fetch: bool | None = None
    temperature: float | None = None


class ModelFactory:
    """Factory for creating LLM instances based on model specifications"""

    MODEL_ALIASES = {
        "gpt51": "responses.gpt-5.1",
        "gpt52": "responses.gpt-5.2",
        "codex": "responses.gpt-5.2-codex",
        "codexplan": "codexresponses.gpt-5.3-codex",
        "codexplan52": "codexresponses.gpt-5.2-codex",
        "codexspark": "codexresponses.gpt-5.3-codex-spark",
        "sonnet": "claude-sonnet-4-6",
        "sonnet4": "claude-sonnet-4-0",
        "sonnet45": "claude-sonnet-4-5",
        "sonnet46": "claude-sonnet-4-6",
        "sonnet35": "claude-3-5-sonnet-latest",
        "sonnet37": "claude-3-7-sonnet-latest",
        "claude": "claude-sonnet-4-6",
        "haiku": "claude-haiku-4-5",
        "haiku3": "claude-3-haiku-20240307",
        "haiku35": "claude-3-5-haiku-latest",
        "haiku45": "claude-haiku-4-5",
        "opus": "claude-opus-4-6",
        "opus4": "claude-opus-4-1",
        "opus45": "claude-opus-4-5",
        "opus46": "claude-opus-4-6",
        "opus3": "claude-3-opus-latest",
        "deepseekv3": "deepseek-chat",
        "deepseek3": "deepseek-chat",
        "deepseek": "deepseek-chat",
        "gemini2": "gemini-2.0-flash",
        "gemini25": "gemini-2.5-flash-preview-09-2025",
        "gemini25pro": "gemini-2.5-pro",
        "gemini3": "gemini-3-pro-preview",
        "gemini3.1": "gemini-3.1-pro-preview",
        "gemini3flash": "gemini-3-flash-preview",
        "grok-4-fast": "xai.grok-4-fast-non-reasoning",
        "grok-4-fast-reasoning": "xai.grok-4-fast-reasoning",
        "kimigroq": "groq.moonshotai/kimi-k2-instruct-0905",
        "minimax": "hf.MiniMaxAI/MiniMax-M2.1:novita",
        "kimi": "hf.moonshotai/Kimi-K2-Instruct-0905:groq",
        "gpt-oss": "hf.openai/gpt-oss-120b:cerebras",
        "gpt-oss-20b": "hf.openai/gpt-oss-20b",
        "glm47": "hf.zai-org/GLM-4.7:cerebras",
        "glm5": "hf.zai-org/GLM-5:novita",
        "glm": "hf.zai-org/GLM-5:novita",
        "qwen3": "hf.Qwen/Qwen3-Next-80B-A3B-Instruct:together",
        "deepseek31": "hf.deepseek-ai/DeepSeek-V3.1",
        "kimithink": "hf.moonshotai/Kimi-K2-Thinking:together",
        "deepseek32": "hf.deepseek-ai/DeepSeek-V3.2:fireworks-ai",
        "kimi25": "hf.moonshotai/Kimi-K2.5:fireworks-ai",
        "kimi-2.5": "hf.moonshotai/Kimi-K2.5:fireworks-ai",
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
    def register_runtime_model_provider(cls, model_name: str, provider: Provider) -> None:
        """Register or override a runtime default provider for an unprefixed model name."""
        ModelDatabase.register_runtime_default_provider(model_name, provider)

    @classmethod
    def unregister_runtime_model_provider(cls, model_name: str) -> None:
        """Remove a runtime default provider override."""
        ModelDatabase.unregister_runtime_default_provider(model_name)

    @classmethod
    def register_runtime_model(
        cls,
        model_name: str,
        provider: Provider,
        llm_class: LLMClass | None = None,
    ) -> None:
        """Register a runtime model provider and optional model-specific class."""
        cls.register_runtime_model_provider(model_name, provider)
        if llm_class is not None:
            cls.MODEL_SPECIFIC_CLASSES[model_name] = llm_class

    @classmethod
    def unregister_runtime_model(cls, model_name: str) -> None:
        """Remove runtime model provider and model-specific class overrides."""
        cls.unregister_runtime_model_provider(model_name)
        cls.MODEL_SPECIFIC_CLASSES.pop(model_name, None)

    @classmethod
    def get_runtime_aliases(cls) -> dict[str, str]:
        """Return parser aliases, including curated catalog aliases."""
        aliases = dict(cls.MODEL_ALIASES)

        from fast_agent.llm.model_selection import ModelSelectionCatalog

        for entry in ModelSelectionCatalog.list_current_entries():
            alias = entry.alias.strip()
            if not alias or "?" in entry.model:
                continue
            aliases.setdefault(alias, entry.model)

        return aliases

    @classmethod
    def parse_model_string(
        cls, model_string: str, aliases: dict[str, str] | None = None
    ) -> ModelConfig:
        """Parse a model string into a ModelConfig object

        Args:
            model_string: The model specification string (e.g. "gpt-4.1", "kimi:groq")
            aliases: Optional custom aliases map. Defaults to MODEL_ALIASES.
        """
        if aliases is None:
            aliases = cls.get_runtime_aliases()

        query_setting: ReasoningEffortSetting | None = None
        query_structured: StructuredOutputMode | None = None
        query_text_verbosity: TextVerbosityLevel | None = None
        query_instant: bool | None = None
        query_long_context: bool = False
        query_transport: Literal["sse", "websocket", "auto"] | None = None
        query_web_search: bool | None = None
        query_web_fetch: bool | None = None
        query_temperature: float | None = None

        def _parse_on_off_query(raw_value: str, query_key: str) -> bool:
            parsed = parse_reasoning_setting(raw_value)
            if parsed is not None and parsed.kind == "toggle":
                return bool(parsed.value)
            raise ModelConfigError(
                f"Invalid {query_key} query value: '{raw_value}' in '{model_string}'. "
                "Use on/off (or true/false, 1/0)."
            )

        if "?" in model_string:
            model_string, _, query = model_string.partition("?")
            query_params = parse_qs(query)
            if "reasoning" in query_params:
                values = query_params.get("reasoning") or []
                raw_value = values[-1] if values else ""
                query_setting = parse_reasoning_setting(raw_value)
                if query_setting is None:
                    raise ModelConfigError(
                        f"Invalid reasoning query value: '{raw_value}' in '{model_string}'"
                    )
            if "verbosity" in query_params:
                values = query_params.get("verbosity") or []
                raw_value = values[-1] if values else ""
                query_text_verbosity = parse_text_verbosity(raw_value)
                if query_text_verbosity is None:
                    raise ModelConfigError(
                        f"Invalid verbosity query value: '{raw_value}' in '{model_string}'"
                    )
            if "structured" in query_params:
                values = query_params.get("structured") or []
                raw_value = values[-1] if values else ""
                query_structured = parse_structured_output_mode(raw_value)
                if query_structured is None:
                    raise ModelConfigError(
                        f"Invalid structured query value: '{raw_value}' in '{model_string}'"
                    )
            if "instant" in query_params:
                values = query_params.get("instant") or []
                raw_value = values[-1] if values else ""
                instant_setting = parse_reasoning_setting(raw_value)
                if instant_setting is None or instant_setting.kind != "toggle":
                    raise ModelConfigError(
                        f"Invalid instant query value: '{raw_value}' in '{model_string}'"
                    )
                query_instant = bool(instant_setting.value)
            if "context" in query_params:
                values = query_params.get("context") or []
                raw_value = (values[-1] if values else "").strip().lower()
                if raw_value == "1m":
                    query_long_context = True
                else:
                    raise ModelConfigError(
                        f"Invalid context query value: '{raw_value}' \u2014 only '1m' is supported"
                    )
            if "transport" in query_params:
                values = query_params.get("transport") or []
                raw_value = (values[-1] if values else "").strip().lower()
                transport_aliases: dict[str, Literal["sse", "websocket", "auto"]] = {
                    "ws": "websocket",
                    "websocket": "websocket",
                    "sse": "sse",
                    "auto": "auto",
                }
                normalized_transport = transport_aliases.get(raw_value)
                if normalized_transport is None:
                    raise ModelConfigError(
                        f"Invalid transport query value: '{raw_value}' in '{model_string}'"
                    )
                query_transport = normalized_transport
            if "web_search" in query_params:
                values = query_params.get("web_search") or []
                raw_value = values[-1] if values else ""
                query_web_search = _parse_on_off_query(raw_value, "web_search")
            if "web_fetch" in query_params:
                values = query_params.get("web_fetch") or []
                raw_value = values[-1] if values else ""
                query_web_fetch = _parse_on_off_query(raw_value, "web_fetch")
            if "temperature" in query_params or "temp" in query_params:
                values = query_params.get("temperature") or query_params.get("temp") or []
                raw_value = values[-1] if values else ""
                try:
                    parsed_temperature = float(raw_value)
                except ValueError as exc:
                    raise ModelConfigError(
                        f"Invalid temperature query value: '{raw_value}' in '{model_string}'"
                    ) from exc

                if not math.isfinite(parsed_temperature):
                    raise ModelConfigError(
                        f"Invalid temperature query value: '{raw_value}' in '{model_string}'"
                    )
                query_temperature = parsed_temperature

        suffix: str | None = None
        if ":" in model_string:
            base, suffix = model_string.rsplit(":", 1)
            if base:
                model_string = base

        model_string = aliases.get(model_string, model_string)

        # If user provided a suffix (e.g., kimi:groq), strip any existing suffix
        # from the resolved alias (e.g., hf.model:cerebras -> hf.model)
        if suffix and ":" in model_string:
            model_string = model_string.rsplit(":", 1)[0]
        provider_override: Provider | None = None
        if "/" in model_string:
            prefix, rest = model_string.split("/", 1)
            if prefix and rest and any(p.value == prefix for p in Provider):
                provider_override = Provider(prefix)
                model_string = rest

        parts = model_string.split(".")

        model_name_str = model_string  # Default full string as model name initially
        provider: Provider | None = provider_override
        reasoning_effort = query_setting
        parts_for_provider_model = []

        # Check for reasoning effort first (last part)
        if len(parts) > 1 and parse_reasoning_setting(parts[-1].lower()):
            suffix_setting = parse_reasoning_setting(parts[-1].lower())
            if suffix_setting and suffix_setting.kind == "effort":
                if query_setting is not None:
                    raise ModelConfigError(
                        f"Multiple reasoning settings provided for '{model_string}'."
                    )
                reasoning_effort = suffix_setting
                parts_for_provider_model = parts[:-1]
            else:
                parts_for_provider_model = parts[:]
        else:
            parts_for_provider_model = parts[:]

        # Try to match longest possible provider string
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

        # Construct model_name from remaining parts
        if identified_provider_parts > 0:
            model_name_str = ".".join(parts_for_provider_model[identified_provider_parts:])
        else:
            # If no provider prefix was matched, the whole string (after effort removal) is the model name
            model_name_str = ".".join(parts_for_provider_model)

        # If provider still None, try to resolve from model metadata.
        if provider is None:
            provider = ModelDatabase.get_default_provider(model_name_str)

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

        if query_instant is not None:
            if reasoning_effort is not None:
                raise ModelConfigError(
                    f"Multiple reasoning settings provided for '{model_string}'."
                )
            base_model = model_name_str.rsplit(":", 1)[0].strip().lower()
            if base_model != "moonshotai/kimi-k2.5":
                raise ModelConfigError(
                    f"Instant mode is only supported for moonshotai/kimi-k2.5, got '{model_name_str}'."
                )
            reasoning_effort = ReasoningEffortSetting(kind="toggle", value=not query_instant)

        if query_transport in {"websocket", "auto"}:
            if provider not in {Provider.CODEX_RESPONSES, Provider.RESPONSES}:
                raise ModelConfigError(
                    "WebSocket transport is experimental and currently supported only for "
                    "the codexresponses and responses providers."
                )
            supports_transport = ModelDatabase.supports_response_transport(
                model_name_str, "websocket"
            )
            if supports_transport is False:
                raise ModelConfigError(
                    f"Transport '{query_transport}' is not supported for model '{model_name_str}'."
                )
            supports_provider = ModelDatabase.supports_response_websocket_provider(
                model_name_str, provider
            )
            if supports_provider is False:
                raise ModelConfigError(
                    f"Transport '{query_transport}' is not supported for model '{model_name_str}' "
                    f"with provider '{provider.value}'."
                )

        return ModelConfig(
            provider=provider,
            model_name=model_name_str,
            reasoning_effort=reasoning_effort,
            text_verbosity=query_text_verbosity,
            structured_output_mode=query_structured,
            long_context=query_long_context,
            transport=query_transport,
            web_search=query_web_search,
            web_fetch=query_web_fetch,
            temperature=query_temperature,
        )

    @classmethod
    def create_factory(
        cls, model_string: str, aliases: dict[str, str] | None = None
    ) -> LLMFactoryProtocol:
        """
        Creates a factory function that follows the attach_llm protocol.

        Args:
            model_string: The model specification string (e.g. "gpt-4.1")
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
            effective_request_params = request_params

            if config.temperature is not None:
                if effective_request_params is None:
                    effective_request_params = RequestParams(temperature=config.temperature)
                else:
                    effective_request_params = effective_request_params.model_copy(
                        update={"temperature": config.temperature}
                    )

            if config.reasoning_effort:
                kwargs["reasoning_effort"] = config.reasoning_effort
            if config.text_verbosity:
                kwargs["text_verbosity"] = config.text_verbosity
            if config.structured_output_mode:
                kwargs["structured_output_mode"] = config.structured_output_mode
            if config.long_context:
                kwargs["long_context"] = True
            if config.transport:
                kwargs["transport"] = config.transport
            if config.web_search is not None and config.provider in {
                Provider.ANTHROPIC,
                Provider.RESPONSES,
                Provider.OPENRESPONSES,
                Provider.CODEX_RESPONSES,
            }:
                kwargs["web_search"] = config.web_search
            if config.web_fetch is not None and config.provider == Provider.ANTHROPIC:
                kwargs["web_fetch"] = config.web_fetch
            llm_args = {
                "model": config.model_name,
                "request_params": effective_request_params,
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
            if provider == Provider.CODEX_RESPONSES:
                from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM

                return CodexResponsesLLM
            if provider == Provider.OPENRESPONSES:
                from fast_agent.llm.provider.openai.openresponses import OpenResponsesLLM

                return OpenResponsesLLM

        except Exception as e:
            raise ModelConfigError(
                f"Provider '{provider.value}' is unavailable or missing dependencies: {e}"
            )
        raise ModelConfigError(f"Unsupported provider: {provider}")
