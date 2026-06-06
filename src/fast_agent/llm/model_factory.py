import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, ClassVar, Literal, Protocol, Self
from urllib.parse import parse_qsl

from pydantic import BaseModel

from fast_agent.core.exceptions import ModelConfigError
from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LLMFactoryProtocol
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import (
    ReasoningEffortSetting,
    parse_reasoning_setting,
)
from fast_agent.llm.request_params import is_structured_tool_policy
from fast_agent.llm.resolved_model import ResolvedModelSpec, resolve_base_model_params
from fast_agent.llm.structured_output_mode import (
    StructuredOutputMode,
    parse_structured_output_mode,
)
from fast_agent.llm.task_budget import parse_task_budget_tokens, validate_task_budget_tokens
from fast_agent.llm.text_verbosity import TextVerbosityLevel, parse_text_verbosity
from fast_agent.types import RequestParams, StructuredToolPolicy
from fast_agent.utils.action_normalization import parse_boolean_alias
from fast_agent.utils.count_display import plural_label
from fast_agent.utils.text import strip_casefold


class LLMClass(Protocol):
    """Constructor for an LLM implementation."""

    @property
    def __name__(self) -> str: ...

    def __call__(self, **kwargs: Any) -> FastAgentLLMProtocol: ...


TransportSetting = Literal["sse", "websocket", "auto"]
ServiceTierSetting = Literal["fast", "flex"]
ModelQueryPairs = Sequence[tuple[str, str]]
_TRANSPORT_QUERY_VALUES: dict[str, TransportSetting] = {
    "ws": "websocket",
    "websocket": "websocket",
    "sse": "sse",
    "auto": "auto",
}
_SERVICE_TIER_QUERY_VALUES: dict[str, ServiceTierSetting] = {
    "fast": "fast",
    "flex": "flex",
}
_WEBSOCKET_TRANSPORT_PROVIDERS = (
    Provider.CODEX_RESPONSES,
    Provider.RESPONSES,
    Provider.XAI,
)
_FLEX_SERVICE_TIER_MODEL_CHECK_PROVIDERS = (
    Provider.RESPONSES,
    Provider.OPENRESPONSES,
)
_SINGLE_VALUE_MODEL_QUERY_KEYS = (
    "reasoning",
    "verbosity",
    "structured",
    "instant",
    "context",
    "transport",
    "service_tier",
)
_STRUCTURED_TOOL_QUERY_KEYS = (
    "structured_tools",
    "structuredToolPolicy",
    "structured_tool_policy",
)
_WEB_TOOL_QUERY_KEYS = ("web_search", "x_search", "web_fetch")
_TASK_BUDGET_QUERY_KEYS = ("task_budget", "taskBudget")
_SAMPLING_QUERY_KEYS = {
    "temperature": ("temperature", "temp"),
    "top_p": ("top_p", "topP"),
    "top_k": ("top_k", "topK"),
    "min_p": ("min_p", "minP"),
    "presence_penalty": ("presence_penalty", "presencePenalty"),
    "repetition_penalty": ("repetition_penalty", "repetitionPenalty"),
}
SUPPORTED_MODEL_QUERY_KEYS = frozenset(
    (
        *_SINGLE_VALUE_MODEL_QUERY_KEYS,
        *_STRUCTURED_TOOL_QUERY_KEYS,
        *_WEB_TOOL_QUERY_KEYS,
        *_TASK_BUDGET_QUERY_KEYS,
        *(
            key
            for key_group in _SAMPLING_QUERY_KEYS.values()
            for key in key_group
        ),
    )
)
_PROVIDER_CLASS_PATHS: dict[Provider, tuple[str, str]] = {
    Provider.FAST_AGENT: ("fast_agent.llm.internal.passthrough", "PassthroughLLM"),
    Provider.ANTHROPIC: ("fast_agent.llm.provider.anthropic.llm_anthropic", "AnthropicLLM"),
    Provider.ANTHROPIC_VERTEX: (
        "fast_agent.llm.provider.anthropic.llm_anthropic_vertex",
        "AnthropicVertexLLM",
    ),
    Provider.OPENAI: ("fast_agent.llm.provider.openai.llm_openai", "OpenAILLM"),
    Provider.DEEPSEEK: ("fast_agent.llm.provider.openai.llm_deepseek", "DeepSeekLLM"),
    Provider.GENERIC: ("fast_agent.llm.provider.openai.llm_generic", "GenericLLM"),
    Provider.GOOGLE_OAI: ("fast_agent.llm.provider.openai.llm_google_oai", "GoogleOaiLLM"),
    Provider.GOOGLE: ("fast_agent.llm.provider.google.llm_google_native", "GoogleNativeLLM"),
    Provider.HUGGINGFACE: (
        "fast_agent.llm.provider.openai.llm_huggingface",
        "HuggingFaceLLM",
    ),
    Provider.XAI: ("fast_agent.llm.provider.openai.xai_responses", "XAIResponsesLLM"),
    Provider.OPENROUTER: ("fast_agent.llm.provider.openai.llm_openrouter", "OpenRouterLLM"),
    Provider.TENSORZERO: (
        "fast_agent.llm.provider.openai.llm_tensorzero_openai",
        "TensorZeroOpenAILLM",
    ),
    Provider.AZURE: ("fast_agent.llm.provider.openai.llm_azure", "AzureOpenAILLM"),
    Provider.ALIYUN: ("fast_agent.llm.provider.openai.llm_aliyun", "AliyunLLM"),
    Provider.BEDROCK: ("fast_agent.llm.provider.bedrock.llm_bedrock", "BedrockLLM"),
    Provider.GROQ: ("fast_agent.llm.provider.openai.llm_groq", "GroqLLM"),
    Provider.RESPONSES: ("fast_agent.llm.provider.openai.responses", "ResponsesLLM"),
    Provider.CODEX_RESPONSES: (
        "fast_agent.llm.provider.openai.codex_responses",
        "CodexResponsesLLM",
    ),
    Provider.OPENRESPONSES: (
        "fast_agent.llm.provider.openai.openresponses",
        "OpenResponsesLLM",
    ),
}
_MODEL_SPECIFIC_CLASS_PATHS: dict[str, tuple[str, str]] = {
    "playback": ("fast_agent.llm.internal.playback", "PlaybackLLM"),
    "silent": ("fast_agent.llm.internal.silent", "SilentLLM"),
    "slow": ("fast_agent.llm.internal.slow", "SlowLLM"),
}


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    provider: Provider
    model_name: str
    reasoning_effort: ReasoningEffortSetting | None = None
    text_verbosity: TextVerbosityLevel | None = None
    structured_output_mode: StructuredOutputMode | None = None
    structured_tool_policy: StructuredToolPolicy | None = None
    long_context: bool = False
    transport: TransportSetting | None = None
    service_tier: ServiceTierSetting | None = None
    web_search: bool | None = None
    x_search: bool | None = None
    web_fetch: bool | None = None
    task_budget_tokens: int | None = None
    task_budget_configured: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None


@dataclass(frozen=True, slots=True)
class ModelQueryOverrides:
    """Typed query overrides parsed from a model spec query string."""

    reasoning_effort: ReasoningEffortSetting | None = None
    instant: bool | None = None
    text_verbosity: TextVerbosityLevel | None = None
    structured_output_mode: StructuredOutputMode | None = None
    structured_tool_policy: StructuredToolPolicy | None = None
    long_context: bool = False
    transport: TransportSetting | None = None
    service_tier: ServiceTierSetting | None = None
    web_search: bool | None = None
    x_search: bool | None = None
    web_fetch: bool | None = None
    task_budget_tokens: int | None = None
    task_budget_configured: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None

    def with_defaults(self, defaults: Self) -> "ModelQueryOverrides":
        """Return a copy with unset values filled from defaults."""
        def coalesce[T](value: T | None, default: T | None) -> T | None:
            return value if value is not None else default

        return ModelQueryOverrides(
            reasoning_effort=coalesce(self.reasoning_effort, defaults.reasoning_effort),
            instant=coalesce(self.instant, defaults.instant),
            text_verbosity=coalesce(self.text_verbosity, defaults.text_verbosity),
            structured_output_mode=coalesce(
                self.structured_output_mode, defaults.structured_output_mode
            ),
            structured_tool_policy=coalesce(
                self.structured_tool_policy, defaults.structured_tool_policy
            ),
            long_context=self.long_context or defaults.long_context,
            transport=coalesce(self.transport, defaults.transport),
            service_tier=coalesce(self.service_tier, defaults.service_tier),
            web_search=coalesce(self.web_search, defaults.web_search),
            x_search=coalesce(self.x_search, defaults.x_search),
            web_fetch=coalesce(self.web_fetch, defaults.web_fetch),
            task_budget_tokens=(
                self.task_budget_tokens
                if self.task_budget_configured
                else defaults.task_budget_tokens
            ),
            task_budget_configured=self.task_budget_configured or defaults.task_budget_configured,
            temperature=coalesce(self.temperature, defaults.temperature),
            top_p=coalesce(self.top_p, defaults.top_p),
            top_k=coalesce(self.top_k, defaults.top_k),
            min_p=coalesce(self.min_p, defaults.min_p),
            presence_penalty=coalesce(self.presence_penalty, defaults.presence_penalty),
            repetition_penalty=coalesce(self.repetition_penalty, defaults.repetition_penalty),
        )


@dataclass(frozen=True, slots=True)
class ParsedModelSpec:
    """Canonical parsed representation of a model specification string."""

    raw_input: str
    expanded_input: str
    provider: Provider
    model_name: str
    reasoning_effort: ReasoningEffortSetting | None
    query_overrides: ModelQueryOverrides

    def to_model_config(self) -> ModelConfig:
        """Convert the parsed spec into the public ModelConfig object."""
        return ModelConfig(
            provider=self.provider,
            model_name=self.model_name,
            reasoning_effort=self.reasoning_effort,
            text_verbosity=self.query_overrides.text_verbosity,
            structured_output_mode=self.query_overrides.structured_output_mode,
            structured_tool_policy=self.query_overrides.structured_tool_policy,
            long_context=self.query_overrides.long_context,
            transport=self.query_overrides.transport,
            service_tier=self.query_overrides.service_tier,
            web_search=self.query_overrides.web_search,
            x_search=self.query_overrides.x_search,
            web_fetch=self.query_overrides.web_fetch,
            task_budget_tokens=self.query_overrides.task_budget_tokens,
            task_budget_configured=self.query_overrides.task_budget_configured,
            temperature=self.query_overrides.temperature,
            top_p=self.query_overrides.top_p,
            top_k=self.query_overrides.top_k,
            min_p=self.query_overrides.min_p,
            presence_penalty=self.query_overrides.presence_penalty,
            repetition_penalty=self.query_overrides.repetition_penalty,
        )


@dataclass(frozen=True, slots=True)
class _ExpandedModelPreset:
    model_spec: str
    query_defaults: ModelQueryOverrides


def _collect_query_values(
    query_params: ModelQueryPairs,
    keys: tuple[str, ...],
) -> list[str]:
    key_set = set(keys)
    return [value for key, value in query_params if key in key_set]


def _has_query_key(query_params: ModelQueryPairs, key: str) -> bool:
    return any(query_key == key for query_key, _ in query_params)


def _has_any_query_key(query_params: ModelQueryPairs, keys: tuple[str, ...]) -> bool:
    key_set = set(keys)
    return any(query_key in key_set for query_key, _ in query_params)


def _parse_transport_query_value(value: str) -> TransportSetting | None:
    return _TRANSPORT_QUERY_VALUES.get(strip_casefold(value))


def _parse_service_tier_query_value(value: str) -> ServiceTierSetting | None:
    return _SERVICE_TIER_QUERY_VALUES.get(strip_casefold(value))


def _provider_values_text(providers: tuple[Provider, ...]) -> str:
    provider_values = tuple(provider.config_name for provider in providers)
    if not provider_values:
        return ""
    if len(provider_values) == 1:
        return provider_values[0]
    return f"{', '.join(provider_values[:-1])}, and {provider_values[-1]}"


def _parse_float_query(
    query_params: ModelQueryPairs,
    model_spec: str,
    *,
    keys: tuple[str, ...],
    label: str,
) -> float | None:
    values = _collect_query_values(query_params, keys)
    if not values:
        return None

    raw_value = values[-1]
    try:
        parsed_value = float(raw_value)
    except ValueError as exc:
        raise ModelConfigError(
            f"Invalid {label} query value: '{raw_value}' in '{model_spec}'"
        ) from exc

    if not math.isfinite(parsed_value):
        raise ModelConfigError(f"Invalid {label} query value: '{raw_value}' in '{model_spec}'")

    return parsed_value


def _parse_int_query(
    query_params: ModelQueryPairs,
    model_spec: str,
    *,
    keys: tuple[str, ...],
    label: str,
) -> int | None:
    values = _collect_query_values(query_params, keys)
    if not values:
        return None

    raw_value = values[-1]
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ModelConfigError(
            f"Invalid {label} query value: '{raw_value}' in '{model_spec}'"
        ) from exc


def _parse_bool_query(raw_value: str, query_key: str, model_spec: str) -> bool:
    parsed_boolean = parse_boolean_alias(raw_value)
    if parsed_boolean is not None:
        return parsed_boolean
    raise ModelConfigError(
        f"Invalid {query_key} query value: '{raw_value}' in '{model_spec}'. "
        "Use on/off (or true/false, 1/0)."
    )


def _raise_for_unsupported_query_keys(
    query_params: ModelQueryPairs,
    model_spec: str,
) -> None:
    unsupported_keys = sorted({key for key, _ in query_params} - SUPPORTED_MODEL_QUERY_KEYS)
    if unsupported_keys:
        joined = ", ".join(f"'{key}'" for key in unsupported_keys)
        parameter_label = plural_label(len(unsupported_keys), "parameter")
        raise ModelConfigError(
            f"Unsupported model query {parameter_label} {joined} in '{model_spec}'"
        )


def _parse_reasoning_query(
    query_params: ModelQueryPairs, model_spec: str
) -> ReasoningEffortSetting | None:
    if _has_query_key(query_params, "reasoning"):
        raw_value = _collect_query_values(query_params, ("reasoning",))[-1]
        parsed_reasoning = parse_reasoning_setting(raw_value)
        if parsed_reasoning is None:
            raise ModelConfigError(
                f"Invalid reasoning query value: '{raw_value}' in '{model_spec}'"
            )
        return parsed_reasoning
    return None


def _parse_verbosity_query(
    query_params: ModelQueryPairs, model_spec: str
) -> TextVerbosityLevel | None:
    if _has_query_key(query_params, "verbosity"):
        raw_value = _collect_query_values(query_params, ("verbosity",))[-1]
        parsed_verbosity = parse_text_verbosity(raw_value)
        if parsed_verbosity is None:
            raise ModelConfigError(
                f"Invalid verbosity query value: '{raw_value}' in '{model_spec}'"
            )
        return parsed_verbosity
    return None


def _parse_structured_query(
    query_params: ModelQueryPairs, model_spec: str
) -> StructuredOutputMode | None:
    if _has_query_key(query_params, "structured"):
        raw_value = _collect_query_values(query_params, ("structured",))[-1]
        parsed_structured = parse_structured_output_mode(raw_value)
        if parsed_structured is None:
            raise ModelConfigError(
                f"Invalid structured query value: '{raw_value}' in '{model_spec}'"
            )
        return parsed_structured
    return None


def _parse_structured_tool_query(
    query_params: ModelQueryPairs, model_spec: str
) -> StructuredToolPolicy | None:
    if not _has_any_query_key(query_params, _STRUCTURED_TOOL_QUERY_KEYS):
        return None
    raw_value = strip_casefold(_collect_query_values(query_params, _STRUCTURED_TOOL_QUERY_KEYS)[-1])
    if not is_structured_tool_policy(raw_value):
        raise ModelConfigError(
            f"Invalid structured_tools query value: '{raw_value}' in '{model_spec}'"
        )
    return raw_value


def _parse_instant_query(query_params: ModelQueryPairs, model_spec: str) -> bool | None:
    if _has_query_key(query_params, "instant"):
        raw_value = _collect_query_values(query_params, ("instant",))[-1]
        return _parse_bool_query(raw_value, "instant", model_spec)
    return None


def _parse_web_tool_queries(
    query_params: ModelQueryPairs,
    model_spec: str,
) -> dict[str, bool | None]:
    return {
        query_key: (
            _parse_bool_query(
                _collect_query_values(query_params, (query_key,))[-1],
                query_key,
                model_spec,
            )
            if _has_query_key(query_params, query_key)
            else None
        )
        for query_key in _WEB_TOOL_QUERY_KEYS
    }


def _parse_context_query(query_params: ModelQueryPairs, model_spec: str) -> bool:
    if _has_query_key(query_params, "context"):
        normalized_value = strip_casefold(
            _collect_query_values(query_params, ("context",))[-1]
        )
        if normalized_value == "1m":
            return True
        raise ModelConfigError(
            f"Invalid context query value: '{normalized_value}' — only '1m' is supported"
        )
    return False


def _parse_transport_query(
    query_params: ModelQueryPairs, model_spec: str
) -> TransportSetting | None:
    if _has_query_key(query_params, "transport"):
        raw_value = _collect_query_values(query_params, ("transport",))[-1]
        normalized_transport = _parse_transport_query_value(raw_value)
        if normalized_transport is None:
            normalized_value = strip_casefold(raw_value)
            raise ModelConfigError(
                f"Invalid transport query value: '{normalized_value}' in '{model_spec}'"
            )
        return normalized_transport
    return None


def _parse_service_tier_query(
    query_params: ModelQueryPairs, model_spec: str
) -> ServiceTierSetting | None:
    if _has_query_key(query_params, "service_tier"):
        raw_value = _collect_query_values(query_params, ("service_tier",))[-1]
        normalized_service_tier = _parse_service_tier_query_value(raw_value)
        if normalized_service_tier is None:
            normalized_value = strip_casefold(raw_value)
            raise ModelConfigError(
                f"Invalid service_tier query value: '{normalized_value}' in '{model_spec}'"
            )
        return normalized_service_tier
    return None


def _parse_task_budget_query(
    query_params: ModelQueryPairs, model_spec: str
) -> tuple[int | None, bool]:
    if not _has_any_query_key(query_params, _TASK_BUDGET_QUERY_KEYS):
        return None, False
    raw_value = _collect_query_values(query_params, _TASK_BUDGET_QUERY_KEYS)[-1]
    try:
        return validate_task_budget_tokens(parse_task_budget_tokens(raw_value)), True
    except ValueError as exc:
        raise ModelConfigError(
            f"Invalid task_budget query value: '{raw_value}' in '{model_spec}'"
        ) from exc


def _parse_query_overrides(
    query_params: ModelQueryPairs,
    model_spec: str,
) -> ModelQueryOverrides:
    _raise_for_unsupported_query_keys(query_params, model_spec)
    task_budget_tokens, task_budget_configured = _parse_task_budget_query(
        query_params, model_spec
    )
    web_tool_overrides = _parse_web_tool_queries(query_params, model_spec)

    return ModelQueryOverrides(
        reasoning_effort=_parse_reasoning_query(query_params, model_spec),
        instant=_parse_instant_query(query_params, model_spec),
        text_verbosity=_parse_verbosity_query(query_params, model_spec),
        structured_output_mode=_parse_structured_query(query_params, model_spec),
        structured_tool_policy=_parse_structured_tool_query(query_params, model_spec),
        long_context=_parse_context_query(query_params, model_spec),
        transport=_parse_transport_query(query_params, model_spec),
        service_tier=_parse_service_tier_query(query_params, model_spec),
        web_search=web_tool_overrides["web_search"],
        x_search=web_tool_overrides["x_search"],
        web_fetch=web_tool_overrides["web_fetch"],
        task_budget_tokens=task_budget_tokens,
        task_budget_configured=task_budget_configured,
        temperature=_parse_float_query(
            query_params,
            model_spec,
            keys=_SAMPLING_QUERY_KEYS["temperature"],
            label="temperature",
        ),
        top_p=_parse_float_query(
            query_params,
            model_spec,
            keys=_SAMPLING_QUERY_KEYS["top_p"],
            label="top_p",
        ),
        top_k=_parse_int_query(
            query_params,
            model_spec,
            keys=_SAMPLING_QUERY_KEYS["top_k"],
            label="top_k",
        ),
        min_p=_parse_float_query(
            query_params,
            model_spec,
            keys=_SAMPLING_QUERY_KEYS["min_p"],
            label="min_p",
        ),
        presence_penalty=_parse_float_query(
            query_params,
            model_spec,
            keys=_SAMPLING_QUERY_KEYS["presence_penalty"],
            label="presence_penalty",
        ),
        repetition_penalty=_parse_float_query(
            query_params,
            model_spec,
            keys=_SAMPLING_QUERY_KEYS["repetition_penalty"],
            label="repetition_penalty",
        ),
    )


def _split_model_spec_and_query(model_string: str) -> tuple[str, ModelQueryOverrides]:
    if "?" not in model_string:
        return model_string, ModelQueryOverrides()

    model_spec, _, query = model_string.partition("?")
    return model_spec, _parse_query_overrides(
        parse_qsl(query, keep_blank_values=True),
        model_spec,
    )


def _split_model_suffix(model_spec: str) -> tuple[str, str | None]:
    if ":" not in model_spec:
        return model_spec, None

    base, suffix = model_spec.rsplit(":", 1)
    if not base:
        return model_spec, None
    return base, suffix


def _expand_model_preset(
    model_spec: str,
    presets: Mapping[str, str],
) -> _ExpandedModelPreset:
    expanded = presets.get(model_spec, model_spec)
    if "?" not in expanded:
        return _ExpandedModelPreset(model_spec=expanded, query_defaults=ModelQueryOverrides())

    expanded_spec, _, preset_query = expanded.partition("?")
    return _ExpandedModelPreset(
        model_spec=expanded_spec,
        query_defaults=_parse_query_overrides(
            parse_qsl(preset_query, keep_blank_values=True),
            expanded_spec,
        ),
    )


def _reject_deprecated_reasoning_suffix(model_spec: str) -> None:
    parts = model_spec.split(".")
    if len(parts) <= 1:
        return

    suffix_setting = parse_reasoning_setting(strip_casefold(parts[-1]))
    if suffix_setting is None or suffix_setting.kind != "effort":
        return

    raise ModelConfigError(
        f"Reasoning suffix syntax is no longer supported for '{model_spec}'. "
        "Use '?reasoning=<value>' instead."
    )


def _provider_from_value(value: str) -> Provider | None:
    try:
        return Provider(value)
    except ValueError:
        return None


def _split_slash_provider_override(model_spec: str) -> tuple[Provider | None, str]:
    if "/" not in model_spec:
        return None, model_spec
    prefix, rest = model_spec.split("/", 1)
    if not prefix or not rest:
        return None, model_spec
    provider = _provider_from_value(prefix)
    if provider is None:
        return None, model_spec
    return provider, rest


def _split_dotted_provider_prefix(model_spec: str) -> tuple[Provider | None, str]:
    parts = model_spec.split(".")
    for provider_part_count in (2, 1):
        if len(parts) < provider_part_count:
            continue
        provider = _provider_from_value(".".join(parts[:provider_part_count]))
        if provider is not None:
            return provider, ".".join(parts[provider_part_count:])
    return None, model_spec


def _default_provider_for_model(
    model_name: str,
    *,
    bedrock_pattern_matches: Callable[[str], bool],
) -> Provider | None:
    provider = ModelDatabase.get_default_provider(model_name)
    if provider is not None:
        return provider
    if bedrock_pattern_matches(model_name):
        return Provider.BEDROCK
    return None


def _resolve_provider_and_model_name(
    model_spec: str,
    *,
    bedrock_pattern_matches: Callable[[str], bool],
) -> tuple[Provider, str]:
    provider, model_name = _split_slash_provider_override(model_spec)
    if provider is None:
        provider, model_name = _split_dotted_provider_prefix(model_name)

    if provider is None:
        provider = _default_provider_for_model(
            model_name,
            bedrock_pattern_matches=bedrock_pattern_matches,
        )
    if provider is None:
        raise ModelConfigError(
            f"Unknown model or provider for: {model_spec}. Model name parsed as '{model_name}'"
        )

    if provider == Provider.TENSORZERO and not model_name:
        raise ModelConfigError(
            f"TensorZero provider requires a function name after the provider "
            f"(e.g., tensorzero.my-function), got: {model_spec}"
        )

    return provider, model_name


def _validate_transport_constraints(
    provider: Provider,
    model_name: str,
    transport: TransportSetting | None,
) -> None:
    if transport not in {"websocket", "auto"}:
        return

    if provider not in _WEBSOCKET_TRANSPORT_PROVIDERS:
        raise ModelConfigError(
            "WebSocket transport is experimental and currently supported only for "
            f"the {_provider_values_text(_WEBSOCKET_TRANSPORT_PROVIDERS)} providers."
        )

    supports_transport = ModelDatabase.supports_response_transport(model_name, "websocket")
    if supports_transport is False:
        raise ModelConfigError(
            f"Transport '{transport}' is not supported for model '{model_name}'."
        )

    supports_provider = ModelDatabase.supports_response_websocket_provider(model_name, provider)
    if supports_provider is False:
        raise ModelConfigError(
            f"Transport '{transport}' is not supported for model '{model_name}' "
            f"with provider '{provider.config_name}'."
        )


def _validate_service_tier_constraints(
    provider: Provider,
    model_name: str,
    service_tier: ServiceTierSetting | None,
) -> None:
    if service_tier != "flex":
        return

    if provider == Provider.CODEX_RESPONSES:
        raise ModelConfigError(
            "Provider 'codexresponses' does not support service_tier=flex. "
            "Allowed values are fast or unset (standard)."
        )

    if provider not in _FLEX_SERVICE_TIER_MODEL_CHECK_PROVIDERS:
        return

    supports_flex = ModelDatabase.supports_response_service_tier(model_name, "flex")
    if supports_flex is False:
        raise ModelConfigError(
            f"Model '{model_name}' does not support service_tier=flex "
            f"with provider '{provider.config_name}'. Allowed values are fast or unset "
            "(standard)."
        )


class ModelFactory:
    """Factory for creating LLM instances based on model specifications"""

    MODEL_PRESETS: ClassVar[dict[str, str]] = {
        "gpt51": "responses.gpt-5.1",
        "gpt52": "responses.gpt-5.2",
        "gpt54": "responses.gpt-5.4",
        "gpt55": "responses.gpt-5.5",
        "gpt54-mini": "responses.gpt-5.4-mini",
        "gpt54-nano": "responses.gpt-5.4-nano",
        "chatgpt": "responses.chat-latest",
        "chat-latest": "responses.chat-latest",
        "codex": "responses.gpt-5.3-codex",
        "codexplan": "codexresponses.gpt-5.5?reasoning=medium",
        "codexplan54": "codexresponses.gpt-5.4?reasoning=high",
        "codexplan53": "codexresponses.gpt-5.3-codex?reasoning=medium",
        "codexspark": "codexresponses.gpt-5.3-codex-spark",
        "sonnet": "claude-sonnet-4-6",
        "sonnet4": "claude-sonnet-4-6",
        "sonnet46": "claude-sonnet-4-6",
        "claude": "claude-sonnet-4-6",
        "haiku": "claude-haiku-4-5",
        "haiku45": "claude-haiku-4-5",
        "opus": "claude-opus-4-8",
        "opus4": "claude-opus-4-8",
        "opus46": "claude-opus-4-6",
        "opus47": "claude-opus-4-7",
        "opus48": "claude-opus-4-8",
        "deepseek": "deepseek.deepseek-v4-pro",
        "deepseek4": "deepseek.deepseek-v4-pro",
        "deepseek4pro": "deepseek.deepseek-v4-pro",
        "deepseekv4pro": "deepseek.deepseek-v4-pro",
        "deepseek-direct": "deepseek.deepseek-v4-pro",
        "deepseek4flash": "deepseek.deepseek-v4-flash",
        "deepseek4pro-direct": "deepseek.deepseek-v4-pro",
        "deepseek-reasoner": "deepseek.deepseek-reasoner",
        "gemini": "gemini-3.1-pro-preview",
        "gemini2": "gemini-2.0-flash",
        "gemini25": "gemini-2.5-flash",
        "gemini25pro": "gemini-2.5-pro",
        "gemini35": "gemini-3.5-flash",
        "gemini35flash": "gemini-3.5-flash",
        "gemini3.5flash": "gemini-3.5-flash",
        "gemini3": "gemini-3-pro-preview",
        "gemini3.1": "gemini-3.1-pro-preview",
        "gemini31pro": "gemini-3.1-pro-preview",
        "gemini3.1flashlite": "gemini-3.1-flash-lite-preview",
        "gemini3flash": "gemini-3-flash-preview",
        "grok": "xai.grok-4.3",
        "grok4": "xai.grok-4.3",
        "grok-4-fast": "xai.grok-4-fast-non-reasoning",
        "grok-4-fast-reasoning": "xai.grok-4-fast-reasoning",
        "minimax": "hf.MiniMaxAI/MiniMax-M2.7:fireworks-ai?temperature=1.0&top_p=0.95&top_k=40",
        "minimax25": "hf.MiniMaxAI/MiniMax-M2.5:fireworks-ai?temperature=1.0&top_p=0.95&top_k=40",
        "minimax27": "hf.MiniMaxAI/MiniMax-M2.7:fireworks-ai?temperature=1.0&top_p=0.95&top_k=40",
        "minimax2.5": "hf.MiniMaxAI/MiniMax-M2.5:novita?temperature=1.0&top_p=0.95&top_k=40",
        "minimax21": "hf.MiniMaxAI/MiniMax-M2.1:novita",
        "kimi": ("hf.moonshotai/Kimi-K2.6:novita?temperature=1.0&top_p=0.95&reasoning=on"),
        "kimithink": "hf.moonshotai/Kimi-K2.6:novita?temperature=1.0&top_p=0.95&reasoning=on",
        "gpt-oss": "hf.openai/gpt-oss-120b:cerebras",
        "gpt-oss-20b": "hf.openai/gpt-oss-20b",
        "glm47": "hf.zai-org/GLM-4.7:cerebras",
        "glm51": "hf.zai-org/GLM-5.1:together",
        "glm5": "hf.zai-org/GLM-5:novita",
        "glm": "hf.zai-org/GLM-5.1:together",
        "deepseek-hf": "hf.deepseek-ai/DeepSeek-V4-Pro:together",
        "deepseek32": "hf.deepseek-ai/DeepSeek-V3.2:fireworks-ai",
        "deepseek4-hf": "hf.deepseek-ai/DeepSeek-V4-Pro:together",
        "deepseek4pro-hf": "hf.deepseek-ai/DeepSeek-V4-Pro:together",
        "deepseekv4pro-hf": "hf.deepseek-ai/DeepSeek-V4-Pro:together",
        "kimi26": "hf.moonshotai/Kimi-K2.6:novita?temperature=1.0&top_p=0.95&reasoning=on",
        "kimi26instant": (
            "hf.moonshotai/Kimi-K2.6:novita?temperature=0.6&top_p=0.95&reasoning=off"
        ),
        "kimi-2.6": "hf.moonshotai/Kimi-K2.6:novita?temperature=1.0&top_p=0.95&reasoning=on",
        "kimi25": ("hf.moonshotai/Kimi-K2.5:novita?temperature=1.0&top_p=0.95&reasoning=on"),
        "kimi25instant": (
            "hf.moonshotai/Kimi-K2.5:novita?temperature=0.6&top_p=0.95&reasoning=off"
        ),
        "kimi-2.5": ("hf.moonshotai/Kimi-K2.5:novita?temperature=1.0&top_p=0.95&reasoning=on"),
        "qwen35": (
            "hf.Qwen/Qwen3.5-397B-A17B:novita"
            "?temperature=0.6&top_p=0.95&top_k=20&min_p=0.0"
            "&presence_penalty=0.0&repetition_penalty=1.0&reasoning=on"
        ),
        "qwen35instruct": (
            "hf.Qwen/Qwen3.5-397B-A17B:novita"
            "?temperature=0.7&top_p=0.8&top_k=20&min_p=0.0"
            "&presence_penalty=1.5&repetition_penalty=1.0&reasoning=off"
        ),
    }

    @staticmethod
    def _bedrock_pattern_matches(model_name: str) -> bool:
        """Return True if model_name matches Bedrock's expected pattern, else False.

        Uses provider's helper if available; otherwise, returns False.
        """
        try:
            from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM

            return BedrockLLM.matches_model_pattern(model_name)
        except Exception:
            return False

    # Mapping of providers to their LLM classes
    PROVIDER_CLASSES: ClassVar[dict[Provider, LLMClass]] = {}

    # Extension point for model names that should use a custom LLM class.
    MODEL_SPECIFIC_CLASSES: ClassVar[dict[str, LLMClass]] = {}

    # Built-in model-specific classes are imported lazily to keep CLI startup light.
    MODEL_SPECIFIC_NAMES: ClassVar[set[str]] = {"playback", "silent", "slow"}

    @classmethod
    def get_runtime_presets(cls) -> dict[str, str]:
        """Return built-in model presets, including curated catalog presets."""
        presets = dict(cls.MODEL_PRESETS)

        from fast_agent.llm.model_selection import ModelSelectionCatalog

        for entry in ModelSelectionCatalog.list_current_entries():
            preset_token = entry.alias.strip()
            if not preset_token:
                continue
            presets.setdefault(preset_token, entry.model)

        presets.update(load_model_overlay_registry().runtime_presets())
        return presets

    @classmethod
    def parse_model_spec(
        cls,
        model_string: str,
        presets: Mapping[str, str] | None = None,
    ) -> ParsedModelSpec:
        """Parse a model string into a canonical ParsedModelSpec.

        Args:
            model_string: The model specification string (e.g. "gpt-4.1", "kimi:groq")
            presets: Optional custom parser preset map.
        """
        if presets is None:
            presets = cls.get_runtime_presets()

        raw_input = model_string
        model_spec, explicit_overrides = _split_model_spec_and_query(model_string)
        model_spec, user_suffix = _split_model_suffix(model_spec)

        expanded_preset = _expand_model_preset(model_spec, presets)
        expanded_model_spec = expanded_preset.model_spec
        if user_suffix and ":" in expanded_model_spec:
            expanded_model_spec = expanded_model_spec.rsplit(":", 1)[0]
        if user_suffix:
            expanded_model_spec = f"{expanded_model_spec}:{user_suffix}"

        merged_overrides = explicit_overrides.with_defaults(expanded_preset.query_defaults)
        _reject_deprecated_reasoning_suffix(expanded_model_spec)

        provider, model_name = _resolve_provider_and_model_name(
            expanded_model_spec,
            bedrock_pattern_matches=cls._bedrock_pattern_matches,
        )

        reasoning_effort = merged_overrides.reasoning_effort
        if merged_overrides.instant is not None:
            if reasoning_effort is not None:
                raise ModelConfigError(
                    f"Multiple reasoning settings provided for '{expanded_model_spec}'."
                )
            base_model = strip_casefold(model_name.rsplit(":", 1)[0])
            if base_model not in {"moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6"}:
                raise ModelConfigError(
                    "Instant mode is only supported for moonshotai/kimi-k2.5 "
                    f"and moonshotai/kimi-k2.6, got '{model_name}'."
                )
            reasoning_effort = ReasoningEffortSetting(
                kind="toggle",
                value=not merged_overrides.instant,
            )

        _validate_transport_constraints(provider, model_name, merged_overrides.transport)
        _validate_service_tier_constraints(provider, model_name, merged_overrides.service_tier)
        return ParsedModelSpec(
            raw_input=raw_input,
            expanded_input=expanded_model_spec,
            provider=provider,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            query_overrides=merged_overrides,
        )

    @classmethod
    def parse_model_string(
        cls,
        model_string: str,
        presets: Mapping[str, str] | None = None,
    ) -> ModelConfig:
        """Parse a model string into a ModelConfig object.

        Args:
            model_string: The model specification string (e.g. "gpt-4.1", "kimi:groq")
            presets: Optional custom parser preset map.
        """
        return cls.parse_model_spec(model_string, presets=presets).to_model_config()

    @classmethod
    def resolve_model_spec(
        cls,
        model_string: str,
        presets: Mapping[str, str] | None = None,
    ) -> ResolvedModelSpec:
        """Hydrate a model selection into a single resolved specification."""
        selected_model_name = model_string.strip()
        overlay_registry = load_model_overlay_registry()
        selected_overlay = overlay_registry.resolve_model_string(model_string)

        if selected_overlay is not None:
            source: Literal["overlay", "preset", "direct"] = "overlay"
            parsed = cls.parse_model_spec(
                model_string,
                presets={selected_overlay.name: selected_overlay.compiled_model_spec},
            )
        else:
            if presets is None:
                presets = cls.get_runtime_presets()

            parsed = cls.parse_model_spec(model_string, presets=presets)
            selected_token = selected_model_name.partition("?")[0].strip()
            source = "preset" if selected_token in presets else "direct"

        model_config = parsed.to_model_config()

        model_params = None
        if selected_overlay is not None:
            model_params = selected_overlay.build_model_parameters()
        if model_params is None:
            model_params = resolve_base_model_params(
                provider=parsed.provider,
                model_name=parsed.model_name,
            )
        wire_model_name = ModelDatabase.resolve_wire_model_name(
            provider=parsed.provider,
            model_name=parsed.model_name,
        )

        return ResolvedModelSpec(
            raw_input=model_string,
            selected_model_name=selected_model_name,
            source=source,
            model_config=model_config,
            provider=model_config.provider,
            wire_model_name=wire_model_name,
            overlay=selected_overlay,
            model_params=model_params,
        )

    @classmethod
    def create_factory(
        cls, model_string: str, presets: Mapping[str, str] | None = None
    ) -> LLMFactoryProtocol:
        """
        Creates a factory function that follows the attach_llm protocol.

        Args:
            model_string: The model specification string (e.g. "gpt-4.1")
            presets: Optional custom parser preset map.

        Returns:
            A callable that takes an agent parameter and returns an LLM instance
        """
        resolved_model = cls.resolve_model_spec(model_string, presets=presets)
        config = resolved_model.model_config

        # Ensure provider is valid before trying to access PROVIDER_CLASSES with it
        # Lazily ensure provider class map is populated and supports this provider
        model_specific_class = cls.MODEL_SPECIFIC_CLASSES.get(config.model_name)
        if model_specific_class is None and config.model_name not in cls.MODEL_SPECIFIC_NAMES:
            llm_class = cls._load_provider_class(config.provider)
            # Stash for next time
            cls.PROVIDER_CLASSES[config.provider] = llm_class

        if model_specific_class is not None:
            llm_class = model_specific_class
        elif config.model_name in cls.MODEL_SPECIFIC_NAMES:
            llm_class = cls._load_model_specific_class(config.model_name)
        else:
            llm_class = cls.PROVIDER_CLASSES[config.provider]

        def factory(
            agent: AgentProtocol, request_params: RequestParams | None = None, **kwargs
        ) -> FastAgentLLMProtocol:
            effective_request_params = resolved_model.apply_request_defaults(request_params)
            llm_args = {
                "model": resolved_model.wire_model_name,
                "resolved_model_spec": resolved_model,
                "request_params": effective_request_params,
                "name": getattr(agent, "name", "fast-agent"),
                "instructions": getattr(agent, "instruction", None),
                **resolved_model.build_llm_kwargs(),
                **kwargs,
            }
            if resolved_model.llm_init_kwargs:
                llm_args = {
                    **llm_args,
                    **resolved_model.llm_init_kwargs,
                }
            llm: FastAgentLLMProtocol = llm_class(**llm_args)
            return llm

        return factory

    @classmethod
    def _load_provider_class(cls, provider: Provider) -> LLMClass:
        """Import provider-specific LLM classes lazily to avoid heavy deps at import time."""
        class_path = _PROVIDER_CLASS_PATHS.get(provider)
        if class_path is None:
            raise ModelConfigError(f"Unsupported provider: {provider}")
        module_name, class_name = class_path
        try:
            llm_class = vars(import_module(module_name))[class_name]
        except Exception as e:
            raise ModelConfigError(
                f"Provider '{provider.value}' is unavailable or missing dependencies: {e}"
            ) from e
        return llm_class

    @classmethod
    def _load_model_specific_class(cls, model_name: str) -> LLMClass:
        """Import built-in model-specific LLM classes lazily."""
        class_path = _MODEL_SPECIFIC_CLASS_PATHS.get(model_name)
        if class_path is not None:
            module_name, class_name = class_path
            return vars(import_module(module_name))[class_name]
        raise ModelConfigError(f"Unsupported fast-agent model: {model_name}")
