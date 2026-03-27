import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Literal, Self, Type, Union
from urllib.parse import parse_qs

from pydantic import BaseModel

from fast_agent.core.exceptions import ModelConfigError
from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LLMFactoryProtocol
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.internal.playback import PlaybackLLM
from fast_agent.llm.internal.silent import SilentLLM
from fast_agent.llm.internal.slow import SlowLLM
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, parse_reasoning_setting
from fast_agent.llm.resolved_model import ResolvedModelSpec, resolve_base_model_params
from fast_agent.llm.structured_output_mode import (
    StructuredOutputMode,
    parse_structured_output_mode,
)
from fast_agent.llm.text_verbosity import TextVerbosityLevel, parse_text_verbosity
from fast_agent.types import RequestParams

# Type alias for LLM classes
LLMClass = Union[Type[PassthroughLLM], Type[PlaybackLLM], Type[SilentLLM], Type[SlowLLM], type]
TransportSetting = Literal["sse", "websocket", "auto"]
ServiceTierSetting = Literal["fast", "flex"]


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    provider: Provider
    model_name: str
    reasoning_effort: ReasoningEffortSetting | None = None
    text_verbosity: TextVerbosityLevel | None = None
    structured_output_mode: StructuredOutputMode | None = None
    long_context: bool = False
    transport: TransportSetting | None = None
    service_tier: ServiceTierSetting | None = None
    web_search: bool | None = None
    web_fetch: bool | None = None
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
    long_context: bool = False
    transport: TransportSetting | None = None
    service_tier: ServiceTierSetting | None = None
    web_search: bool | None = None
    web_fetch: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None

    def with_defaults(self, defaults: Self) -> "ModelQueryOverrides":
        """Return a copy with unset values filled from defaults."""
        return ModelQueryOverrides(
            reasoning_effort=(
                self.reasoning_effort
                if self.reasoning_effort is not None
                else defaults.reasoning_effort
            ),
            instant=self.instant if self.instant is not None else defaults.instant,
            text_verbosity=(
                self.text_verbosity if self.text_verbosity is not None else defaults.text_verbosity
            ),
            structured_output_mode=(
                self.structured_output_mode
                if self.structured_output_mode is not None
                else defaults.structured_output_mode
            ),
            long_context=self.long_context or defaults.long_context,
            transport=self.transport if self.transport is not None else defaults.transport,
            service_tier=(
                self.service_tier if self.service_tier is not None else defaults.service_tier
            ),
            web_search=self.web_search if self.web_search is not None else defaults.web_search,
            web_fetch=self.web_fetch if self.web_fetch is not None else defaults.web_fetch,
            temperature=self.temperature if self.temperature is not None else defaults.temperature,
            top_p=self.top_p if self.top_p is not None else defaults.top_p,
            top_k=self.top_k if self.top_k is not None else defaults.top_k,
            min_p=self.min_p if self.min_p is not None else defaults.min_p,
            presence_penalty=(
                self.presence_penalty
                if self.presence_penalty is not None
                else defaults.presence_penalty
            ),
            repetition_penalty=(
                self.repetition_penalty
                if self.repetition_penalty is not None
                else defaults.repetition_penalty
            ),
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
            long_context=self.query_overrides.long_context,
            transport=self.query_overrides.transport,
            service_tier=self.query_overrides.service_tier,
            web_search=self.query_overrides.web_search,
            web_fetch=self.query_overrides.web_fetch,
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
    query_params: Mapping[str, list[str]],
    keys: tuple[str, ...],
) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(query_params.get(key) or [])
    return values


def _parse_float_query(
    query_params: Mapping[str, list[str]],
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
    query_params: Mapping[str, list[str]],
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


def _parse_on_off_query(raw_value: str, query_key: str, model_spec: str) -> bool:
    parsed = parse_reasoning_setting(raw_value)
    if parsed is not None and parsed.kind == "toggle":
        return bool(parsed.value)
    raise ModelConfigError(
        f"Invalid {query_key} query value: '{raw_value}' in '{model_spec}'. "
        "Use on/off (or true/false, 1/0)."
    )


def _parse_query_overrides(
    query_params: Mapping[str, list[str]],
    model_spec: str,
) -> ModelQueryOverrides:
    supported_keys = {
        "reasoning",
        "verbosity",
        "structured",
        "instant",
        "context",
        "transport",
        "service_tier",
        "web_search",
        "web_fetch",
        "temperature",
        "temp",
        "top_p",
        "topP",
        "top_k",
        "topK",
        "min_p",
        "minP",
        "presence_penalty",
        "presencePenalty",
        "repetition_penalty",
        "repetitionPenalty",
    }
    unsupported_keys = sorted(set(query_params) - supported_keys)
    if unsupported_keys:
        joined = ", ".join(f"'{key}'" for key in unsupported_keys)
        raise ModelConfigError(
            f"Unsupported model query parameter(s) {joined} in '{model_spec}'"
        )

    reasoning_effort: ReasoningEffortSetting | None = None
    text_verbosity: TextVerbosityLevel | None = None
    structured_output_mode: StructuredOutputMode | None = None
    instant: bool | None = None
    long_context = False
    transport: TransportSetting | None = None
    service_tier: ServiceTierSetting | None = None
    web_search: bool | None = None
    web_fetch: bool | None = None

    if "reasoning" in query_params:
        raw_value = _collect_query_values(query_params, ("reasoning",))[-1]
        parsed_reasoning = parse_reasoning_setting(raw_value)
        if parsed_reasoning is None:
            raise ModelConfigError(
                f"Invalid reasoning query value: '{raw_value}' in '{model_spec}'"
            )
        reasoning_effort = parsed_reasoning

    if "verbosity" in query_params:
        raw_value = _collect_query_values(query_params, ("verbosity",))[-1]
        parsed_verbosity = parse_text_verbosity(raw_value)
        if parsed_verbosity is None:
            raise ModelConfigError(
                f"Invalid verbosity query value: '{raw_value}' in '{model_spec}'"
            )
        text_verbosity = parsed_verbosity

    if "structured" in query_params:
        raw_value = _collect_query_values(query_params, ("structured",))[-1]
        parsed_structured = parse_structured_output_mode(raw_value)
        if parsed_structured is None:
            raise ModelConfigError(
                f"Invalid structured query value: '{raw_value}' in '{model_spec}'"
            )
        structured_output_mode = parsed_structured

    if "instant" in query_params:
        raw_value = _collect_query_values(query_params, ("instant",))[-1]
        instant_setting = parse_reasoning_setting(raw_value)
        if instant_setting is None or instant_setting.kind != "toggle":
            raise ModelConfigError(f"Invalid instant query value: '{raw_value}' in '{model_spec}'")
        instant = bool(instant_setting.value)

    if "context" in query_params:
        raw_value = _collect_query_values(query_params, ("context",))[-1].strip().lower()
        if raw_value == "1m":
            long_context = True
        else:
            raise ModelConfigError(
                f"Invalid context query value: '{raw_value}' — only '1m' is supported"
            )

    if "transport" in query_params:
        raw_value = _collect_query_values(query_params, ("transport",))[-1].strip().lower()
        transport_presets: dict[str, TransportSetting] = {
            "ws": "websocket",
            "websocket": "websocket",
            "sse": "sse",
            "auto": "auto",
        }
        normalized_transport = transport_presets.get(raw_value)
        if normalized_transport is None:
            raise ModelConfigError(
                f"Invalid transport query value: '{raw_value}' in '{model_spec}'"
            )
        transport = normalized_transport

    if "service_tier" in query_params:
        raw_value = _collect_query_values(query_params, ("service_tier",))[-1].strip().lower()
        if raw_value not in {"fast", "flex"}:
            raise ModelConfigError(
                f"Invalid service_tier query value: '{raw_value}' in '{model_spec}'"
            )
        service_tier = "fast" if raw_value == "fast" else "flex"

    if "web_search" in query_params:
        raw_value = _collect_query_values(query_params, ("web_search",))[-1]
        web_search = _parse_on_off_query(raw_value, "web_search", model_spec)

    if "web_fetch" in query_params:
        raw_value = _collect_query_values(query_params, ("web_fetch",))[-1]
        web_fetch = _parse_on_off_query(raw_value, "web_fetch", model_spec)

    return ModelQueryOverrides(
        reasoning_effort=reasoning_effort,
        instant=instant,
        text_verbosity=text_verbosity,
        structured_output_mode=structured_output_mode,
        long_context=long_context,
        transport=transport,
        service_tier=service_tier,
        web_search=web_search,
        web_fetch=web_fetch,
        temperature=_parse_float_query(
            query_params,
            model_spec,
            keys=("temperature", "temp"),
            label="temperature",
        ),
        top_p=_parse_float_query(
            query_params,
            model_spec,
            keys=("top_p", "topP"),
            label="top_p",
        ),
        top_k=_parse_int_query(
            query_params,
            model_spec,
            keys=("top_k", "topK"),
            label="top_k",
        ),
        min_p=_parse_float_query(
            query_params,
            model_spec,
            keys=("min_p", "minP"),
            label="min_p",
        ),
        presence_penalty=_parse_float_query(
            query_params,
            model_spec,
            keys=("presence_penalty", "presencePenalty"),
            label="presence_penalty",
        ),
        repetition_penalty=_parse_float_query(
            query_params,
            model_spec,
            keys=("repetition_penalty", "repetitionPenalty"),
            label="repetition_penalty",
        ),
    )


def _split_model_spec_and_query(model_string: str) -> tuple[str, ModelQueryOverrides]:
    if "?" not in model_string:
        return model_string, ModelQueryOverrides()

    model_spec, _, query = model_string.partition("?")
    return model_spec, _parse_query_overrides(parse_qs(query), model_spec)


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
        query_defaults=_parse_query_overrides(parse_qs(preset_query), expanded_spec),
    )


def _reject_deprecated_reasoning_suffix(model_spec: str) -> None:
    parts = model_spec.split(".")
    if len(parts) <= 1:
        return

    suffix_setting = parse_reasoning_setting(parts[-1].lower())
    if suffix_setting is None or suffix_setting.kind != "effort":
        return

    raise ModelConfigError(
        f"Reasoning suffix syntax is no longer supported for '{model_spec}'. "
        "Use '?reasoning=<value>' instead."
    )


def _resolve_provider_and_model_name(
    model_spec: str,
    *,
    bedrock_pattern_matches: Callable[[str], bool],
) -> tuple[Provider, str]:
    provider_override: Provider | None = None
    normalized_spec = model_spec
    if "/" in normalized_spec:
        prefix, rest = normalized_spec.split("/", 1)
        if prefix and rest and any(p.value == prefix for p in Provider):
            provider_override = Provider(prefix)
            normalized_spec = rest

    parts = normalized_spec.split(".")
    provider = provider_override
    identified_provider_parts = 0

    if provider is None and len(parts) >= 2:
        potential_provider_str = f"{parts[0]}.{parts[1]}"
        if any(p.value == potential_provider_str for p in Provider):
            provider = Provider(potential_provider_str)
            identified_provider_parts = 2

    if provider is None and len(parts) >= 1:
        potential_provider_str = parts[0]
        if any(p.value == potential_provider_str for p in Provider):
            provider = Provider(potential_provider_str)
            identified_provider_parts = 1

    if identified_provider_parts > 0:
        model_name = ".".join(parts[identified_provider_parts:])
    else:
        model_name = ".".join(parts)

    if provider is None:
        provider = ModelDatabase.get_default_provider(model_name)
        if provider is None and bedrock_pattern_matches(model_name):
            provider = Provider.BEDROCK

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

    if provider not in {Provider.CODEX_RESPONSES, Provider.RESPONSES}:
        raise ModelConfigError(
            "WebSocket transport is experimental and currently supported only for "
            "the codexresponses and responses providers."
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
            f"with provider '{provider.value}'."
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

    if provider not in {Provider.RESPONSES, Provider.OPENRESPONSES}:
        return

    supports_flex = ModelDatabase.supports_response_service_tier(model_name, "flex")
    if supports_flex is False:
        raise ModelConfigError(
            f"Model '{model_name}' does not support service_tier=flex "
            f"with provider '{provider.value}'. Allowed values are fast or unset "
            "(standard)."
        )


class ModelFactory:
    """Factory for creating LLM instances based on model specifications"""

    MODEL_PRESETS = {
        "gpt51": "responses.gpt-5.1",
        "gpt52": "responses.gpt-5.2",
        "gpt54": "responses.gpt-5.4",
        "gpt54-mini": "responses.gpt-5.4-mini",
        "gpt54-nano": "responses.gpt-5.4-nano",
        "chatgpt": "responses.gpt-5.3-chat-latest",
        "codex": "responses.gpt-5.3-codex",
        "codexplan": "codexresponses.gpt-5.4",
        "codexplan53": "codexresponses.gpt-5.3-codex",
        "codexplan52": "codexresponses.gpt-5.2-codex",
        "codexplan51": "codexresponses.gpt-5.1-codex",
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
        "minimax": "hf.MiniMaxAI/MiniMax-M2.5:novita",
        "minimax25": "hf.MiniMaxAI/MiniMax-M2.5:fireworks-ai?temperature=1.0&top_p=0.95&top_k=40",
        "minimax2.5": "hf.MiniMaxAI/MiniMax-M2.5:novita?temperature=1.0&top_p=0.95&top_k=40",
        "minimax21": "hf.MiniMaxAI/MiniMax-M2.1:novita",
        "kimi": "hf.moonshotai/Kimi-K2-Instruct-0905:groq",
        "gpt-oss": "hf.openai/gpt-oss-120b:sambanova",
        "gpt-oss-20b": "hf.openai/gpt-oss-20b",
        "glm47": "hf.zai-org/GLM-4.7:cerebras",
        "glm5": "hf.zai-org/GLM-5:novita",
        "glm": "hf.zai-org/GLM-5:novita",
        "qwen3": "hf.Qwen/Qwen3-Next-80B-A3B-Instruct:together",
        "deepseek31": "hf.deepseek-ai/DeepSeek-V3.1",
        "kimithink": "hf.moonshotai/Kimi-K2-Thinking:fireworks-ai",
        "deepseek32": "hf.deepseek-ai/DeepSeek-V3.2:fireworks-ai",
        "kimi25": ("hf.moonshotai/Kimi-K2.5:fireworks-ai?temperature=1.0&top_p=0.95&reasoning=on"),
        # "kimi25instant": (
        #     "hf.moonshotai/Kimi-K2.5:fireworks-ai"
        #     "?temperature=0.6&top_p=0.95&reasoning=off"
        # ),
        "kimi-2.5": (
            "hf.moonshotai/Kimi-K2.5:fireworks-ai?temperature=1.0&top_p=0.95&reasoning=on"
        ),
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
    PROVIDER_CLASSES: dict[Provider, LLMClass] = {}

    # Mapping of special model names to their specific LLM classes
    # This overrides the provider-based class selection
    MODEL_SPECIFIC_CLASSES: dict[str, LLMClass] = {
        "playback": PlaybackLLM,
        "silent": SilentLLM,
        "slow": SlowLLM,
    }

    @classmethod
    def get_runtime_presets(cls) -> dict[str, str]:
        """Return built-in model presets, including curated catalog presets."""
        presets = dict(cls.MODEL_PRESETS)

        from fast_agent.llm.model_selection import ModelSelectionCatalog

        for entry in ModelSelectionCatalog.list_current_entries():
            preset_token = entry.alias.strip()
            if not preset_token or "?" in entry.model:
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
            base_model = model_name.rsplit(":", 1)[0].strip().lower()
            if base_model != "moonshotai/kimi-k2.5":
                raise ModelConfigError(
                    f"Instant mode is only supported for moonshotai/kimi-k2.5, got '{model_name}'."
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
            if selected_token in presets:
                source = "preset"
            else:
                source = "direct"

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
    def _load_provider_class(cls, provider: Provider) -> type:
        """Import provider-specific LLM classes lazily to avoid heavy deps at import time."""
        try:
            if provider == Provider.FAST_AGENT:
                return PassthroughLLM
            if provider == Provider.ANTHROPIC:
                from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM

                return AnthropicLLM
            if provider == Provider.ANTHROPIC_VERTEX:
                from fast_agent.llm.provider.anthropic.llm_anthropic_vertex import (
                    AnthropicVertexLLM,
                )

                return AnthropicVertexLLM
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
