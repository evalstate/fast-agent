from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fast_agent.config import get_settings
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_selection import CatalogModelEntry, ModelSelectionCatalog
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import available_reasoning_values, format_reasoning_setting

if TYPE_CHECKING:
    from pathlib import Path

ModelSource = Literal["curated", "all"]
KEEP_VALUE = "__keep__"
DEFAULT_VALUE = "__default__"

PICKER_PROVIDER_ORDER: tuple[Provider, ...] = (
    Provider.RESPONSES,
    Provider.CODEX_RESPONSES,
    Provider.ANTHROPIC,
    Provider.HUGGINGFACE,
    Provider.OPENAI,
    Provider.GOOGLE,
    Provider.XAI,
    Provider.GROQ,
    Provider.DEEPSEEK,
    Provider.ALIYUN,
    Provider.OPENROUTER,
    Provider.AZURE,
    Provider.BEDROCK,
)

REFER_TO_DOCS_PROVIDERS: tuple[Provider, ...] = (
    Provider.OPENROUTER,
    Provider.AZURE,
    Provider.BEDROCK,
)


@dataclass(frozen=True)
class ProviderOption:
    provider: Provider
    active: bool
    curated_entries: tuple[CatalogModelEntry, ...]


@dataclass(frozen=True)
class ModelOption:
    spec: str
    label: str
    alias: str | None = None
    fast: bool = False
    curated: bool = False


@dataclass(frozen=True)
class ModelCapabilities:
    provider: Provider
    model_name: str
    reasoning_values: tuple[str, ...]
    current_reasoning: str
    default_reasoning: str
    web_search_supported: bool
    current_web_search: bool | None
    supports_long_context: bool
    current_long_context: bool
    long_context_window: int | None
    cache_ttl_default: str | None


@dataclass(frozen=True)
class ModelPickerSnapshot:
    providers: tuple[ProviderOption, ...]
    config_payload: dict[str, Any]


def _provider_is_active(provider: Provider, config_payload: dict[str, Any]) -> bool:
    config_key = ProviderKeyManager.get_config_file_key(provider.config_name, config_payload)
    if config_key:
        return True

    if ProviderKeyManager.get_env_var(provider.config_name):
        return True

    if provider == Provider.GOOGLE:
        google_cfg = config_payload.get("google")
        if isinstance(google_cfg, dict):
            vertex_cfg = google_cfg.get("vertex_ai")
            if isinstance(vertex_cfg, dict) and bool(vertex_cfg.get("enabled")):
                return True

    if provider == Provider.AZURE:
        azure_cfg = config_payload.get("azure")
        if isinstance(azure_cfg, dict):
            use_default = bool(azure_cfg.get("use_default_azure_credential"))
            base_url = azure_cfg.get("base_url")
            if use_default and isinstance(base_url, str) and bool(base_url.strip()):
                return True

    if provider == Provider.CODEX_RESPONSES:
        try:
            from fast_agent.llm.provider.openai.codex_oauth import get_codex_token_status

            status = get_codex_token_status()
            if bool(status.get("present")):
                return True
        except Exception:
            pass

    return False


def build_snapshot(config_path: str | Path | None = None) -> ModelPickerSnapshot:
    settings = get_settings(str(config_path) if config_path else None)
    config_payload = settings.model_dump()

    active_providers = set(ModelSelectionCatalog.configured_providers(config_payload))
    for provider in PICKER_PROVIDER_ORDER:
        if _provider_is_active(provider, config_payload):
            active_providers.add(provider)

    providers: list[ProviderOption] = []
    for provider in PICKER_PROVIDER_ORDER:
        entries = ModelSelectionCatalog.CATALOG_ENTRIES_BY_PROVIDER.get(provider, ())
        providers.append(
            ProviderOption(
                provider=provider,
                active=provider in active_providers,
                curated_entries=entries,
            )
        )

    return ModelPickerSnapshot(providers=tuple(providers), config_payload=config_payload)


def find_provider(snapshot: ModelPickerSnapshot, provider_name: str) -> ProviderOption:
    for option in snapshot.providers:
        if option.provider.config_name == provider_name:
            return option
    raise ValueError(f"Unknown provider: {provider_name}")


def build_provider_label(option: ProviderOption) -> str:
    status = "active" if option.active else "inactive"
    curated_count = len(option.curated_entries)
    curated_text = f"{curated_count} curated model"
    if curated_count != 1:
        curated_text += "s"
    return f"{option.provider.display_name:<16} [{status}] · {curated_text}"


def active_provider_names(snapshot: ModelPickerSnapshot) -> list[str]:
    return [option.provider.display_name for option in snapshot.providers if option.active]


def _model_identity(model_spec: str) -> tuple[Provider, str] | None:
    try:
        parsed = ModelFactory.parse_model_string(model_spec)
    except Exception:
        return None
    return parsed.provider, parsed.model_name


def _static_provider_models(provider: Provider) -> list[str]:
    models: list[str] = []
    for model in ModelDatabase.list_models():
        if ModelDatabase.get_default_provider(model) != provider:
            continue
        models.append(f"{provider.config_name}.{model}")
    return models


def model_options_for_provider(
    snapshot: ModelPickerSnapshot,
    provider: Provider,
    *,
    source: ModelSource,
) -> list[ModelOption]:
    if provider in REFER_TO_DOCS_PROVIDERS:
        return [
            ModelOption(
                spec=f"{provider.config_name}.refer-to-docs",
                label="Refer to docs (provider-specific setup not yet modeled)",
            )
        ]

    provider_option = find_provider(snapshot, provider.config_name)

    curated_options: list[ModelOption] = []
    for entry in provider_option.curated_entries:
        tags: list[str] = []
        if entry.fast:
            tags.append("fast")
        if not entry.current:
            tags.append("legacy")

        suffix = f" ({', '.join(tags)})" if tags else ""
        label = f"{entry.alias:<18} → {entry.model}{suffix}"
        curated_options.append(
            ModelOption(
                spec=entry.model,
                label=label,
                alias=entry.alias,
                fast=entry.fast,
                curated=entry.current,
            )
        )

    if source == "curated":
        return curated_options

    seen_identities: set[tuple[Provider, str]] = set()
    options: list[ModelOption] = list(curated_options)

    for curated in curated_options:
        model_identity = _model_identity(curated.spec)
        if model_identity is not None:
            seen_identities.add(model_identity)

    for spec in _static_provider_models(provider):
        model_identity = _model_identity(spec)
        if model_identity is not None and model_identity in seen_identities:
            continue
        if model_identity is not None:
            seen_identities.add(model_identity)
        options.append(ModelOption(spec=spec, label=f"{spec} (catalog)"))

    return options


def _supports_web_search(provider: Provider, model_name: str) -> bool:
    if provider in {
        Provider.RESPONSES,
        Provider.CODEX_RESPONSES,
        Provider.OPENRESPONSES,
    }:
        return True

    if provider == Provider.ANTHROPIC:
        return ModelDatabase.get_anthropic_web_search_version(model_name) is not None

    return False


def model_capabilities(model_spec: str) -> ModelCapabilities:
    parsed = ModelFactory.parse_model_string(model_spec)
    reasoning_spec = ModelDatabase.get_reasoning_effort_spec(parsed.model_name)
    reasoning_values = tuple(available_reasoning_values(reasoning_spec))
    default_reasoning = format_reasoning_setting(
        reasoning_spec.default if reasoning_spec is not None else None
    )
    long_context_window = ModelDatabase.get_long_context_window(parsed.model_name)
    cache_ttl_default = ModelDatabase.get_cache_ttl(parsed.model_name)

    return ModelCapabilities(
        provider=parsed.provider,
        model_name=parsed.model_name,
        reasoning_values=reasoning_values,
        current_reasoning=format_reasoning_setting(parsed.reasoning_effort),
        default_reasoning=default_reasoning,
        web_search_supported=_supports_web_search(parsed.provider, parsed.model_name),
        current_web_search=parsed.web_search,
        supports_long_context=long_context_window is not None,
        current_long_context=parsed.long_context,
        long_context_window=long_context_window,
        cache_ttl_default=cache_ttl_default,
    )


def _update_query_param(model_spec: str, *, key: str, value: str | None) -> str:
    parsed = urlsplit(model_spec)
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != key]

    if value is not None:
        query_pairs.append((key, value))

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_pairs),
            parsed.fragment,
        )
    )


def apply_option_overrides(
    model_spec: str,
    *,
    reasoning_value: str | None = None,
    web_search_value: str | None = None,
    context_value: str | None = None,
) -> str:
    """Apply optional overrides to a model spec query string.

    Values:
    - ``None``: keep existing value
    - ``DEFAULT_VALUE``: remove explicit query override
    - anything else: set query override to that value
    """

    result = model_spec

    if reasoning_value is not None:
        target = None if reasoning_value == DEFAULT_VALUE else reasoning_value
        result = _update_query_param(result, key="reasoning", value=target)

    if web_search_value is not None:
        target = None if web_search_value == DEFAULT_VALUE else web_search_value
        result = _update_query_param(result, key="web_search", value=target)

    if context_value is not None:
        target = None if context_value == DEFAULT_VALUE else context_value
        result = _update_query_param(result, key="context", value=target)

    return result


def web_search_display(value: bool | None) -> str:
    if value is None:
        return "default"
    return "on" if value else "off"
