from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from fast_agent.config import get_settings
from fast_agent.constants import DEFAULT_HOME_DIR, FAST_AGENT_RUNTIME_HOME
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.model_selection import CatalogModelEntry, ModelSelectionCatalog
from fast_agent.llm.provider.anthropic.vertex_config import (
    anthropic_vertex_ready,
)
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import available_reasoning_values, format_reasoning_setting
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.count_display import format_count
from fast_agent.utils.huggingface_hub import is_huggingface_hub_logged_in
from fast_agent.utils.text import strip_str_to_none

if TYPE_CHECKING:
    from pathlib import Path

ModelSource = Literal["curated", "all"]
ModelAvailability = Literal["active", "attention", "inactive"]


PICKER_PROVIDER_ORDER: tuple[Provider, ...] = (
    Provider.RESPONSES,
    Provider.OPENRESPONSES,
    Provider.CODEX_RESPONSES,
    Provider.ANTHROPIC,
    Provider.HUGGINGFACE,
    Provider.GOOGLE,
    Provider.XAI,
    Provider.META_AI,
    Provider.DEEPSEEK,
    Provider.GENERIC,
    Provider.ANTHROPIC_VERTEX,
    Provider.OPENAI,
    Provider.GROQ,
    Provider.AZURE,
    Provider.BEDROCK,
    Provider.ALIYUN,
    Provider.OPENROUTER,
    Provider.FAST_AGENT,
)

REFER_TO_DOCS_PROVIDERS: tuple[Provider, ...] = (
    Provider.OPENROUTER,
    Provider.AZURE,
    Provider.BEDROCK,
)

GENERIC_CUSTOM_MODEL_SENTINEL = "generic.__custom__"
LLAMACPP_PROVIDER_KEY = "llamacpp"
LLAMACPP_IMPORT_SENTINEL = "llamacpp.__import__"
PROVIDER_PREFIX_DELIMITERS = ("/", ".")
ProviderActiveCheck = Callable[[dict[str, Any]], bool]
ModelSpecTransform = Callable[[str], str]


@dataclass(frozen=True)
class ProviderActivation:
    provider: Provider


@dataclass(frozen=True)
class ProviderOption:
    provider: Provider | None
    active: bool
    curated_entries: tuple[CatalogModelEntry, ...]
    key: str | None = None
    display_name: str | None = None
    overlay_group: bool = False
    disabled_reason: str | None = None

    @property
    def option_key(self) -> str:
        if self.key is not None:
            return self.key
        if self.provider is None:
            raise ValueError("Provider option requires key when provider is unset")
        return self.provider.config_name


@dataclass(frozen=True)
class ModelOption:
    spec: str
    label: str
    preset_token: str | None = None
    fast: bool = False
    activation_action: ProviderActivation | None = None


SyntheticProviderOptionFactory = Callable[[Provider], list[ModelOption] | None]


@dataclass(frozen=True)
class ModelCapabilities:
    provider: Provider
    model_name: str
    reasoning_values: tuple[str, ...]
    default_reasoning: str
    web_search_supported: bool
    supports_long_context: bool
    long_context_window: int | None
    cache_ttl_default: str | None


@dataclass(frozen=True)
class ModelPickerSnapshot:
    providers: tuple[ProviderOption, ...]
    config_payload: dict[str, Any]


def _provider_is_active(provider: Provider, config_payload: dict[str, Any]) -> bool:
    if provider == Provider.ANTHROPIC_VERTEX:
        ready, _ = anthropic_vertex_ready(config_payload)
        return ready

    config_key = ProviderKeyManager.get_config_file_key(provider.config_name, config_payload)
    if config_key:
        return True

    if ProviderKeyManager.get_env_var(provider.config_name):
        return True

    if active_check := _PROVIDER_ACTIVE_CHECKS.get(provider):
        return active_check(config_payload)

    return provider in {Provider.FAST_AGENT, Provider.GENERIC}


def _google_vertex_is_active(config_payload: dict[str, Any]) -> bool:
    google_cfg = config_payload.get("google")
    if not isinstance(google_cfg, dict):
        return False
    vertex_cfg = google_cfg.get("vertex_ai")
    return isinstance(vertex_cfg, dict) and bool(vertex_cfg.get("enabled"))


def _azure_default_credential_is_active(config_payload: dict[str, Any]) -> bool:
    azure_cfg = config_payload.get("azure")
    if not isinstance(azure_cfg, dict):
        return False
    use_default = bool(azure_cfg.get("use_default_azure_credential"))
    base_url = azure_cfg.get("base_url")
    normalized_base_url = strip_str_to_none(base_url)
    return use_default and normalized_base_url is not None


def _huggingface_hub_is_active(_config_payload: dict[str, Any]) -> bool:
    return is_huggingface_hub_logged_in()


_PROVIDER_ACTIVE_CHECKS: dict[Provider, ProviderActiveCheck] = {
    Provider.GOOGLE: _google_vertex_is_active,
    Provider.AZURE: _azure_default_credential_is_active,
    Provider.HUGGINGFACE: _huggingface_hub_is_active,
}


def _catalog_options_from_entries(
    entries: tuple[CatalogModelEntry, ...],
    *,
    provider: Provider,
    source: ModelSource,
    spec_transform: ModelSpecTransform | None = None,
    filter_current: bool = True,
) -> list[ModelOption]:
    transform = spec_transform or _identity_model_spec

    entry_options: list[ModelOption] = []
    for entry in entries:
        spec = transform(entry.model)
        entry_options.append(
            ModelOption(
                spec=spec,
                label=format_catalog_model_entry_label(entry, spec=spec),
                preset_token=entry.alias,
                fast=entry.fast,
            )
        )

    if source == "curated":
        if not filter_current:
            return entry_options
        return [
            option for entry, option in zip(entries, entry_options, strict=True) if entry.current
        ]

    seen_identities: set[tuple[Provider, str]] = set()
    options: list[ModelOption] = list(entry_options)
    for option in entry_options:
        identity = model_identity(option.spec)
        if identity is not None:
            seen_identities.add(identity)

    for spec in _static_provider_models(provider):
        transformed_spec = transform(spec)
        identity = model_identity(transformed_spec)
        if identity is not None and identity in seen_identities:
            continue
        if identity is not None:
            seen_identities.add(identity)
        options.append(ModelOption(spec=transformed_spec, label=f"{transformed_spec} (catalog)"))

    return options


def _identity_model_spec(model_spec: str) -> str:
    return model_spec


def catalog_model_entry_tags(entry: CatalogModelEntry) -> tuple[str, ...]:
    tags: list[str] = []
    if entry.local:
        tags.append("local")
    if entry.fast:
        tags.append("fast")
    if not entry.current:
        tags.append("legacy")
    return tuple(tags)


def format_catalog_model_entry_label(entry: CatalogModelEntry, *, spec: str | None = None) -> str:
    resolved_spec = spec or entry.model
    tags = catalog_model_entry_tags(entry)
    suffix = f" ({', '.join(tags)})" if tags else ""
    label = f"{(entry.display_label or entry.alias):<19} → {resolved_spec}{suffix}"
    if entry.description:
        label = f"{label} — {entry.description}"
    return label


def _generic_provider_model_options(_provider: Provider) -> list[ModelOption]:
    return [
        ModelOption(
            spec=GENERIC_CUSTOM_MODEL_SENTINEL,
            label="Enter local model string (e.g. llama3.2)",
        )
    ]


def _refer_to_docs_provider_model_options(provider: Provider) -> list[ModelOption]:
    return [
        ModelOption(
            spec=f"{provider.config_name}.refer-to-docs",
            label="Refer to docs (provider-specific setup)",
        )
    ]


_SYNTHETIC_PROVIDER_OPTION_FACTORIES: dict[Provider, SyntheticProviderOptionFactory] = {
    Provider.GENERIC: _generic_provider_model_options,
    **{provider: _refer_to_docs_provider_model_options for provider in REFER_TO_DOCS_PROVIDERS},
}


def _synthetic_provider_model_options(provider: Provider) -> list[ModelOption] | None:
    factory = _SYNTHETIC_PROVIDER_OPTION_FACTORIES.get(provider)
    if factory is None:
        return None
    return factory(provider)


def model_options_for_option(
    option: ProviderOption,
    *,
    source: ModelSource,
) -> list[ModelOption]:
    provider = option.provider
    if provider is not None:
        synthetic_options = _synthetic_provider_model_options(provider)
        if synthetic_options is not None:
            return synthetic_options

    if option.option_key == LLAMACPP_PROVIDER_KEY:
        return [
            ModelOption(
                spec=LLAMACPP_IMPORT_SENTINEL,
                label="Discover local llama.cpp models and write overlay",
            )
        ]

    if option.overlay_group:
        return _catalog_options_from_entries(
            option.curated_entries,
            provider=Provider.ANTHROPIC,
            source="curated",
            filter_current=False,
        )

    if provider is None:
        raise ValueError(f"Provider option '{option.option_key}' has no model provider")
    return _catalog_options_from_entries(
        option.curated_entries,
        provider=provider,
        source=source,
    )


def build_snapshot(
    config_path: str | Path | None = None,
    *,
    config_payload: dict[str, Any] | None = None,
    start_path: Path | None = None,
) -> ModelPickerSnapshot:
    if config_payload is None:
        settings = get_settings(str(config_path) if config_path else None)
        config_payload = settings.model_dump()

    active_providers = set(ModelSelectionCatalog.configured_providers(config_payload))
    for provider in PICKER_PROVIDER_ORDER:
        if _provider_is_active(provider, config_payload):
            active_providers.add(provider)

    providers: list[ProviderOption] = []
    overlay_registry = _load_overlay_registry_for_snapshot(
        config_path=config_path,
        config_payload=config_payload,
        start_path=start_path,
    )
    overlay_entries = tuple(
        CatalogModelEntry(
            alias=overlay.name,
            model=overlay.compiled_model_spec,
            current=overlay.current,
            fast=overlay.fast,
            local=True,
            display_label=overlay.display_label,
            description=overlay.description,
        )
        for overlay in overlay_registry.overlays
    )
    overlay_group_active = bool(overlay_entries)
    providers.append(
        ProviderOption(
            provider=None,
            active=overlay_group_active,
            curated_entries=overlay_entries,
            key="overlays",
            display_name="Overlays",
            overlay_group=True,
        )
    )
    for provider in PICKER_PROVIDER_ORDER:
        entries = tuple(
            entry
            for entry in ModelSelectionCatalog.list_entries(
                provider,
                overlay_registry=overlay_registry,
            )
            if not entry.local
        )
        has_special_picker_flow = (
            provider in REFER_TO_DOCS_PROVIDERS or provider == Provider.GENERIC
        )
        if not entries and not has_special_picker_flow:
            continue
        if provider == Provider.DEEPSEEK:
            providers.append(
                ProviderOption(
                    provider=None,
                    active=False,
                    curated_entries=(),
                    key=LLAMACPP_PROVIDER_KEY,
                    display_name="llama.cpp",
                )
            )
        providers.append(
            ProviderOption(
                provider=provider,
                active=provider in active_providers,
                curated_entries=entries,
                display_name=("Generic (ollama)" if provider == Provider.GENERIC else None),
                disabled_reason=(
                    anthropic_vertex_ready(config_payload)[1]
                    if provider == Provider.ANTHROPIC_VERTEX and provider not in active_providers
                    else None
                ),
            )
        )

    return ModelPickerSnapshot(providers=tuple(providers), config_payload=config_payload)


def _load_overlay_registry_for_snapshot(
    *,
    config_path: str | Path | None,
    config_payload: dict[str, Any],
    start_path: Path | None,
):
    home = config_payload.get("home")
    normalized_home = _normalized_overlay_home(home)
    candidate_starts = _overlay_candidate_starts(
        config_path=config_path,
        home=home,
        normalized_home=normalized_home,
        start_path=start_path,
    )

    if normalized_home is None and (config_path is not None or start_path is not None):
        normalized_home = DEFAULT_HOME_DIR

    return _first_overlay_registry_with_entries(
        _dedupe_paths(candidate_starts),
        home=normalized_home,
    )


def _normalized_overlay_home(home: object) -> str | Path | None:
    from pathlib import Path as _Path

    if isinstance(home, (str, _Path)):
        return home
    return os.getenv(FAST_AGENT_RUNTIME_HOME) or os.getenv("FAST_AGENT_HOME")


def _overlay_candidate_starts(
    *,
    config_path: str | Path | None,
    home: object,
    normalized_home: str | Path | None,
    start_path: Path | None,
) -> list[Path]:
    from pathlib import Path as _Path

    if config_path is not None:
        return _config_overlay_candidate_starts(
            config_path=config_path,
            home=home,
            normalized_home=normalized_home,
        )

    if start_path is not None:
        return [_Path(start_path).expanduser().resolve()]
    return [_Path.cwd().resolve()]


def _config_overlay_candidate_starts(
    *,
    config_path: str | Path,
    home: object,
    normalized_home: str | Path | None,
) -> list[Path]:
    from pathlib import Path as _Path

    config_file = _Path(config_path).expanduser().resolve()
    candidate_starts: list[Path] = [config_file.parent]
    relative_home = _relative_overlay_home(
        home=home,
        normalized_home=normalized_home,
    )
    project_root = _project_root_for_env_config(config_file.parent, relative_home)
    if project_root is not None:
        candidate_starts.append(project_root)
    return candidate_starts


def _relative_overlay_home(
    *,
    home: object,
    normalized_home: str | Path | None,
) -> Path | None:
    from pathlib import Path as _Path

    if home is None:
        return _Path(DEFAULT_HOME_DIR)
    if normalized_home is None:
        raise ValueError("home must be a string or path")

    home_path = _Path(normalized_home).expanduser()
    if home_path.is_absolute():
        return None
    return home_path


def _project_root_for_env_config(config_dir: Path, relative_home: Path | None) -> Path | None:
    if relative_home is None or not relative_home.parts:
        return None

    env_parts = relative_home.parts
    parent_parts = config_dir.parts
    if len(env_parts) > len(parent_parts) or parent_parts[-len(env_parts) :] != env_parts:
        return None

    project_root = config_dir
    for _ in env_parts:
        project_root = project_root.parent
    if project_root == config_dir:
        return None
    return project_root


def _dedupe_paths(candidate_starts: list[Path]) -> list[Path]:
    return unique_preserve_order(candidate_starts)


def _first_overlay_registry_with_entries(
    ordered_starts: list[Path],
    *,
    home: str | Path | None,
):
    from pathlib import Path as _Path

    fallback_registry = None
    for overlay_start_path in ordered_starts:
        registry = load_model_overlay_registry(
            start_path=overlay_start_path,
            home=home,
        )
        if fallback_registry is None:
            fallback_registry = registry
        if registry.overlays:
            return registry

    if fallback_registry is not None:
        return fallback_registry
    return load_model_overlay_registry(start_path=_Path.cwd().resolve(), home=home)


def find_provider(snapshot: ModelPickerSnapshot, provider_name: str) -> ProviderOption:
    for option in snapshot.providers:
        if option.option_key == provider_name:
            return option
    raise ValueError(f"Unknown provider: {provider_name}")


def provider_option_count_label(option: ProviderOption) -> str:
    if option.option_key == LLAMACPP_PROVIDER_KEY:
        return "import flow"

    curated_count = len(option.curated_entries)
    if option.overlay_group:
        return format_count(curated_count, "overlay")
    return format_count(curated_count, "curated model")


def has_explicit_provider_prefix(model_spec: str) -> bool:
    provider_names = {provider.config_name for provider in Provider}
    for delimiter in PROVIDER_PREFIX_DELIMITERS:
        prefix, separator, rest = model_spec.partition(delimiter)
        if prefix and separator and rest and prefix in provider_names:
            return True
    return False


def normalize_generic_model_spec(raw_model: str) -> str | None:
    candidate = raw_model.strip()
    if not candidate:
        return None

    if has_explicit_provider_prefix(candidate):
        return candidate

    return f"generic.{candidate}"


def infer_initial_picker_provider(model_spec: str | None) -> str | None:
    if model_spec is None:
        return None

    normalized = model_spec.strip()
    if not normalized:
        return None

    from fast_agent.llm.model_factory import ModelFactory

    try:
        parsed = ModelFactory.parse_model_string(
            normalized,
            presets=ModelFactory.MODEL_PRESETS,
        )
    except Exception:
        return None

    config_name = parsed.provider.config_name.strip()
    return config_name or None


def provider_activation_action(
    snapshot: ModelPickerSnapshot,
    provider: Provider,
) -> ProviderActivation | None:
    option = find_provider(snapshot, provider.config_name)
    if provider in {Provider.CODEX_RESPONSES, Provider.XAI} and not option.active:
        return ProviderActivation(provider)
    return None


def model_identity(model_spec: str) -> tuple[Provider, str] | None:
    from fast_agent.llm.model_factory import ModelFactory

    try:
        parsed = ModelFactory.parse_model_string(model_spec)
    except Exception:
        return None
    return parsed.provider, parsed.model_name


def _static_provider_models(provider: Provider) -> list[str]:
    models: list[str] = []
    for model in ModelDatabase.list_models():
        default_provider = ModelDatabase.get_default_provider(model)
        if provider == Provider.ANTHROPIC_VERTEX:
            if default_provider != Provider.ANTHROPIC:
                continue
            models.append(f"{provider.config_name}.{model}")
            continue
        if default_provider != provider:
            continue
        models.append(f"{provider.config_name}.{model}")
    return models


def model_options_for_provider(
    snapshot: ModelPickerSnapshot,
    provider: Provider,
    *,
    source: ModelSource,
) -> list[ModelOption]:
    synthetic_options = _synthetic_provider_model_options(provider)
    if synthetic_options is not None:
        return synthetic_options

    provider_option = find_provider(snapshot, provider.config_name)
    activation_action = provider_activation_action(snapshot, provider)
    if activation_action is not None:
        return [
            replace(option, activation_action=activation_action)
            for option in _catalog_options_from_entries(
                provider_option.curated_entries,
                provider=provider,
                source=source,
            )
        ]
    return _catalog_options_from_entries(
        provider_option.curated_entries,
        provider=provider,
        source=source,
    )


def model_capabilities(model_spec: str) -> ModelCapabilities:
    from fast_agent.llm.model_factory import ModelFactory

    resolved = ModelFactory.resolve_model_spec(model_spec)
    parsed = resolved.model_config
    reasoning_spec = resolved.reasoning_effort_spec
    reasoning_values = tuple(available_reasoning_values(reasoning_spec))
    default_reasoning = format_reasoning_setting(
        reasoning_spec.default if reasoning_spec is not None else None
    )
    long_context_window = resolved.long_context_window
    cache_ttl_default = resolved.cache_ttl
    supports_long_context = long_context_window is not None

    return ModelCapabilities(
        provider=parsed.provider,
        model_name=parsed.model_name,
        reasoning_values=reasoning_values,
        default_reasoning=default_reasoning,
        web_search_supported=(
            parsed.provider in {Provider.RESPONSES, Provider.CODEX_RESPONSES, Provider.XAI}
            or (
                parsed.provider == Provider.ANTHROPIC
                and resolved.anthropic_web_search_version is not None
            )
        ),
        supports_long_context=supports_long_context,
        long_context_window=long_context_window,
        cache_ttl_default=cache_ttl_default,
    )
