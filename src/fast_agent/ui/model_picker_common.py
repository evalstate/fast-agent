from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fast_agent.config import get_settings
from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.model_selection import CatalogModelEntry, ModelSelectionCatalog
from fast_agent.llm.provider.anthropic.vertex_config import (
    anthropic_vertex_intent,
    anthropic_vertex_ready,
)
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import available_reasoning_values, format_reasoning_setting

if TYPE_CHECKING:
    from pathlib import Path

ModelSource = Literal["curated", "all"]
ProviderActivationAction = Literal["codex-login"]
KEEP_VALUE = "__keep__"
DEFAULT_VALUE = "__default__"

PICKER_PROVIDER_ORDER: tuple[Provider, ...] = (
    Provider.RESPONSES,
    Provider.OPENRESPONSES,
    Provider.CODEX_RESPONSES,
    Provider.ANTHROPIC,
    Provider.ANTHROPIC_VERTEX,
    Provider.HUGGINGFACE,
    Provider.OPENAI,
    Provider.GENERIC,
    Provider.GOOGLE,
    Provider.XAI,
    Provider.GROQ,
    Provider.DEEPSEEK,
    Provider.ALIYUN,
    Provider.OPENROUTER,
    Provider.AZURE,
    Provider.BEDROCK,
    Provider.FAST_AGENT,
)

REFER_TO_DOCS_PROVIDERS: tuple[Provider, ...] = (
    Provider.OPENROUTER,
    Provider.AZURE,
    Provider.BEDROCK,
)

GENERIC_CUSTOM_MODEL_SENTINEL = "generic.__custom__"
CODEX_LOGIN_SENTINEL = "codexresponses.__login__"
ANTHROPIC_VERTEX_PROVIDER_KEY = "anthropic-vertex"


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
        assert self.provider is not None
        return self.provider.config_name

    @property
    def option_display_name(self) -> str:
        if self.display_name is not None:
            return self.display_name
        assert self.provider is not None
        return self.provider.display_name


@dataclass(frozen=True)
class ModelOption:
    spec: str
    label: str
    preset_token: str | None = None
    fast: bool = False
    curated: bool = False
    activation_action: ProviderActivationAction | None = None


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
    if provider == Provider.ANTHROPIC_VERTEX:
        ready, _ = anthropic_vertex_ready(config_payload)
        return ready

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

    if provider in {Provider.FAST_AGENT, Provider.GENERIC}:
        return True

    return False

def _catalog_options_from_entries(
    entries: tuple[CatalogModelEntry, ...],
    *,
    provider: Provider,
    source: ModelSource,
    spec_transform: Any = None,
) -> list[ModelOption]:
    transform = spec_transform or (lambda value: value)

    curated_options: list[ModelOption] = []
    for entry in entries:
        spec = transform(entry.model)
        tags: list[str] = []
        if entry.local:
            tags.append("local")
        if entry.fast:
            tags.append("fast")
        if not entry.current:
            tags.append("legacy")

        suffix = f" ({', '.join(tags)})" if tags else ""
        entry_label = entry.display_label or entry.alias
        label = f"{entry_label:<19} → {spec}{suffix}"
        if entry.description:
            label = f"{label} — {entry.description}"
        curated_options.append(
            ModelOption(
                spec=spec,
                label=label,
                preset_token=entry.alias,
                fast=entry.fast,
                curated=entry.current,
            )
        )

    if source == "curated":
        return curated_options

    seen_identities: set[tuple[Provider, str]] = set()
    options: list[ModelOption] = list(curated_options)
    for curated in curated_options:
        identity = model_identity(curated.spec)
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


def model_options_for_option(
    snapshot: ModelPickerSnapshot,
    option: ProviderOption,
    *,
    source: ModelSource,
) -> list[ModelOption]:
    if option.overlay_group:
        return _catalog_options_from_entries(
            option.curated_entries,
            provider=Provider.ANTHROPIC,
            source="curated",
        )

    provider = option.provider
    assert provider is not None
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
    if overlay_entries:
        overlay_group_active = True
    else:
        overlay_group_active = False
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
        if provider == Provider.ANTHROPIC_VERTEX and not anthropic_vertex_intent(config_payload):
            continue
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
        providers.append(
            ProviderOption(
                provider=provider,
                active=provider in active_providers,
                curated_entries=entries,
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
    from pathlib import Path as _Path

    env_dir = config_payload.get("environment_dir")
    normalized_env_dir = env_dir if isinstance(env_dir, (str, _Path)) else None

    candidate_starts: list[_Path] = []
    if config_path is not None:
        config_file = _Path(config_path).expanduser().resolve()
        candidate_starts.append(config_file.parent)

        relative_env_dir: _Path | None = None
        if normalized_env_dir is None:
            relative_env_dir = _Path(DEFAULT_ENVIRONMENT_DIR)
        else:
            env_path = _Path(normalized_env_dir).expanduser()
            if not env_path.is_absolute():
                relative_env_dir = env_path

        if relative_env_dir is not None and relative_env_dir.parts:
            env_parts = relative_env_dir.parts
            parent_parts = config_file.parent.parts
            if len(env_parts) <= len(parent_parts) and parent_parts[-len(env_parts) :] == env_parts:
                project_root = config_file.parent
                for _ in env_parts:
                    project_root = project_root.parent
                if project_root != config_file.parent:
                    candidate_starts.append(project_root)

    if config_path is None and start_path is not None:
        candidate_starts.append(_Path(start_path).expanduser().resolve())

    if config_path is None and start_path is None:
        candidate_starts.append(_Path.cwd().resolve())

    seen: set[_Path] = set()
    ordered_starts: list[_Path] = []
    for candidate in candidate_starts:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_starts.append(candidate)

    fallback_registry = None
    for start_path in ordered_starts:
        registry = load_model_overlay_registry(start_path=start_path, env_dir=normalized_env_dir)
        if fallback_registry is None:
            fallback_registry = registry
        if registry.overlays:
            return registry

    assert fallback_registry is not None
    return fallback_registry


def find_provider(snapshot: ModelPickerSnapshot, provider_name: str) -> ProviderOption:
    for option in snapshot.providers:
        if option.option_key == provider_name:
            return option
    raise ValueError(f"Unknown provider: {provider_name}")


def build_provider_label(option: ProviderOption) -> str:
    status = "active" if option.active else "disabled" if option.disabled_reason else "inactive"
    curated_count = len(option.curated_entries)
    if option.overlay_group:
        entry_text = "overlay" if curated_count == 1 else "overlays"
        count_text = f"{curated_count} {entry_text}"
    else:
        count_text = f"{curated_count} curated model"
        if curated_count != 1:
            count_text += "s"
    return f"{option.option_display_name:<16} [{status}] · {count_text}"


def active_provider_names(snapshot: ModelPickerSnapshot) -> list[str]:
    return [option.option_display_name for option in snapshot.providers if option.active]


def has_explicit_provider_prefix(model_spec: str) -> bool:
    provider_names = {provider.config_name for provider in Provider}

    slash_prefix, _, slash_rest = model_spec.partition("/")
    if slash_prefix and slash_rest and slash_prefix in provider_names:
        return True

    dot_prefix, _, dot_rest = model_spec.partition(".")
    if dot_prefix and dot_rest and dot_prefix in provider_names:
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
) -> ProviderActivationAction | None:
    option = find_provider(snapshot, provider.config_name)
    if provider == Provider.CODEX_RESPONSES and not option.active:
        return "codex-login"
    return None


def model_identity(model_spec: str) -> tuple[Provider, str] | None:
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
    if provider == Provider.GENERIC:
        return [
            ModelOption(
                spec=GENERIC_CUSTOM_MODEL_SENTINEL,
                label="Enter local model string (e.g. llama3.2)",
            )
        ]

    if provider in REFER_TO_DOCS_PROVIDERS:
        return [
            ModelOption(
                spec=f"{provider.config_name}.refer-to-docs",
                label="Refer to docs (provider-specific setup)",
            )
        ]

    provider_option = find_provider(snapshot, provider.config_name)
    activation_action = provider_activation_action(snapshot, provider)
    if activation_action is not None:
        return [
            ModelOption(
                spec=CODEX_LOGIN_SENTINEL,
                label="Log in to enable Codex (Plan)",
                activation_action=activation_action,
            )
        ]
    return _catalog_options_from_entries(
        provider_option.curated_entries,
        provider=provider,
        source=source,
    )


def model_capabilities(model_spec: str) -> ModelCapabilities:
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
        current_reasoning=format_reasoning_setting(parsed.reasoning_effort),
        default_reasoning=default_reasoning,
        web_search_supported=(
            parsed.provider in {Provider.RESPONSES, Provider.CODEX_RESPONSES}
            or (
                parsed.provider == Provider.ANTHROPIC
                and resolved.anthropic_web_search_version is not None
            )
        ),
        current_web_search=parsed.web_search,
        supports_long_context=supports_long_context,
        current_long_context=parsed.long_context and supports_long_context,
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
