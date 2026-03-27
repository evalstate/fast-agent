"""Local model overlay discovery and resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

import fast_agent.config as config_module
from fast_agent.config import load_yaml_mapping, resolve_environment_config_file
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.provider_types import Provider

logger = get_logger(__name__)


def _normalize_reasoning_value(value: str | bool | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        return str(value)
    normalized = value.strip()
    return normalized or None


def _normalize_toggle_value(value: bool | None) -> str | None:
    if value is None:
        return None
    return "on" if value else "off"


def _prefixed_model_spec(provider: Provider, model_name: str) -> str:
    if model_name.startswith(f"{provider.value}.") or model_name.startswith(f"{provider.value}/"):
        return model_name
    return f"{provider.value}.{model_name}"


def _overlay_model_key(provider: Provider, model_name: str) -> str:
    normalized = model_name.strip().lower()
    if provider == Provider.HUGGINGFACE and ":" in normalized:
        return normalized.rsplit(":", 1)[0]
    return normalized


def _existing_model_params(provider: Provider, model_name: str) -> ModelParameters | None:
    return ModelDatabase.get_model_params(model_name, provider=provider)


class ModelOverlayConnection(BaseModel):
    """Connection overrides attached to a local model overlay."""

    model_config = ConfigDict(extra="ignore")

    base_url: str | None = None
    auth: Literal["none", "env", "secret_ref"] | None = None
    api_key_env: str | None = None
    secret_ref: str | None = None
    default_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_auth_configuration(self) -> "ModelOverlayConnection":
        if self.auth == "env" and not self.api_key_env:
            raise ValueError("connection.api_key_env is required when connection.auth is 'env'")
        if self.auth == "secret_ref" and not self.secret_ref:
            raise ValueError("connection.secret_ref is required when connection.auth is 'secret_ref'")
        return self

    def auth_mode(self) -> Literal["none", "env", "secret_ref"] | None:
        if self.auth is not None:
            return self.auth
        if self.api_key_env:
            return "env"
        if self.secret_ref:
            return "secret_ref"
        return None


class ModelOverlayDefaults(BaseModel):
    """Request defaults applied by a local overlay."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str | bool | int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    transport: Literal["sse", "websocket", "auto"] | None = None
    service_tier: Literal["fast", "flex"] | None = None
    web_search: bool | None = None
    web_fetch: bool | None = None
    max_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
    )

    def to_query_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        reasoning = _normalize_reasoning_value(self.reasoning)
        if reasoning is not None:
            pairs.append(("reasoning", reasoning))
        if self.temperature is not None:
            pairs.append(("temperature", str(self.temperature)))
        if self.top_p is not None:
            pairs.append(("top_p", str(self.top_p)))
        if self.top_k is not None:
            pairs.append(("top_k", str(self.top_k)))
        if self.min_p is not None:
            pairs.append(("min_p", str(self.min_p)))
        if self.presence_penalty is not None:
            pairs.append(("presence_penalty", str(self.presence_penalty)))
        if self.repetition_penalty is not None:
            pairs.append(("repetition_penalty", str(self.repetition_penalty)))
        if self.transport is not None:
            pairs.append(("transport", self.transport))
        if self.service_tier is not None:
            pairs.append(("service_tier", self.service_tier))
        web_search = _normalize_toggle_value(self.web_search)
        if web_search is not None:
            pairs.append(("web_search", web_search))
        web_fetch = _normalize_toggle_value(self.web_fetch)
        if web_fetch is not None:
            pairs.append(("web_fetch", web_fetch))
        return pairs


class ModelOverlayMetadata(BaseModel):
    """Runtime metadata attached to a local overlay."""

    model_config = ConfigDict(extra="ignore")

    context_window: int | None = None
    max_output_tokens: int | None = None
    tokenizes: list[str] | None = None
    # Legacy fallback retained for older overlay files. New overlays should use
    # defaults.temperature instead.
    default_temperature: float | None = None
    fast: bool | None = None


class ModelOverlayPicker(BaseModel):
    """Picker presentation metadata for a local overlay."""

    model_config = ConfigDict(extra="ignore")

    label: str | None = None
    description: str | None = None
    current: bool = True
    featured: bool = False


class ModelOverlaySecretEntry(BaseModel):
    """Secret companion entry for a model overlay."""

    model_config = ConfigDict(extra="ignore")

    api_key: str | None = None
    default_headers: dict[str, str] | None = None


class ModelOverlayManifest(BaseModel):
    """User-authored local model overlay manifest."""

    model_config = ConfigDict(extra="ignore")

    name: str
    provider: Provider
    model: str
    connection: ModelOverlayConnection = Field(default_factory=ModelOverlayConnection)
    defaults: ModelOverlayDefaults = Field(
        default_factory=ModelOverlayDefaults,
        validation_alias=AliasChoices("defaults", "request_defaults"),
    )
    metadata: ModelOverlayMetadata = Field(
        default_factory=ModelOverlayMetadata,
        validation_alias=AliasChoices("metadata", "model_metadata"),
    )
    picker: ModelOverlayPicker = Field(default_factory=ModelOverlayPicker)


@dataclass(frozen=True, slots=True)
class LoadedModelOverlay:
    """Loaded overlay plus file provenance and secret companion data."""

    manifest: ModelOverlayManifest
    manifest_path: Path
    secret_entry: ModelOverlaySecretEntry | None = None

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def provider(self) -> Provider:
        return self.manifest.provider

    @property
    def model_name(self) -> str:
        return self.manifest.model

    @property
    def display_label(self) -> str:
        label = self.manifest.picker.label
        return label if label is not None and label.strip() else self.name

    @property
    def description(self) -> str | None:
        description = self.manifest.picker.description
        if description is None:
            return None
        normalized = description.strip()
        return normalized or None

    @property
    def current(self) -> bool:
        return self.manifest.picker.current

    @property
    def featured(self) -> bool:
        return self.manifest.picker.featured

    @property
    def fast(self) -> bool:
        return bool(self.manifest.metadata.fast)

    @property
    def compiled_model_spec(self) -> str:
        model_spec = _prefixed_model_spec(self.provider, self.model_name)
        query = self.manifest.defaults.to_query_pairs()
        if not query:
            return model_spec
        return f"{model_spec}?{urlencode(query)}"

    def resolved_default_headers(self) -> dict[str, str] | None:
        headers = dict(self.manifest.connection.default_headers)
        if self.secret_entry and self.secret_entry.default_headers:
            headers.update(self.secret_entry.default_headers)
        return headers or None

    def resolved_api_key(self) -> str | None:
        auth_mode = self.manifest.connection.auth_mode()
        if auth_mode == "none":
            return ""
        if auth_mode == "env":
            env_name = self.manifest.connection.api_key_env
            if not env_name:
                raise ModelConfigError(
                    f"Overlay '{self.name}' is missing connection.api_key_env",
                    "Set connection.api_key_env or change connection.auth.",
                )
            api_key = os.getenv(env_name)
            if api_key is None:
                raise ModelConfigError(
                    f"Overlay '{self.name}' requires environment variable '{env_name}'",
                    f"Set {env_name} before using overlay '{self.name}'.",
                )
            return api_key
        if auth_mode == "secret_ref":
            secret_ref = self.manifest.connection.secret_ref
            if not secret_ref:
                raise ModelConfigError(
                    f"Overlay '{self.name}' is missing connection.secret_ref",
                    "Set connection.secret_ref or change connection.auth.",
                )
            if self.secret_entry is None or self.secret_entry.api_key is None:
                raise ModelConfigError(
                    f"Overlay '{self.name}' secret ref '{secret_ref}' could not be resolved",
                    "Add an api_key entry to .fast-agent/model-overlays.secrets.yaml.",
                )
            return self.secret_entry.api_key
        return None

    def llm_init_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if self.manifest.connection.base_url is not None:
            kwargs["base_url"] = self.manifest.connection.base_url
        api_key = self.resolved_api_key()
        if api_key is not None:
            kwargs["api_key"] = api_key
        default_headers = self.resolved_default_headers()
        if default_headers is not None:
            kwargs["default_headers"] = default_headers
        return kwargs

    def build_model_parameters(self) -> ModelParameters | None:
        existing = _existing_model_params(self.provider, self.model_name)
        context_window = self.manifest.metadata.context_window
        if context_window is None and existing is not None:
            context_window = existing.context_window

        max_output_tokens = self.manifest.metadata.max_output_tokens
        if max_output_tokens is None:
            max_output_tokens = self.manifest.defaults.max_tokens
        if max_output_tokens is None and existing is not None:
            max_output_tokens = existing.max_output_tokens

        if context_window is None or max_output_tokens is None:
            return None

        default_temperature = self.manifest.defaults.temperature
        if default_temperature is None:
            default_temperature = self.manifest.metadata.default_temperature

        if existing is not None:
            update_payload: dict[str, object] = {
                "context_window": context_window,
                "max_output_tokens": max_output_tokens,
                "default_provider": self.provider,
            }
            if self.manifest.metadata.tokenizes is not None:
                update_payload["tokenizes"] = self.manifest.metadata.tokenizes
            if default_temperature is not None:
                update_payload["default_temperature"] = default_temperature
            if self.manifest.metadata.fast is not None:
                update_payload["fast"] = self.manifest.metadata.fast
            return existing.model_copy(update=update_payload)

        tokenizes = self.manifest.metadata.tokenizes or list(ModelDatabase.TEXT_ONLY)
        return ModelParameters(
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            tokenizes=tokenizes,
            default_provider=self.provider,
            default_temperature=default_temperature,
            fast=bool(self.manifest.metadata.fast),
        )


@dataclass(frozen=True, slots=True)
class ModelOverlayRegistry:
    """Resolved local overlay registry."""

    overlays: tuple[LoadedModelOverlay, ...]
    env_root: Path

    def by_name(self) -> dict[str, LoadedModelOverlay]:
        return {overlay.name: overlay for overlay in self.overlays}

    def runtime_presets(self) -> dict[str, str]:
        return {overlay.name: overlay.compiled_model_spec for overlay in self.overlays}

    def providers(self) -> tuple[Provider, ...]:
        seen: set[Provider] = set()
        ordered: list[Provider] = []
        for overlay in self.overlays:
            if overlay.provider in seen:
                continue
            seen.add(overlay.provider)
            ordered.append(overlay.provider)
        return tuple(ordered)

    def entries_for_provider(self, provider: Provider) -> tuple[LoadedModelOverlay, ...]:
        overlays = [overlay for overlay in self.overlays if overlay.provider == provider]
        overlays.sort(key=lambda overlay: (not overlay.current, not overlay.featured, overlay.name))
        return tuple(overlays)

    def resolve_model_string(self, model_string: str) -> LoadedModelOverlay | None:
        raw_token = model_string.partition("?")[0].strip()
        if not raw_token:
            return None
        return self.by_name().get(raw_token)


@dataclass(frozen=True, slots=True)
class ModelOverlayPaths:
    """Filesystem paths used by the model overlay registry."""

    env_root: Path
    overlays_dir: Path
    secrets_path: Path


def _load_secret_entries(path: Path) -> dict[str, ModelOverlaySecretEntry]:
    payload = load_yaml_mapping(path)
    if not payload:
        return {}

    raw_entries = payload.get("overlays")
    if raw_entries is None:
        raw_entries = payload
    if not isinstance(raw_entries, dict):
        return {}

    entries: dict[str, ModelOverlaySecretEntry] = {}
    for name, entry_payload in raw_entries.items():
        if not isinstance(name, str) or not isinstance(entry_payload, dict):
            continue
        try:
            entries[name] = ModelOverlaySecretEntry.model_validate(entry_payload)
        except Exception as exc:
            logger.warning(
                "Skipping invalid model overlay secret entry",
                secret_ref=name,
                path=str(path),
                error=str(exc),
            )
    return entries


def _load_overlay_file(
    path: Path,
    *,
    secret_entries: dict[str, ModelOverlaySecretEntry],
) -> LoadedModelOverlay | None:
    payload = load_yaml_mapping(path)
    if not payload:
        return None

    try:
        manifest = ModelOverlayManifest.model_validate(payload)
    except Exception as exc:
        logger.warning(
            "Skipping invalid model overlay manifest",
            path=str(path),
            error=str(exc),
        )
        return None

    secret_entry = None
    secret_ref = manifest.connection.secret_ref
    if secret_ref:
        secret_entry = secret_entries.get(secret_ref)

    return LoadedModelOverlay(
        manifest=manifest,
        manifest_path=path,
        secret_entry=secret_entry,
    )


def resolve_model_overlay_paths(
    *,
    start_path: Path | None = None,
    env_dir: str | Path | None = None,
) -> ModelOverlayPaths:
    """Resolve the active model overlay storage paths."""

    base_path = (start_path or Path.cwd()).resolve()
    override = env_dir
    if override is None:
        override = os.getenv("ENVIRONMENT_DIR")
    if override is None:
        configured = _settings_environment_override(start_path=start_path)
        if configured is not None:
            base_path, override = configured

    env_root = resolve_environment_config_file(base_path, env_dir=override).parent
    return ModelOverlayPaths(
        env_root=env_root,
        overlays_dir=env_root / "model-overlays",
        secrets_path=env_root / "model-overlays.secrets.yaml",
    )


def load_model_overlay_secret_entries(
    *,
    start_path: Path | None = None,
    env_dir: str | Path | None = None,
) -> dict[str, ModelOverlaySecretEntry]:
    """Load companion overlay secret entries from the active environment."""

    paths = resolve_model_overlay_paths(start_path=start_path, env_dir=env_dir)
    return _load_secret_entries(paths.secrets_path)


def serialize_model_overlay_manifest(manifest: ModelOverlayManifest) -> str:
    """Serialize a model overlay manifest to YAML."""

    payload = manifest.model_dump(mode="json", exclude_none=True)
    return f"{yaml.safe_dump(payload, sort_keys=False).rstrip()}\n"


def write_model_overlay_manifest(
    manifest: ModelOverlayManifest,
    *,
    start_path: Path | None = None,
    env_dir: str | Path | None = None,
    replace: bool = False,
) -> Path:
    """Write a model overlay manifest into the active environment directory."""

    paths = resolve_model_overlay_paths(start_path=start_path, env_dir=env_dir)
    paths.overlays_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.overlays_dir / f"{_safe_overlay_filename(manifest.name)}.yaml"
    if output_path.exists() and not replace:
        raise FileExistsError(f"Model overlay already exists at {output_path}")

    output_path.write_text(
        serialize_model_overlay_manifest(manifest),
        encoding="utf-8",
    )
    return output_path


def _safe_overlay_filename(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        return "overlay"

    safe_chars: list[str] = []
    last_was_dash = False
    for char in normalized:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
            last_was_dash = False
            continue
        if last_was_dash:
            continue
        safe_chars.append("-")
        last_was_dash = True
    safe_name = "".join(safe_chars).strip("-.")
    return safe_name or "overlay"


def _settings_environment_override(
    *,
    start_path: Path | None,
) -> tuple[Path, str | Path | None] | None:
    settings = getattr(config_module, "_settings", None)
    if settings is None:
        return None

    environment_dir = getattr(settings, "environment_dir", None)
    if environment_dir is None:
        return None

    base_path = (start_path or Path.cwd()).resolve()
    config_file = getattr(settings, "_config_file", None)
    if start_path is None and isinstance(config_file, str) and config_file.strip():
        base_path = Path(config_file).expanduser().resolve().parent

    return base_path, environment_dir


def load_model_overlay_registry(
    *,
    start_path: Path | None = None,
    env_dir: str | Path | None = None,
) -> ModelOverlayRegistry:
    """Load static model overlays from the active environment directory."""

    paths = resolve_model_overlay_paths(start_path=start_path, env_dir=env_dir)
    secret_entries = _load_secret_entries(paths.secrets_path)

    loaded: dict[str, LoadedModelOverlay] = {}
    if paths.overlays_dir.exists():
        for path in sorted(paths.overlays_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            overlay = _load_overlay_file(path, secret_entries=secret_entries)
            if overlay is None:
                continue
            if overlay.name in loaded:
                logger.warning(
                    "Duplicate model overlay name detected; replacing earlier manifest",
                    overlay_name=overlay.name,
                    replaced_path=str(loaded[overlay.name].manifest_path),
                    path=str(path),
                )
            loaded[overlay.name] = overlay

    return ModelOverlayRegistry(
        overlays=tuple(loaded.values()),
        env_root=paths.env_root,
    )
