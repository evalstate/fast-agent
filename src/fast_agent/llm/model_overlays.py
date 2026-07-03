"""Local model overlay discovery and resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

import fast_agent.config as config_module
from fast_agent.config import load_yaml_mapping
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.home import resolve_fast_agent_home
from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.provider_types import Provider
from fast_agent.paths import resolve_settings_start_path
from fast_agent.utils.action_normalization import normalize_action_token, on_off_label
from fast_agent.utils.text import starts_with_casefold, strip_casefold, strip_to_none

logger = get_logger(__name__)

_SLASH_PROVIDER_PREFIXES: dict[str, Provider] = {
    provider.config_name: provider for provider in Provider if provider is not Provider.OPENAI
}
_SLASH_PROVIDER_PREFIXES["huggingface"] = Provider.HUGGINGFACE
_MODEL_OVERLAY_SUFFIXES = {".yaml", ".yml"}


def _normalize_reasoning_value(value: str | bool | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return on_off_label(value)
    if isinstance(value, int):
        return str(value)
    return strip_to_none(value)


def _normalize_toggle_value(value: bool | None) -> str | None:
    if value is None:
        return None
    return on_off_label(value)


def _reject_bool_numeric_overlay_value(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("Numeric model overlay values must not be boolean values.")
    return value


def _prefixed_model_spec(provider: Provider, model_name: str) -> str:
    if model_name.startswith((f"{provider.value}.", f"{provider.value}/")):
        return model_name
    return f"{provider.value}.{model_name}"


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
            raise ValueError(
                "connection.secret_ref is required when connection.auth is 'secret_ref'"
            )
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

    @field_validator(
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "max_tokens",
        mode="before",
    )
    @classmethod
    def _reject_bool_numeric_values(cls, value: object) -> object:
        return _reject_bool_numeric_overlay_value(value)

    def to_query_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        reasoning = _normalize_reasoning_value(self.reasoning)
        if reasoning is not None:
            pairs.append(("reasoning", reasoning))

        scalar_values: tuple[tuple[str, object | None], ...] = (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("min_p", self.min_p),
            ("presence_penalty", self.presence_penalty),
            ("repetition_penalty", self.repetition_penalty),
            ("transport", self.transport),
            ("service_tier", self.service_tier),
        )
        pairs.extend((name, str(value)) for name, value in scalar_values if value is not None)

        toggle_values = (
            ("web_search", _normalize_toggle_value(self.web_search)),
            ("web_fetch", _normalize_toggle_value(self.web_fetch)),
        )
        pairs.extend((name, value) for name, value in toggle_values if value is not None)
        return pairs


class ModelOverlayMetadata(BaseModel):
    """Runtime metadata attached to a local overlay."""

    model_config = ConfigDict(extra="ignore")

    context_window: int | None = None
    max_output_tokens: int | None = None
    tokenizes: list[str] | None = None
    json_mode: Literal["schema", "object"] | None = None
    structured_tool_policy: Literal["always", "defer", "no_tools"] | None = None
    model_specific: str | None = None
    # Legacy fallback retained for older overlay files. New overlays should use
    # defaults.temperature instead.
    default_temperature: float | None = None
    fast: bool | None = None

    @field_validator(
        "context_window",
        "max_output_tokens",
        "default_temperature",
        mode="before",
    )
    @classmethod
    def _reject_bool_numeric_values(cls, value: object) -> object:
        return _reject_bool_numeric_overlay_value(value)

    @field_validator("json_mode", mode="before")
    @classmethod
    def _normalize_json_mode(cls, value: object) -> object:
        if isinstance(value, str) and strip_casefold(value) == "none":
            return None
        return value


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
        return strip_to_none(self.manifest.picker.label) or self.name

    @property
    def description(self) -> str | None:
        return strip_to_none(self.manifest.picker.description)

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
        context_window, max_output_tokens = self._resolved_model_limits(existing)

        if context_window is None or max_output_tokens is None:
            return None

        if existing is not None:
            return existing.model_copy(
                update=self._existing_model_update_payload(
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                )
            )

        return ModelParameters(
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            tokenizes=self.manifest.metadata.tokenizes or list(ModelDatabase.TEXT_ONLY),
            json_mode=self._new_model_json_mode(),
            structured_tool_policy=self.manifest.metadata.structured_tool_policy,
            model_specific=self.manifest.metadata.model_specific,
            default_provider=self.provider,
            default_temperature=self._default_temperature(),
            fast=bool(self.manifest.metadata.fast),
        )

    def _resolved_model_limits(
        self,
        existing: ModelParameters | None,
    ) -> tuple[int | None, int | None]:
        context_window = self.manifest.metadata.context_window
        if context_window is None and existing is not None:
            context_window = existing.context_window

        max_output_tokens = self.manifest.metadata.max_output_tokens
        if max_output_tokens is None:
            max_output_tokens = self.manifest.defaults.max_tokens
        if max_output_tokens is None and existing is not None:
            max_output_tokens = existing.max_output_tokens

        return context_window, max_output_tokens

    def _default_temperature(self) -> float | None:
        if self.manifest.defaults.temperature is not None:
            return self.manifest.defaults.temperature
        return self.manifest.metadata.default_temperature

    def _existing_model_update_payload(
        self,
        *,
        context_window: int,
        max_output_tokens: int,
    ) -> dict[str, object]:
        metadata = self.manifest.metadata
        update_payload: dict[str, object] = {
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
            "default_provider": self.provider,
        }
        if metadata.tokenizes is not None:
            update_payload["tokenizes"] = metadata.tokenizes
        if "json_mode" in metadata.model_fields_set:
            update_payload["json_mode"] = metadata.json_mode
        if metadata.structured_tool_policy is not None:
            update_payload["structured_tool_policy"] = metadata.structured_tool_policy
        if metadata.model_specific is not None:
            update_payload["model_specific"] = metadata.model_specific
        if self._default_temperature() is not None:
            update_payload["default_temperature"] = self._default_temperature()
        if metadata.fast is not None:
            update_payload["fast"] = metadata.fast
        return update_payload

    def _new_model_json_mode(self) -> str | None:
        if "json_mode" in self.manifest.metadata.model_fields_set:
            return self.manifest.metadata.json_mode
        return "schema"


@dataclass(frozen=True, slots=True)
class ModelOverlayRegistry:
    """Resolved local overlay registry."""

    overlays: tuple[LoadedModelOverlay, ...]
    home_root: Path

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

    home_root: Path
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


def _is_model_overlay_file(path: Path) -> bool:
    return path.is_file() and strip_casefold(path.suffix) in _MODEL_OVERLAY_SUFFIXES


def resolve_model_overlay_paths(
    *,
    start_path: Path | None = None,
    home: str | Path | None = None,
) -> ModelOverlayPaths:
    """Resolve the active model overlay storage paths."""

    base_path = (start_path or Path.cwd()).resolve()
    override = home
    if override is None:
        configured = _settings_home_override(start_path=start_path)
        if configured is not None:
            base_path, override = configured

    resolved_home = resolve_fast_agent_home(cwd=base_path, cli_override=override)
    home_root = resolved_home.path if resolved_home is not None else base_path
    return ModelOverlayPaths(
        home_root=home_root,
        overlays_dir=home_root / "model-overlays",
        secrets_path=home_root / "model-overlays.secrets.yaml",
    )


def load_model_overlay_secret_entries(
    *,
    start_path: Path | None = None,
    home: str | Path | None = None,
) -> dict[str, ModelOverlaySecretEntry]:
    """Load companion overlay secret entries from the active fast-agent home."""

    paths = resolve_model_overlay_paths(start_path=start_path, home=home)
    return _load_secret_entries(paths.secrets_path)


def serialize_model_overlay_manifest(manifest: ModelOverlayManifest) -> str:
    """Serialize a model overlay manifest to YAML."""

    payload = manifest.model_dump(mode="json", exclude_none=True)
    return f"{yaml.safe_dump(payload, sort_keys=False).rstrip()}\n"


def _split_provider_prefix(model_name: str) -> tuple[Provider | None, str]:
    """Split an optional provider prefix from a model string.

    Accepts forms like "openrouter.gpt-4o", "anthropic/claude-4-sonnet-20250514",
    "hf.openai/gpt-oss-120b:cerebras", or bare namespaced HF IDs like "openai/gpt-oss-120b".

    Hugging Face namespaced models (those containing "/") are special: the namespace
    may collide with another provider name (e.g. "openai/...", "meta-llama/...").
    In those cases we must not strip the first segment as a provider prefix.
    """
    raw = model_name.strip()
    if not raw:
        return None, raw

    # Explicit dot-prefixed providers must win before slash handling so specs
    # like "openrouter.moonshotai/kimi-k2" do not look like bare HF repo IDs.
    if "." in raw:
        head, tail = raw.split(".", 1)
        provider = Provider.HUGGINGFACE if normalize_action_token(head) == "huggingface" else None
        if provider is None:
            try:
                provider = Provider(normalize_action_token(head))
            except ValueError:
                provider = None
        if provider is not None:
            return provider, tail

    # Slash-prefixed providers are opt-in because bare HF repos may start with
    # provider-looking namespaces such as "openai/...".
    if "/" in raw:
        head, tail = raw.split("/", 1)
        provider = _SLASH_PROVIDER_PREFIXES.get(normalize_action_token(head))
        if provider is not None:
            return provider, tail

    return None, raw


def build_model_overlay_manifest_from_database(
    model_name: str,
    *,
    provider: Provider | None = None,
    overlay_name: str | None = None,
    description: str | None = None,
) -> ModelOverlayManifest:
    """Create a ModelOverlayManifest seeded from a ModelDatabase entry.

    This allows users to export a known-good catalog model (with its model_specific
    prompt text, modalities, context window, etc.) into a local overlay that they
    can then customize.

    If model_name contains an explicit provider prefix (e.g. "openrouter.gpt-4o"),
    that prefix is used unless an explicit provider override is supplied.
    """
    prefix_provider, bare_model = _split_provider_prefix(model_name)

    lookup_name = model_name if prefix_provider is not None else bare_model
    if prefix_provider == Provider.HUGGINGFACE and starts_with_casefold(model_name, "huggingface."):
        lookup_name = f"{Provider.HUGGINGFACE.value}.{bare_model}"
    effective_provider = provider or prefix_provider
    if effective_provider is None and "/" in bare_model:
        effective_provider = Provider.HUGGINGFACE

    existing = None
    if effective_provider is not None:
        existing = _existing_model_params(effective_provider, lookup_name)
    if existing is None:
        existing = ModelDatabase.get_model_params(lookup_name)

    if existing is None:
        raise ModelConfigError(
            f"Model '{model_name}' was not found in the model database",
            "Check the model name or use a fully-qualified provider.model string.",
        )

    # Final provider decision: explicit arg > parsed prefix > catalog default > OPENAI
    resolved_provider = effective_provider or existing.default_provider or Provider.OPENAI

    # The manifest should store the bare model name (without our own prefix)
    manifest_model = bare_model
    name = overlay_name or _default_export_overlay_name(resolved_provider, manifest_model)

    # Map ModelParameters fields that are supported by ModelOverlayMetadata
    json_mode: Literal["schema", "object"] | None = None
    if existing.json_mode in ("schema", "object"):
        json_mode = "schema" if existing.json_mode == "schema" else "object"

    metadata = ModelOverlayMetadata(
        context_window=existing.context_window,
        max_output_tokens=existing.max_output_tokens,
        tokenizes=existing.tokenizes if existing.tokenizes else None,
        model_specific=existing.model_specific,
        json_mode=json_mode,
        structured_tool_policy=existing.structured_tool_policy,
        fast=existing.fast,
        default_temperature=existing.default_temperature,
    )

    return ModelOverlayManifest(
        name=name,
        provider=resolved_provider,
        model=manifest_model,
        metadata=metadata,
        picker=ModelOverlayPicker(
            label=name,
            description=description or "Exported from model database",
            current=True,
            featured=False,
        ),
    )


def write_model_overlay_manifest(
    manifest: ModelOverlayManifest,
    *,
    start_path: Path | None = None,
    home: str | Path | None = None,
    replace: bool = False,
) -> Path:
    """Write a model overlay manifest into the active home."""

    paths = resolve_model_overlay_paths(start_path=start_path, home=home)
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


def _default_export_overlay_name(provider: Provider, model: str) -> str:
    model_without_query = model.split("?", 1)[0]
    return _safe_overlay_filename(f"{provider.value}-{model_without_query}")


def _settings_home_override(
    *,
    start_path: Path | None,
) -> tuple[Path, str | Path | None] | None:
    settings = getattr(config_module, "_settings", None)
    if settings is None:
        return None

    raw_config_file = getattr(settings, "_config_file", None)
    config_file = (
        raw_config_file if isinstance(raw_config_file, str) and raw_config_file.strip() else None
    )
    home = getattr(settings, "home", None)
    if home is None and (
        os.getenv("FAST_AGENT_HOME") or os.getenv("FAST_AGENT_RUNTIME_HOME")
    ):
        return None
    if home is None and not (start_path is None and config_file is not None):
        return None

    base_path = (
        Path(start_path).resolve()
        if start_path is not None
        else resolve_settings_start_path(settings)
    )

    return base_path, home


def load_model_overlay_registry(
    *,
    start_path: Path | None = None,
    home: str | Path | None = None,
) -> ModelOverlayRegistry:
    """Load static model overlays from the active home."""

    paths = resolve_model_overlay_paths(start_path=start_path, home=home)
    secret_entries = _load_secret_entries(paths.secrets_path)

    loaded: dict[str, LoadedModelOverlay] = {}
    if paths.overlays_dir.exists():
        for path in sorted(paths.overlays_dir.iterdir()):
            if not _is_model_overlay_file(path):
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
        home_root=paths.home_root,
    )
