"""Unified endpoint discovery for `--model-url` (vLLM, llama.cpp, generic OpenAI-compatible)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from fast_agent.cli.commands.url_parser import normalize_auth_token
from fast_agent.llm.llamacpp_discovery import (
    DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    LlamaCppChatTemplateCaps,
    LlamaCppDiscoveryError,
    LlamaCppServerEndpoints,
    match_llamacpp_model_database_entry,
    normalize_llamacpp_url,
    parse_llamacpp_props_details,
    resolve_live_reasoning_effort_spec,
)
from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.model_overlays import (
    LoadedModelOverlay,
    ModelOverlayConnection,
    ModelOverlayDefaults,
    ModelOverlayManifest,
    ModelOverlayMetadata,
    ModelOverlayPicker,
)
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.utils.numeric import positive_int_or_none
from fast_agent.utils.text import strip_casefold, strip_to_none

_HF_ENDPOINT_HOST_SUFFIX: Final[str] = ".endpoints.huggingface.cloud"
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1"})
_LLAMACPP_PROPS_MARKER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "default_generation_settings",
        "modalities",
        "model_alias",
        "chat_template_caps",
        "total_slots",
        "build_info",
    }
)

type EndpointServerKind = Literal["vllm", "llamacpp", "generic"]
type EndpointAuthSource = Literal[
    "model_auth",
    "api_key_env",
    "hf_token_env",
    "hf_secrets_file",
    "generic_api_key_env",
    "none",
]


class ModelUrlDiscoveryError(RuntimeError):
    """Raised when `--model-url` endpoint discovery or authentication fails."""


@dataclass(frozen=True, slots=True)
class ModelUrlAuth:
    """Resolved provider and credential source for a `--model-url` endpoint."""

    provider: Provider
    source: EndpointAuthSource
    token: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    is_hf_endpoint: bool = False

    def headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        _validate_header_token(self.token, source_label=self.source)
        return {"Authorization": f"Bearer {self.token}"}

    def overlay_connection_auth(
        self,
    ) -> tuple[Literal["none", "env"] | None, str | None, str | None]:
        """Return `(connection.auth, connection.api_key_env, override_api_key)`."""
        if self.source in {"model_auth", "hf_secrets_file"}:
            return None, None, self.token
        if self.source in {"api_key_env", "hf_token_env", "generic_api_key_env"}:
            return "env", self.api_key_env, None
        return "none", None, None


@dataclass(frozen=True, slots=True)
class ModelUrlModelListing:
    """A model entry discovered from an OpenAI-compatible `/v1/models` endpoint."""

    model_id: str
    owned_by: str | None = None
    root: str | None = None
    max_model_len: int | None = None
    training_context_window: int | None = None
    aliases: tuple[str, ...] = ()
    status: str | None = None


@dataclass(frozen=True, slots=True)
class ModelUrlDiscoveryCatalog:
    """Discovered endpoint catalog and detected backend kind."""

    endpoints: LlamaCppServerEndpoints
    server_kind: EndpointServerKind
    server_version: str | None
    models_url: str
    models: tuple[ModelUrlModelListing, ...]
    auth: ModelUrlAuth
    initial_props_payload: object | None = None
    initial_props_url: str | None = None


@dataclass(frozen=True, slots=True)
class ModelUrlDiscoveredModel:
    """Resolved model metadata merged with live endpoint discovery truth."""

    listing: ModelUrlModelListing
    server_kind: EndpointServerKind
    server_version: str | None = None
    runtime_context_window: int | None = None
    max_output_tokens: int | None = None
    tokenizes: tuple[str, ...] | None = None
    model_alias: str | None = None
    chat_template_caps: LlamaCppChatTemplateCaps | None = None
    matched_database_model: str | None = None

    @property
    def effective_context_window(self) -> int | None:
        return (
            self.runtime_context_window
            or self.listing.max_model_len
            or self.listing.training_context_window
        )


class _EndpointModelMetaPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_ctx_train: object = None


class _EndpointModelStatusPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: str | None = None


class _EndpointModelPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    owned_by: str | None = None
    root: str | None = None
    max_model_len: object = None
    aliases: list[str] | None = None
    meta: _EndpointModelMetaPayload | None = None
    status: _EndpointModelStatusPayload | str | None = None


def _normalized_hostname(url: str) -> str | None:
    raw = url.strip()
    if not raw:
        return None
    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    hostname = parsed.hostname
    if not hostname:
        return None
    return hostname.rstrip(".").lower()


def _validate_header_token(token: str, *, source_label: str) -> str:
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in token):
        raise ModelUrlDiscoveryError(
            f"Invalid credential from {source_label}: header values cannot contain control characters."
        )
    return token


def is_huggingface_endpoint_url(url: str) -> bool:
    """Return True when `url` targets a Hugging Face Inference Endpoint host."""
    hostname = _normalized_hostname(url)
    if not hostname:
        return False
    return hostname.endswith(_HF_ENDPOINT_HOST_SUFFIX)


def normalize_model_url(url: str) -> LlamaCppServerEndpoints:
    """Normalize a bare origin or `.../v1` endpoint URL."""
    raw = url.strip()
    if not raw:
        raise ModelUrlDiscoveryError("Model URL cannot be empty.")

    if "://" not in raw:
        default_scheme = "https" if is_huggingface_endpoint_url(raw) else "http"
        raw = f"{default_scheme}://{raw}"

    try:
        parsed = urlsplit(raw)
        _ = parsed.port
    except ValueError:
        raise ModelUrlDiscoveryError(
            "Invalid model URL. Check the scheme, host, and port (and do not embed credentials)."
        ) from None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ModelUrlDiscoveryError(
            f"Unsupported model URL scheme {parsed.scheme!r}; expected http or https."
        )

    if parsed.username is not None or parsed.password is not None:
        raise ModelUrlDiscoveryError(
            "Model URL must not contain embedded credentials (userinfo). "
            "Use --model-api-key-env or --model-auth instead."
        )

    if not parsed.hostname:
        raise ModelUrlDiscoveryError("Model URL must include a valid host.")

    try:
        endpoints = normalize_llamacpp_url(raw)
        httpx.URL(endpoints.request_base_url)
        httpx.URL(endpoints.server_url)
    except (LlamaCppDiscoveryError, ValueError, httpx.InvalidURL):
        raise ModelUrlDiscoveryError(
            "Invalid model URL. Check the scheme, host, and port."
        ) from None
    return endpoints


def resolve_model_url_auth(
    url: str,
    *,
    model_auth: str | None = None,
    api_key_env: str | None = None,
    config: object | None = None,
) -> ModelUrlAuth:
    """Resolve provider and fail-closed credentials for `--model-url`.

    Precedence:
    1. Explicit `--model-auth <token>` (in-memory only, optional `Bearer ` prefix).
    2. Explicit `--model-api-key-env <VAR>` (fails closed if unset/empty).
    3. For `*.endpoints.huggingface.cloud`: `Provider.HUGGINGFACE` (requires `https://`),
       using `HF_TOKEN` then `hf.api_key` in config/secrets (never `~/.cache/huggingface/token`).
    4. For non-HF hosts: `Provider.GENERIC`, using `GENERIC_API_KEY` only for `https://`
       or loopback (`localhost` / `127.0.0.1` / `::1`) URLs, else unauthenticated.
    """
    endpoints = normalize_model_url(url)
    parsed = urlsplit(endpoints.request_base_url)
    scheme = parsed.scheme.lower()
    hostname = _normalized_hostname(endpoints.request_base_url)
    is_hf = bool(hostname and hostname.endswith(_HF_ENDPOINT_HOST_SUFFIX))
    is_loopback = bool(hostname and hostname in _LOOPBACK_HOSTS)
    provider = Provider.HUGGINGFACE if is_hf else Provider.GENERIC

    if is_hf and scheme != "https":
        raise ModelUrlDiscoveryError(
            "Hugging Face Inference Endpoints require https://; refusing cleartext http:// URL."
        )

    if model_auth is not None:
        try:
            normalized_token = normalize_auth_token(model_auth)
        except ValueError as exc:
            raise ModelUrlDiscoveryError(f"Invalid --model-auth value: {exc}") from exc
        _validate_header_token(normalized_token, source_label="--model-auth")
        return ModelUrlAuth(
            provider=provider,
            source="model_auth",
            token=normalized_token,
            is_hf_endpoint=is_hf,
        )

    if api_key_env is not None:
        env_name = strip_to_none(api_key_env)
        if env_name is None:
            raise ModelUrlDiscoveryError("--model-api-key-env cannot be empty.")
        env_value = os.getenv(env_name)
        if not env_value:
            raise ModelUrlDiscoveryError(
                f"Environment variable '{env_name}' is not set or empty for --model-url. "
                f"Export {env_name} or pass --model-auth."
            )
        _validate_header_token(env_value, source_label=f"environment variable '{env_name}'")
        return ModelUrlAuth(
            provider=provider,
            source="api_key_env",
            token=env_value,
            api_key_env=env_name,
            is_hf_endpoint=is_hf,
        )

    if is_hf:
        hf_env_token = ProviderKeyManager.get_env_var("hf")
        if hf_env_token:
            _validate_header_token(hf_env_token, source_label="HF_TOKEN")
            return ModelUrlAuth(
                provider=Provider.HUGGINGFACE,
                source="hf_token_env",
                token=hf_env_token,
                api_key_env="HF_TOKEN",
                is_hf_endpoint=True,
            )
        if config is not None:
            hf_config_token = ProviderKeyManager.get_config_file_key("hf", config)
            if hf_config_token:
                _validate_header_token(hf_config_token, source_label="hf.api_key")
                return ModelUrlAuth(
                    provider=Provider.HUGGINGFACE,
                    source="hf_secrets_file",
                    token=hf_config_token,
                    is_hf_endpoint=True,
                )
        return ModelUrlAuth(
            provider=Provider.HUGGINGFACE,
            source="none",
            token=None,
            is_hf_endpoint=True,
        )

    if scheme == "https" or is_loopback:
        generic_env_token = ProviderKeyManager.get_env_var("generic")
        if generic_env_token:
            _validate_header_token(generic_env_token, source_label="GENERIC_API_KEY")
            return ModelUrlAuth(
                provider=Provider.GENERIC,
                source="generic_api_key_env",
                token=generic_env_token,
                api_key_env="GENERIC_API_KEY",
                is_hf_endpoint=False,
            )
    return ModelUrlAuth(
        provider=Provider.GENERIC,
        source="none",
        token=None,
        is_hf_endpoint=False,
    )


async def discover_model_url_catalog(
    url: str,
    *,
    model_auth: str | None = None,
    api_key_env: str | None = None,
    config: object | None = None,
    timeout_seconds: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelUrlDiscoveryCatalog:
    """Probe `<base_url>/models` and auto-detect vLLM, llama.cpp, or generic endpoints."""
    endpoints = normalize_model_url(url)
    resolved_auth = resolve_model_url_auth(
        endpoints.request_base_url,
        model_auth=model_auth,
        api_key_env=api_key_env,
        config=config,
    )
    headers = resolved_auth.headers()
    models_url = endpoints.models_urls()[0]

    async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport) as client:
        try:
            response = await client.get(models_url, headers=headers)
        except httpx.HTTPError as exc:
            raise ModelUrlDiscoveryError(
                f"Unable to reach model endpoint {models_url} ({type(exc).__name__})."
            ) from None

        if response.status_code in {401, 403}:
            raise ModelUrlDiscoveryError(
                _format_unauthorized_error(
                    models_url,
                    auth=resolved_auth,
                    status_code=response.status_code,
                )
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelUrlDiscoveryError(
                f"Model endpoint {models_url} returned HTTP {response.status_code}."
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelUrlDiscoveryError(
                f"Model endpoint {models_url} returned invalid JSON: {exc}"
            ) from exc

        models = _parse_models_payload(payload)
        if not models:
            raise ModelUrlDiscoveryError(f"Model endpoint {models_url} returned no models.")

        server_kind: EndpointServerKind
        server_version: str | None = None
        initial_props_payload: object | None = None
        initial_props_url: str | None = None

        if _has_llamacpp_owner(models):
            props_probe = await _probe_llamacpp_props(
                client=client,
                endpoints=endpoints,
                headers=headers,
                models=models,
            )
            if props_probe is not None:
                server_kind = "llamacpp"
                initial_props_payload, initial_props_url = props_probe
            else:
                server_kind = "llamacpp"
        elif _looks_like_vllm(models):
            server_kind = "vllm"
            server_version = await _probe_vllm_version(
                client=client,
                server_url=endpoints.server_url,
                headers=headers,
            )
        else:
            props_probe = await _probe_llamacpp_props(
                client=client,
                endpoints=endpoints,
                headers=headers,
                models=models,
            )
            if props_probe is not None:
                server_kind = "llamacpp"
                initial_props_payload, initial_props_url = props_probe
            else:
                server_kind = "generic"

    return ModelUrlDiscoveryCatalog(
        endpoints=endpoints,
        server_kind=server_kind,
        server_version=server_version,
        models_url=models_url,
        models=models,
        auth=resolved_auth,
        initial_props_payload=initial_props_payload,
        initial_props_url=initial_props_url,
    )


async def interrogate_model_url_model(
    *,
    catalog: ModelUrlDiscoveryCatalog,
    model_id: str,
    timeout_seconds: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelUrlDiscoveredModel:
    """Resolve runtime limits, modalities, and ModelDatabase match for a discovered model."""
    listing = next((item for item in catalog.models if item.model_id == model_id), None)
    if listing is None:
        raise ModelUrlDiscoveryError(
            f"Model {model_id!r} was not present in the discovered catalog for "
            f"{catalog.endpoints.request_base_url}."
        )

    runtime_context_window: int | None = listing.max_model_len
    max_output_tokens: int | None = None
    tokenizes: tuple[str, ...] | None = None
    model_alias: str | None = None
    chat_template_caps: LlamaCppChatTemplateCaps | None = None

    if catalog.server_kind == "llamacpp":
        props_payload = catalog.initial_props_payload if len(catalog.models) == 1 else None
        if props_payload is None:
            props_payload = await _fetch_llamacpp_props_for_model(
                catalog=catalog,
                model_id=model_id,
                timeout_seconds=timeout_seconds,
                transport=transport,
            )
        if props_payload is not None:
            try:
                details = parse_llamacpp_props_details(props_payload)
            except LlamaCppDiscoveryError as exc:
                raise ModelUrlDiscoveryError(str(exc)) from exc
            if details.runtime_context_window is not None:
                runtime_context_window = details.runtime_context_window
            max_output_tokens = details.max_output_tokens
            tokenizes = details.tokenizes
            model_alias = details.model_alias
            chat_template_caps = details.chat_template_caps

    matched = match_llamacpp_model_database_entry(
        listing.model_id,
        root=listing.root,
        model_alias=model_alias,
        aliases=listing.aliases,
    )

    return ModelUrlDiscoveredModel(
        listing=listing,
        server_kind=catalog.server_kind,
        server_version=catalog.server_version,
        runtime_context_window=runtime_context_window,
        max_output_tokens=max_output_tokens,
        tokenizes=tokenizes,
        model_alias=model_alias,
        chat_template_caps=chat_template_caps,
        matched_database_model=matched[0] if matched is not None else None,
    )


def build_model_url_runtime_model_parameters(
    discovered: ModelUrlDiscoveredModel,
    *,
    provider: Provider,
) -> ModelParameters:
    """Merge matched ModelDatabase entry with live endpoint limits."""
    base_params = (
        ModelDatabase.MODELS.get(discovered.matched_database_model)
        if discovered.matched_database_model is not None
        else None
    )

    context_window = (
        discovered.effective_context_window
        or (base_params.context_window if base_params is not None else None)
        or 32768
    )
    if discovered.max_output_tokens is not None:
        max_output_tokens = min(discovered.max_output_tokens, context_window)
    elif base_params is not None:
        max_output_tokens = min(base_params.max_output_tokens, context_window)
    else:
        max_output_tokens = min(context_window, 16384)

    if discovered.tokenizes is not None:
        tokenizes = list(discovered.tokenizes)
    elif base_params is not None:
        tokenizes = list(base_params.tokenizes)
    else:
        tokenizes = list(ModelDatabase.TEXT_ONLY)

    live_reasoning_spec = resolve_live_reasoning_effort_spec(
        base_spec=base_params.reasoning_effort_spec if base_params is not None else None,
        caps=discovered.chat_template_caps,
    )

    if base_params is not None:
        return base_params.model_copy(
            update={
                "context_window": context_window,
                "max_output_tokens": max_output_tokens,
                "tokenizes": tokenizes,
                "reasoning_effort_spec": live_reasoning_spec,
                "default_provider": provider,
            }
        )

    reasoning_mode: Literal["reasoning_content"] | None = (
        "reasoning_content"
        if (
            discovered.chat_template_caps is not None
            and (
                discovered.chat_template_caps.supports_reasoning_effort
                or discovered.chat_template_caps.supports_preserve_reasoning
            )
        )
        else None
    )
    return ModelParameters(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        tokenizes=tokenizes,
        json_mode="schema",
        reasoning=reasoning_mode,
        reasoning_effort_spec=live_reasoning_spec,
        default_provider=provider,
    )


def default_model_url_overlay_name(
    model_id: str,
    *,
    server_kind: EndpointServerKind,
) -> str:
    """Return a readable ephemeral overlay name for a `--model-url` model."""
    leaf = model_id.rsplit("/", 1)[-1].strip() or model_id.strip()
    slug_chars: list[str] = []
    last_was_dash = False
    for char in strip_casefold(leaf):
        if char.isalnum():
            slug_chars.append(char)
            last_was_dash = False
            continue
        if last_was_dash:
            continue
        slug_chars.append("-")
        last_was_dash = True
    slug = "".join(slug_chars).strip("-") or "model"
    return f"{server_kind}-{slug}"


def build_ephemeral_model_url_overlay(
    *,
    discovered: ModelUrlDiscoveredModel,
    catalog: ModelUrlDiscoveryCatalog,
    overlay_name: str | None = None,
) -> LoadedModelOverlay:
    """Build an in-memory runnable overlay for `fast-agent --model-url`."""
    resolved_name = overlay_name or default_model_url_overlay_name(
        discovered.listing.model_id,
        server_kind=catalog.server_kind,
    )
    runtime_params = build_model_url_runtime_model_parameters(
        discovered,
        provider=catalog.auth.provider,
    )
    conn_auth, conn_api_key_env, override_api_key = catalog.auth.overlay_connection_auth()
    label = (
        discovered.model_alias
        or discovered.listing.model_id.rsplit("/", 1)[-1]
        or discovered.listing.model_id
    )
    clamped_live_max_output = (
        min(discovered.max_output_tokens, runtime_params.context_window)
        if discovered.max_output_tokens is not None
        else None
    )
    manifest = ModelOverlayManifest(
        name=resolved_name,
        provider=catalog.auth.provider,
        model=discovered.listing.model_id,
        profile=discovered.matched_database_model,
        connection=ModelOverlayConnection(
            base_url=catalog.endpoints.request_base_url,
            auth=conn_auth,
            api_key_env=conn_api_key_env,
        ),
        defaults=ModelOverlayDefaults(max_tokens=clamped_live_max_output),
        metadata=ModelOverlayMetadata(
            context_window=runtime_params.context_window,
            max_output_tokens=clamped_live_max_output,
            tokenizes=list(runtime_params.tokenizes),
        ),
        picker=ModelOverlayPicker(
            label=label,
            description=f"Connected via --model-url ({catalog.server_kind})",
            current=True,
        ),
    )
    return LoadedModelOverlay(
        manifest=manifest,
        manifest_path=Path("<model-url>"),
        secret_entry=None,
        override_model_parameters=runtime_params,
        override_api_key=override_api_key,
        preserve_wire_model_name=True,
        isolate_origin_headers=True,
    )


def _format_unauthorized_error(
    url: str,
    *,
    auth: ModelUrlAuth,
    status_code: int,
) -> str:
    env_hint = (
        f" Check that {auth.api_key_env} holds a valid token, or pass --model-auth."
        if auth.api_key_env is not None
        else ""
    )
    if auth.is_hf_endpoint:
        return (
            f"Authentication failed (HTTP {status_code}) querying {url}. "
            "Set HF_TOKEN, configure hf.api_key in fastagent.secrets.yaml, "
            f"pass --model-auth <token>, or pass --model-api-key-env <ENV_VAR>.{env_hint}"
        )
    return (
        f"Authentication failed (HTTP {status_code}) querying {url}. "
        "Set GENERIC_API_KEY, pass --model-auth <token>, "
        f"or pass --model-api-key-env <ENV_VAR>.{env_hint}"
    )


def _parse_models_payload(payload: object) -> tuple[ModelUrlModelListing, ...]:
    raw_items: object
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("data")
        if not isinstance(raw_items, list):
            raw_items = payload.get("models")
    else:
        raw_items = None

    if not isinstance(raw_items, list):
        raise ModelUrlDiscoveryError(
            "Model endpoint returned an unexpected /models payload (expected a 'data' list)."
        )

    models: list[ModelUrlModelListing] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        try:
            parsed = _EndpointModelPayload.model_validate(raw_item)
        except ValidationError:
            continue
        model_id = strip_to_none(parsed.id)
        if model_id is None:
            continue
        aliases = tuple(
            alias
            for raw_alias in (parsed.aliases or [])
            if (alias := strip_to_none(raw_alias)) is not None
        )
        raw_status = (
            parsed.status
            if isinstance(parsed.status, str)
            else (parsed.status.value if parsed.status is not None else None)
        )
        models.append(
            ModelUrlModelListing(
                model_id=model_id,
                owned_by=strip_to_none(parsed.owned_by),
                root=strip_to_none(parsed.root),
                max_model_len=positive_int_or_none(parsed.max_model_len),
                training_context_window=positive_int_or_none(
                    parsed.meta.n_ctx_train if parsed.meta is not None else None
                ),
                aliases=aliases,
                status=strip_to_none(raw_status),
            )
        )
    return tuple(models)


def _has_llamacpp_owner(models: tuple[ModelUrlModelListing, ...]) -> bool:
    return any(
        item.owned_by is not None and strip_casefold(item.owned_by) == "llamacpp" for item in models
    )


def _looks_like_vllm(models: tuple[ModelUrlModelListing, ...]) -> bool:
    for item in models:
        if item.owned_by is not None and strip_casefold(item.owned_by) == "vllm":
            return True
        if item.max_model_len is not None:
            return True
    return False


async def _probe_vllm_version(
    *,
    client: httpx.AsyncClient,
    server_url: str,
    headers: dict[str, str],
) -> str | None:
    version_url = f"{server_url.rstrip('/')}/version"
    try:
        response = await client.get(version_url, headers=headers)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_version = payload.get("version")
    return strip_to_none(raw_version) if isinstance(raw_version, str) else None


async def _probe_llamacpp_props(
    *,
    client: httpx.AsyncClient,
    endpoints: LlamaCppServerEndpoints,
    headers: dict[str, str],
    models: tuple[ModelUrlModelListing, ...],
) -> tuple[object, str] | None:
    props_url = endpoints.props_urls()[0]
    try:
        response = await client.get(props_url, headers=headers)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    has_llamacpp_owner = any(
        item.owned_by is not None and strip_casefold(item.owned_by) == "llamacpp" for item in models
    )
    if has_llamacpp_owner or any(key in payload for key in _LLAMACPP_PROPS_MARKER_KEYS):
        return payload, props_url
    return None


async def _fetch_llamacpp_props_for_model(
    *,
    catalog: ModelUrlDiscoveryCatalog,
    model_id: str,
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> object | None:
    headers = catalog.auth.headers()
    is_multi_model = len(catalog.models) > 1
    if is_multi_model:
        props_urls: tuple[str, ...] = (catalog.endpoints.props_urls(model_id=model_id)[0],)
    else:
        props_urls = catalog.endpoints.props_urls(model_id=model_id)

    async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport) as client:
        for props_url in props_urls:
            try:
                response = await client.get(props_url, headers=headers)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                payload = response.json()
            except ValueError:
                continue
            if isinstance(payload, dict):
                return payload
    if is_multi_model:
        return None
    return catalog.initial_props_payload
