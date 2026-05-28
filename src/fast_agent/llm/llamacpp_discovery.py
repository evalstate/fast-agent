"""llama.cpp discovery helpers for overlay import flows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Literal
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fast_agent.llm.model_overlays import (
    ModelOverlayConnection,
    ModelOverlayDefaults,
    ModelOverlayManifest,
    ModelOverlayMetadata,
    ModelOverlayPicker,
)
from fast_agent.llm.provider_types import Provider

DEFAULT_LLAMA_CPP_URL: Final[str] = "http://localhost:8080/v1"
DEFAULT_DISCOVERY_TIMEOUT_SECONDS: Final[float] = 10.0
_TEXT_TOKENIZES: Final[tuple[str, ...]] = ("text/plain",)
_VISION_TOKENIZES: Final[tuple[str, ...]] = (
    "image/jpeg",
    "image/png",
    "image/webp",
)
type LlamaCppAuthMode = Literal["none", "env", "secret_ref"]


class LlamaCppDiscoveryError(RuntimeError):
    """Raised when llama.cpp discovery fails."""


@dataclass(frozen=True, slots=True)
class LlamaCppServerEndpoints:
    """Normalized URLs for llama.cpp discovery and runtime use."""

    requested_url: str
    server_url: str
    request_base_url: str

    def models_urls(self) -> tuple[str, ...]:
        """Return candidate model-listing URLs in preferred order."""
        return _dedupe_urls(
            (
                _join_url(self.request_base_url, "/models"),
            )
        )

    def props_urls(self, *, model_id: str | None = None) -> tuple[str, ...]:
        """Return candidate props URLs in preferred order."""
        query_suffix = ""
        if model_id is not None and model_id.strip():
            query_suffix = f"?model={quote(model_id.strip(), safe='')}"
        return _dedupe_urls(
            (
                f"{_join_url(self.server_url, '/props')}{query_suffix}",
                _join_url(self.server_url, "/props"),
            )
        )

    def slots_urls(self) -> tuple[str, ...]:
        """Return candidate slots URLs in preferred order."""
        return (_join_url(self.server_url, "/slots"),)


@dataclass(frozen=True, slots=True)
class LlamaCppModelListing:
    """A model entry discovered from the llama.cpp model list endpoint."""

    model_id: str
    owned_by: str | None
    training_context_window: int | None
    child_server_url: str | None = None


@dataclass(frozen=True, slots=True)
class LlamaCppDiscoveryCatalog:
    """The discovered llama.cpp catalog."""

    endpoints: LlamaCppServerEndpoints
    models: tuple[LlamaCppModelListing, ...]
    models_url: str
    router_mode: bool = False


@dataclass(frozen=True, slots=True)
class LlamaCppDiscoveredModel:
    """Normalized runtime metadata for a discovered llama.cpp model."""

    listing: LlamaCppModelListing
    props_url: str
    runtime_context_window: int | None
    max_output_tokens: int | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    min_p: float | None
    tokenizes: tuple[str, ...]
    model_alias: str | None

    @property
    def effective_context_window(self) -> int | None:
        return self.runtime_context_window or self.listing.training_context_window

    @property
    def context_window_source(self) -> Literal["runtime", "catalog"] | None:
        if self.runtime_context_window is not None:
            return "runtime"
        if self.listing.training_context_window is not None:
            return "catalog"
        return None


class _LlamaCppModelMetaPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_ctx_train: int | None = None


class _LlamaCppModelPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    owned_by: str | None = None
    meta: _LlamaCppModelMetaPayload | None = None


class _LlamaCppModelStatusPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: str | None = None
    args: list[str] = Field(default_factory=list)


class _LlamaCppGenerationParamsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    n_predict: int | None = None


class _LlamaCppGenerationSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_ctx: int | None = None
    params: _LlamaCppGenerationParamsPayload | None = None


class _LlamaCppModalitiesPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    vision: bool = False
    audio: bool = False


class _LlamaCppPropsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default_generation_settings: _LlamaCppGenerationSettingsPayload = Field(
        default_factory=_LlamaCppGenerationSettingsPayload
    )
    model_alias: str | None = None
    modalities: _LlamaCppModalitiesPayload = Field(default_factory=_LlamaCppModalitiesPayload)


class _LlamaCppSlotParamsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_tokens: int | None = None
    n_predict: int | None = None


class _LlamaCppSlotPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_processing: bool = False
    params: _LlamaCppSlotParamsPayload | None = None


def normalize_llamacpp_url(url: str) -> LlamaCppServerEndpoints:
    """Normalize a user-supplied llama.cpp URL.

    The returned `request_base_url` is suitable for persisted overlay runtime use.
    The returned `server_url` is suitable for llama.cpp-specific discovery endpoints.
    """

    raw_url = url.strip() or DEFAULT_LLAMA_CPP_URL
    if "://" not in raw_url:
        raw_url = f"http://{raw_url}"

    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid llama.cpp URL: {url!r}")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        server_path = path[: -len("/v1")]
        request_path = path
    else:
        server_path = path
        request_path = f"{path}/v1" if path else "/v1"

    server_url = _compose_base_url(parsed.scheme, parsed.netloc, server_path)
    request_base_url = _compose_base_url(parsed.scheme, parsed.netloc, request_path)
    return LlamaCppServerEndpoints(
        requested_url=raw_url,
        server_url=server_url,
        request_base_url=request_base_url,
    )


async def discover_llamacpp_models(
    *,
    url: str = DEFAULT_LLAMA_CPP_URL,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
) -> LlamaCppDiscoveryCatalog:
    """Discover models from a llama.cpp-compatible server."""

    endpoints = normalize_llamacpp_url(url)
    headers = _discovery_headers(api_key)
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        payload, resolved_url = await _get_json_from_candidates(
            client,
            endpoints.models_urls(),
            headers=headers,
            endpoint_name="llama.cpp model discovery",
        )
        models, router_mode = _parse_models_payload(payload)
        if router_mode:
            models = await _enrich_router_models_with_child_metadata(
                client=client,
                headers=headers,
                models=models,
            )
    if not models:
        raise LlamaCppDiscoveryError(
            f"No models were returned by llama.cpp discovery at {resolved_url}."
        )

    return LlamaCppDiscoveryCatalog(
        endpoints=endpoints,
        models=models,
        models_url=resolved_url,
        router_mode=router_mode,
    )


async def interrogate_llamacpp_model(
    *,
    catalog: LlamaCppDiscoveryCatalog,
    model_id: str,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
) -> LlamaCppDiscoveredModel:
    """Load runtime defaults and metadata for a selected llama.cpp model."""

    listing = next((item for item in catalog.models if item.model_id == model_id), None)
    if listing is None:
        raise LlamaCppDiscoveryError(
            f"Model {model_id!r} was not present in the discovered llama.cpp catalog."
        )

    headers = _discovery_headers(api_key)
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        payload, resolved_url = await _get_json_from_candidates(
            client,
            catalog.endpoints.props_urls(model_id=model_id),
            headers=headers,
            endpoint_name="llama.cpp props discovery",
        )
        props = _parse_props_payload(payload)
        generation = props.default_generation_settings
        params = generation.params or _LlamaCppGenerationParamsPayload()
        max_output_tokens = _positive_int_or_none(params.n_predict)
        if max_output_tokens is None:
            max_output_tokens = _positive_int_or_none(params.max_tokens)
        if max_output_tokens is None:
            max_output_tokens = await _discover_slots_max_output_tokens(
                client=client,
                catalog=catalog,
                headers=headers,
            )
        if catalog.router_mode and listing.training_context_window is None:
            listing = await _refresh_router_model_listing(
                client=client,
                catalog=catalog,
                headers=headers,
                listing=listing,
            )

    return LlamaCppDiscoveredModel(
        listing=listing,
        props_url=resolved_url,
        runtime_context_window=_positive_int_or_none(generation.n_ctx),
        max_output_tokens=max_output_tokens,
        temperature=_normalize_llamacpp_float(params.temperature),
        top_k=_positive_int_or_none(params.top_k),
        top_p=_normalize_llamacpp_float(params.top_p),
        min_p=_normalize_llamacpp_float(params.min_p),
        tokenizes=_derive_tokenizes(props.modalities),
        model_alias=_normalize_text(props.model_alias),
    )


def build_llamacpp_overlay_manifest(
    *,
    overlay_name: str,
    discovered_model: LlamaCppDiscoveredModel,
    base_url: str,
    auth: LlamaCppAuthMode | None,
    api_key_env: str | None,
    secret_ref: str | None,
    current: bool,
    include_sampling_defaults: bool = False,
) -> ModelOverlayManifest:
    """Build a runnable model overlay manifest from llama.cpp discovery data."""

    return ModelOverlayManifest(
        name=overlay_name,
        provider=Provider.OPENRESPONSES,
        model=discovered_model.listing.model_id,
        connection=ModelOverlayConnection(
            base_url=base_url,
            auth=auth,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        ),
        defaults=ModelOverlayDefaults(
            temperature=(
                _normalize_llamacpp_float(discovered_model.temperature)
                if include_sampling_defaults
                else None
            ),
            top_k=discovered_model.top_k if include_sampling_defaults else None,
            top_p=(
                _normalize_llamacpp_float(discovered_model.top_p)
                if include_sampling_defaults
                else None
            ),
            min_p=(
                _normalize_llamacpp_float(discovered_model.min_p)
                if include_sampling_defaults
                else None
            ),
            max_tokens=discovered_model.max_output_tokens,
        ),
        metadata=ModelOverlayMetadata(
            context_window=discovered_model.effective_context_window,
            max_output_tokens=discovered_model.max_output_tokens,
            tokenizes=list(discovered_model.tokenizes),
        ),
        picker=ModelOverlayPicker(
            label=_picker_label(discovered_model),
            description="Imported from llama.cpp",
            current=current,
        ),
    )


def default_overlay_name_for_model(model_id: str) -> str:
    """Return a readable default overlay name for a discovered model."""

    leaf = model_id.rsplit("/", 1)[-1].strip() or model_id.strip()
    slug = _slugify_name(leaf)
    return f"llamacpp-{slug or 'model'}"


def uniquify_overlay_name(name: str, *, existing_names: set[str]) -> str:
    """Return a unique overlay name, appending a numeric suffix when needed."""

    candidate = name.strip()
    if not candidate:
        raise ValueError("Overlay name cannot be empty.")
    if candidate not in existing_names:
        return candidate

    index = 2
    while True:
        numbered = f"{candidate}-{index}"
        if numbered not in existing_names:
            return numbered
        index += 1


def _compose_base_url(scheme: str, netloc: str, path: str) -> str:
    normalized_path = path or ""
    return urlunsplit((scheme, netloc, normalized_path, "", "")).rstrip("/")


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _dedupe_urls(urls: tuple[str, ...]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return tuple(deduped)


def _discovery_headers(api_key: str | None) -> dict[str, str]:
    if api_key is None:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _normalize_llamacpp_float(value: float | None) -> float | None:
    if value is None:
        return None
    normalized = round(value, 6)
    if normalized == 0:
        return 0.0
    return normalized


async def _get_json_from_candidates(
    client: httpx.AsyncClient,
    urls: tuple[str, ...],
    *,
    headers: dict[str, str],
    endpoint_name: str,
) -> tuple[object, str]:
    errors: list[str] = []
    for url in urls:
        try:
            response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            errors.append(f"{url}: {exc}")
            continue

        if response.status_code in {401, 403}:
            raise LlamaCppDiscoveryError(
                f"{endpoint_name} at {url} rejected the supplied credentials."
            )
        if response.status_code == 404:
            errors.append(f"{url}: HTTP 404")
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            errors.append(f"{url}: HTTP {response.status_code} ({exc})")
            continue

        try:
            return response.json(), url
        except ValueError as exc:
            errors.append(f"{url}: invalid JSON ({exc})")

    attempted = ", ".join(urls)
    details = "; ".join(errors) if errors else "no responses"
    raise LlamaCppDiscoveryError(
        f"Unable to query {endpoint_name}. Tried {attempted}. Details: {details}."
    )


async def _maybe_get_json_from_candidates(
    client: httpx.AsyncClient,
    urls: tuple[str, ...],
    *,
    headers: dict[str, str],
) -> tuple[object, str] | None:
    for url in urls:
        try:
            response = await client.get(url, headers=headers)
        except httpx.HTTPError:
            continue

        if response.status_code in {401, 403, 404}:
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            continue

        try:
            return response.json(), url
        except ValueError:
            continue

    return None


def _parse_models_payload(payload: object) -> tuple[tuple[LlamaCppModelListing, ...], bool]:
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        data = _string_keyed_payload(payload).get("data")
        raw_items = data if isinstance(data, list) else []
    else:
        raise LlamaCppDiscoveryError("Unexpected llama.cpp models payload shape.")

    models: list[LlamaCppModelListing] = []
    router_mode = False
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        raw_payload = _string_keyed_payload(raw_item)
        try:
            parsed = _LlamaCppModelPayload.model_validate(raw_item)
        except ValidationError as exc:
            raise LlamaCppDiscoveryError("Unexpected llama.cpp models payload shape.") from exc
        raw_status = raw_payload.get("status")
        if isinstance(raw_status, dict):
            router_mode = True
        model_id = parsed.id.strip()
        if not model_id:
            continue
        status = None
        if isinstance(raw_status, dict):
            status = _LlamaCppModelStatusPayload.model_validate(raw_status)
        models.append(
            LlamaCppModelListing(
                model_id=model_id,
                owned_by=_normalize_text(parsed.owned_by),
                training_context_window=(
                    _positive_int_or_none(parsed.meta.n_ctx_train) if parsed.meta is not None else None
                ),
                child_server_url=_child_server_url_from_status(status),
            )
        )
    return tuple(models), router_mode


def _parse_props_payload(payload: object) -> _LlamaCppPropsPayload:
    if not isinstance(payload, dict):
        raise LlamaCppDiscoveryError("Unexpected llama.cpp props payload shape.")
    try:
        return _LlamaCppPropsPayload.model_validate(payload)
    except ValidationError as exc:
        raise LlamaCppDiscoveryError("Unexpected llama.cpp props payload shape.") from exc


async def _discover_slots_max_output_tokens(
    *,
    client: httpx.AsyncClient,
    catalog: LlamaCppDiscoveryCatalog,
    headers: dict[str, str],
) -> int | None:
    slots_payload = await _maybe_get_json_from_candidates(
        client,
        catalog.endpoints.slots_urls(),
        headers=headers,
    )
    if slots_payload is None:
        return None

    payload, _resolved_url = slots_payload
    return _parse_slots_max_output_tokens(payload)


def _parse_slots_max_output_tokens(payload: object) -> int | None:
    if not isinstance(payload, list):
        return None

    slots: list[_LlamaCppSlotPayload] = []
    for raw_slot in payload:
        if not isinstance(raw_slot, dict):
            continue
        try:
            slots.append(_LlamaCppSlotPayload.model_validate(raw_slot))
        except ValidationError as exc:
            raise LlamaCppDiscoveryError("Unexpected llama.cpp slots payload shape.") from exc

    for slot in slots:
        if not slot.is_processing or slot.params is None:
            continue
        n_predict = _positive_int_or_none(slot.params.n_predict)
        if n_predict is not None:
            return n_predict
        max_tokens = _positive_int_or_none(slot.params.max_tokens)
        if max_tokens is not None:
            return max_tokens

    for slot in slots:
        if slot.params is None:
            continue
        n_predict = _positive_int_or_none(slot.params.n_predict)
        if n_predict is not None:
            return n_predict
        max_tokens = _positive_int_or_none(slot.params.max_tokens)
        if max_tokens is not None:
            return max_tokens

    return None


def _derive_tokenizes(modalities: _LlamaCppModalitiesPayload) -> tuple[str, ...]:
    tokenizes = list(_TEXT_TOKENIZES)
    if modalities.vision:
        tokenizes.extend(_VISION_TOKENIZES)
    return tuple(tokenizes)


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _positive_int_or_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _child_server_url_from_status(status: _LlamaCppModelStatusPayload | None) -> str | None:
    if status is None or _normalize_text(status.value) != "loaded":
        return None

    host: str | None = None
    port: int | None = None
    args = status.args
    for index, arg in enumerate(args):
        if arg == "--host" and index + 1 < len(args):
            host = _normalize_text(args[index + 1])
        if arg == "--port" and index + 1 < len(args):
            port = _positive_int_or_none(_parse_int_or_none(args[index + 1]))

    if host is None or port is None:
        return None
    return f"http://{host}:{port}"


def _parse_int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


async def _enrich_router_models_with_child_metadata(
    *,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    models: tuple[LlamaCppModelListing, ...],
) -> tuple[LlamaCppModelListing, ...]:
    enriched: list[LlamaCppModelListing] = []
    for model in models:
        if model.training_context_window is not None or model.child_server_url is None:
            enriched.append(model)
            continue
        enriched.append(
            await _refresh_model_listing_from_child(
                client=client,
                headers=headers,
                listing=model,
            )
        )
    return tuple(enriched)


async def _refresh_router_model_listing(
    *,
    client: httpx.AsyncClient,
    catalog: LlamaCppDiscoveryCatalog,
    headers: dict[str, str],
    listing: LlamaCppModelListing,
) -> LlamaCppModelListing:
    payload, _resolved_url = await _get_json_from_candidates(
        client,
        catalog.endpoints.models_urls(),
        headers=headers,
        endpoint_name="llama.cpp model discovery",
    )
    models, _router_mode = _parse_models_payload(payload)
    updated = next((item for item in models if item.model_id == listing.model_id), None)
    if updated is None:
        return listing
    if updated.training_context_window is not None or updated.child_server_url is None:
        return updated
    return await _refresh_model_listing_from_child(
        client=client,
        headers=headers,
        listing=updated,
    )


async def _refresh_model_listing_from_child(
    *,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    listing: LlamaCppModelListing,
) -> LlamaCppModelListing:
    child_server_url = listing.child_server_url
    if child_server_url is None:
        return listing

    payload, _resolved_url = await _get_json_from_candidates(
        client,
        (_join_url(child_server_url, "/v1/models"), _join_url(child_server_url, "/models")),
        headers=headers,
        endpoint_name="llama.cpp child model discovery",
    )
    child_models, _router_mode = _parse_models_payload(payload)
    child_listing = next((item for item in child_models if item.model_id == listing.model_id), None)
    if child_listing is None and child_models:
        child_listing = child_models[0]
    if child_listing is None or child_listing.training_context_window is None:
        return listing
    return replace(listing, training_context_window=child_listing.training_context_window)


def _slugify_name(value: str) -> str:
    slug_chars: list[str] = []
    last_was_dash = False
    for char in value.strip().lower():
        if char.isalnum():
            slug_chars.append(char)
            last_was_dash = False
            continue
        if last_was_dash:
            continue
        slug_chars.append("-")
        last_was_dash = True
    return "".join(slug_chars).strip("-")


def _picker_label(discovered_model: LlamaCppDiscoveredModel) -> str:
    return (
        discovered_model.model_alias
        or discovered_model.listing.model_id.rsplit("/", 1)[-1]
        or discovered_model.listing.model_id
    )


def _string_keyed_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if isinstance(key, str)
    }
