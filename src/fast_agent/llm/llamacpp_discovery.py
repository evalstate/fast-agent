"""llama.cpp discovery helpers for overlay import flows."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.model_overlays import (
    LoadedModelOverlay,
    ModelOverlayConnection,
    ModelOverlayDefaults,
    ModelOverlayManifest,
    ModelOverlayMetadata,
    ModelOverlayPicker,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import (
    ReasoningEffortLevel,
    ReasoningEffortSetting,
    ReasoningEffortSpec,
)
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.numeric import positive_int_or_none
from fast_agent.utils.text import strip_casefold, strip_to_none

DEFAULT_LLAMA_CPP_URL: Final[str] = "http://localhost:9931/v1"
LEGACY_LLAMA_CPP_URL: Final[str] = "http://localhost:8080/v1"
DEFAULT_DISCOVERY_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_LOCAL_PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
_TEXT_TOKENIZES: Final[tuple[str, ...]] = ("text/plain",)
_VISION_TOKENIZES: Final[tuple[str, ...]] = (
    "image/jpeg",
    "image/png",
    "image/webp",
)
_AUDIO_TOKENIZES: Final[tuple[str, ...]] = (
    "audio/wav",
    "audio/mpeg",
    "audio/ogg",
    "audio/flac",
)
_VIDEO_TOKENIZES: Final[tuple[str, ...]] = (
    "video/mp4",
    "video/webm",
)
_GGUF_PUBLISHER_PREFIXES: Final[frozenset[str]] = frozenset(
    {
        "ggml-org",
        "unsloth",
        "bartowski",
        "lmstudio-community",
        "mradermacher",
        "mlx-community",
        "tensorblock",
        "quantfactory",
        "thebloke",
        "second-state",
        "bullerwins",
        "ubergarm",
        "local",
    }
)
_GGUF_SUFFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:[-_.]+"
    r"(?:"
    r"gguf|"
    r"ud[-_](?:i?q\d+(?:[-_][a-z0-9]+)*)|"
    r"iq\d+(?:[-_][a-z0-9]+)*|"
    r"q\d+(?:[-_][a-z0-9]+)*|"
    r"bf16|fp16|f16|f32|fp8|fp4|mxfp4|nvfp4|int8|int4|awq|gptq|exl2|"
    r"mtp|qat"
    r"))$",
    re.IGNORECASE,
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
        return _dedupe_urls((_join_url(self.request_base_url, "/models"),))

    def props_urls(
        self,
        *,
        model_id: str | None = None,
        autoload: bool | None = None,
    ) -> tuple[str, ...]:
        """Return candidate props URLs in preferred order."""
        props_base = _join_url(self.server_url, "/props")
        normalized_model_id = strip_to_none(model_id)
        if normalized_model_id is not None:
            query = f"model={quote(normalized_model_id, safe='')}"
            if autoload is False:
                return (f"{props_base}?{query}&autoload=false",)
            if autoload is True:
                query = f"{query}&autoload=true"
            return _dedupe_urls((f"{props_base}?{query}", props_base))
        return (props_base,)


@dataclass(frozen=True, slots=True)
class LlamaCppChatTemplateCaps:
    """Chat template capabilities advertised by llama.cpp `/props`."""

    supports_tools: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supports_preserve_reasoning: bool | None = None
    supports_reasoning_effort: bool | None = None
    reasoning_effort_levels: tuple[str, ...] = ()
    reasoning_effort_rejected: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LlamaCppPropsDetails:
    """Parsed metadata from a llama.cpp `/props` response."""

    runtime_context_window: int | None
    max_output_tokens: int | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    min_p: float | None
    tokenizes: tuple[str, ...]
    model_alias: str | None
    chat_template_caps: LlamaCppChatTemplateCaps | None


@dataclass(frozen=True, slots=True)
class LlamaCppModelListing:
    """A model entry discovered from the llama.cpp model list endpoint."""

    model_id: str
    owned_by: str | None
    training_context_window: int | None
    child_server_url: str | None = None
    status: str | None = None
    aliases: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ()


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
    chat_template_caps: LlamaCppChatTemplateCaps | None = None
    matched_database_model: str | None = None

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


class _LlamaCppModelArchitecturePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_modalities: list[str] = Field(default_factory=list)


class _LlamaCppModelPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    owned_by: str | None = None
    meta: _LlamaCppModelMetaPayload | None = None
    context_length: int | None = None
    aliases: list[str] = Field(default_factory=list)
    architecture: _LlamaCppModelArchitecturePayload | None = None


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
    video: bool = False


class _LlamaCppChatTemplateCapsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    supports_tools: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supports_preserve_reasoning: bool | None = None
    supports_reasoning_effort: bool | None = None
    reasoning_effort_levels: list[str] = Field(default_factory=list)
    reasoning_effort_rejected: list[str] = Field(default_factory=list)


class _LlamaCppPropsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default_generation_settings: _LlamaCppGenerationSettingsPayload = Field(
        default_factory=_LlamaCppGenerationSettingsPayload
    )
    model_alias: str | None = None
    modalities: _LlamaCppModalitiesPayload = Field(default_factory=_LlamaCppModalitiesPayload)
    chat_template_caps: _LlamaCppChatTemplateCapsPayload | None = None


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


def candidate_local_llamacpp_urls(explicit_url: str | None = None) -> tuple[str, ...]:
    """Return local llama.cpp base URLs to probe in priority order."""
    normalized_explicit = strip_to_none(explicit_url)
    if normalized_explicit is not None:
        return (normalize_llamacpp_url(normalized_explicit).request_base_url,)

    candidates: list[str] = []
    env_host = strip_to_none(os.environ.get("LLAMA_ARG_HOST"))
    env_port = strip_to_none(os.environ.get("LLAMA_ARG_PORT"))
    if env_host is not None or env_port is not None:
        host = env_host or "localhost"
        if host in {"0.0.0.0", "::", "[::]"}:
            host = "localhost"
        port = env_port or "9931"
        candidates.append(normalize_llamacpp_url(f"http://{host}:{port}/v1").request_base_url)

    candidates.append(DEFAULT_LLAMA_CPP_URL)
    candidates.append(LEGACY_LLAMA_CPP_URL)
    return _dedupe_urls(tuple(candidates))


async def discover_local_llamacpp_catalog(
    *,
    url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_LOCAL_PROBE_TIMEOUT_SECONDS,
) -> LlamaCppDiscoveryCatalog:
    """Probe local llama.cpp candidate URLs and return the first responding catalog."""
    candidates = candidate_local_llamacpp_urls(url)
    errors: list[str] = []
    for candidate_url in candidates:
        try:
            return await discover_llamacpp_models(
                url=candidate_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        except LlamaCppDiscoveryError as exc:
            errors.append(f"{candidate_url}: {exc}")
    attempted = ", ".join(candidates)
    raise LlamaCppDiscoveryError(
        f"No running llama.cpp server found. Tried {attempted}. "
        f"Start llama-server (default port 9931 or 8080) or pass --url."
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
    autoload: bool | None = None,
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
            catalog.endpoints.props_urls(model_id=model_id, autoload=autoload),
            headers=headers,
            endpoint_name="llama.cpp props discovery",
        )
        props = _parse_props_payload(payload)
        generation = props.default_generation_settings
        params = generation.params or _LlamaCppGenerationParamsPayload()
        max_output_tokens = positive_int_or_none(params.n_predict)
        if max_output_tokens is None:
            max_output_tokens = positive_int_or_none(params.max_tokens)
        if catalog.router_mode and listing.training_context_window is None:
            listing = await _refresh_router_model_listing(
                client=client,
                catalog=catalog,
                headers=headers,
                listing=listing,
            )

    model_alias = _normalize_text(props.model_alias)
    matched = match_llamacpp_model_database_entry(
        listing.model_id,
        model_alias=model_alias,
        aliases=listing.aliases,
    )
    caps = _parse_chat_template_caps(props.chat_template_caps)

    return LlamaCppDiscoveredModel(
        listing=listing,
        props_url=resolved_url,
        runtime_context_window=positive_int_or_none(generation.n_ctx),
        max_output_tokens=max_output_tokens,
        temperature=_normalize_llamacpp_float(params.temperature),
        top_k=positive_int_or_none(params.top_k),
        top_p=_normalize_llamacpp_float(params.top_p),
        min_p=_normalize_llamacpp_float(params.min_p),
        tokenizes=_derive_tokenizes(props.modalities),
        model_alias=model_alias,
        chat_template_caps=caps,
        matched_database_model=matched[0] if matched is not None else None,
    )


def _strip_gguf_quant_suffixes(candidate: str) -> str:
    if ":" in candidate:
        candidate = candidate.split(":", 1)[0]
    candidate = candidate.strip()
    while True:
        stripped = _GGUF_SUFFIX_PATTERN.sub("", candidate).strip()
        if not stripped or stripped == candidate:
            break
        candidate = stripped
    return candidate.lower()


def _normalize_gguf_candidate(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    if candidate.lower().endswith(".gguf"):
        candidate = candidate.rsplit("/", 1)[-1]
        candidate = candidate[:-5]
    elif candidate.count("/") > 1:
        candidate = candidate.rsplit("/", 1)[-1]
    return _strip_gguf_quant_suffixes(candidate)


def _normalize_endpoint_candidates(value: str) -> list[str]:
    candidate = value.strip().strip("/")
    if not candidate:
        return []
    results: list[str] = []
    if not candidate.lower().endswith(".gguf") and candidate.count("/") >= 2:
        parts = [part for part in candidate.split("/") if part]
        if len(parts) >= 2:
            tail_pair = _strip_gguf_quant_suffixes(f"{parts[-2]}/{parts[-1]}")
            if tail_pair:
                results.append(tail_pair)
    norm = _normalize_gguf_candidate(value)
    if norm and norm not in results:
        results.append(norm)
    if "/" in norm:
        prefix, leaf = norm.split("/", 1)
        if prefix in _GGUF_PUBLISHER_PREFIXES and leaf and leaf not in results:
            results.append(leaf)
        elif leaf and leaf not in results:
            results.append(leaf)
    return results


def match_llamacpp_model_database_entry(
    model_id: str,
    *,
    root: str | None = None,
    model_alias: str | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[str, ModelParameters] | None:
    """Match a discovered endpoint model name (GGUF/vLLM/OpenAI-compatible) against `ModelDatabase.MODELS`."""
    raw_candidates: list[str] = [model_id]
    if root:
        raw_candidates.append(root)
    if model_alias:
        raw_candidates.append(model_alias)
    raw_candidates.extend(aliases)

    normalized_candidates: list[str] = []
    for raw in raw_candidates:
        for norm in _normalize_endpoint_candidates(raw):
            if norm not in normalized_candidates:
                normalized_candidates.append(norm)

    if not normalized_candidates:
        return None

    models_map = ModelDatabase.MODELS
    # Exact key match first (e.g. "thinkingmachines/inkling-2-mini").
    for candidate in normalized_candidates:
        if candidate in models_map:
            return candidate, models_map[candidate]

    # Suffix / leaf match against org/model entries in ModelDatabase.MODELS.
    leaf_index: dict[str, str] = {}
    for db_key in models_map:
        leaf = db_key.rsplit("/", 1)[-1]
        leaf_index.setdefault(leaf, db_key)

    for candidate in normalized_candidates:
        if candidate in leaf_index:
            matched_key = leaf_index[candidate]
            return matched_key, models_map[matched_key]

    return None


def parse_llamacpp_props_details(payload: object) -> LlamaCppPropsDetails:
    """Parse runtime metadata from a llama.cpp `/props` JSON payload."""
    props = _parse_props_payload(payload)
    generation = props.default_generation_settings
    params = generation.params or _LlamaCppGenerationParamsPayload()
    max_output_tokens = positive_int_or_none(params.n_predict)
    if max_output_tokens is None:
        max_output_tokens = positive_int_or_none(params.max_tokens)
    return LlamaCppPropsDetails(
        runtime_context_window=positive_int_or_none(generation.n_ctx),
        max_output_tokens=max_output_tokens,
        temperature=_normalize_llamacpp_float(params.temperature),
        top_k=positive_int_or_none(params.top_k),
        top_p=_normalize_llamacpp_float(params.top_p),
        min_p=_normalize_llamacpp_float(params.min_p),
        tokenizes=_derive_tokenizes(props.modalities),
        model_alias=_normalize_text(props.model_alias),
        chat_template_caps=_parse_chat_template_caps(props.chat_template_caps),
    )


def resolve_live_reasoning_effort_spec(
    *,
    base_spec: ReasoningEffortSpec | None,
    caps: LlamaCppChatTemplateCaps | None,
) -> ReasoningEffortSpec | None:
    if caps is None:
        return base_spec
    if caps.supports_reasoning_effort is False:
        return None
    if not caps.reasoning_effort_levels:
        return base_spec

    valid_levels: dict[str, ReasoningEffortLevel] = {
        "none": "none",
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
    allowed: list[ReasoningEffortLevel] = []
    for level in caps.reasoning_effort_levels:
        mapped = valid_levels.get(level.strip().lower())
        if mapped is not None and mapped not in allowed:
            allowed.append(mapped)
    if not allowed:
        return base_spec

    default_setting: ReasoningEffortSetting | None = None
    if (
        base_spec is not None
        and base_spec.default is not None
        and base_spec.default.kind == "effort"
        and str(base_spec.default.value) in allowed
    ):
        default_setting = base_spec.default
    elif "medium" in allowed:
        default_setting = ReasoningEffortSetting(kind="effort", value="medium")
    else:
        default_setting = ReasoningEffortSetting(kind="effort", value=allowed[0])

    return ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=allowed,
        allow_auto=False,
        default=default_setting,
    )


def build_llamacpp_runtime_model_parameters(
    discovered_model: LlamaCppDiscoveredModel,
) -> ModelParameters:
    """Merge matched ModelDatabase preferences with live llama.cpp server truth."""
    matched = match_llamacpp_model_database_entry(
        discovered_model.listing.model_id,
        model_alias=discovered_model.model_alias,
        aliases=discovered_model.listing.aliases,
    )
    base_params = matched[1] if matched is not None else None

    context_window = (
        discovered_model.effective_context_window
        or (base_params.context_window if base_params is not None else None)
        or 32768
    )
    if discovered_model.max_output_tokens is not None:
        max_output_tokens = discovered_model.max_output_tokens
    elif base_params is not None:
        max_output_tokens = min(base_params.max_output_tokens, context_window)
    else:
        max_output_tokens = min(context_window, 16384)

    live_tokenizes = list(discovered_model.tokenizes)
    live_reasoning_spec = resolve_live_reasoning_effort_spec(
        base_spec=base_params.reasoning_effort_spec if base_params is not None else None,
        caps=discovered_model.chat_template_caps,
    )

    if base_params is not None:
        return base_params.model_copy(
            update={
                "context_window": context_window,
                "max_output_tokens": max_output_tokens,
                "tokenizes": live_tokenizes,
                "reasoning_effort_spec": live_reasoning_spec,
            }
        )

    reasoning_mode: Literal["reasoning_content"] | None = (
        "reasoning_content"
        if (
            discovered_model.chat_template_caps is not None
            and (
                discovered_model.chat_template_caps.supports_reasoning_effort
                or discovered_model.chat_template_caps.supports_preserve_reasoning
            )
        )
        else None
    )
    return ModelParameters(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        tokenizes=live_tokenizes,
        json_mode="schema",
        reasoning=reasoning_mode,
        reasoning_effort_spec=live_reasoning_spec,
    )


def build_ephemeral_llamacpp_overlay(
    *,
    discovered_model: LlamaCppDiscoveredModel,
    base_url: str,
    overlay_name: str | None = None,
) -> LoadedModelOverlay:
    """Build an in-memory runnable overlay for `fast-agent --local`."""
    resolved_name = overlay_name or default_overlay_name_for_model(
        discovered_model.listing.model_id
    )
    manifest = build_llamacpp_overlay_manifest(
        overlay_name=resolved_name,
        discovered_model=discovered_model,
        base_url=base_url,
        auth="none",
        api_key_env=None,
        secret_ref=None,
        current=True,
        include_sampling_defaults=False,
    )
    runtime_params = build_llamacpp_runtime_model_parameters(discovered_model)
    return LoadedModelOverlay(
        manifest=manifest,
        manifest_path=Path("<local>"),
        secret_entry=None,
        override_model_parameters=runtime_params,
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
    return tuple(unique_preserve_order(urls))


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
        training_ctx = (
            positive_int_or_none(parsed.meta.n_ctx_train) if parsed.meta is not None else None
        )
        if training_ctx is None:
            training_ctx = positive_int_or_none(parsed.context_length)
        input_modalities = (
            tuple(m.strip().lower() for m in parsed.architecture.input_modalities if m.strip())
            if parsed.architecture is not None
            else ()
        )
        models.append(
            LlamaCppModelListing(
                model_id=model_id,
                owned_by=_normalize_text(parsed.owned_by),
                training_context_window=training_ctx,
                child_server_url=_child_server_url_from_status(status),
                status=_normalize_text(status.value) if status is not None else None,
                aliases=tuple(a.strip() for a in parsed.aliases if a.strip()),
                input_modalities=input_modalities,
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


def _parse_chat_template_caps(
    payload: _LlamaCppChatTemplateCapsPayload | None,
) -> LlamaCppChatTemplateCaps | None:
    if payload is None:
        return None
    return LlamaCppChatTemplateCaps(
        supports_tools=payload.supports_tools,
        supports_parallel_tool_calls=payload.supports_parallel_tool_calls,
        supports_preserve_reasoning=payload.supports_preserve_reasoning,
        supports_reasoning_effort=payload.supports_reasoning_effort,
        reasoning_effort_levels=tuple(
            level.strip().lower() for level in payload.reasoning_effort_levels if level.strip()
        ),
        reasoning_effort_rejected=tuple(
            level.strip().lower() for level in payload.reasoning_effort_rejected if level.strip()
        ),
    )


def _derive_tokenizes(modalities: _LlamaCppModalitiesPayload) -> tuple[str, ...]:
    tokenizes = list(_TEXT_TOKENIZES)
    if modalities.vision:
        tokenizes.extend(_VISION_TOKENIZES)
    if modalities.audio:
        tokenizes.extend(_AUDIO_TOKENIZES)
    if modalities.video:
        tokenizes.extend(_VIDEO_TOKENIZES)
    return tuple(tokenizes)


def _normalize_text(value: str | None) -> str | None:
    return strip_to_none(value)


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
            port = positive_int_or_none(_parse_int_or_none(args[index + 1]))

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
    for char in strip_casefold(value):
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
    return {key: value for key, value in payload.items() if isinstance(key, str)}
