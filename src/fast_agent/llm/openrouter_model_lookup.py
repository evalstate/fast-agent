"""OpenRouter model discovery and runtime metadata registration."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.provider_types import Provider
from fast_agent.utils.async_utils import run_sync

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    OpenRouterLookupFn = Callable[[str, str, float], Awaitable["OpenRouterModelLookupResult"]]


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DISCOVERY_CACHE_TTL_SECONDS = 300.0


class OpenRouterArchitecture(BaseModel):
    """OpenRouter model architecture summary."""

    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)


class OpenRouterTopProvider(BaseModel):
    """Top provider limits for a model in OpenRouter."""

    context_length: int | None = None
    max_completion_tokens: int | None = None


class OpenRouterModel(BaseModel):
    """OpenRouter model entry from /models/user."""

    id: str
    name: str | None = None
    context_length: int | None = None
    architecture: OpenRouterArchitecture = Field(default_factory=OpenRouterArchitecture)
    top_provider: OpenRouterTopProvider | None = None
    supported_parameters: list[str] = Field(default_factory=list)


class OpenRouterModelLookupResult(BaseModel):
    """Lookup result for OpenRouter model discovery."""

    models: list[OpenRouterModel] = Field(default_factory=list)
    error: str | None = None

    @property
    def has_models(self) -> bool:
        return bool(self.models)


_OPENROUTER_MODEL_CACHE: dict[str, tuple[float, OpenRouterModelLookupResult]] = {}


def _normalize_base_url(base_url: str | None) -> str:
    url = (base_url or DEFAULT_OPENROUTER_BASE_URL).strip()
    return url.rstrip("/")


def _cache_key(api_key: str, base_url: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"{base_url}::{digest}"


def clear_openrouter_model_cache() -> None:
    """Clear in-memory OpenRouter model discovery cache."""
    _OPENROUTER_MODEL_CACHE.clear()


async def lookup_openrouter_models(
    api_key: str,
    base_url: str | None = None,
    timeout: float = 10.0,
    *,
    force_refresh: bool = False,
    lookup_fn: OpenRouterLookupFn | None = None,
) -> OpenRouterModelLookupResult:
    """Look up models available to the current OpenRouter user/key."""
    if lookup_fn is not None:
        return await lookup_fn(api_key, _normalize_base_url(base_url), timeout)

    resolved_base_url = _normalize_base_url(base_url)
    cache_key = _cache_key(api_key, resolved_base_url)
    now = time.time()

    if not force_refresh:
        cached = _OPENROUTER_MODEL_CACHE.get(cache_key)
        if cached is not None:
            expires_at, cached_result = cached
            if expires_at > now:
                return cached_result

    url = f"{resolved_base_url}/models/user"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)

        if response.status_code in {401, 403}:
            return OpenRouterModelLookupResult(
                models=[],
                error="OpenRouter API key rejected while listing available models",
            )

        response.raise_for_status()
        payload = response.json()
        entries = payload.get("data", []) if isinstance(payload, dict) else []
        models = [OpenRouterModel.model_validate(entry) for entry in entries]

        result = OpenRouterModelLookupResult(models=models)
        _OPENROUTER_MODEL_CACHE[cache_key] = (
            now + OPENROUTER_DISCOVERY_CACHE_TTL_SECONDS,
            result,
        )
        return result

    except httpx.TimeoutException:
        return OpenRouterModelLookupResult(models=[], error="Timeout listing OpenRouter models")
    except httpx.HTTPStatusError as exc:
        return OpenRouterModelLookupResult(
            models=[],
            error=f"HTTP error {exc.response.status_code} listing OpenRouter models",
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return OpenRouterModelLookupResult(
            models=[],
            error=f"Error listing OpenRouter models: {exc}",
        )


def lookup_openrouter_models_sync(
    api_key: str,
    base_url: str | None = None,
    timeout: float = 10.0,
    *,
    force_refresh: bool = False,
) -> OpenRouterModelLookupResult:
    """Synchronous wrapper for OpenRouter model lookup."""
    result = run_sync(
        lookup_openrouter_models,
        api_key,
        base_url,
        timeout,
        force_refresh=force_refresh,
    )
    if result is None:
        raise RuntimeError("OpenRouter model lookup returned no result")
    return result


def _map_modalities_to_tokenizes(input_modalities: list[str]) -> list[str]:
    normalized = {mode.strip().lower() for mode in input_modalities if mode}
    tokenizes: list[str] = []

    if "text" in normalized:
        tokenizes.append("text/plain")
    if "image" in normalized:
        tokenizes.extend(["image/jpeg", "image/png", "image/webp"])
    if "file" in normalized:
        tokenizes.append("application/pdf")
    if "audio" in normalized:
        tokenizes.extend(["audio/wav", "audio/mpeg", "audio/mp3"])

    if not tokenizes:
        tokenizes.append("text/plain")

    deduped: list[str] = []
    for mime in tokenizes:
        if mime not in deduped:
            deduped.append(mime)
    return deduped


def _resolve_json_mode(supported_parameters: list[str]) -> str | None:
    supported = {param.strip().lower() for param in supported_parameters if param}
    if "structured_outputs" in supported:
        return "schema"
    if "response_format" in supported:
        return "object"
    return None


def _to_model_parameters(model: OpenRouterModel) -> ModelParameters:
    top_provider = model.top_provider or OpenRouterTopProvider()
    context_window = top_provider.context_length or model.context_length or 128000

    max_output_tokens = top_provider.max_completion_tokens
    if max_output_tokens is None or max_output_tokens <= 0:
        max_output_tokens = 16384

    return ModelParameters(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        tokenizes=_map_modalities_to_tokenizes(model.architecture.input_modalities),
        json_mode=_resolve_json_mode(model.supported_parameters),
        default_provider=Provider.OPENROUTER,
    )


def register_runtime_openrouter_models(result: OpenRouterModelLookupResult) -> int:
    """Register runtime metadata for discovered OpenRouter models."""
    count = 0
    for model in result.models:
        model_id = (model.id or "").strip()
        if not model_id:
            continue

        normalized = ModelDatabase.normalize_model_name(f"openrouter.{model_id}")
        if not normalized:
            continue

        # Keep curated/static metadata for known models intact.
        if normalized in ModelDatabase.MODELS:
            continue

        ModelDatabase.register_runtime_model_params(
            f"openrouter.{model_id}", _to_model_parameters(model)
        )
        count += 1

    return count


def discover_openrouter_models_sync(
    api_key: str,
    base_url: str | None = None,
    timeout: float = 10.0,
    *,
    force_refresh: bool = False,
) -> OpenRouterModelLookupResult:
    """Discover OpenRouter models and register runtime model metadata."""
    result = lookup_openrouter_models_sync(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        force_refresh=force_refresh,
    )
    if result.has_models:
        register_runtime_openrouter_models(result)
    return result


def list_openrouter_model_specs_sync(
    api_key: str,
    base_url: str | None = None,
    timeout: float = 10.0,
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Return model specs suitable for direct use in fast-agent.

    Example: ``openrouter.openai/gpt-4.1-mini``
    """
    result = discover_openrouter_models_sync(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        force_refresh=force_refresh,
    )
    if not result.has_models:
        return []

    seen: set[str] = set()
    specs: list[str] = []
    for entry in result.models:
        model_id = (entry.id or "").strip()
        if not model_id:
            continue
        spec = f"openrouter.{model_id}"
        if spec in seen:
            continue
        seen.add(spec)
        specs.append(spec)
    return specs


__all__ = [
    "DEFAULT_OPENROUTER_BASE_URL",
    "OPENROUTER_DISCOVERY_CACHE_TTL_SECONDS",
    "OpenRouterModel",
    "OpenRouterModelLookupResult",
    "clear_openrouter_model_cache",
    "discover_openrouter_models_sync",
    "list_openrouter_model_specs_sync",
    "lookup_openrouter_models",
    "lookup_openrouter_models_sync",
    "register_runtime_openrouter_models",
]
