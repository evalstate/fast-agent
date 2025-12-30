"""Utility to lookup HuggingFace inference providers for a model.

This module provides functionality to check whether a HuggingFace model
has inference providers available through the HuggingFace Inference API.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field, computed_field

from fast_agent.utils.async_utils import run_sync

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    # Type alias for lookup function - allows dependency injection for testing
    InferenceLookupFn = Callable[[str], Awaitable["InferenceProviderLookupResult"]]


class InferenceProviderStatus(str, Enum):
    """Status of an inference provider for a model."""

    LIVE = "live"
    STAGING = "staging"


class InferenceProvider(BaseModel):
    """Information about an inference provider for a model."""

    name: str
    status: InferenceProviderStatus = InferenceProviderStatus.LIVE
    provider_id: str = Field(default="", alias="providerId")
    task: str = ""
    is_model_author: bool = Field(default=False, alias="isModelAuthor")

    model_config = {"populate_by_name": True}

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "InferenceProvider":
        """Create an InferenceProvider from API response data."""
        return cls(name=name, **data)


class InferenceProviderLookupResult(BaseModel):
    """Result of looking up inference providers for a model."""

    model_id: str
    exists: bool
    providers: list[InferenceProvider] = Field(default_factory=list)
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_providers(self) -> bool:
        """Return True if the model has any live inference providers."""
        return len(self.live_providers) > 0

    @property
    def live_providers(self) -> list[InferenceProvider]:
        """Return only providers with 'live' status."""
        return [p for p in self.providers if p.status == InferenceProviderStatus.LIVE]

    def format_provider_list(self) -> str:
        """Format the list of live providers as a comma-separated string."""
        return ", ".join(p.name for p in self.live_providers)

    def format_model_strings(self) -> list[str]:
        """Format model strings with provider suffixes for each live provider.

        Returns strings like: model_id:provider_name
        """
        return [f"{self.model_id}:{p.name}" for p in self.live_providers]


class ModelValidationResult(BaseModel):
    """Result of validating an HF model for /set-model."""

    valid: bool
    display_message: str = ""
    error: str | None = None


HF_API_BASE = "https://huggingface.co/api/models"


def normalize_hf_model_id(model: str) -> str | None:
    """Normalize an HF model spec to a bare model_id, or return None if not HF."""
    model_id = model

    if model_id.startswith("hf."):
        model_id = model_id[3:]

    if ":" in model_id:
        model_id = model_id.rsplit(":", 1)[0]

    if "/" not in model_id:
        return None

    return model_id


def format_provider_help_message(result: InferenceProviderLookupResult) -> str | None:
    """Format the provider help message for /set-model when providers exist."""
    if result.has_providers:
        providers = result.format_provider_list()
        example = random.choice(result.format_model_strings())
        return (
            f"**Available providers:** {providers}\n\n"
            f"**Autoroutes if no provider specified. Example use:** `/set-model {example}`"
        )
    if result.exists:
        return "No inference providers currently available for this model."
    return None


def format_provider_summary(result: InferenceProviderLookupResult) -> str | None:
    """Format a brief provider summary for status output."""
    if result.has_providers:
        providers = result.format_provider_list()
        return f"Available providers: {providers}"
    if result.exists:
        return "No inference providers available"
    return None


async def lookup_inference_providers(
    model_id: str,
    timeout: float = 10.0,
    *,
    lookup_fn: InferenceLookupFn | None = None,
) -> InferenceProviderLookupResult:
    """Look up available inference providers for a HuggingFace model.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Thinking")
        timeout: Request timeout in seconds
        lookup_fn: Optional function to use for lookup (for testing)

    Returns:
        InferenceProviderLookupResult with provider information

    Example:
        >>> result = await lookup_inference_providers("moonshotai/Kimi-K2-Thinking")
        >>> if result.has_providers:
        ...     print(f"Available providers: {result.format_provider_list()}")
        ...     for model_str in result.format_model_strings():
        ...         print(f"  hf.{model_str}")
    """
    # Allow test injection
    if lookup_fn is not None:
        return await lookup_fn(model_id)

    # Normalize model_id - strip any hf. prefix
    if model_id.startswith("hf."):
        model_id = model_id[3:]

    # Strip any existing provider suffix (e.g., model:provider -> model)
    if ":" in model_id:
        model_id = model_id.rsplit(":", 1)[0]

    url = f"{HF_API_BASE}/{model_id}"
    params = {"expand[]": "inferenceProviderMapping"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params)

            if response.status_code == 401:
                # Model does not exist
                return InferenceProviderLookupResult(
                    model_id=model_id,
                    exists=False,
                    providers=[],
                    error=f"Model '{model_id}' not found on HuggingFace",
                )

            response.raise_for_status()
            data = response.json()

            # Parse inference provider mapping
            provider_mapping = data.get("inferenceProviderMapping", {})
            providers = [
                InferenceProvider.from_dict(name, info) for name, info in provider_mapping.items()
            ]

            return InferenceProviderLookupResult(
                model_id=model_id,
                exists=True,
                providers=providers,
            )

    except httpx.TimeoutException:
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=False,
            providers=[],
            error=f"Timeout looking up model '{model_id}'",
        )
    except httpx.HTTPStatusError as e:
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=False,
            providers=[],
            error=f"HTTP error {e.response.status_code} looking up model '{model_id}'",
        )
    except Exception as e:
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=False,
            providers=[],
            error=f"Error looking up model '{model_id}': {e}",
        )


def lookup_inference_providers_sync(
    model_id: str,
    timeout: float = 10.0,
) -> InferenceProviderLookupResult:
    """Synchronous wrapper for lookup_inference_providers.

    Args:
        model_id: The HuggingFace model ID
        timeout: Request timeout in seconds

    Returns:
        InferenceProviderLookupResult with provider information
    """
    result = run_sync(lookup_inference_providers, model_id, timeout)
    if result is None:
        raise RuntimeError("Inference provider lookup returned no result")
    return result


def format_inference_lookup_message(result: InferenceProviderLookupResult) -> str:
    """Format the lookup result as a user-friendly message.

    Args:
        result: The lookup result to format

    Returns:
        A formatted string suitable for display
    """
    if result.error:
        return f"**Error:** {result.error}"

    if not result.exists:
        return f"Model `{result.model_id}` not found on HuggingFace."

    if not result.has_providers:
        return (
            f"Model `{result.model_id}` exists on HuggingFace but has no "
            f"inference providers available.\n\n"
            f"You may still be able to use it with a locally hosted inference endpoint."
        )

    providers = result.live_providers
    lines = [
        f"Model `{result.model_id}` has **{len(providers)}** inference provider(s) available:\n",
    ]

    for provider in providers:
        lines.append(f"- **{provider.name}**")

    lines.extend(
        [
            "",
            "**Usage:**",
            "```",
            f"/set-model hf.{result.model_id}:<provider>",
            "```",
            "",
            "**Examples:**",
        ]
    )

    for model_str in result.format_model_strings()[:3]:  # Show up to 3 examples
        lines.append(f"- `hf.{model_str}`")

    return "\n".join(lines)


async def validate_hf_model(
    model: str,
    *,
    aliases: dict[str, str] | None = None,
    lookup_fn: InferenceLookupFn | None = None,
) -> ModelValidationResult:
    """Validate that an HF model exists and has inference providers.

    Args:
        model: The model string (e.g., "hf.moonshotai/Kimi-K2-Thinking:together")
            Can also be an alias like "kimi" or "glm" that resolves to an HF model.
        aliases: Optional dict of model aliases (e.g., {"kimi": "hf.moonshotai/..."}).
            If not provided, no alias resolution is performed.
        lookup_fn: Optional function to use for lookup (for testing)

    Returns:
        ModelValidationResult with validation status and messages
    """
    # Resolve aliases first (e.g., "kimi" -> "hf.moonshotai/Kimi-K2-Instruct-0905:groq")
    if aliases:
        model = aliases.get(model, model)

    model_id = normalize_hf_model_id(model)
    if model_id is None:
        # Not an HF model - skip validation (let ModelFactory handle it)
        return ModelValidationResult(valid=True)

    try:
        result = await lookup_inference_providers(model_id, lookup_fn=lookup_fn)

        if not result.exists:
            return ModelValidationResult(
                valid=False,
                error=f"Error: Model `{model_id}` not found on HuggingFace",
            )

        if not result.has_providers:
            return ModelValidationResult(
                valid=False,
                error=f"Error: Model `{model_id}` exists but has no inference providers available",
            )

        display_message = format_provider_help_message(result) or ""
        return ModelValidationResult(valid=True, display_message=display_message)

    except Exception as e:
        return ModelValidationResult(
            valid=False,
            error=f"Error: Failed to validate model `{model_id}`: {e}",
        )
