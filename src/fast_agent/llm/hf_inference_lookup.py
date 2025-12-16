"""Utility to lookup HuggingFace inference providers for a model.

This module provides functionality to check whether a HuggingFace model
has inference providers available through the HuggingFace Inference API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from typing import Any


class InferenceProviderStatus(Enum):
    """Status of an inference provider for a model."""

    LIVE = "live"
    STAGING = "staging"


@dataclass
class InferenceProvider:
    """Information about an inference provider for a model."""

    name: str
    status: InferenceProviderStatus
    provider_id: str
    task: str
    is_model_author: bool

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "InferenceProvider":
        """Create an InferenceProvider from API response data."""
        return cls(
            name=name,
            status=InferenceProviderStatus(data.get("status", "live")),
            provider_id=data.get("providerId", ""),
            task=data.get("task", ""),
            is_model_author=data.get("isModelAuthor", False),
        )


@dataclass
class InferenceProviderLookupResult:
    """Result of looking up inference providers for a model."""

    model_id: str
    exists: bool
    providers: list[InferenceProvider]
    error: str | None = None

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


HF_API_BASE = "https://huggingface.co/api/models"


async def lookup_inference_providers(
    model_id: str,
    timeout: float = 10.0,
) -> InferenceProviderLookupResult:
    """Look up available inference providers for a HuggingFace model.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Thinking")
        timeout: Request timeout in seconds

    Returns:
        InferenceProviderLookupResult with provider information

    Example:
        >>> result = await lookup_inference_providers("moonshotai/Kimi-K2-Thinking")
        >>> if result.has_providers:
        ...     print(f"Available providers: {result.format_provider_list()}")
        ...     for model_str in result.format_model_strings():
        ...         print(f"  hf.{model_str}")
    """
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
                InferenceProvider.from_dict(name, info)
                for name, info in provider_mapping.items()
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
    return asyncio.run(lookup_inference_providers(model_id, timeout))


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

    lines.extend([
        "",
        "**Usage:**",
        "```",
        f"/set-model hf.{result.model_id}:<provider>",
        "```",
        "",
        "**Examples:**",
    ])

    for model_str in result.format_model_strings()[:3]:  # Show up to 3 examples
        lines.append(f"- `hf.{model_str}`")

    return "\n".join(lines)
