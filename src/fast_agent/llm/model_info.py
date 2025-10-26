"""
Typed model information helpers.

Provides a small, pythonic interface to query model/provider and
capabilities (Text/Document/Vision), backed by the model database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    # Import behind TYPE_CHECKING to avoid import cycles at runtime
    from fast_agent.interfaces import FastAgentLLMProtocol


@dataclass(frozen=True)
class ModelInfo:
    """Resolved model information with convenient capability accessors."""

    name: str
    provider: Provider
    context_window: Optional[int]
    max_output_tokens: Optional[int]
    tokenizes: List[str]
    json_mode: Optional[str]
    reasoning: Optional[str]

    @property
    def supports_text(self) -> bool:
        if "text/plain" in (self.tokenizes or []):
            return True
        return ModelDatabase.supports_mime(self.name, "text/plain")

    @property
    def supports_document(self) -> bool:
        # Document support currently keyed off PDF support
        if "application/pdf" in (self.tokenizes or []):
            return True
        return ModelDatabase.supports_mime(self.name, "pdf")

    @property
    def supports_vision(self) -> bool:
        # Any common image format indicates vision support
        tokenizes = self.tokenizes or []
        if any(mt in tokenizes for mt in ("image/jpeg", "image/png", "image/webp")):
            return True

        return any(
            ModelDatabase.supports_mime(self.name, mt)
            for mt in ("image/jpeg", "image/png", "image/webp")
        )

    @property
    def tdv_flags(self) -> tuple[bool, bool, bool]:
        """Convenience tuple: (text, document, vision)."""
        return (self.supports_text, self.supports_document, self.supports_vision)

    @classmethod
    def from_llm(cls, llm: "FastAgentLLMProtocol") -> Optional["ModelInfo"]:
        name = getattr(llm, "model_name", None)
        provider = getattr(llm, "provider", None)
        if not name or not provider:
            return None
        return cls.from_name(name, provider)

    @classmethod
    def from_name(cls, name: str, provider: Provider | None = None) -> Optional["ModelInfo"]:
        canonical_name = ModelFactory.MODEL_ALIASES.get(name, name)
        params = ModelDatabase.get_model_params(canonical_name)
        if not params:
            # Unknown model: return a conservative default that supports text only.
            # This matches the desired behavior for TDV display fallbacks.
            if provider is None:
                provider = Provider.GENERIC
            return ModelInfo(
                name=canonical_name,
                provider=provider,
                context_window=None,
                max_output_tokens=None,
                tokenizes=["text/plain"],
                json_mode=None,
                reasoning=None,
            )

        if provider is None:
            provider = ModelFactory.DEFAULT_PROVIDERS.get(canonical_name, Provider.GENERIC)

        return ModelInfo(
            name=canonical_name,
            provider=provider,
            context_window=params.context_window,
            max_output_tokens=params.max_output_tokens,
            tokenizes=params.tokenizes,
            json_mode=params.json_mode,
            reasoning=params.reasoning,
        )
