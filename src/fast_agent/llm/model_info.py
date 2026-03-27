"""
Typed model information helpers.

Provides a small, pythonic interface to query model/provider and
capabilities (Text/Document/Vision), backed by the model database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.llm.model_database import ModelDatabase, ResourceSource
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.mime_utils import DOCUMENT_MIME_TYPES, normalize_mime_type

if TYPE_CHECKING:
    # Import behind TYPE_CHECKING to avoid import cycles at runtime
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.resolved_model import ResolvedModelSpec


@dataclass(frozen=True)
class ModelInfo:
    """Resolved model information with convenient capability accessors."""

    name: str
    provider: Provider
    context_window: int | None
    max_output_tokens: int | None
    tokenizes: list[str]
    json_mode: str | None
    reasoning: str | None

    def _supports_provider_document_mime(
        self,
        normalized: str | None,
        *,
        resource_source: ResourceSource | None = None,
    ) -> bool | None:
        if not normalized or normalized not in DOCUMENT_MIME_TYPES:
            return None

        if (
            resource_source == "link"
            and self.provider in {Provider.ANTHROPIC, Provider.ANTHROPIC_VERTEX}
            and normalized != "application/pdf"
        ):
            return False

        multimodal_tokens = [mime.lower() for mime in (self.tokenizes or [])]
        has_multimodal_io = any(mime.startswith("image/") for mime in multimodal_tokens)

        if self.provider in {
            Provider.RESPONSES,
            Provider.OPENRESPONSES,
            Provider.CODEX_RESPONSES,
            Provider.ANTHROPIC,
        }:
            return has_multimodal_io

        if self.provider in {
            Provider.OPENAI,
            Provider.AZURE,
            Provider.ALIYUN,
            Provider.GOOGLE_OAI,
        }:
            return normalized == "application/pdf"

        return None

    def supports_mime(
        self,
        mime_type: str,
        *,
        resource_source: ResourceSource | None = None,
    ) -> bool:
        tokenizes = [mime.lower() for mime in (self.tokenizes or [])]
        mt = (mime_type or "").strip().lower()
        if mt.endswith("/*") and "/" in mt:
            prefix = mt.split("/", 1)[0] + "/"
            if any(supported.startswith(prefix) for supported in tokenizes):
                return True

        normalized = normalize_mime_type(mime_type)
        provider_override = self._supports_provider_document_mime(
            normalized,
            resource_source=resource_source,
        )
        if provider_override is not None:
            return provider_override
        if normalized and normalized.lower() in tokenizes:
            return True

        return ModelDatabase.supports_mime(
            self.name,
            mime_type,
            provider=self.provider,
            resource_source=resource_source,
        )

    def supports_any_mime(
        self,
        mime_types: list[str],
        *,
        resource_source: ResourceSource | None = None,
    ) -> bool:
        return any(
            self.supports_mime(mime_type, resource_source=resource_source)
            for mime_type in mime_types
        )

    @property
    def supports_text(self) -> bool:
        return self.supports_mime("text/plain")

    @property
    def supports_document(self) -> bool:
        return self.supports_any_mime(list(DOCUMENT_MIME_TYPES))

    @property
    def supports_vision(self) -> bool:
        return self.supports_any_mime(["image/jpeg", "image/png", "image/webp"])

    @property
    def tdv_flags(self) -> tuple[bool, bool, bool]:
        """Convenience tuple: (text, document, vision)."""
        return (self.supports_text, self.supports_document, self.supports_vision)

    @classmethod
    def from_llm(cls, llm: "FastAgentLLMProtocol") -> "ModelInfo" | None:
        """Build ModelInfo from an LLM instance.

        Delegates to ``llm.model_info`` so that provider-level overrides
        for explicit extended-context requests are reflected automatically.
        """
        return llm.model_info

    @classmethod
    def from_resolved_model(
        cls,
        resolved_model: "ResolvedModelSpec",
        *,
        context_window_override: int | None = None,
    ) -> "ModelInfo" | None:
        """Build ModelInfo from a resolved model specification."""
        return resolved_model.build_model_info(context_window_override=context_window_override)

    @classmethod
    def from_name(cls, name: str, provider: Provider | None = None) -> "ModelInfo" | None:
        canonical_name = ModelFactory.MODEL_PRESETS.get(name, name)
        params = ModelDatabase.get_model_params(canonical_name, provider=provider)
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
            provider = ModelDatabase.get_default_provider(canonical_name) or Provider.GENERIC

        return ModelInfo(
            name=canonical_name,
            provider=provider,
            context_window=params.context_window,
            max_output_tokens=params.max_output_tokens,
            tokenizes=params.tokenizes,
            json_mode=params.json_mode,
            reasoning=params.reasoning,
        )
