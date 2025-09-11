"""
Typed model information helpers.

Provides a small, pythonic interface to query model/provider and
capabilities (Text/Document/Vision), backed by the model database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    # Import behind TYPE_CHECKING to avoid import cycles at runtime
    from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol


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
        return ModelDatabase.supports_mime(self.name, "text/plain")

    @property
    def supports_document(self) -> bool:
        # Document support currently keyed off PDF support
        return ModelDatabase.supports_mime(self.name, "pdf")

    @property
    def supports_vision(self) -> bool:
        # Any common image format indicates vision support
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
        params = ModelDatabase.get_model_params(name)
        if not params:
            # Unknown model: return a conservative default that supports text only.
            # This matches the desired behavior for TDV display fallbacks.
            if provider is None:
                provider = Provider.GENERIC
            return ModelInfo(
                name=name,
                provider=provider,
                context_window=None,
                max_output_tokens=None,
                tokenizes=["text/plain"],
                json_mode=None,
                reasoning=None,
            )

        return ModelInfo(
            name=name,
            provider=provider or Provider.GENERIC,
            context_window=params.context_window,
            max_output_tokens=params.max_output_tokens,
            tokenizes=params.tokenizes,
            json_mode=params.json_mode,
            reasoning=params.reasoning,
        )


def get_model_info(
    subject: Union["AgentProtocol", "FastAgentLLMProtocol", str, None],
    provider: Provider | None = None,
) -> Optional[ModelInfo]:
    """Resolve a ModelInfo from an Agent, LLM, or model name.

    Keeps the public API small while enabling type-safe access to model
    capabilities across the codebase.
    """
    if subject is None:
        return None

    # Agent → LLM
    try:
        from fast_agent.interfaces import AgentProtocol as _AgentProtocol
    except Exception:
        _AgentProtocol = None  # type: ignore

    if _AgentProtocol and isinstance(subject, _AgentProtocol):  # type: ignore[arg-type]
        return ModelInfo.from_llm(subject.llm)

    # LLM → ModelInfo
    try:
        from fast_agent.interfaces import FastAgentLLMProtocol as _LLMProtocol
    except Exception:
        _LLMProtocol = None  # type: ignore

    if _LLMProtocol and isinstance(subject, _LLMProtocol):  # type: ignore[arg-type]
        return ModelInfo.from_llm(subject)

    # String model name
    if isinstance(subject, str):
        return ModelInfo.from_name(subject, provider)

    return None
