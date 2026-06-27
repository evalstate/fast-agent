"""FastMCP-compatible auth providers used by fast-agent."""

from fast_agent.mcp.auth.providers.huggingface import (
    DEFAULT_HUGGINGFACE_SCOPES,
    HuggingFaceProvider,
    HuggingFaceTokenVerifier,
)

__all__ = [
    "DEFAULT_HUGGINGFACE_SCOPES",
    "HuggingFaceProvider",
    "HuggingFaceTokenVerifier",
]
