"""Backward-compatible Hugging Face bearer token verifier import."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.mcp.auth.providers._huggingface_compat import (
    DEFAULT_HUGGINGFACE_SCOPES,
)
from fast_agent.mcp.auth.providers._huggingface_compat import (
    HuggingFaceTokenVerifier as _FastMCPHuggingFaceTokenVerifier,
)

if TYPE_CHECKING:
    import httpx


class HuggingFaceTokenVerifier(_FastMCPHuggingFaceTokenVerifier):
    """FastMCP-compatible Hugging Face verifier with legacy constructor args."""

    def __init__(
        self,
        provider: str = "huggingface",
        scopes: list[str] | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: int = 10,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        del base_url
        super().__init__(required_scopes=None, timeout_seconds=timeout_seconds, http_client=http_client)
        self.provider = provider
        self.scopes = scopes or list(DEFAULT_HUGGINGFACE_SCOPES)


__all__ = ["HuggingFaceTokenVerifier"]
