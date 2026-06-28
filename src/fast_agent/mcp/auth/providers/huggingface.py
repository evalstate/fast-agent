"""Hugging Face auth provider shim.

Prefer FastMCP's native provider when available. Until fast-agent's minimum
FastMCP version includes it, fall back to a compatibility copy with the same
public shape.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

try:
    _native = import_module("fastmcp.server.auth.providers.huggingface")
except ImportError:
    from fast_agent.mcp.auth.providers._huggingface_compat import (
        DEFAULT_HUGGINGFACE_SCOPES,
        HUGGINGFACE_AUTHORIZATION_ENDPOINT,
        HUGGINGFACE_TOKEN_ENDPOINT,
        HUGGINGFACE_USERINFO_ENDPOINT,
        HUGGINGFACE_WHOAMI_ENDPOINT,
        HuggingFaceProvider,
        HuggingFaceTokenVerifier,
    )
else:
    DEFAULT_HUGGINGFACE_SCOPES = cast(
        "list[str]",
        _native.DEFAULT_HUGGINGFACE_SCOPES,
    )
    HUGGINGFACE_AUTHORIZATION_ENDPOINT = cast(
        "str",
        _native.HUGGINGFACE_AUTHORIZATION_ENDPOINT,
    )
    HUGGINGFACE_TOKEN_ENDPOINT = cast("str", _native.HUGGINGFACE_TOKEN_ENDPOINT)
    HUGGINGFACE_USERINFO_ENDPOINT = cast("str", _native.HUGGINGFACE_USERINFO_ENDPOINT)
    HUGGINGFACE_WHOAMI_ENDPOINT = cast("str", _native.HUGGINGFACE_WHOAMI_ENDPOINT)
    HuggingFaceProvider = cast("type[Any]", _native.HuggingFaceProvider)
    HuggingFaceTokenVerifier = cast("type[Any]", _native.HuggingFaceTokenVerifier)

__all__ = [
    "DEFAULT_HUGGINGFACE_SCOPES",
    "HUGGINGFACE_AUTHORIZATION_ENDPOINT",
    "HUGGINGFACE_TOKEN_ENDPOINT",
    "HUGGINGFACE_USERINFO_ENDPOINT",
    "HUGGINGFACE_WHOAMI_ENDPOINT",
    "HuggingFaceProvider",
    "HuggingFaceTokenVerifier",
]
