"""Authentication modules for MCP server."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "HFAuthHeaderMiddleware",
    "HuggingFaceOAuthOrHubTokenVerifier",
    "request_bearer_token",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "HFAuthHeaderMiddleware": ("fast_agent.mcp.auth.middleware", "HFAuthHeaderMiddleware"),
    "HuggingFaceOAuthOrHubTokenVerifier": (
        "fast_agent.mcp.auth.huggingface",
        "HuggingFaceOAuthOrHubTokenVerifier",
    ),
    "request_bearer_token": ("fast_agent.mcp.auth.context", "request_bearer_token"),
}

if TYPE_CHECKING:
    from fast_agent.mcp.auth.context import request_bearer_token as request_bearer_token
    from fast_agent.mcp.auth.huggingface import (
        HuggingFaceOAuthOrHubTokenVerifier as HuggingFaceOAuthOrHubTokenVerifier,
    )
    from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware as HFAuthHeaderMiddleware


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
