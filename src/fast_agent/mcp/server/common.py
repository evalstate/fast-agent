"""Shared MCP server helpers."""

from __future__ import annotations

import os
from importlib.metadata import version as get_version
from typing import Literal

from fastmcp.server.auth.providers.huggingface import DEFAULT_HUGGINGFACE_SCOPES
from fastmcp.utilities.auth import parse_scopes

from fast_agent.utils.text import strip_casefold

TransportMode = Literal["http", "stdio"]


def get_fast_agent_version() -> str | None:
    for package_name in ("fast-agent-mcp", "fast-agent"):
        try:
            return get_version(package_name)
        except Exception:
            continue
    return None


def normalize_serve_oauth_provider(provider: str | None) -> str | None:
    if provider is None:
        return None

    oauth_provider = strip_casefold(provider)
    if oauth_provider in {"hf", "huggingface"}:
        return "huggingface"
    if not oauth_provider:
        return None
    return oauth_provider


def get_oauth_config() -> tuple[str | None, list[str], str]:
    oauth_provider = normalize_serve_oauth_provider(os.environ.get("FAST_AGENT_SERVE_OAUTH"))
    oauth_scopes_str = os.environ.get("FAST_AGENT_OAUTH_SCOPES") or os.environ.get(
        "OAUTH_SCOPES",
        "",
    )
    oauth_scopes = parse_scopes(oauth_scopes_str) or list(DEFAULT_HUGGINGFACE_SCOPES)
    resource_url = os.environ.get("FAST_AGENT_OAUTH_RESOURCE_URL")
    if resource_url is None and oauth_provider == "huggingface" and os.environ.get("SPACE_HOST"):
        resource_url = f"https://{os.environ['SPACE_HOST']}"
    if resource_url is None:
        resource_url = "http://localhost:8000"
    return oauth_provider, oauth_scopes, resource_url
