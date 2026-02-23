"""Shared MCP runtime connect target parsing and resolution helpers."""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlparse

from fast_agent.cli.commands.url_parser import parse_server_urls

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings


_FAST_AGENT_CONNECT_FLAG_NAMES: tuple[str, ...] = (
    "--auth",
    "--oauth",
    "--no-oauth",
    "--timeout",
    "--name",
    "-n",
    "--reconnect",
    "--no-reconnect",
)


def _find_embedded_fast_agent_flag_for_url_target(target_text: str) -> str | None:
    """Return the first fast-agent CLI flag embedded in a URL target string."""
    if infer_connect_mode(target_text) != "url":
        return None

    try:
        tokens = shlex.split(target_text)
    except ValueError:
        # Preserve existing error behavior for invalid shell quoting.
        return None

    if len(tokens) <= 1:
        return None

    for token in tokens[1:]:
        if token in _FAST_AGENT_CONNECT_FLAG_NAMES:
            return token
        if token.startswith("--auth="):
            return "--auth"
        if token.startswith("--timeout="):
            return "--timeout"
        if token.startswith("--name="):
            return "--name"

    return None


def _build_url_target_flag_error(*, source_path: str, flag: str) -> str:
    if flag == "--auth":
        return (
            f"`{source_path}` must be a pure target string. "
            "Move --auth to `headers` or `auth` settings."
        )
    return (
        f"`{source_path}` must be a pure target string. "
        "Move fast-agent flags to structured server settings."
    )


def infer_connect_mode(target_text: str) -> str:
    """Infer runtime connect mode from a target string."""
    stripped = target_text.strip()
    if stripped.startswith(("http://", "https://")):
        return "url"
    if stripped.startswith("@"):
        return "npx"
    if stripped.startswith("npx "):
        return "npx"
    if stripped.startswith("uvx "):
        return "uvx"
    return "stdio"


def _slugify_server_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return normalized or "mcp-server"


def infer_server_name(target_text: str, mode: str | None = None) -> str:
    """Infer an MCP server name from runtime connect target text."""
    resolved_mode = mode or infer_connect_mode(target_text)
    tokens = shlex.split(target_text)
    if resolved_mode == "url":
        parsed = urlparse(target_text)
        if parsed.hostname:
            return _slugify_server_name(parsed.hostname)

    if resolved_mode in {"npx", "uvx"} and tokens:
        if tokens[0].startswith("@"):
            package = tokens[0]
        elif len(tokens) >= 2:
            package = tokens[1]
        else:
            package = tokens[0]

        if package.startswith("@"):
            package = package.rsplit("@", 1)[0] if package.count("@") > 1 else package
        else:
            package = package.split("@", 1)[0]
        package = package.rsplit("/", 1)[-1]
        return _slugify_server_name(package)

    if tokens:
        return _slugify_server_name(tokens[0].rsplit("/", 1)[-1])

    return "mcp-server"


def build_server_config_from_target(
    target_text: str,
    *,
    server_name: str | None = None,
    auth_token: str | None = None,
) -> tuple[str, MCPServerSettings]:
    """Build a runtime MCP server configuration from a connect target string."""
    from fast_agent.config import MCPServerSettings

    normalized_target = target_text.strip()
    if not normalized_target:
        raise ValueError("Connection target is required")

    mode = infer_connect_mode(normalized_target)
    resolved_name = (server_name or infer_server_name(normalized_target, mode)).strip()
    if not resolved_name:
        raise ValueError("Server name could not be resolved from connection target")

    if mode == "url":
        parsed_urls = parse_server_urls(normalized_target, auth_token=auth_token)
        if not parsed_urls:
            raise ValueError("Connection target is required")
        _parsed_name, transport, parsed_url, headers = parsed_urls[0]
        return resolved_name, MCPServerSettings(
            name=resolved_name,
            transport=transport,
            url=parsed_url,
            headers=headers,
        )

    tokens = shlex.split(normalized_target)
    if not tokens:
        raise ValueError("Connection target is required")

    if mode == "npx" and tokens[0].startswith("@"):
        return resolved_name, MCPServerSettings(
            name=resolved_name,
            transport="stdio",
            command="npx",
            args=tokens,
        )

    return resolved_name, MCPServerSettings(
        name=resolved_name,
        transport="stdio",
        command=tokens[0],
        args=tokens[1:],
    )


def resolve_target_entry(
    target: str,
    *,
    default_name: str | None,
    overrides: Mapping[str, Any],
    source_path: str,
) -> tuple[str, MCPServerSettings]:
    """Resolve target shorthand + explicit overrides into canonical settings."""
    from fast_agent.config import MCPServerSettings

    normalized_target = target.strip()
    if not normalized_target:
        raise ValueError(f"`{source_path}` must be a non-empty string")

    embedded_flag = _find_embedded_fast_agent_flag_for_url_target(normalized_target)
    if embedded_flag is not None:
        raise ValueError(_build_url_target_flag_error(source_path=source_path, flag=embedded_flag))

    resolved_default_name = default_name.strip() if isinstance(default_name, str) else None
    resolved_name, derived_settings = build_server_config_from_target(
        normalized_target,
        server_name=resolved_default_name,
    )

    merged_payload: dict[str, Any] = derived_settings.model_dump(mode="python")
    merged_payload.update(dict(overrides))

    final_name_value = merged_payload.get("name")
    if isinstance(final_name_value, str) and final_name_value.strip():
        final_name = final_name_value.strip()
    elif resolved_default_name:
        final_name = resolved_default_name
    else:
        final_name = resolved_name
    merged_payload["name"] = final_name

    resolved_settings = MCPServerSettings.model_validate(merged_payload)
    return final_name, resolved_settings
