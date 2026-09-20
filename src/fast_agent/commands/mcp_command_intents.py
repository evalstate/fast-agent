"""Shared MCP command-intent parsing across TUI and ACP surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fast_agent.utils.text import strip_to_none

McpTopLevelAction = Literal[
    "error",
    "auth",
    "list",
    "status",
    "attach",
    "connect",
    "disconnect",
    "reconnect",
    "cache",
    "refresh",
]
McpServerNameAction = Literal["attach", "disconnect", "reconnect"]

MCP_TOP_LEVEL_ACTIONS: tuple[McpTopLevelAction, ...] = (
    "cache",
    "refresh",
    "list",
    "status",
    "error",
    "auth",
    "attach",
    "connect",
    "disconnect",
    "reconnect",
)
MCP_SERVER_NAME_ACTIONS: tuple[McpServerNameAction, ...] = (
    "attach",
    "disconnect",
    "reconnect",
)
MCP_TOP_LEVEL_ACTION_DESCRIPTIONS: dict[str, str] = {
    "cache": "Show or clear tool caches",
    "refresh": "Refresh tool catalogs from servers",
    "error": "Show startup failures and recovery guidance",
    "auth": "Show authentication diagnostics and recovery guidance",
    "list": "List configured and attached MCP servers",
    "status": "Show detailed MCP server status",
    "attach": "Attach a configured MCP server",
    "connect": "Connect an ad-hoc MCP target",
    "disconnect": "Disconnect an attached MCP server",
    "reconnect": "Reconnect an attached MCP server",
}


@dataclass(frozen=True, slots=True)
class McpServerNameIntent:
    server_name: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class McpNoArgsIntent:
    error: str | None


def is_mcp_top_level_action(action: str) -> bool:
    return action in MCP_TOP_LEVEL_ACTIONS


def is_mcp_server_name_action(action: str) -> bool:
    return action in MCP_SERVER_NAME_ACTIONS


def parse_mcp_server_name_tokens(tokens: list[str], *, usage: str) -> McpServerNameIntent:
    if len(tokens) != 2:
        return McpServerNameIntent(server_name=None, error=usage)
    server_name = strip_to_none(tokens[1])
    if server_name is None:
        return McpServerNameIntent(server_name=None, error=usage)
    return McpServerNameIntent(server_name=server_name, error=None)


def parse_mcp_no_args_tokens(tokens: list[str], *, usage: str) -> McpNoArgsIntent:
    if len(tokens) != 1:
        return McpNoArgsIntent(error=usage)
    return McpNoArgsIntent(error=None)


@dataclass(frozen=True, slots=True)
class McpCacheIntent:
    action: Literal["summary", "clear", "refresh"]
    server_name: str | None = None
    error: str | None = None


def parse_mcp_cache_tokens(tokens: list[str]) -> McpCacheIntent:
    tokens = [tokens[0].lower(), *tokens[1:]] if tokens else []
    usage = "Usage: /mcp cache [clear [server|all]] or /mcp refresh [server|all]"
    if tokens == ["cache"]:
        return McpCacheIntent("summary")
    if tokens and tokens[0] == "refresh" and len(tokens) <= 2:
        action = "refresh"
        targets = tokens[1:]
    elif tokens[:2] == ["cache", "clear"] and len(tokens) <= 3:
        action = "clear"
        targets = tokens[2:]
    else:
        return McpCacheIntent("summary", error=usage)
    target = targets[0] if targets else None
    if target is not None and not target.strip():
        return McpCacheIntent("summary", error=usage)
    return McpCacheIntent(action, None if target == "all" else target)
