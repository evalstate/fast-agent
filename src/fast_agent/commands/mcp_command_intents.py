"""Shared MCP command-intent parsing across TUI and ACP surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fast_agent.utils.text import strip_to_none

McpTopLevelAction = Literal["list", "status", "attach", "connect", "disconnect", "reconnect"]
McpServerNameAction = Literal["attach", "disconnect", "reconnect"]

MCP_TOP_LEVEL_ACTIONS: tuple[McpTopLevelAction, ...] = (
    "list",
    "status",
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
