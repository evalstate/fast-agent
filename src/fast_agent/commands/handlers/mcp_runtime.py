"""Runtime MCP connect/list/disconnect command handlers."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from fast_agent.commands.results import CommandOutcome
from fast_agent.config import MCPServerSettings
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions


class McpRuntimeManager(Protocol):
    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> object: ...

    async def detach_mcp_server(self, agent_name: str, server_name: str) -> object: ...

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]: ...

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class ParsedMcpConnectInput:
    target_text: str
    server_name: str | None
    timeout_seconds: float | None
    trigger_oauth: bool | None
    reconnect_on_disconnect: bool | None
    force_reconnect: bool


def infer_connect_mode(target_text: str) -> str:
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


def _infer_server_name(target_text: str, mode: str) -> str:
    tokens = shlex.split(target_text)
    if mode == "url":
        parsed = urlparse(target_text)
        if parsed.hostname:
            return _slugify_server_name(parsed.hostname)
    if mode in {"npx", "uvx"} and tokens:
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


def parse_connect_input(target_text: str) -> ParsedMcpConnectInput:
    tokens = shlex.split(target_text)
    target_tokens: list[str] = []
    server_name: str | None = None
    timeout_seconds: float | None = None
    trigger_oauth: bool | None = None
    reconnect_on_disconnect: bool | None = None
    force_reconnect = False

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--name", "-n"}:
            idx += 1
            if idx >= len(tokens):
                raise ValueError("Missing value for --name")
            server_name = tokens[idx]
        elif token == "--timeout":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("Missing value for --timeout")
            timeout_seconds = float(tokens[idx])
        elif token == "--oauth":
            trigger_oauth = True
        elif token == "--no-oauth":
            trigger_oauth = False
        elif token == "--reconnect":
            force_reconnect = True
        elif token == "--no-reconnect":
            reconnect_on_disconnect = False
        else:
            target_tokens.append(token)
        idx += 1

    normalized_target = " ".join(target_tokens).strip()
    if not normalized_target:
        raise ValueError("Connection target is required")

    return ParsedMcpConnectInput(
        target_text=normalized_target,
        server_name=server_name,
        timeout_seconds=timeout_seconds,
        trigger_oauth=trigger_oauth,
        reconnect_on_disconnect=reconnect_on_disconnect,
        force_reconnect=force_reconnect,
    )


def _build_server_config(target_text: str, server_name: str) -> MCPServerSettings:
    mode = infer_connect_mode(target_text)
    if mode == "url":
        return MCPServerSettings(name=server_name, transport="http", url=target_text)

    tokens = shlex.split(target_text)
    if not tokens:
        raise ValueError("Connection target is required")

    if mode == "npx" and tokens[0].startswith("@"):
        return MCPServerSettings(
            name=server_name,
            transport="stdio",
            command="npx",
            args=tokens,
        )

    return MCPServerSettings(
        name=server_name,
        transport="stdio",
        command=tokens[0],
        args=tokens[1:],
    )


async def handle_mcp_list(ctx, *, manager: McpRuntimeManager, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    attached = await manager.list_attached_mcp_servers(agent_name)
    detached: list[str] = []
    try:
        detached = await manager.list_configured_detached_mcp_servers(agent_name)
    except Exception:
        detached = []

    if not attached:
        outcome.add_message("No MCP servers attached.", channel="warning", right_info="mcp")
    else:
        outcome.add_message(
            "Attached MCP servers: " + ", ".join(attached),
            right_info="mcp",
            agent_name=agent_name,
        )

    if detached:
        outcome.add_message(
            "Configured but detached: " + ", ".join(detached),
            channel="info",
            right_info="mcp",
            agent_name=agent_name,
        )

    return outcome


async def handle_mcp_connect(
    ctx,
    *,
    manager: McpRuntimeManager,
    agent_name: str,
    target_text: str,
) -> CommandOutcome:
    del ctx
    outcome = CommandOutcome()
    try:
        parsed = parse_connect_input(target_text)
    except ValueError as exc:
        outcome.add_message(f"Invalid MCP connect arguments: {exc}", channel="error")
        return outcome

    mode = infer_connect_mode(parsed.target_text)
    server_name = parsed.server_name or _infer_server_name(parsed.target_text, mode)

    try:
        config = _build_server_config(parsed.target_text, server_name)
        attach_options = MCPAttachOptions(
            startup_timeout_seconds=parsed.timeout_seconds or 10.0,
            trigger_oauth=True if parsed.trigger_oauth is None else parsed.trigger_oauth,
            force_reconnect=parsed.force_reconnect,
            reconnect_on_disconnect=parsed.reconnect_on_disconnect,
        )
        result = await manager.attach_mcp_server(
            agent_name,
            server_name,
            server_config=config,
            options=attach_options,
        )
    except Exception as exc:
        outcome.add_message(f"Failed to connect MCP server: {exc}", channel="error")
        return outcome

    tools_added = getattr(result, "tools_added", [])
    prompts_added = getattr(result, "prompts_added", [])
    warnings = getattr(result, "warnings", [])
    already_attached = bool(getattr(result, "already_attached", False))

    if already_attached:
        outcome.add_message(
            (
                f"MCP server '{server_name}' is already attached. "
                "Use --reconnect to force reconnect and refresh tools."
            ),
            channel="warning",
            right_info="mcp",
            agent_name=agent_name,
        )
    else:
        outcome.add_message(
            f"Connected MCP server '{server_name}' ({mode}).",
            right_info="mcp",
            agent_name=agent_name,
        )
    if tools_added:
        outcome.add_message(
            "Tools added: " + ", ".join(tools_added),
            channel="info",
            right_info="mcp",
            agent_name=agent_name,
        )
    if prompts_added:
        outcome.add_message(
            "Prompts added: " + ", ".join(prompts_added),
            channel="info",
            right_info="mcp",
            agent_name=agent_name,
        )
    for warning in warnings:
        outcome.add_message(warning, channel="warning", right_info="mcp", agent_name=agent_name)

    detached = await manager.list_configured_detached_mcp_servers(agent_name)
    if detached:
        outcome.add_message(
            "Configured but detached: " + ", ".join(detached),
            channel="info",
            right_info="mcp",
            agent_name=agent_name,
        )

    return outcome


async def handle_mcp_disconnect(
    ctx,
    *,
    manager: McpRuntimeManager,
    agent_name: str,
    server_name: str,
) -> CommandOutcome:
    del ctx
    outcome = CommandOutcome()
    try:
        result = await manager.detach_mcp_server(agent_name, server_name)
    except Exception as exc:
        outcome.add_message(f"Failed to disconnect MCP server: {exc}", channel="error")
        return outcome

    detached = bool(getattr(result, "detached", False))
    if not detached:
        outcome.add_message(
            f"MCP server '{server_name}' was not attached.",
            channel="warning",
            right_info="mcp",
            agent_name=agent_name,
        )
        return outcome

    tools_removed = getattr(result, "tools_removed", [])
    prompts_removed = getattr(result, "prompts_removed", [])

    outcome.add_message(
        f"Disconnected MCP server '{server_name}'.",
        right_info="mcp",
        agent_name=agent_name,
    )
    if tools_removed:
        outcome.add_message(
            "Tools removed: " + ", ".join(tools_removed),
            channel="info",
            right_info="mcp",
            agent_name=agent_name,
        )
    if prompts_removed:
        outcome.add_message(
            "Prompts removed: " + ", ".join(prompts_removed),
            channel="info",
            right_info="mcp",
            agent_name=agent_name,
        )

    return outcome
