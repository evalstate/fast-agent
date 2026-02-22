"""Runtime MCP connect/list/disconnect command handlers."""

from __future__ import annotations

import math
import os
import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.mcp.connect_targets import (
    build_server_config_from_target,
    infer_server_name,
)
from fast_agent.mcp.connect_targets import (
    infer_connect_mode as infer_connect_mode_shared,
)
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.oauth_client import OAuthEvent


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
    auth_token: str | None


_AUTH_ENV_BRACED_RE = re.compile(
    r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>.*))?\}$"
)
_AUTH_ENV_SIMPLE_RE = re.compile(r"^\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")


def _resolve_auth_token_value(raw_value: str) -> str:
    """Resolve --auth values that reference environment variables.

    Supported forms:
    - ``$VAR``
    - ``${VAR}``
    - ``${VAR:default}``
    """

    match = _AUTH_ENV_BRACED_RE.match(raw_value)
    if match:
        env_name = match.group("name")
        default = match.group("default")
        resolved = os.environ.get(env_name)
        if resolved is not None:
            return resolved
        if default is not None:
            return default
        raise ValueError(f"Environment variable '{env_name}' is not set for --auth")

    match = _AUTH_ENV_SIMPLE_RE.match(raw_value)
    if match:
        env_name = match.group("name")
        resolved = os.environ.get(env_name)
        if resolved is None:
            raise ValueError(f"Environment variable '{env_name}' is not set for --auth")
        return resolved

    return raw_value


def infer_connect_mode(target_text: str) -> str:
    return infer_connect_mode_shared(target_text)


def _infer_server_name(target_text: str, mode: str) -> str:
    """Backward-compatible private wrapper used by interactive UI code."""
    return infer_server_name(target_text, mode)


def _rebuild_target_text(tokens: list[str]) -> str:
    """Rebuild target text while preserving whitespace grouping for later shlex parsing."""
    if not tokens:
        return ""

    rebuilt_parts: list[str] = []
    for token in tokens:
        if token == "" or any(char.isspace() for char in token):
            rebuilt_parts.append(shlex.quote(token))
        else:
            rebuilt_parts.append(token)
    return " ".join(rebuilt_parts)


def parse_connect_input(target_text: str) -> ParsedMcpConnectInput:
    tokens = shlex.split(target_text)
    target_tokens: list[str] = []
    server_name: str | None = None
    timeout_seconds: float | None = None
    trigger_oauth: bool | None = None
    reconnect_on_disconnect: bool | None = None
    force_reconnect = False
    auth_token: str | None = None

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
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError(
                    "Invalid value for --timeout: expected a finite number greater than 0"
                )
        elif token == "--oauth":
            trigger_oauth = True
        elif token == "--no-oauth":
            trigger_oauth = False
        elif token == "--reconnect":
            force_reconnect = True
        elif token == "--no-reconnect":
            reconnect_on_disconnect = False
        elif token == "--auth":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("Missing value for --auth")
            auth_token = _resolve_auth_token_value(tokens[idx])
        elif token.startswith("--auth="):
            auth_token = token.split("=", 1)[1]
            if not auth_token:
                raise ValueError("Missing value for --auth")
            auth_token = _resolve_auth_token_value(auth_token)
        else:
            target_tokens.append(token)
        idx += 1

    normalized_target = _rebuild_target_text(target_tokens).strip()
    if not normalized_target:
        raise ValueError("Connection target is required")

    return ParsedMcpConnectInput(
        target_text=normalized_target,
        server_name=server_name,
        timeout_seconds=timeout_seconds,
        trigger_oauth=trigger_oauth,
        reconnect_on_disconnect=reconnect_on_disconnect,
        force_reconnect=force_reconnect,
        auth_token=auth_token,
    )


def _build_server_config(
    target_text: str,
    server_name: str,
    *,
    auth_token: str | None = None,
) -> tuple[str, MCPServerSettings]:
    return build_server_config_from_target(
        target_text,
        server_name=server_name,
        auth_token=auth_token,
    )


async def _resolve_configured_server_alias(
    *,
    manager: McpRuntimeManager,
    agent_name: str,
    target_text: str,
    explicit_server_name: str | None,
    auth_token: str | None,
) -> str | None:
    """Return configured server name when target text is an alias.

    We treat a single stdio token as a server alias only when no explicit
    --name override or URL auth token is provided.
    """

    if explicit_server_name is not None or auth_token is not None:
        return None

    if infer_connect_mode(target_text) != "stdio":
        return None

    tokens = shlex.split(target_text)
    if len(tokens) != 1:
        return None

    candidate = tokens[0]
    if not candidate or candidate.startswith("-"):
        return None

    configured_names: set[str] = set()
    try:
        configured_names.update(await manager.list_configured_detached_mcp_servers(agent_name))
    except Exception:
        pass

    try:
        configured_names.update(await manager.list_attached_mcp_servers(agent_name))
    except Exception:
        pass

    return candidate if candidate in configured_names else None


def _format_added_summary(tools_added_count: int, prompts_added_count: int) -> Text:
    tool_word = "tool" if tools_added_count == 1 else "tools"
    prompt_word = "prompt" if prompts_added_count == 1 else "prompts"

    summary = Text()
    summary.append("Added ", style="dim")
    summary.append(str(tools_added_count), style="bold bright_cyan")
    summary.append(f" {tool_word} and ", style="dim")
    summary.append(str(prompts_added_count), style="bold bright_cyan")
    summary.append(f" {prompt_word}.", style="dim")
    return summary


def _format_refreshed_summary(
    *,
    tools_refreshed_count: int,
    prompts_refreshed_count: int,
    tools_added_count: int,
    prompts_added_count: int,
) -> Text:
    tool_word = "tool" if tools_refreshed_count == 1 else "tools"
    prompt_word = "prompt" if prompts_refreshed_count == 1 else "prompts"
    new_count = tools_added_count + prompts_added_count

    summary = Text()
    summary.append("Refreshed ", style="dim")
    summary.append(str(tools_refreshed_count), style="bold bright_cyan")
    summary.append(f" {tool_word} and ", style="dim")
    summary.append(str(prompts_refreshed_count), style="bold bright_cyan")
    summary.append(f" {prompt_word} (", style="dim")
    summary.append(str(new_count), style="bold bright_cyan")
    summary.append(" new).", style="dim")
    return summary


def _format_removed_summary(tools_removed_count: int, prompts_removed_count: int) -> Text:
    tool_word = "tool" if tools_removed_count == 1 else "tools"
    prompt_word = "prompt" if prompts_removed_count == 1 else "prompts"

    summary = Text()
    summary.append("Removed ", style="dim")
    summary.append(str(tools_removed_count), style="bold bright_cyan")
    summary.append(f" {tool_word} and ", style="dim")
    summary.append(str(prompts_removed_count), style="bold bright_cyan")
    summary.append(f" {prompt_word}.", style="dim")
    return summary


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
    on_progress: Callable[[str], Awaitable[None]] | None = None,
    on_oauth_event: Callable[[OAuthEvent], Awaitable[None]] | None = None,
) -> CommandOutcome:
    del ctx
    outcome = CommandOutcome()

    async def emit_progress(message: str) -> None:
        if on_progress is None:
            return
        try:
            await on_progress(message)
        except Exception:
            return

    await emit_progress("Preparing MCP connection…")

    oauth_links_seen: set[str] = set()
    oauth_links_ordered: list[str] = []
    oauth_paste_fallback_enabled = on_progress is None and on_oauth_event is None

    async def emit_oauth_event(event: OAuthEvent) -> None:
        if on_oauth_event is not None:
            try:
                await on_oauth_event(event)
            except Exception:
                pass

        if event.event_type == "authorization_url" and event.url:
            if event.url not in oauth_links_seen:
                oauth_links_seen.add(event.url)
                oauth_links_ordered.append(event.url)
                await emit_progress(f"Open this link to authorize: {event.url}")
            return

        if event.event_type == "wait_start":
            await emit_progress(event.message or "Waiting for OAuth callback (startup timer paused)…")
            return

        if event.event_type == "wait_end":
            await emit_progress(event.message or "OAuth callback wait complete.")
            return

        if event.event_type == "callback_received":
            await emit_progress(event.message or "OAuth callback received. Completing token exchange…")
            return

        if event.event_type == "oauth_error" and event.message:
            await emit_progress(f"OAuth status: {event.message}")

    try:
        parsed = parse_connect_input(target_text)
    except ValueError as exc:
        outcome.add_message(f"Invalid MCP connect arguments: {exc}", channel="error")
        return outcome

    configured_alias = await _resolve_configured_server_alias(
        manager=manager,
        agent_name=agent_name,
        target_text=parsed.target_text,
        explicit_server_name=parsed.server_name,
        auth_token=parsed.auth_token,
    )

    mode = "configured" if configured_alias is not None else infer_connect_mode(parsed.target_text)
    server_name = configured_alias or parsed.server_name or infer_server_name(parsed.target_text, mode)
    await emit_progress(f"Connecting MCP server '{server_name}' via {mode}…")

    trigger_oauth = True if parsed.trigger_oauth is None else parsed.trigger_oauth
    startup_timeout_seconds = parsed.timeout_seconds
    if startup_timeout_seconds is None:
        # OAuth-backed URL servers often need additional non-callback time for
        # metadata discovery and token exchange after the browser callback.
        startup_timeout_seconds = 30.0 if (mode == "url" and trigger_oauth) else 10.0

    try:
        config: MCPServerSettings | None
        if configured_alias is not None:
            config = None
        else:
            server_name, config = _build_server_config(
                parsed.target_text,
                server_name,
                auth_token=parsed.auth_token,
            )
        attach_options = MCPAttachOptions(
            startup_timeout_seconds=startup_timeout_seconds,
            trigger_oauth=trigger_oauth,
            force_reconnect=parsed.force_reconnect,
            reconnect_on_disconnect=parsed.reconnect_on_disconnect,
            oauth_event_handler=emit_oauth_event
            if (on_progress is not None or on_oauth_event is not None)
            else None,
            allow_oauth_paste_fallback=oauth_paste_fallback_enabled,
        )
        result = await manager.attach_mcp_server(
            agent_name,
            server_name,
            server_config=config,
            options=attach_options,
        )
    except Exception as exc:
        await emit_progress(f"Failed to connect MCP server '{server_name}'.")
        error_text = str(exc)
        outcome.add_message(f"Failed to connect MCP server: {error_text}", channel="error")

        normalized_error = error_text.lower()
        oauth_related = "oauth" in normalized_error
        oauth_registration_404 = (
            "oauth" in normalized_error and "registration failed: 404" in normalized_error
        )
        fallback_disabled = (
            "paste fallback is disabled" in normalized_error
            or "non-interactive connection mode" in normalized_error
        )
        oauth_timeout = "oauth" in normalized_error and "time" in normalized_error

        if oauth_registration_404:
            outcome.add_message(
                (
                    "OAuth client registration returned HTTP 404. "
                    "This server likely does not allow dynamic client registration."
                ),
                channel="warning",
                right_info="mcp",
                agent_name=agent_name,
            )
            outcome.add_message(
                (
                    "Try either `--client-metadata-url <https-url>` (CIMD) "
                    "or connect with bearer auth via `--auth <token>`."
                ),
                channel="info",
                right_info="mcp",
                agent_name=agent_name,
            )
            if "githubcopilot.com" in normalized_error:
                outcome.add_message(
                    (
                        "For GitHub Copilot MCP, token auth is commonly required. "
                        "Try `--auth $GITHUB_TOKEN`."
                    ),
                    channel="info",
                    right_info="mcp",
                    agent_name=agent_name,
                )

        if oauth_related and (fallback_disabled or oauth_timeout or not oauth_paste_fallback_enabled):
            outcome.add_message(
                (
                    "OAuth could not be completed in this connection mode. "
                    "Run `fast-agent auth login <server-name-or-identity>` on the fast-agent host, "
                    "then retry `/mcp connect ...`."
                ),
                channel="warning",
                right_info="mcp",
                agent_name=agent_name,
            )
            outcome.add_message(
                (
                    "To cancel an in-flight ACP connection, use your client's Stop/Cancel control "
                    "(ACP `session/cancel`)."
                ),
                channel="info",
                right_info="mcp",
                agent_name=agent_name,
            )

        return outcome

    tools_added = getattr(result, "tools_added", [])
    prompts_added = getattr(result, "prompts_added", [])
    tools_total = getattr(result, "tools_total", None)
    prompts_total = getattr(result, "prompts_total", None)
    warnings = getattr(result, "warnings", [])
    already_attached = bool(getattr(result, "already_attached", False))

    tools_added_count = len(tools_added)
    prompts_added_count = len(prompts_added)
    tools_refreshed_count = (
        tools_total
        if isinstance(tools_total, int) and tools_total >= 0
        else tools_added_count
    )
    prompts_refreshed_count = (
        prompts_total
        if isinstance(prompts_total, int) and prompts_total >= 0
        else prompts_added_count
    )

    if already_attached and not parsed.force_reconnect:
        outcome.add_message(
            (
                f"MCP server '{server_name}' is already attached. "
                "Use --reconnect to force reconnect and refresh tools."
            ),
            channel="warning",
            right_info="mcp",
            agent_name=agent_name,
        )
        await emit_progress(f"MCP server '{server_name}' is already connected.")
    else:
        action = "Reconnected" if already_attached and parsed.force_reconnect else "Connected"
        outcome.add_message(
            f"{action} MCP server '{server_name}' ({mode}).",
            right_info="mcp",
            agent_name=agent_name,
        )
        if action == "Reconnected":
            outcome.add_message(
                _format_refreshed_summary(
                    tools_refreshed_count=tools_refreshed_count,
                    prompts_refreshed_count=prompts_refreshed_count,
                    tools_added_count=tools_added_count,
                    prompts_added_count=prompts_added_count,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
        else:
            outcome.add_message(
                _format_added_summary(
                    tools_added_count=tools_added_count,
                    prompts_added_count=prompts_added_count,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
        await emit_progress(f"{action} MCP server '{server_name}'.")
    for warning in warnings:
        outcome.add_message(warning, channel="warning", right_info="mcp", agent_name=agent_name)

    if oauth_links_ordered:
        outcome.add_message(
            f"OAuth authorization link: {oauth_links_ordered[-1]}",
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
    outcome.add_message(
        _format_removed_summary(
            tools_removed_count=len(tools_removed),
            prompts_removed_count=len(prompts_removed),
        ),
        right_info="mcp",
        agent_name=agent_name,
    )

    return outcome
