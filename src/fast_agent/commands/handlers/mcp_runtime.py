"""Runtime MCP connect/list/disconnect command handlers."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import get_terminal_size
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Protocol, cast

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.mcp.connect_targets import (
    build_server_config_from_target,
    infer_server_name,
)
from fast_agent.mcp.connect_targets import (
    infer_connect_mode as infer_connect_mode_shared,
)
from fast_agent.mcp.experimental_session_client import ExperimentalSessionClient, SessionJarEntry
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


class SessionClientProtocol(Protocol):
    async def list_jar(self) -> list[SessionJarEntry]: ...

    async def resolve_server_name(self, server_identifier: str | None) -> str: ...

    async def list_server_cookies(
        self, server_identifier: str | None
    ) -> tuple[str, str | None, str | None, list[dict[str, Any]]]: ...

    async def create_session(
        self,
        server_identifier: str | None,
        *,
        title: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]: ...

    async def resume_session(
        self,
        server_identifier: str | None,
        *,
        session_id: str,
    ) -> tuple[str, dict[str, Any]]: ...

    async def clear_cookie(self, server_identifier: str | None) -> str: ...

    async def clear_all_cookies(self) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class ParsedMcpConnectInput:
    target_text: str
    server_name: str | None
    timeout_seconds: float | None
    trigger_oauth: bool | None
    reconnect_on_disconnect: bool | None
    force_reconnect: bool
    auth_token: str | None


_AUTH_ENV_BRACED_RE = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>.*))?\}$")
_AUTH_ENV_SIMPLE_RE = re.compile(r"^\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")


def _normalize_auth_token_value(raw_value: str) -> str:
    """Normalize user-provided --auth values before environment lookup.

    ``--auth`` takes the raw token value. If a user passes an Authorization
    header style value (``Bearer <token>``), strip the prefix so downstream
    code can still compose a single valid ``Authorization: Bearer ...`` header.
    """

    normalized = raw_value.strip()
    if normalized.lower().startswith("bearer "):
        normalized = normalized[7:].strip()
    return normalized


def _resolve_auth_token_value(raw_value: str) -> str:
    """Resolve --auth values that reference environment variables.

    Supported forms:
    - ``$VAR``
    - ``${VAR}``
    - ``${VAR:default}``
    """

    normalized_value = _normalize_auth_token_value(raw_value)
    if not normalized_value:
        raise ValueError("Missing value for --auth")

    match = _AUTH_ENV_BRACED_RE.match(normalized_value)
    if match:
        env_name = match.group("name")
        default = match.group("default")
        resolved = os.environ.get(env_name)
        if resolved is not None:
            return resolved
        if default is not None:
            return default
        raise ValueError(f"Environment variable '{env_name}' is not set for --auth")

    match = _AUTH_ENV_SIMPLE_RE.match(normalized_value)
    if match:
        env_name = match.group("name")
        resolved = os.environ.get(env_name)
        if resolved is None:
            raise ValueError(f"Environment variable '{env_name}' is not set for --auth")
        return resolved

    return normalized_value


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


def _describe_server_config_source(server_config: Any) -> str | None:
    """Return a concise url/command description for an MCP server config."""

    if isinstance(server_config, dict):
        url_value = server_config.get("url")
        command_value = server_config.get("command")
        args_value = server_config.get("args")
    else:
        url_value = getattr(server_config, "url", None)
        command_value = getattr(server_config, "command", None)
        args_value = getattr(server_config, "args", None)

    if isinstance(url_value, str):
        url = url_value.strip()
        if url:
            return url

    if isinstance(command_value, str):
        command = command_value.strip()
        if command:
            args: list[str] = []
            if isinstance(args_value, list):
                args = [str(value) for value in args_value]
            return shlex.join([command, *args])

    return None


def _resolve_configured_source_from_context(ctx, server_name: str) -> str | None:
    """Resolve configured server description from runtime settings."""

    try:
        settings = ctx.resolve_settings()
    except Exception:
        return None

    mcp_settings = getattr(settings, "mcp", None)
    server_map = getattr(mcp_settings, "servers", None)
    if not isinstance(server_map, dict):
        return None

    server_config = server_map.get(server_name)
    if server_config is None:
        return None
    return _describe_server_config_source(server_config)


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


McpSessionAction = Literal["jar", "new", "use", "clear", "list"]


def _resolve_session_client(ctx, *, agent_name: str) -> SessionClientProtocol:
    agent = ctx.agent_provider._agent(agent_name)
    aggregator = getattr(agent, "aggregator", None)
    if aggregator is None:
        raise RuntimeError(f"Agent '{agent_name}' does not expose an MCP aggregator.")

    client = getattr(aggregator, "experimental_sessions", None)
    required_methods = (
        "list_jar",
        "resolve_server_name",
        "list_server_cookies",
        "create_session",
        "resume_session",
        "clear_cookie",
        "clear_all_cookies",
    )
    if isinstance(client, ExperimentalSessionClient) or all(
        hasattr(client, method) for method in required_methods
    ):
        return cast("SessionClientProtocol", client)

    # Backward-compatible fallback for older aggregators exposing a different property name.
    fallback = getattr(aggregator, "session_client", None)
    if isinstance(fallback, ExperimentalSessionClient) or all(
        hasattr(fallback, method) for method in required_methods
    ):
        return cast("SessionClientProtocol", fallback)

    raise RuntimeError(f"Agent '{agent_name}' does not expose MCP session controls.")


def _render_cookie(cookie: dict[str, Any] | None) -> str:
    if not cookie:
        return "null"
    return json.dumps(cookie, indent=2, sort_keys=True, ensure_ascii=False)


def _render_jar_entry(entry: SessionJarEntry) -> str:
    features = ", ".join(entry.features) if entry.features else "none"
    supported = (
        "yes" if entry.supported is True else "no" if entry.supported is False else "unknown"
    )
    mcp_name = entry.server_identity or "(unset)"
    target = entry.target or "(unset)"
    title = entry.title or "(none)"

    return (
        f"server={entry.server_name}\n"
        f"target={target}\n"
        f"session={_extract_cookie_id(entry.cookie) or '-'}\n"
        f"mcp_name={mcp_name}\n"
        f"exp_session_supported={supported}\n"
        f"features={features}\n"
        f"title={title}\n"
        f"last_used_id={entry.last_used_id or '-'}\n"
        f"session=\n{_render_cookie(entry.cookie)}"
    )


def _truncate_cell(value: str, max_len: int = 28) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def _left_truncate_with_ellipsis(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length == 1:
        return "…"
    return f"…{text[-(max_length - 1) :]}"


def _format_parent_current_path(path_text: str) -> str:
    normalized = os.path.normpath(path_text)
    path = Path(normalized)
    current = path.name or normalized
    parent = path.parent.name
    if parent:
        return f"{parent}/{current}"
    return current


def _fit_path_for_display(path_text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""

    compact = _format_parent_current_path(path_text)
    if len(compact) <= max_length:
        return compact

    current = Path(path_text).name or path_text
    if len(current) <= max_length:
        return current

    return _left_truncate_with_ellipsis(current, max_length)


def _format_target_for_display(
    target: str | None,
    *,
    width: int,
) -> tuple[str, str | None]:
    if not target:
        return "-", None

    if target.startswith("cmd:"):
        payload = target[4:].strip()
        command, separator, cwd = payload.partition(" @ ")
        command_display = f"cmd: {command}" if command else "cmd"
        if not separator or not cwd:
            return command_display, None

        path_width = max(12, width - len("cwd: "))
        return command_display, f"cwd: {_fit_path_for_display(cwd, path_width)}"

    if target.startswith("url:"):
        url = target[4:].strip()
        path_width = max(12, width)
        return f"url: {_left_truncate_with_ellipsis(url, path_width)}", None

    display_width = max(12, width)
    return _left_truncate_with_ellipsis(target, display_width), None


def _cookie_size_display(summary: dict[str, Any]) -> str:
    raw_size = summary.get("cookieSizeBytes")
    if isinstance(raw_size, int) and raw_size > 0:
        return f"{raw_size} bytes"
    return "-"


def _extract_cookie_id(cookie: dict[str, Any] | None) -> str | None:
    if not isinstance(cookie, dict):
        return None
    raw_id = cookie.get("sessionId")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    return None


def _extract_session_title(payload: dict[str, Any]) -> str:
    direct_title = payload.get("title")
    if isinstance(direct_title, str) and direct_title.strip():
        return direct_title.strip()

    data = payload.get("data")
    if isinstance(data, dict):
        data_title = data.get("title") or data.get("label")
        if isinstance(data_title, str) and data_title.strip():
            return data_title.strip()

    return "-"


def _extract_session_expiry(payload: dict[str, Any]) -> str:
    expiry = payload.get("expiresAt")
    if isinstance(expiry, str) and expiry:
        return expiry
    return "-"


def _extract_session_created(payload: dict[str, Any]) -> str:
    for key in ("created", "created_at", "createdAt"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw:
            return raw

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("created", "created_at", "createdAt"):
            raw = data.get(key)
            if isinstance(raw, str) and raw:
                return raw

    session_id = payload.get("sessionId")
    if isinstance(session_id, str):
        match = re.match(r"^(\d{10})-[A-Za-z0-9]+$", session_id)
        if match:
            token = match.group(1)
            try:
                parsed = datetime.strptime(token, "%y%m%d%H%M")
            except ValueError:
                return "-"
            return parsed.isoformat()

    return "-"


def _format_expiry_compact(expiry: str | None) -> str:
    if not expiry or expiry == "-":
        return "-"
    try:
        parsed = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
    except ValueError:
        return _truncate_cell(expiry, 14)
    return parsed.strftime("%d/%m/%y %H:%M")


def _format_session_window(start: str | None, end: str | None) -> str:
    start_display = start if start and start != "-" else "unknown"
    end_display = end if end and end != "-" else "∞"
    return f"({start_display} → {end_display})"


def _resolve_terminal_width() -> int:
    try:
        from fast_agent.ui.console import console

        width = console.size.width
    except Exception:
        width = 0
    if width <= 0:
        width = get_terminal_size(fallback=(100, 20)).columns
    return width


def _resolve_store_size_display(session_client: SessionClientProtocol) -> str:
    size_getter = getattr(session_client, "store_size_bytes", None)
    if not callable(size_getter):
        return "-"
    try:
        size = size_getter()
    except Exception:
        return "-"
    if not isinstance(size, int) or size < 0:
        return "-"
    return f"{size} bytes"


def _render_jar_table(entries: list[SessionJarEntry], *, store_size_display: str) -> Text:
    if not entries:
        return Text("No MCP session jar entries available.", style="dim")

    grouped: dict[str, list[SessionJarEntry]] = {}
    for entry in entries:
        key = entry.target or entry.server_identity or entry.server_name
        grouped.setdefault(key, []).append(entry)

    labels = sorted(grouped)
    target_word = "target" if len(labels) == 1 else "targets"
    content = Text()
    content.append(f"▎ MCP session jar ({len(labels)} {target_word}):", style="bold")
    content.append("\n\n")

    for index, label in enumerate(labels, 1):
        grouped_entries = grouped[label]
        unsupported_connected = any(
            entry.connected is True and entry.supported is False for entry in grouped_entries
        )
        primary = next(
            (
                entry
                for entry in grouped_entries
                if entry.connected is True and _extract_cookie_id(entry.cookie)
            ),
            next(
                (entry for entry in grouped_entries if _extract_cookie_id(entry.cookie)),
                grouped_entries[0],
            ),
        )

        combined_cookies: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in grouped_entries:
            for summary in entry.cookies:
                if not isinstance(summary, dict):
                    continue
                raw_id = summary.get("id")
                if not isinstance(raw_id, str) or not raw_id:
                    continue
                if raw_id in seen_ids:
                    continue
                seen_ids.add(raw_id)
                combined_cookies.append(dict(summary))

        combined_cookies.sort(key=lambda item: str(item.get("updatedAt") or ""), reverse=True)

        active_session_id = primary.last_used_id
        if active_session_id is None:
            active_session_id = next(
                (
                    item.get("id")
                    for item in combined_cookies
                    if isinstance(item.get("id"), str) and item.get("active") is True
                ),
                None,
            )

        if active_session_id is not None:
            for item in combined_cookies:
                item_id = item.get("id")
                item["active"] = isinstance(item_id, str) and item_id == active_session_id

        section_header = Text()
        section_header.append("▎ ", style="dim")
        section_header.append(primary.server_name, style="white")
        if primary.server_identity:
            section_header.append(" • ", style="dim")
            section_header.append(primary.server_identity, style="dim")
        section_header.append(" • ", style="dim")
        is_connected = any(entry.connected is True for entry in grouped_entries)
        section_header.append(
            "connected" if is_connected else "disconnected",
            style="bright_green" if is_connected else "dim",
        )
        if unsupported_connected:
            section_header.append(" • ", style="dim")
            section_header.append("unsupported", style="dim red")
        content.append_text(section_header)
        content.append("\n")

        sessions_supported: bool | None
        if unsupported_connected:
            sessions_supported = False
        else:
            sessions_supported = primary.supported

        table = _render_server_cookies_table(
            server_name=primary.server_name,
            server_identity=primary.server_identity,
            target=primary.target,
            sessions_supported=sessions_supported,
            cookies=combined_cookies,
            active_session_id=active_session_id if isinstance(active_session_id, str) else None,
            store_size_display=store_size_display,
            include_store=False,
            include_mcp_name=False,
        )
        content.append_text(table)

        if index != len(labels):
            content.append("\n")

    content.append("\n")
    content.append("▎  ", style="dim")
    content.append("store file: ", style="dim")
    content.append(store_size_display, style="white")

    return content


def _render_server_cookies_table(
    *,
    server_name: str | None,
    server_identity: str | None,
    target: str | None,
    sessions_supported: bool | None,
    cookies: list[dict[str, Any]],
    active_session_id: str | None,
    store_size_display: str,
    include_store: bool = True,
    include_mcp_name: bool = True,
) -> Text:
    content = Text()
    width = _resolve_terminal_width()
    command_display, path_display = _format_target_for_display(target, width=width - 6)

    content.append("▎  ", style="dim")
    content.append(f"target: {command_display}", style="bold")
    content.append("\n")
    if path_display:
        content.append("▎    ", style="dim")
        content.append(path_display, style="dim")
        content.append("\n")

    content.append("▎  ", style="dim")
    if include_mcp_name:
        content.append("mcp name: ", style="dim")
        content.append(server_identity or server_name or "-", style="white")
        content.append(" • ", style="dim")
    content.append("cookies: ", style="dim")
    content.append(str(len(cookies)), style="white")
    if include_store:
        content.append("\n")
        content.append("▎  ", style="dim")
        content.append("store file: ", style="dim")
        content.append(store_size_display, style="white")
    content.append("\n")

    if sessions_supported is False:
        content.append("Experimental sessions feature not supported.", style="dim red")
        content.append("\n")
    elif not cookies:
        content.append("No sessions found for this server.", style="dim")
        content.append("\n")
    else:
        index_width = max(2, len(str(len(cookies))))

        for index, item in enumerate(cookies, 1):
            raw_session_id = item.get("id")
            session_id = (
                raw_session_id if isinstance(raw_session_id, str) and raw_session_id else "-"
            )
            is_active = active_session_id is not None and session_id == active_session_id
            is_invalidated = bool(item.get("invalidated"))
            if is_invalidated:
                marker = "○"
                marker_style = "dim red"
                session_style = "dim"
            elif is_active:
                marker = "▶"
                marker_style = "bright_green"
                session_style = "bright_green"
            else:
                marker = "•"
                marker_style = "dim"
                session_style = "white"

            updated_value = (
                item.get("updatedAt") if isinstance(item.get("updatedAt"), str) else None
            )
            updated_compact = _format_expiry_compact(updated_value)
            expiry_compact = _format_expiry_compact(_extract_session_expiry(item))
            window_display = _format_session_window(updated_compact, expiry_compact)
            store_display = _cookie_size_display(item)

            line = Text()
            line.append(f"[{index:>{index_width}}] ", style="dim cyan")
            line.append(f"{marker} ", style=marker_style)
            line.append(session_id, style=session_style)
            line.append(" ", style="dim")
            line.append(window_display, style="dim")
            line.append(" • ", style="dim")
            line.append("store: ", style="dim")
            line.append(store_display, style="white")
            if is_invalidated:
                line.append(" • invalid", style="dim red")
            content.append_text(line)
            content.append("\n")

    return content


def _render_connected_server_cookies_table(
    rows: list[tuple[str, str | None, str | None, bool | None, str | None, list[dict[str, Any]]]],
) -> Text:
    content = Text()
    server_word = "server" if len(rows) == 1 else "servers"
    content.append(f"▎ MCP sessions ({len(rows)} connected {server_word}):", style="bold")
    content.append("\n\n")

    for index, (server_name, server_identity, target, supported, active_session_id, cookies) in enumerate(rows, 1):
        section_header = Text()
        section_header.append("▎ ", style="dim")
        section_header.append(server_name, style="white")
        if server_identity:
            section_header.append(" • ", style="dim")
            section_header.append(server_identity, style="dim")
        content.append_text(section_header)
        content.append("\n")

        table = _render_server_cookies_table(
            server_name=server_name,
            server_identity=server_identity,
            target=target,
            sessions_supported=supported,
            cookies=cookies,
            active_session_id=active_session_id,
            store_size_display="-",
            include_store=False,
            include_mcp_name=False,
        )
        content.append_text(table)
        if index != len(rows):
            content.append("\n\n")

    return content


def _render_session_action_result(
    *,
    heading: str,
    server_name: str,
    server_identity: str | None,
    target: str | None,
    sessions_supported: bool | None,
    cookies: list[dict[str, Any]],
    active_session_id: str | None,
) -> Text:
    content = Text()
    content.append(heading, style="bold")
    content.append("\n\n")
    content.append_text(
        _render_server_cookies_table(
            server_name=server_name,
            server_identity=server_identity,
            target=target,
            sessions_supported=sessions_supported,
            cookies=cookies,
            active_session_id=active_session_id,
            store_size_display="-",
            include_store=False,
        )
    )
    return content
def _target_by_server(entries: list[SessionJarEntry]) -> dict[str, str]:
    targets: dict[str, str] = {}
    for entry in entries:
        if entry.server_name not in targets and isinstance(entry.target, str) and entry.target:
            targets[entry.server_name] = entry.target
    return targets


def _target_for_identity_or_name(
    entries: list[SessionJarEntry],
    *,
    server_name: str | None,
    server_identity: str | None,
) -> str | None:
    if server_name:
        by_server = _target_by_server(entries)
        target = by_server.get(server_name)
        if target:
            return target

    if server_identity:
        for entry in entries:
            if entry.server_identity == server_identity and isinstance(entry.target, str) and entry.target:
                return entry.target

    return None


def _support_for_identity_or_name(
    entries: list[SessionJarEntry],
    *,
    server_name: str | None,
    server_identity: str | None,
) -> bool | None:
    for entry in entries:
        if server_name is not None and entry.server_name == server_name:
            return entry.supported
        if server_identity is not None and entry.server_identity == server_identity:
            return entry.supported
    return None


def _render_clear_all_result(servers: list[str]) -> Text:
    content = Text()
    content.append("Cleared MCP session entries:", style="bold")
    content.append("\n\n")

    index_width = max(2, len(str(len(servers))))
    for index, server in enumerate(servers, 1):
        content.append(f"[{index:>{index_width}}] ", style="dim cyan")
        content.append(server, style="white")
        content.append("\n")

    return content


async def handle_mcp_session(
    ctx,
    *,
    agent_name: str,
    action: McpSessionAction,
    server_identity: str | None,
    session_id: str | None,
    title: str | None,
    clear_all: bool,
) -> CommandOutcome:
    outcome = CommandOutcome()

    try:
        session_client = _resolve_session_client(ctx, agent_name=agent_name)
    except Exception as exc:
        outcome.add_message(str(exc), channel="error", right_info="mcp")
        return outcome

    try:
        store_size_display = _resolve_store_size_display(session_client)

        if action == "jar":
            entries = await session_client.list_jar()
            if server_identity:
                resolved = await session_client.resolve_server_name(server_identity)
                entries = [entry for entry in entries if entry.server_name == resolved]

            if not entries:
                outcome.add_message(
                    "No MCP session jar entries available.",
                    channel="warning",
                    right_info="mcp",
                    agent_name=agent_name,
                )
                return outcome

            rendered = _render_jar_table(entries, store_size_display=store_size_display)
            outcome.add_message(
                rendered,
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        if action == "list":
            target: str | None = None
            sessions_supported: bool | None = None
            if server_identity is None:
                entries = await session_client.list_jar()
                targets_by_server = _target_by_server(entries)
                support_by_server = {
                    entry.server_name: entry.supported
                    for entry in entries
                    if isinstance(entry.server_name, str) and entry.server_name
                }
                connected_servers = sorted(
                    {
                        entry.server_name
                        for entry in entries
                        if isinstance(entry.server_name, str)
                        and entry.server_name
                        and entry.connected is True
                    }
                )

                if not connected_servers:
                    outcome.add_message(
                        "No connected MCP servers available.",
                        channel="warning",
                        right_info="mcp",
                        agent_name=agent_name,
                    )
                    return outcome

                rows: list[
                    tuple[
                        str,
                        str | None,
                        str | None,
                        bool | None,
                        str | None,
                        list[dict[str, Any]],
                    ]
                ] = []
                for connected_server in connected_servers:
                    (
                        listed_server,
                        listed_identity,
                        listed_active_session,
                        listed_cookies,
                    ) = await session_client.list_server_cookies(connected_server)
                    rows.append(
                        (
                            listed_server,
                            listed_identity,
                            targets_by_server.get(listed_server),
                            support_by_server.get(listed_server),
                            listed_active_session,
                            listed_cookies,
                        )
                    )

                outcome.add_message(
                    _render_connected_server_cookies_table(rows),
                    right_info="mcp",
                    agent_name=agent_name,
                )
                return outcome
            else:
                (
                    listed_server,
                    server_id,
                    active_session_id,
                    cookies,
                ) = await session_client.list_server_cookies(server_identity)
                entries = await session_client.list_jar()
                target = _target_for_identity_or_name(
                    entries,
                    server_name=listed_server,
                    server_identity=server_id,
                )
                sessions_supported = _support_for_identity_or_name(
                    entries,
                    server_name=listed_server,
                    server_identity=server_id,
                )
            outcome.add_message(
                _render_server_cookies_table(
                    server_name=listed_server,
                    server_identity=server_id,
                    target=target,
                    sessions_supported=sessions_supported,
                    cookies=cookies,
                    active_session_id=active_session_id,
                    store_size_display=store_size_display,
                    include_store=False,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        if action == "new":
            server_name, _cookie = await session_client.create_session(server_identity, title=title)
            (
                listed_server,
                listed_identity,
                active_session_id,
                cookies,
            ) = await session_client.list_server_cookies(server_name)
            entries = await session_client.list_jar()
            target = _target_for_identity_or_name(
                entries,
                server_name=listed_server,
                server_identity=listed_identity,
            )
            sessions_supported = _support_for_identity_or_name(
                entries,
                server_name=listed_server,
                server_identity=listed_identity,
            )
            outcome.add_message(
                _render_session_action_result(
                    heading=f"Created new MCP session for {server_name}.",
                    server_name=listed_server,
                    server_identity=listed_identity,
                    target=target,
                    sessions_supported=sessions_supported,
                    cookies=cookies,
                    active_session_id=active_session_id,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        if action == "use":
            if not session_id:
                raise ValueError("Session id is required for use.")
            server_name, _cookie = await session_client.resume_session(
                server_identity,
                session_id=session_id,
            )
            (
                listed_server,
                listed_identity,
                active_session_id,
                cookies,
            ) = await session_client.list_server_cookies(server_name)
            entries = await session_client.list_jar()
            target = _target_for_identity_or_name(
                entries,
                server_name=listed_server,
                server_identity=listed_identity,
            )
            sessions_supported = _support_for_identity_or_name(
                entries,
                server_name=listed_server,
                server_identity=listed_identity,
            )
            outcome.add_message(
                _render_session_action_result(
                    heading=f"Selected MCP session for {server_name}.",
                    server_name=listed_server,
                    server_identity=listed_identity,
                    target=target,
                    sessions_supported=sessions_supported,
                    cookies=cookies,
                    active_session_id=active_session_id,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        if action == "clear":
            if clear_all:
                cleared = await session_client.clear_all_cookies()
                if not cleared:
                    outcome.add_message(
                        "No attached MCP servers to clear.",
                        channel="warning",
                        right_info="mcp",
                        agent_name=agent_name,
                    )
                    return outcome
                outcome.add_message(
                    _render_clear_all_result(cleared),
                    right_info="mcp",
                    agent_name=agent_name,
                )
                return outcome

            server_name = await session_client.clear_cookie(server_identity)
            (
                listed_server,
                listed_identity,
                active_session_id,
                cookies,
            ) = await session_client.list_server_cookies(server_name)
            entries = await session_client.list_jar()
            target = _target_for_identity_or_name(
                entries,
                server_name=listed_server,
                server_identity=listed_identity,
            )
            sessions_supported = _support_for_identity_or_name(
                entries,
                server_name=listed_server,
                server_identity=listed_identity,
            )
            outcome.add_message(
                _render_session_action_result(
                    heading=f"Cleared MCP session entry for {server_name}.",
                    server_name=listed_server,
                    server_identity=listed_identity,
                    target=target,
                    sessions_supported=sessions_supported,
                    cookies=cookies,
                    active_session_id=active_session_id,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        outcome.add_message(
            f"Unsupported /mcp session action: {action}",
            channel="error",
            right_info="mcp",
            agent_name=agent_name,
        )
    except Exception as exc:
        outcome.add_message(
            str(exc),
            channel="error",
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
            await emit_progress(
                event.message or "Waiting for OAuth callback (startup timer paused)…"
            )
            return

        if event.event_type == "wait_end":
            await emit_progress(event.message or "OAuth callback wait complete.")
            return

        if event.event_type == "callback_received":
            await emit_progress(
                event.message or "OAuth callback received. Completing token exchange…"
            )
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
    server_name = (
        configured_alias or parsed.server_name or infer_server_name(parsed.target_text, mode)
    )
    if mode == "configured":
        await emit_progress(f"Connecting MCP server '{server_name}' from config file…")
    else:
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

        if oauth_related and (
            fallback_disabled or oauth_timeout or not oauth_paste_fallback_enabled
        ):
            outcome.add_message(
                (
                    "OAuth could not be completed in this connection mode. "
                    "Run `fast-agent auth login <server-name-or-mcp-name>` on the fast-agent host, "
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
        tools_total if isinstance(tools_total, int) and tools_total >= 0 else tools_added_count
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
        if mode == "configured":
            configured_source = _resolve_configured_source_from_context(ctx, server_name)
            source_text = configured_source or parsed.target_text
            message_text = f"{action} MCP server '{server_name}' from configuration: {source_text}."
        else:
            message_text = f"{action} MCP server '{server_name}' ({mode})."
        outcome.add_message(
            message_text,
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


async def handle_mcp_reconnect(
    ctx,
    *,
    manager: McpRuntimeManager,
    agent_name: str,
    server_name: str,
) -> CommandOutcome:
    del ctx
    outcome = CommandOutcome()

    try:
        attached_servers = await manager.list_attached_mcp_servers(agent_name)
    except Exception as exc:
        outcome.add_message(f"Failed to list attached MCP servers: {exc}", channel="error")
        return outcome

    if server_name not in attached_servers:
        outcome.add_message(
            (
                f"MCP server '{server_name}' is not currently attached. "
                "Use `/mcp connect <target>` to attach it first."
            ),
            channel="warning",
            right_info="mcp",
            agent_name=agent_name,
        )
        return outcome

    try:
        result = await manager.attach_mcp_server(
            agent_name,
            server_name,
            server_config=None,
            options=MCPAttachOptions(force_reconnect=True),
        )
    except Exception as exc:
        outcome.add_message(f"Failed to reconnect MCP server: {exc}", channel="error")
        return outcome

    tools_added = getattr(result, "tools_added", [])
    prompts_added = getattr(result, "prompts_added", [])
    tools_total = getattr(result, "tools_total", None)
    prompts_total = getattr(result, "prompts_total", None)
    warnings = getattr(result, "warnings", [])

    tools_added_count = len(tools_added)
    prompts_added_count = len(prompts_added)
    tools_refreshed_count = (
        tools_total if isinstance(tools_total, int) and tools_total >= 0 else tools_added_count
    )
    prompts_refreshed_count = (
        prompts_total
        if isinstance(prompts_total, int) and prompts_total >= 0
        else prompts_added_count
    )

    outcome.add_message(
        f"Reconnected MCP server '{server_name}'.",
        right_info="mcp",
        agent_name=agent_name,
    )
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

    for warning in warnings:
        outcome.add_message(warning, channel="warning", right_info="mcp", agent_name=agent_name)

    return outcome
