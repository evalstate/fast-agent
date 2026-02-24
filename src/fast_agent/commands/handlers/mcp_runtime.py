"""Runtime MCP connect/list/disconnect command handlers."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
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

    raise RuntimeError(f"Agent '{agent_name}' does not expose experimental session controls.")


def _render_cookie(cookie: dict[str, Any] | None) -> str:
    if not cookie:
        return "null"
    return json.dumps(cookie, indent=2, sort_keys=True, ensure_ascii=False)


def _render_jar_entry(entry: SessionJarEntry) -> str:
    features = ", ".join(entry.features) if entry.features else "none"
    supported = (
        "yes" if entry.supported is True else "no" if entry.supported is False else "unknown"
    )
    identity = entry.server_identity or "(unset)"
    title = entry.title or "(none)"

    return (
        f"server={entry.server_name}\n"
        f"identity={identity}\n"
        f"exp_session_supported={supported}\n"
        f"features={features}\n"
        f"title={title}\n"
        f"last_used_id={entry.last_used_id or '-'}\n"
        f"cookie=\n{_render_cookie(entry.cookie)}"
    )


def _truncate_cell(value: str, max_len: int = 28) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def _extract_cookie_id(cookie: dict[str, Any] | None) -> str | None:
    if not isinstance(cookie, dict):
        return None
    raw_id = cookie.get("id")
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
    expiry = payload.get("expiry")
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

    session_id = payload.get("id")
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


def _session_suffix(session_id: str | None, *, digits: int = 5) -> str:
    if not session_id:
        return "none"
    if len(session_id) <= digits:
        return session_id
    return f"…{session_id[-digits:]}"


def _experimental_version(entry: SessionJarEntry) -> str:
    if entry.supported is not True:
        return "-"
    feature_set = {feature.strip().lower() for feature in entry.features}
    return "v2" if {"create", "delete", "list"}.issubset(feature_set) else "v1"


def _render_jar_table(entries: list[SessionJarEntry]) -> Text:
    if not entries:
        return Text("No MCP session jar entries available.", style="dim")

    grouped: dict[str, list[SessionJarEntry]] = {}
    for entry in entries:
        key = entry.server_identity or entry.server_name
        grouped.setdefault(key, []).append(entry)

    labels = sorted(grouped)
    content = Text()
    content.append("MCP session jar:", style="bold")
    content.append("\n\n")
    width = _resolve_terminal_width()
    index_width = max(2, len(str(len(labels))))

    for index, label in enumerate(labels, 1):
        grouped_entries = grouped[label]
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
        version = _experimental_version(primary)

        active_summary: dict[str, Any] | None = None
        if primary.cookies:
            active_summary = next(
                (
                    summary
                    for summary in primary.cookies
                    if isinstance(summary, dict) and summary.get("active") is True
                ),
                primary.cookies[0] if isinstance(primary.cookies[0], dict) else None,
            )

        active_cookie_id = _extract_cookie_id(primary.cookie)
        if isinstance(active_summary, dict):
            raw_id = active_summary.get("id")
            if isinstance(raw_id, str) and raw_id:
                active_cookie_id = raw_id

        updated = "-"
        if isinstance(active_summary, dict):
            raw_updated = active_summary.get("updatedAt")
            updated = _format_expiry_compact(raw_updated if isinstance(raw_updated, str) else None)

        created = "-"
        expiry = "-"
        if isinstance(primary.cookie, dict):
            created = _format_expiry_compact(_extract_session_created(primary.cookie))
            raw_expiry = primary.cookie.get("expiry")
            expiry = _format_expiry_compact(raw_expiry if isinstance(raw_expiry, str) else None)
        if isinstance(active_summary, dict):
            raw_expiry = active_summary.get("expiry")
            active_expiry = _format_expiry_compact(
                raw_expiry if isinstance(raw_expiry, str) else None
            )
            if active_expiry != "-":
                expiry = active_expiry

        if created != "-":
            time_display = created
        elif updated != "-":
            time_display = updated
        else:
            time_display = None

        connection_state = (
            "connected" if any(e.connected is True for e in grouped_entries) else "disconnected"
        )
        version_display = version if version != "-" else "unknown"

        summary_title = active_summary.get("title") if isinstance(active_summary, dict) else None
        title_raw = (
            summary_title
            if isinstance(summary_title, str) and summary_title.strip()
            else (primary.title or "")
        )
        title_text = title_raw.strip() if isinstance(title_raw, str) else ""
        if not title_text:
            title_text = "(untitled)"

        window_display = _format_session_window(time_display, expiry)

        active_display = "none"
        if isinstance(active_cookie_id, str) and active_cookie_id:
            active_display = (
                active_cookie_id
                if len(active_cookie_id) <= 24
                else f"{active_cookie_id[:10]}…{active_cookie_id[-6:]}"
            )

        cookie_count = sum(len(entry.cookies) for entry in grouped_entries)

        header = Text()
        header.append(f"[{index:>{index_width}}] ", style="dim cyan")
        header.append(_truncate_cell(label, max_len=30), style="white")
        header.append(" • ", style="dim")
        header.append(
            connection_state,
            style="bright_green" if connection_state == "connected" else "dim",
        )
        header.append(" • ", style="dim")
        header.append(version_display, style="cyan" if version_display != "unknown" else "dim")
        content.append_text(header)
        content.append("\n")

        meta = Text()
        meta.append("active: ", style="dim")
        meta.append(active_display, style="white")
        meta.append(" • ", style="dim")
        meta.append("cookies: ", style="dim")
        meta.append(str(cookie_count), style="white")
        content.append_text(meta)
        content.append("\n")

        details = Text()
        title_reserved = details.cell_len + 1 + len(window_display)
        title_width = max(12, width - title_reserved)
        details.append(_truncate_cell(title_text, max_len=title_width), style="white")
        details.append(" ", style="dim")
        details.append(window_display, style="dim")
        content.append_text(details)
        content.append("\n")

        if index != len(labels):
            content.append("\n")

    return content


def _render_server_cookies_table(
    *,
    server_identity: str | None,
    cookies: list[dict[str, Any]],
    active_session_id: str | None,
) -> Text:
    content = Text()
    content.append("MCP sessions:", style="bold")
    content.append("\n\n")

    if not cookies:
        content.append("No session cookies found for this server.", style="dim")
        content.append("\n")
    else:
        width = _resolve_terminal_width()
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
            title_raw = _extract_session_title(item)
            if title_raw == "-":
                title_raw = "(untitled)"
            window_display = _format_session_window(updated_compact, expiry_compact)

            line = Text()
            line.append(f"[{index:>{index_width}}] ", style="dim cyan")
            line.append(f"{marker} ", style=marker_style)
            line.append(session_id, style=session_style)

            invalid_segment = Text()
            if is_invalidated:
                invalid_segment.append(" • invalid", style="dim red")

            title_prefix = Text(" • ", style="dim")
            reserved = (
                line.cell_len
                + title_prefix.cell_len
                + invalid_segment.cell_len
                + 1
                + len(window_display)
            )
            title_width = max(12, width - reserved)
            title_display = _truncate_cell(title_raw, max_len=title_width)

            line.append_text(title_prefix)
            line.append(title_display, style="white")
            line.append_text(invalid_segment)
            line.append(" ", style="dim")
            line.append(window_display, style="dim")
            content.append_text(line)
            content.append("\n")

    content.append("\n")
    content.append("▎• ", style="dim")
    content.append("identity: ", style="dim")
    content.append(server_identity or "-", style="white")
    content.append(" • ", style="dim")
    content.append("cookies: ", style="dim")
    content.append(str(len(cookies)), style="white")

    return content


def _render_single_cookie_result(
    *,
    heading: str,
    server_name: str,
    cookie: dict[str, Any] | None,
) -> Text:
    content = Text()
    content.append(heading, style="bold")
    content.append("\n\n")

    if isinstance(cookie, dict):
        session_id = _extract_cookie_id(cookie) or "-"
        title_raw = _extract_session_title(cookie)
        created = _format_expiry_compact(_extract_session_created(cookie))
        expiry = _format_expiry_compact(_extract_session_expiry(cookie))
        window_display = _format_session_window(created, expiry)

        width = _resolve_terminal_width()
        reserved = 5 + len(session_id) + 3 + 1 + len(window_display)
        title_width = max(12, width - reserved)
        title = _truncate_cell(title_raw, max_len=title_width)

        line = Text()
        line.append("[ 1] ", style="dim cyan")
        line.append("▶ ", style="bright_green")
        line.append(session_id, style="bright_green")
        line.append(" • ", style="dim")
        line.append(title, style="white")
        line.append(" ", style="dim")
        line.append(window_display, style="dim")
        content.append_text(line)
        content.append("\n\n")
    else:
        content.append("No session cookie returned.", style="dim")
        content.append("\n\n")

    content.append("▎• ", style="dim")
    content.append("server: ", style="dim")
    content.append(server_name, style="white")
    return content


def _render_clear_all_result(servers: list[str]) -> Text:
    content = Text()
    content.append("Cleared experimental session cookies:", style="bold")
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

            rendered = _render_jar_table(entries)
            outcome.add_message(
                rendered,
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        if action == "list":
            if server_identity is None:
                try:
                    (
                        _server_name,
                        server_id,
                        active_session_id,
                        cookies,
                    ) = await session_client.list_server_cookies(None)
                except ValueError as exc:
                    if "Multiple MCP servers are attached" not in str(exc):
                        raise
                    entries = await session_client.list_jar()
                    outcome.add_message(
                        _render_jar_table(entries),
                        right_info="mcp",
                        agent_name=agent_name,
                    )
                    return outcome
            else:
                (
                    _server_name,
                    server_id,
                    active_session_id,
                    cookies,
                ) = await session_client.list_server_cookies(server_identity)
            outcome.add_message(
                _render_server_cookies_table(
                    server_identity=server_id,
                    cookies=cookies,
                    active_session_id=active_session_id,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        if action == "new":
            server_name, cookie = await session_client.create_session(server_identity, title=title)
            outcome.add_message(
                _render_single_cookie_result(
                    heading=f"Created experimental session for {server_name}.",
                    server_name=server_name,
                    cookie=cookie,
                ),
                right_info="mcp",
                agent_name=agent_name,
            )
            return outcome

        if action == "use":
            if not session_id:
                raise ValueError("Session id is required for use.")
            server_name, cookie = await session_client.resume_session(
                server_identity,
                session_id=session_id,
            )
            outcome.add_message(
                _render_single_cookie_result(
                    heading=f"Selected experimental session cookie for {server_name}.",
                    server_name=server_name,
                    cookie=cookie,
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
            outcome.add_message(
                _render_single_cookie_result(
                    heading=f"Cleared experimental session cookie for {server_name}.",
                    server_name=server_name,
                    cookie=None,
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
