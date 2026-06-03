"""Runtime MCP connect/list/disconnect command handlers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    cast,
    runtime_checkable,
)

from rich.text import Text

from fast_agent.commands.handlers._text_formatting import resolve_terminal_width
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.summary_utils import optional_string
from fast_agent.mcp.connect_targets import (
    McpConnectMode,
    NormalizedMcpTarget,
    ParsedMcpConnectRequest,
    build_server_config_from_target,
    infer_server_name,
    render_normalized_target,
)
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.commandline import join_commandline
from fast_agent.utils.count_display import format_count, format_count_parts
from fast_agent.utils.numeric import nonnegative_int_or_none, positive_int_or_none
from fast_agent.utils.path_display import fit_path_for_display, left_truncate_with_ellipsis
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from fast_agent.commands.mcp_command_intents import McpSessionAction
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.experimental_session_client import (
        ExperimentalSessionClient,
        ServerCookiesView,
        SessionJarEntry,
    )
    from fast_agent.mcp.oauth_client import OAuthEvent


_McpConnectRuntimeMode: TypeAlias = Literal["configured", "url", "stdio", "npx", "uvx"]
_EXPERIMENTAL_SESSION_SUPPORT_LABELS: dict[bool | None, str] = {
    True: "yes",
    False: "no",
    None: "unknown",
}


def _connect_runtime_mode(
    *,
    configured_alias: str | None,
    target_mode: McpConnectMode,
) -> _McpConnectRuntimeMode:
    if configured_alias is not None:
        return "configured"
    return target_mode


class McpRuntimeManager(Protocol):
    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult: ...

    async def detach_mcp_server(self, agent_name: str, server_name: str) -> MCPDetachResult: ...

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]: ...

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]: ...


class SessionClientProtocol(Protocol):
    def store_size_bytes(self) -> int | None: ...

    async def list_jar(self) -> list[SessionJarEntry]: ...

    async def resolve_server_name(self, server_identifier: str | None) -> str: ...

    async def list_server_cookies(
        self, server_identifier: str | None
    ) -> ServerCookiesView: ...

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


@dataclass(slots=True)
class _OAuthProgressState:
    links_seen: set[str]
    links_ordered: list[str]


@dataclass(frozen=True, slots=True)
class _McpConnectPlan:
    mode: _McpConnectRuntimeMode
    server_name: str
    config: "MCPServerSettings | None"
    attach_options: MCPAttachOptions


@dataclass(frozen=True, slots=True)
class _McpConnectFailureClassification:
    oauth_related: bool
    oauth_registration_404: bool
    oauth_fallback_unavailable: bool
    oauth_timeout: bool


@runtime_checkable
class SessionClientAgentProtocol(Protocol):
    @property
    def experimental_sessions(self) -> ExperimentalSessionClient: ...


_AUTH_ENV_BRACED_RE = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>.*))?\}$")
_AUTH_ENV_SIMPLE_RE = re.compile(r"^\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")


def _normalize_auth_token_value(raw_value: str) -> str:
    """Normalize user-provided --auth values before environment lookup.

    ``--auth`` takes the raw token value. If a user passes an Authorization
    header style value (``Bearer <token>``), strip the prefix so downstream
    code can still compose a single valid ``Authorization: Bearer ...`` header.
    """

    normalized = strip_to_none(raw_value) or ""
    if normalize_action_token(normalized).startswith("bearer "):
        normalized = strip_to_none(normalized[7:]) or ""
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


def _resolve_request_auth(request: ParsedMcpConnectRequest) -> ParsedMcpConnectRequest:
    auth_token = request.options.auth_token
    if auth_token is None:
        return request
    return replace(
        request,
        options=replace(
            request.options,
            auth_token=_resolve_auth_token_value(auth_token),
        ),
    )


def _describe_server_config_source(
    server_config: Mapping[str, object] | MCPServerSettings,
) -> str | None:
    """Return a concise url/command description for an MCP server config."""

    if isinstance(server_config, Mapping):
        config_mapping = cast("Mapping[str, object]", server_config)
        url_value = config_mapping.get("url")
        command_value = config_mapping.get("command")
        args_value = config_mapping.get("args")
    else:
        url_value = server_config.url
        command_value = server_config.command
        args_value = server_config.args

    url = strip_to_none(url_value) if isinstance(url_value, str) else None
    if url is not None:
        return url

    command = strip_to_none(command_value) if isinstance(command_value, str) else None
    if command is not None:
        args: list[str] = []
        if isinstance(args_value, list):
            args = [str(value) for value in args_value]
        return join_commandline([command, *args], syntax="posix")

    return None


def _resolve_configured_source_from_context(ctx, server_name: str) -> str | None:
    """Resolve configured server description from runtime settings."""

    try:
        settings = ctx.resolve_settings()
    except Exception:
        return None

    mcp_settings = settings.mcp
    if mcp_settings is None:
        return None

    server_config = mcp_settings.servers.get(server_name)
    if server_config is None:
        return None
    return _describe_server_config_source(server_config)


async def _resolve_configured_server_alias(
    *,
    manager: McpRuntimeManager,
    agent_name: str,
    request: ParsedMcpConnectRequest,
) -> str | None:
    """Return configured server name when target text is an alias.

    We treat a single stdio token as a server alias only when no explicit
    --name override or URL auth token is provided.
    """

    if request.target.server_name is not None or request.options.auth_token is not None:
        return None

    if request.target.mode != "stdio":
        return None

    if not request.target.command or request.target.args:
        return None

    candidate = request.target.command
    if not candidate or candidate.startswith("-"):
        return None

    configured_names: set[str] = set()
    with suppress(Exception):
        configured_names.update(await manager.list_configured_detached_mcp_servers(agent_name))

    with suppress(Exception):
        configured_names.update(await manager.list_attached_mcp_servers(agent_name))

    return candidate if candidate in configured_names else None


@dataclass(frozen=True, slots=True)
class _McpAttachCounts:
    tools_added_count: int
    prompts_added_count: int
    tools_refreshed_count: int
    prompts_refreshed_count: int
    skills_count: int | None = None

    @property
    def new_count(self) -> int:
        return self.tools_added_count + self.prompts_added_count

    @property
    def added(self) -> "_McpResourceCounts":
        return _McpResourceCounts(
            tools=self.tools_added_count,
            prompts=self.prompts_added_count,
        )

    @property
    def refreshed(self) -> "_McpResourceCounts":
        return _McpResourceCounts(
            tools=self.tools_refreshed_count,
            prompts=self.prompts_refreshed_count,
        )


@dataclass(frozen=True, slots=True)
class _McpResourceCounts:
    tools: int
    prompts: int


def _format_added_summary(counts: _McpAttachCounts) -> Text:
    summary = _format_resource_change_summary("Added", counts.added, trailing_period=False)
    _append_skills_count(summary, counts.skills_count)
    summary.append(".", style="dim")
    return summary


def _format_refreshed_summary(counts: _McpAttachCounts) -> Text:
    summary = _format_resource_change_summary(
        "Refreshed",
        counts.refreshed,
        trailing_period=False,
    )
    summary.append(" (", style="dim")
    summary.append(str(counts.new_count), style="bold bright_cyan")
    summary.append(" new)", style="dim")
    _append_skills_count(summary, counts.skills_count)
    summary.append(".", style="dim")
    return summary


def _format_removed_summary(counts: _McpResourceCounts) -> Text:
    return _format_resource_change_summary("Removed", counts)


def _format_resource_change_summary(
    action: str,
    counts: _McpResourceCounts,
    *,
    trailing_period: bool = True,
) -> Text:
    summary = Text()
    summary.append(f"{action} ", style="dim")
    _append_resource_pair(summary, counts)
    if trailing_period:
        summary.append(".", style="dim")
    return summary


def _append_resource_pair(summary: Text, counts: _McpResourceCounts) -> None:
    _append_counted_resource(summary, counts.tools, "tool")
    summary.append(" and ", style="dim")
    _append_counted_resource(summary, counts.prompts, "prompt")


def _append_skills_count(summary: Text, skills_count: int | None) -> None:
    if skills_count is None:
        return
    summary.append("; ", style="dim")
    _append_counted_resource(summary, skills_count, "skill")
    summary.append(" available from the server", style="dim")


def _append_counted_resource(summary: Text, count: int, singular: str) -> None:
    count_text, label = format_count_parts(count, singular)
    summary.append(count_text, style="bold bright_cyan")
    summary.append(f" {label}", style="dim")


def _mcp_attach_counts(result: MCPAttachResult) -> _McpAttachCounts:
    tools_added_count = len(result.tools_added)
    prompts_added_count = len(result.prompts_added)
    tools_total = nonnegative_int_or_none(result.tools_total)
    prompts_total = nonnegative_int_or_none(result.prompts_total)
    skills_total = nonnegative_int_or_none(result.skills_total)
    tools_refreshed_count = tools_total if tools_total is not None else tools_added_count
    prompts_refreshed_count = prompts_total if prompts_total is not None else prompts_added_count
    return _McpAttachCounts(
        tools_added_count=tools_added_count,
        prompts_added_count=prompts_added_count,
        tools_refreshed_count=tools_refreshed_count,
        prompts_refreshed_count=prompts_refreshed_count,
        skills_count=skills_total,
    )


async def handle_mcp_list(*, manager: McpRuntimeManager, agent_name: str) -> CommandOutcome:
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


McpSessionActionHandler = Callable[
    [SessionClientProtocol, "McpSessionRequest"],
    Awaitable[CommandOutcome],
]
McpSessionServerOperation = Callable[
    [SessionClientProtocol, "McpSessionRequest"],
    Awaitable[str],
]


@dataclass(frozen=True, slots=True)
class McpSessionRequest:
    agent_name: str
    server_identity: str | None
    session_id: str | None
    title: str | None
    clear_all: bool
    store_size_display: str


@dataclass(frozen=True, slots=True)
class _DisplayTarget:
    primary: str
    secondary: str | None = None


def _resolve_session_client(ctx, *, agent_name: str) -> SessionClientProtocol:
    agent = ctx.agent_provider._agent(agent_name)
    if not isinstance(agent, SessionClientAgentProtocol):
        raise RuntimeError(f"Agent '{agent_name}' does not expose an MCP aggregator.")
    return cast("SessionClientProtocol", agent.experimental_sessions)


def _render_cookie(cookie: dict[str, Any] | None) -> str:
    if not cookie:
        return "null"
    return json.dumps(cookie, indent=2, sort_keys=True, ensure_ascii=False)


def _experimental_session_support_label(supported: bool | None) -> str:
    return _EXPERIMENTAL_SESSION_SUPPORT_LABELS[supported]


def _render_jar_entry(entry: SessionJarEntry) -> str:
    features = ", ".join(entry.features) if entry.features else "none"
    supported = _experimental_session_support_label(entry.supported)
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


def _format_target_for_display(
    target: str | None,
    *,
    width: int,
) -> _DisplayTarget:
    if not target:
        return _DisplayTarget(primary="-")

    if target.startswith("cmd:"):
        payload = strip_to_none(target[4:]) or ""
        command, separator, cwd = payload.partition(" @ ")
        command_display = f"cmd: {command}" if command else "cmd"
        if not separator or not cwd:
            return _DisplayTarget(primary=command_display)

        path_width = max(12, width - len("cwd: "))
        return _DisplayTarget(
            primary=command_display,
            secondary=f"cwd: {fit_path_for_display(cwd, path_width)}",
        )

    if target.startswith("url:"):
        url = strip_to_none(target[4:]) or ""
        path_width = max(12, width)
        return _DisplayTarget(primary=f"url: {left_truncate_with_ellipsis(url, path_width)}")

    display_width = max(12, width)
    return _DisplayTarget(primary=left_truncate_with_ellipsis(target, display_width))


def _cookie_size_display(summary: dict[str, Any]) -> str:
    size = positive_int_or_none(summary.get("cookieSizeBytes"))
    if size is not None:
        return f"{size} bytes"
    return "-"


def _extract_cookie_id(cookie: dict[str, Any] | None) -> str | None:
    if not isinstance(cookie, dict):
        return None
    return optional_string(cookie.get("sessionId"))


def _extract_session_title(payload: dict[str, Any]) -> str:
    direct_title = optional_string(payload.get("title"))
    if direct_title is not None:
        return direct_title

    data = payload.get("data")
    if isinstance(data, dict):
        data_title = optional_string(data.get("title")) or optional_string(data.get("label"))
        if data_title is not None:
            return data_title

    return "-"


def _first_optional_string(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    return next(
        (value for key in keys if (value := optional_string(payload.get(key))) is not None),
        None,
    )


def _extract_session_expiry(payload: dict[str, Any]) -> str:
    return _first_optional_string(payload, ("expiresAt", "expiry")) or "-"


def _extract_session_created(payload: dict[str, Any]) -> str:
    created = _first_optional_string(payload, ("created", "created_at", "createdAt"))
    if created is not None:
        return created

    data = payload.get("data")
    if isinstance(data, dict):
        created = _first_optional_string(data, ("created", "created_at", "createdAt"))
        if created is not None:
            return created

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
        parsed = datetime.fromisoformat(expiry)
    except ValueError:
        return _truncate_cell(expiry, 14)
    return parsed.strftime("%d/%m/%y %H:%M")


def _format_session_window(start: str | None, end: str | None) -> str:
    start_display = start if start and start != "-" else "unknown"
    end_display = end if end and end != "-" else "∞"
    return f"({start_display} → {end_display})"


def _resolve_store_size_display(session_client: SessionClientProtocol) -> str:
    parsed_size = nonnegative_int_or_none(_store_size_bytes(session_client))
    if parsed_size is None:
        return "-"
    return f"{parsed_size} bytes"


def _store_size_bytes(session_client: SessionClientProtocol) -> object | None:
    with suppress(Exception):
        return session_client.store_size_bytes()
    return None


def _group_jar_entries(entries: list[SessionJarEntry]) -> dict[str, list[SessionJarEntry]]:
    grouped: dict[str, list[SessionJarEntry]] = {}
    for entry in entries:
        key = entry.target or entry.server_identity or entry.server_name
        grouped.setdefault(key, []).append(entry)
    return grouped


def _select_primary_jar_entry(entries: list[SessionJarEntry]) -> SessionJarEntry:
    return next(
        (
            entry
            for entry in entries
            if entry.connected is True and _extract_cookie_id(entry.cookie)
        ),
        next(
            (entry for entry in entries if _extract_cookie_id(entry.cookie)),
            entries[0],
        ),
    )


def _combined_jar_cookies(entries: list[SessionJarEntry]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        for summary in entry.cookies:
            if not isinstance(summary, dict):
                continue
            raw_id = summary.get("id")
            if not isinstance(raw_id, str) or not raw_id or raw_id in seen_ids:
                continue
            seen_ids.add(raw_id)
            combined.append(dict(summary))
    combined.sort(key=lambda item: str(item.get("updatedAt") or ""), reverse=True)
    return combined


def _resolve_active_jar_session_id(
    primary: SessionJarEntry,
    cookies: list[dict[str, Any]],
) -> str | None:
    if primary.last_used_id is not None:
        return primary.last_used_id
    return next(
        (
            item.get("id")
            for item in cookies
            if isinstance(item.get("id"), str) and item.get("active") is True
        ),
        None,
    )


def _mark_active_jar_cookie(cookies: list[dict[str, Any]], active_session_id: str | None) -> None:
    if active_session_id is None:
        return
    for item in cookies:
        item_id = item.get("id")
        item["active"] = isinstance(item_id, str) and item_id == active_session_id


def _append_jar_section_header(
    content: Text,
    *,
    primary: SessionJarEntry,
    grouped_entries: list[SessionJarEntry],
    unsupported_connected: bool,
) -> None:
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


def _append_server_cookies_summary_line(
    content: Text,
    *,
    server_name: str | None,
    server_identity: str | None,
    cookie_count: int,
    store_size_display: str,
    include_store: bool,
    include_mcp_name: bool,
) -> None:
    content.append("▎  ", style="dim")
    if include_mcp_name:
        content.append("mcp name: ", style="dim")
        content.append(server_identity or server_name or "-", style="white")
        content.append(" • ", style="dim")
    content.append("cookies: ", style="dim")
    content.append(str(cookie_count), style="white")
    if include_store:
        content.append("\n")
        content.append("▎  ", style="dim")
        content.append("store file: ", style="dim")
        content.append(store_size_display, style="white")
    content.append("\n")


def _cookie_row_marker(
    *,
    is_active: bool,
    is_invalidated: bool,
) -> tuple[str, str, str]:
    if is_invalidated:
        return "○", "dim red", "dim"
    if is_active:
        return "▶", "bright_green", "bright_green"
    return "•", "dim", "white"


def _render_cookie_row(
    item: dict[str, Any],
    *,
    index: int,
    index_width: int,
    active_session_id: str | None,
) -> Text:
    raw_session_id = item.get("id")
    session_id = raw_session_id if isinstance(raw_session_id, str) and raw_session_id else "-"
    is_active = active_session_id is not None and session_id == active_session_id
    is_invalidated = bool(item.get("invalidated"))
    marker, marker_style, session_style = _cookie_row_marker(
        is_active=is_active,
        is_invalidated=is_invalidated,
    )

    updated_value = item.get("updatedAt") if isinstance(item.get("updatedAt"), str) else None
    window_display = _format_session_window(
        _format_expiry_compact(updated_value),
        _format_expiry_compact(_extract_session_expiry(item)),
    )

    line = Text()
    line.append(f"[{index:>{index_width}}] ", style="dim cyan")
    line.append(f"{marker} ", style=marker_style)
    line.append(session_id, style=session_style)
    line.append(" ", style="dim")
    line.append(window_display, style="dim")
    line.append(" • ", style="dim")
    line.append("store: ", style="dim")
    line.append(_cookie_size_display(item), style="white")
    if is_invalidated:
        line.append(" • invalid", style="dim red")
    return line


def _append_cookie_rows(
    content: Text,
    cookies: list[dict[str, Any]],
    *,
    active_session_id: str | None,
) -> None:
    index_width = max(2, len(str(len(cookies))))
    for index, item in enumerate(cookies, 1):
        content.append_text(
            _render_cookie_row(
                item,
                index=index,
                index_width=index_width,
                active_session_id=active_session_id,
            )
        )
        content.append("\n")


def _render_jar_table(entries: list[SessionJarEntry], *, store_size_display: str) -> Text:
    if not entries:
        return Text("No MCP session jar entries available.", style="dim")

    grouped = _group_jar_entries(entries)
    labels = sorted(grouped)
    content = Text()
    content.append(f"▎ MCP session jar ({format_count(len(labels), 'target')}):", style="bold")
    content.append("\n\n")

    for index, label in enumerate(labels, 1):
        grouped_entries = grouped[label]
        unsupported_connected = any(
            entry.connected is True and entry.supported is False for entry in grouped_entries
        )
        primary = _select_primary_jar_entry(grouped_entries)
        combined_cookies = _combined_jar_cookies(grouped_entries)
        active_session_id = _resolve_active_jar_session_id(primary, combined_cookies)
        _mark_active_jar_cookie(combined_cookies, active_session_id)
        _append_jar_section_header(
            content,
            primary=primary,
            grouped_entries=grouped_entries,
            unsupported_connected=unsupported_connected,
        )

        sessions_supported = False if unsupported_connected else primary.supported

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
    width = resolve_terminal_width()
    display_target = _format_target_for_display(target, width=width - 6)

    content.append("▎  ", style="dim")
    content.append(f"target: {display_target.primary}", style="bold")
    content.append("\n")
    if display_target.secondary:
        content.append("▎    ", style="dim")
        content.append(display_target.secondary, style="dim")
        content.append("\n")

    _append_server_cookies_summary_line(
        content,
        server_name=server_name,
        server_identity=server_identity,
        cookie_count=len(cookies),
        store_size_display=store_size_display,
        include_store=include_store,
        include_mcp_name=include_mcp_name,
    )

    if sessions_supported is False:
        content.append("Experimental sessions feature not supported.", style="dim red")
        content.append("\n")
    elif not cookies:
        content.append("No sessions found for this server.", style="dim")
        content.append("\n")
    else:
        _append_cookie_rows(content, cookies, active_session_id=active_session_id)

    return content


def _render_connected_server_cookies_table(
    rows: list["ServerCookiesView"],
) -> Text:
    content = Text()
    content.append(
        f"▎ MCP sessions ({format_count(len(rows), 'connected server')}):",
        style="bold",
    )
    content.append("\n\n")

    for index, row in enumerate(rows, 1):
        section_header = Text()
        section_header.append("▎ ", style="dim")
        section_header.append(row.server_name, style="white")
        if row.server_identity:
            section_header.append(" • ", style="dim")
            section_header.append(row.server_identity, style="dim")
        content.append_text(section_header)
        content.append("\n")

        table = _render_server_cookies_table(
            server_name=row.server_name,
            server_identity=row.server_identity,
            target=row.target,
            sessions_supported=row.sessions_supported,
            cookies=row.cookies,
            active_session_id=row.active_session_id,
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


def _connected_jar_server_names(entries: list[SessionJarEntry]) -> list[str]:
    return sorted(
        {
            entry.server_name
            for entry in entries
            if isinstance(entry.server_name, str)
            and entry.server_name
            and entry.connected is True
        }
    )


async def _handle_mcp_session_jar(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> CommandOutcome:
    outcome = CommandOutcome()
    entries = await session_client.list_jar()
    if request.server_identity:
        resolved = await session_client.resolve_server_name(request.server_identity)
        entries = [entry for entry in entries if entry.server_name == resolved]

    if not entries:
        outcome.add_message(
            "No MCP session jar entries available.",
            channel="warning",
            right_info="mcp",
            agent_name=request.agent_name,
        )
        return outcome

    rendered = _render_jar_table(entries, store_size_display=request.store_size_display)
    outcome.add_message(
        rendered,
        right_info="mcp",
        agent_name=request.agent_name,
    )
    return outcome


async def _handle_mcp_session_list(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> CommandOutcome:
    outcome = CommandOutcome()
    if request.server_identity is None:
        entries = await session_client.list_jar()
        connected_servers = _connected_jar_server_names(entries)

        if not connected_servers:
            outcome.add_message(
                "No connected MCP servers available.",
                channel="warning",
                right_info="mcp",
                agent_name=request.agent_name,
            )
            return outcome

        rows: list[ServerCookiesView] = []
        for connected_server in connected_servers:
            server_cookies = await session_client.list_server_cookies(connected_server)
            rows.append(server_cookies)

        outcome.add_message(
            _render_connected_server_cookies_table(rows),
            right_info="mcp",
            agent_name=request.agent_name,
        )
        return outcome

    server_cookies = await session_client.list_server_cookies(request.server_identity)
    outcome.add_message(
        _render_server_cookies_table(
            server_name=server_cookies.server_name,
            server_identity=server_cookies.server_identity,
            target=server_cookies.target,
            sessions_supported=server_cookies.sessions_supported,
            cookies=server_cookies.cookies,
            active_session_id=server_cookies.active_session_id,
            store_size_display=request.store_size_display,
            include_store=True,
        ),
        right_info="mcp",
        agent_name=request.agent_name,
    )
    return outcome


async def _render_session_action_outcome(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
    *,
    server_name: str,
    heading: str,
) -> CommandOutcome:
    outcome = CommandOutcome()
    server_cookies = await session_client.list_server_cookies(server_name)
    outcome.add_message(
        _render_session_action_result(
            heading=heading,
            server_name=server_cookies.server_name,
            server_identity=server_cookies.server_identity,
            target=server_cookies.target,
            sessions_supported=server_cookies.sessions_supported,
            cookies=server_cookies.cookies,
            active_session_id=server_cookies.active_session_id,
        ),
        right_info="mcp",
        agent_name=request.agent_name,
    )
    return outcome


async def _run_session_action(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
    *,
    operation: McpSessionServerOperation,
    heading: Callable[[str], str],
) -> CommandOutcome:
    server_name = await operation(session_client, request)
    return await _render_session_action_outcome(
        session_client,
        request,
        server_name=server_name,
        heading=heading(server_name),
    )


async def _create_session_for_request(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> str:
    server_name, _cookie = await session_client.create_session(
        request.server_identity,
        title=request.title,
    )
    return server_name


async def _resume_session_for_request(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> str:
    if not request.session_id:
        raise ValueError("Session id is required for use.")
    server_name, _cookie = await session_client.resume_session(
        request.server_identity,
        session_id=request.session_id,
    )
    return server_name


async def _clear_session_for_request(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> str:
    return await session_client.clear_cookie(request.server_identity)


async def _handle_mcp_session_new(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> CommandOutcome:
    return await _run_session_action(
        session_client,
        request,
        operation=_create_session_for_request,
        heading=lambda server_name: f"Created new MCP session for {server_name}.",
    )


async def _handle_mcp_session_use(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> CommandOutcome:
    return await _run_session_action(
        session_client,
        request,
        operation=_resume_session_for_request,
        heading=lambda server_name: f"Selected MCP session for {server_name}.",
    )


async def _handle_mcp_session_clear(
    session_client: SessionClientProtocol,
    request: McpSessionRequest,
) -> CommandOutcome:
    outcome = CommandOutcome()
    if request.clear_all:
        cleared = await session_client.clear_all_cookies()
        if not cleared:
            outcome.add_message(
                "No attached MCP servers to clear.",
                channel="warning",
                right_info="mcp",
                agent_name=request.agent_name,
            )
            return outcome
        outcome.add_message(
            _render_clear_all_result(cleared),
            right_info="mcp",
            agent_name=request.agent_name,
        )
        return outcome

    return await _run_session_action(
        session_client,
        request,
        operation=_clear_session_for_request,
        heading=lambda server_name: f"Cleared MCP session entry for {server_name}.",
    )


_MCP_SESSION_ACTION_HANDLERS: dict[McpSessionAction, McpSessionActionHandler] = {
    "jar": _handle_mcp_session_jar,
    "list": _handle_mcp_session_list,
    "new": _handle_mcp_session_new,
    "use": _handle_mcp_session_use,
    "clear": _handle_mcp_session_clear,
}


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
        request = McpSessionRequest(
            agent_name=agent_name,
            server_identity=server_identity,
            session_id=session_id,
            title=title,
            clear_all=clear_all,
            store_size_display=store_size_display,
        )
        action_handler = _MCP_SESSION_ACTION_HANDLERS.get(action)
        if action_handler is not None:
            return await action_handler(session_client, request)

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


async def _emit_connect_progress(
    on_progress: Callable[[str], Awaitable[None]] | None,
    message: str,
) -> None:
    if on_progress is None:
        return
    with suppress(Exception):
        await on_progress(message)


async def _emit_connect_oauth_event(
    event: OAuthEvent,
    *,
    on_progress: Callable[[str], Awaitable[None]] | None,
    on_oauth_event: Callable[[OAuthEvent], Awaitable[None]] | None,
    progress_state: _OAuthProgressState,
) -> None:
    if on_oauth_event is not None:
        with suppress(Exception):
            await on_oauth_event(event)

    if event.event_type == "authorization_url" and event.url:
        if event.url not in progress_state.links_seen:
            progress_state.links_seen.add(event.url)
            progress_state.links_ordered.append(event.url)
            await _emit_connect_progress(
                on_progress,
                f"Open this link to authorize: {event.url}",
            )
        return

    oauth_progress_messages = {
        "wait_start": event.message or "Waiting for OAuth callback (startup timer paused)…",
        "wait_end": event.message or "OAuth callback wait complete.",
        "callback_received": event.message
        or "OAuth callback received. Completing token exchange…",
    }
    if event.event_type in oauth_progress_messages:
        await _emit_connect_progress(on_progress, oauth_progress_messages[event.event_type])
        return

    if event.event_type == "oauth_error" and event.message:
        await _emit_connect_progress(on_progress, f"OAuth status: {event.message}")


def _default_connect_timeout_seconds(
    *,
    mode: _McpConnectRuntimeMode,
    trigger_oauth: bool | None,
) -> float:
    # OAuth-backed URL servers often need additional non-callback time for
    # metadata discovery and token exchange after the browser callback.
    return 30.0 if (mode == "url" and trigger_oauth is not False) else 10.0


def _connect_server_config(
    *,
    configured_alias: str | None,
    parsed: ParsedMcpConnectRequest,
) -> tuple[str, MCPServerSettings | None]:
    if configured_alias is not None:
        return configured_alias, None

    built_config = build_server_config_from_target(
        parsed.target,
        auth_token=parsed.options.auth_token,
    )
    return built_config.server_name, built_config.settings


def _build_connect_plan(
    *,
    parsed: ParsedMcpConnectRequest,
    configured_alias: str | None,
    oauth_event_handler: Callable[[OAuthEvent], Awaitable[None]] | None,
    allow_oauth_paste_fallback: bool,
) -> _McpConnectPlan:
    mode = _connect_runtime_mode(
        configured_alias=configured_alias,
        target_mode=parsed.target.mode,
    )
    server_name = configured_alias or infer_server_name(parsed.target)
    startup_timeout_seconds = parsed.options.timeout_seconds
    if startup_timeout_seconds is None:
        startup_timeout_seconds = _default_connect_timeout_seconds(
            mode=mode,
            trigger_oauth=parsed.options.trigger_oauth,
        )

    server_name, config = _connect_server_config(configured_alias=configured_alias, parsed=parsed)
    attach_options = MCPAttachOptions(
        startup_timeout_seconds=startup_timeout_seconds,
        trigger_oauth=parsed.options.trigger_oauth,
        force_reconnect=parsed.options.force_reconnect,
        reconnect_on_disconnect=parsed.options.reconnect_on_disconnect,
        oauth_event_handler=oauth_event_handler,
        allow_oauth_paste_fallback=allow_oauth_paste_fallback,
    )
    return _McpConnectPlan(
        mode=mode,
        server_name=server_name,
        config=config,
        attach_options=attach_options,
    )


def _add_oauth_registration_404_guidance(
    outcome: CommandOutcome,
    *,
    error_text: str,
    agent_name: str,
) -> None:
    normalized_error = strip_casefold(error_text)
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


def _add_oauth_recovery_guidance(outcome: CommandOutcome, *, agent_name: str) -> None:
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


def _add_connect_failure_guidance(
    outcome: CommandOutcome,
    *,
    error_text: str,
    oauth_paste_fallback_enabled: bool,
    agent_name: str,
) -> None:
    classification = _classify_connect_failure(error_text)

    if classification.oauth_registration_404:
        _add_oauth_registration_404_guidance(
            outcome,
            error_text=error_text,
            agent_name=agent_name,
        )

    if classification.oauth_related and (
        classification.oauth_fallback_unavailable
        or classification.oauth_timeout
        or not oauth_paste_fallback_enabled
    ):
        _add_oauth_recovery_guidance(outcome, agent_name=agent_name)


def _classify_connect_failure(error_text: str) -> _McpConnectFailureClassification:
    normalized_error = strip_casefold(error_text)
    non_oauth_startup_timeout = "non-oauth startup budget" in normalized_error
    oauth_related = "oauth" in normalized_error and not non_oauth_startup_timeout
    oauth_registration_404 = (
        "oauth" in normalized_error and "registration failed: 404" in normalized_error
    )
    oauth_fallback_unavailable = (
        "paste fallback is disabled" in normalized_error
        or "non-interactive connection mode" in normalized_error
    )
    oauth_timeout = oauth_related and any(
        phrase in normalized_error
        for phrase in (
            "timed out",
            "timeout",
            "deadline",
            "callback wait",
        )
    )
    return _McpConnectFailureClassification(
        oauth_related=oauth_related,
        oauth_registration_404=oauth_registration_404,
        oauth_fallback_unavailable=oauth_fallback_unavailable,
        oauth_timeout=oauth_timeout,
    )


def _connect_success_message(
    ctx,
    *,
    action: str,
    mode: _McpConnectRuntimeMode,
    server_name: str,
    target: NormalizedMcpTarget,
) -> str:
    if mode != "configured":
        return f"{action} MCP server '{server_name}' ({mode})."

    configured_source = _resolve_configured_source_from_context(ctx, server_name)
    source_text = configured_source or render_normalized_target(target)
    return f"{action} MCP server '{server_name}' from configuration: {source_text}."


async def _add_connect_success_messages(
    ctx,
    outcome: CommandOutcome,
    *,
    result: MCPAttachResult,
    parsed: ParsedMcpConnectRequest,
    counts: _McpAttachCounts,
    mode: _McpConnectRuntimeMode,
    server_name: str,
    agent_name: str,
    on_progress: Callable[[str], Awaitable[None]] | None,
) -> None:
    if result.already_attached and not parsed.options.force_reconnect:
        message_text = (
            f"MCP server '{server_name}' is already attached. "
            "Use --reconnect to force reconnect and refresh tools."
        )
        outcome.add_message(
            message_text,
            channel="warning",
            right_info="mcp",
            agent_name=agent_name,
            metadata={
                "mcp_connect_status": "already_attached",
                "mcp_connect_details": message_text,
            },
        )
        await _emit_connect_progress(
            on_progress,
            f"MCP server '{server_name}' is already connected.",
        )
        return

    reconnected = result.already_attached and parsed.options.force_reconnect
    action = "Reconnected" if reconnected else "Connected"
    message_text = _connect_success_message(
        ctx,
        action=action,
        mode=mode,
        server_name=server_name,
        target=parsed.target,
    )
    outcome.add_message(
        message_text,
        right_info="mcp",
        agent_name=agent_name,
        metadata={
            "mcp_connect_status": "reconnected" if reconnected else "connected",
            "mcp_connect_details": message_text,
        },
    )
    summary = (
        _format_refreshed_summary(counts)
        if reconnected
        else _format_added_summary(counts)
    )
    outcome.add_message(summary, right_info="mcp", agent_name=agent_name)
    await _emit_connect_progress(on_progress, f"{action} MCP server '{server_name}'.")


async def handle_mcp_connect(
    ctx,
    *,
    manager: McpRuntimeManager,
    agent_name: str,
    request: ParsedMcpConnectRequest,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
    on_oauth_event: Callable[[OAuthEvent], Awaitable[None]] | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    await _emit_connect_progress(on_progress, "Preparing MCP connection…")

    oauth_progress_state = _OAuthProgressState(links_seen=set(), links_ordered=[])
    oauth_paste_fallback_enabled = on_progress is None and on_oauth_event is None

    async def emit_oauth_event(event: OAuthEvent) -> None:
        await _emit_connect_oauth_event(
            event,
            on_progress=on_progress,
            on_oauth_event=on_oauth_event,
            progress_state=oauth_progress_state,
        )

    try:
        parsed = _resolve_request_auth(request)
    except ValueError as exc:
        outcome.add_message(f"Invalid MCP connect arguments: {exc}", channel="error")
        return outcome

    configured_alias = await _resolve_configured_server_alias(
        manager=manager,
        agent_name=agent_name,
        request=parsed,
    )

    plan = _build_connect_plan(
        parsed=parsed,
        configured_alias=configured_alias,
        oauth_event_handler=emit_oauth_event
        if (on_progress is not None or on_oauth_event is not None)
        else None,
        allow_oauth_paste_fallback=oauth_paste_fallback_enabled,
    )
    if plan.mode == "configured":
        await _emit_connect_progress(
            on_progress, f"Connecting MCP server '{plan.server_name}' from config file…"
        )
    else:
        await _emit_connect_progress(
            on_progress, f"Connecting MCP server '{plan.server_name}' via {plan.mode}…"
        )

    try:
        result = await manager.attach_mcp_server(
            agent_name,
            plan.server_name,
            server_config=plan.config,
            options=plan.attach_options,
        )
    except Exception as exc:
        await _emit_connect_progress(
            on_progress, f"Failed to connect MCP server '{plan.server_name}'."
        )
        error_text = str(exc)
        outcome.add_message(f"Failed to connect MCP server: {error_text}", channel="error")
        _add_connect_failure_guidance(
            outcome,
            error_text=error_text,
            oauth_paste_fallback_enabled=oauth_paste_fallback_enabled,
            agent_name=agent_name,
        )
        return outcome

    counts = _mcp_attach_counts(result)
    await _add_connect_success_messages(
        ctx,
        outcome,
        result=result,
        parsed=parsed,
        counts=counts,
        mode=plan.mode,
        server_name=plan.server_name,
        agent_name=agent_name,
        on_progress=on_progress,
    )
    for warning in result.warnings:
        outcome.add_message(warning, channel="warning", right_info="mcp", agent_name=agent_name)

    if oauth_progress_state.links_ordered:
        outcome.add_message(
            f"OAuth authorization link: {oauth_progress_state.links_ordered[-1]}",
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

    if not result.detached:
        outcome.add_message(
            f"MCP server '{server_name}' was not attached.",
            channel="warning",
            right_info="mcp",
            agent_name=agent_name,
        )
        return outcome

    outcome.add_message(
        f"Disconnected MCP server '{server_name}'.",
        right_info="mcp",
        agent_name=agent_name,
    )
    outcome.add_message(
        _format_removed_summary(
            _McpResourceCounts(tools=len(result.tools_removed), prompts=len(result.prompts_removed))
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

    counts = _mcp_attach_counts(result)

    message_text = f"Reconnected MCP server '{server_name}'."
    outcome.add_message(
        message_text,
        right_info="mcp",
        agent_name=agent_name,
        metadata={
            "mcp_connect_status": "reconnected",
            "mcp_connect_details": message_text,
        },
    )
    outcome.add_message(
        _format_refreshed_summary(counts),
        right_info="mcp",
        agent_name=agent_name,
    )

    for warning in result.warnings:
        outcome.add_message(warning, channel="warning", right_info="mcp", agent_name=agent_name)

    return outcome
