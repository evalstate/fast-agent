"""Runtime MCP connect/list/disconnect command handlers."""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypeAlias,
    cast,
)

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
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
from fast_agent.utils.count_display import format_count_parts
from fast_agent.utils.numeric import nonnegative_int_or_none
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.oauth_client import OAuthEvent


_McpConnectRuntimeMode: TypeAlias = Literal["configured", "url", "stdio", "npx", "uvx"]


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
        "callback_received": event.message or "OAuth callback received. Completing token exchange…",
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
            "Run `fast-agent auth mcp login <server-name-or-mcp-name>` on the fast-agent host, "
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
    summary = _format_refreshed_summary(counts) if reconnected else _format_added_summary(counts)
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
