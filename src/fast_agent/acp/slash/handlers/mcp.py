"""MCP slash command handlers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fast_agent.acp.slash.tool_updates import (
    ToolCallStatus,
    send_fetch_tool_call_start,
    send_tool_call_progress,
)
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.mcp_command_intents import (
    MCP_TOP_LEVEL_ACTIONS,
    McpServerNameIntent,
    is_mcp_top_level_action,
    parse_mcp_no_args_tokens,
    parse_mcp_server_name_tokens,
)
from fast_agent.mcp.connect_targets import (
    parse_connect_command_text,
    render_connect_request,
    render_normalized_target,
)
from fast_agent.utils.action_normalization import is_help_flag
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.slash_commands import split_subcommand_and_remainder
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult

    _McpCommandHandler = Callable[..., Awaitable[str]]


@dataclass(frozen=True, slots=True)
class _ConnectOutcomeSummary:
    has_error: bool
    failure_details: str | None = None
    completion_details: str | None = None


@dataclass(frozen=True, slots=True)
class _ConnectProgressMessage:
    message: str
    oauth_authorization_url: str | None


@dataclass(frozen=True, slots=True)
class _AcpMcpRuntimeManager:
    handler: "SlashCommandHandler"

    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: "MCPServerSettings | None" = None,
        options: "MCPAttachOptions | None" = None,
    ) -> "MCPAttachResult":
        callback = self.handler._attach_mcp_server_callback
        if callback is None:
            raise RuntimeError("Runtime MCP server attachment is not available.")
        result = await callback(agent_name, server_name, server_config, options)
        return cast("MCPAttachResult", result)

    async def detach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
    ) -> "MCPDetachResult":
        callback = self.handler._detach_mcp_server_callback
        if callback is None:
            raise RuntimeError("Runtime MCP server detachment is not available.")
        result = await callback(agent_name, server_name)
        return cast("MCPDetachResult", result)

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]:
        callback = self.handler._list_attached_mcp_servers_callback
        if callback is None:
            raise RuntimeError("Runtime MCP server listing is not available.")
        return await callback(agent_name)

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]:
        callback = self.handler._list_configured_detached_mcp_servers_callback
        if callback is None:
            return []
        return await callback(agent_name)


def _parse_mcp_server_name_argument(
    tokens: list[str],
    *,
    heading: str,
    subcommand: str,
) -> McpServerNameIntent:
    intent = parse_mcp_server_name_tokens(
        tokens,
        usage=f"Usage: /mcp {subcommand} <server_name>",
    )
    if intent.error:
        return McpServerNameIntent(server_name=None, error=f"{heading}\n\n{intent.error}")
    return intent


def _parse_mcp_list_arguments(tokens: list[str], *, heading: str) -> str | None:
    intent = parse_mcp_no_args_tokens(tokens, usage="Usage: /mcp list")
    if intent.error:
        return f"{heading}\n\n{intent.error}"
    return None


def _mcp_usage_text(heading: str) -> str:
    return (
        f"{heading}\n\n"
        "Usage:\n"
        "- /mcp list\n"
        "- /mcp connect <target> [--name <server>] [--auth <token>] [--timeout <seconds>] "
        "[--oauth|--no-oauth] [--reconnect|--no-reconnect]\n"
        '  Example: /mcp connect "C:\\Program Files\\Tool\\tool.exe" --flag\n'
        "- /mcp disconnect <server_name>\n"
        "- /mcp reconnect <server_name>"
    )


async def _refresh_acp_instruction_cache(handler: "SlashCommandHandler") -> None:
    if not handler._acp_context:
        return
    agent = handler._get_current_agent()
    await handler._acp_context.invalidate_instruction_cache(
        handler.current_agent_name,
        agent.instruction if agent else None,
    )
    await handler._acp_context.send_available_commands_update()


async def _send_connect_tool_update(
    handler: "SlashCommandHandler",
    *,
    tool_call_id: str,
    title: str,
    status: ToolCallStatus,
    message: str | None = None,
) -> None:
    if handler._acp_context is None:
        return
    await send_tool_call_progress(
        handler._acp_context,
        tool_call_id=tool_call_id,
        title=title,
        status=status,
        message=message,
    )


def _connect_tool_call_title(request) -> str:
    connect_label = "MCP server"
    if request.target.server_name:
        connect_label = f"MCP server '{request.target.server_name}'"
    else:
        target_text = render_normalized_target(request.target)
        try:
            target_tokens = split_commandline(target_text, syntax="posix")
        except ValueError:
            target_tokens = target_text.split()
        first_target_token = target_tokens[0] if target_tokens else target_text
        connect_label = f"MCP target '{first_target_token}'"
    return f"Connect {connect_label}"


def _rewrite_connect_progress_message(
    handler: "SlashCommandHandler",
    *,
    message: str,
    oauth_authorization_url: str | None,
) -> _ConnectProgressMessage:
    if message.startswith("Open this link to authorize:"):
        oauth_authorization_url = strip_to_none(message.split(":", 1)[1])

    if (
        oauth_authorization_url
        and (
            "Waiting for OAuth callback" in message
            or "Waiting for pasted OAuth callback URL" in message
        )
        and "OAuth authorization link:" not in message
    ):
        message = f"{message}\nOAuth authorization link: {oauth_authorization_url}"

    if handler._acp_context is not None and "Waiting for OAuth callback" in message:
        if "Stop/Cancel" not in message:
            message = f"{message}\nTo cancel, use your ACP client's Stop/Cancel action."
        if "fast-agent auth mcp login" not in message:
            message = (
                f"{message}\n"
                "If the browser cannot reach the callback host, run "
                "`fast-agent auth mcp login <server-name-or-mcp-name>` on the "
                "fast-agent host, then retry `/mcp connect ...`."
            )

    return _ConnectProgressMessage(
        message=message,
        oauth_authorization_url=oauth_authorization_url,
    )


async def _start_connect_tool_call(
    handler: "SlashCommandHandler",
    *,
    tool_call_id: str,
    tool_call_title: str,
    display_target: str,
) -> None:
    if handler._acp_context is None:
        return
    started = await send_fetch_tool_call_start(
        handler._acp_context,
        tool_call_id=tool_call_id,
        title=f"{tool_call_title} (open for details)",
    )
    if not started:
        return
    await _send_connect_tool_update(
        handler,
        tool_call_id=tool_call_id,
        title=tool_call_title,
        status="in_progress",
        message=(
            f"{display_target}\nOpen this tool call to view OAuth links and live connection status."
        ),
    )


def _summarize_connect_outcome(outcome) -> _ConnectOutcomeSummary:
    has_error = any(msg.channel == "error" for msg in outcome.messages)
    if has_error:
        first_error = next((msg for msg in outcome.messages if msg.channel == "error"), None)
        failure_details = str(first_error.text) if first_error is not None else None
        return _ConnectOutcomeSummary(has_error=True, failure_details=failure_details)

    success_message = _connect_success_message_from_metadata(outcome) or next(
        (
            str(msg.text)
            for msg in outcome.messages
            if (
                "Connected MCP server" in str(msg.text)
                or "Reconnected MCP server" in str(msg.text)
                or "already attached" in strip_casefold(str(msg.text))
            )
        ),
        "MCP connection complete.",
    )
    oauth_link_message = next(
        (
            str(msg.text)
            for msg in outcome.messages
            if str(msg.text).startswith("OAuth authorization link:")
        ),
        None,
    )
    completion_details = (
        f"{success_message}\n{oauth_link_message}" if oauth_link_message else success_message
    )
    return _ConnectOutcomeSummary(has_error=False, completion_details=completion_details)


def _connect_success_message_from_metadata(outcome) -> str | None:
    for msg in outcome.messages:
        metadata = getattr(msg, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if metadata.get("mcp_connect_status") in {
            "already_attached",
            "connected",
            "reconnected",
        }:
            details = metadata.get("mcp_connect_details")
            return str(details) if details is not None else str(msg.text)
    return None


def _strip_oauth_link_messages(outcome, *, oauth_authorization_url: str | None) -> None:
    if oauth_authorization_url is None:
        return
    outcome.messages = [
        message
        for message in outcome.messages
        if not str(message.text).startswith("OAuth authorization link:")
    ]


async def _emit_connect_completion_progress(
    handler: "SlashCommandHandler",
    *,
    summary: _ConnectOutcomeSummary,
) -> None:
    if summary.has_error:
        if summary.failure_details:
            await handler._send_progress_update(f"❌ {summary.failure_details}")
        return
    if summary.completion_details:
        await handler._send_progress_update(f"✅ {summary.completion_details}")


async def _handle_mcp_list_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    tokens: list[str] | None = None,
) -> str:
    del ctx
    usage_error = _parse_mcp_list_arguments(tokens or ["list"], heading=heading)
    if usage_error:
        return usage_error
    if handler._list_attached_mcp_servers_callback is None:
        return "mcp\n\nRuntime MCP server listing is not available."
    outcome = await mcp_runtime_handlers.handle_mcp_list(
        manager=manager,
        agent_name=handler.current_agent_name,
    )
    return handler._format_outcome_as_markdown(outcome, heading, io=io)


async def _handle_mcp_connect_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    remainder: str,
) -> str:
    if handler._attach_mcp_server_callback is None:
        return "mcp\n\nRuntime MCP server attachment is not available."
    if not remainder:
        return (
            f"{heading}\n\nUsage: /mcp connect <target> [--name <server>] [--auth <token>] "
            "[--timeout <seconds>] [--oauth|--no-oauth] [--reconnect|--no-reconnect]"
        )
    try:
        request = parse_connect_command_text(remainder)
    except ValueError as exc:
        return f"{heading}\n\n{exc}"

    display_target = render_connect_request(request, redact_auth=True)
    tool_call_id = handler._build_tool_call_id()
    oauth_authorization_url: str | None = None
    tool_call_title = _connect_tool_call_title(request)

    async def _send_connect_progress(message: str) -> None:
        nonlocal oauth_authorization_url

        progress_message = _rewrite_connect_progress_message(
            handler,
            message=message,
            oauth_authorization_url=oauth_authorization_url,
        )
        message = progress_message.message
        oauth_authorization_url = progress_message.oauth_authorization_url

        if handler._acp_context is None:
            await handler._send_progress_update(message)
            return
        await _send_connect_tool_update(
            handler,
            tool_call_id=tool_call_id,
            title=tool_call_title,
            status="in_progress",
            message=message,
        )

    await _start_connect_tool_call(
        handler,
        tool_call_id=tool_call_id,
        tool_call_title=tool_call_title,
        display_target=display_target,
    )

    try:
        outcome = await mcp_runtime_handlers.handle_mcp_connect(
            ctx,
            manager=manager,
            agent_name=handler.current_agent_name,
            request=request,
            on_progress=_send_connect_progress,
        )
    except asyncio.CancelledError:
        await _send_connect_tool_update(
            handler,
            tool_call_id=tool_call_id,
            title="Connection cancelled",
            status="failed",
            message="Connection cancelled by client.",
        )
        raise

    if handler._acp_context is not None:
        _strip_oauth_link_messages(
            outcome,
            oauth_authorization_url=oauth_authorization_url,
        )

    summary = _summarize_connect_outcome(outcome)
    await _send_connect_tool_update(
        handler,
        tool_call_id=tool_call_id,
        title=tool_call_title,
        status="failed" if summary.has_error else "completed",
        message=summary.failure_details if summary.has_error else summary.completion_details,
    )

    await _emit_connect_completion_progress(
        handler,
        summary=summary,
    )

    if handler._acp_context:
        agent = handler._get_current_agent()
        await handler._acp_context.invalidate_instruction_cache(
            handler.current_agent_name,
            agent.instruction if agent else None,
        )
        await handler._acp_context.send_available_commands_update()
    return handler._format_outcome_as_markdown(outcome, heading, io=io)


async def _handle_mcp_disconnect_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    tokens: list[str],
) -> str:
    if handler._detach_mcp_server_callback is None:
        return "mcp\n\nRuntime MCP server detachment is not available."
    intent = _parse_mcp_server_name_argument(
        tokens,
        heading=heading,
        subcommand="disconnect",
    )
    if intent.error:
        return intent.error
    outcome = await mcp_runtime_handlers.handle_mcp_disconnect(
        ctx,
        manager=manager,
        agent_name=handler.current_agent_name,
        server_name=cast("str", intent.server_name),
    )
    await _refresh_acp_instruction_cache(handler)
    return handler._format_outcome_as_markdown(outcome, heading, io=io)


async def _handle_mcp_reconnect_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    tokens: list[str],
) -> str:
    if handler._attach_mcp_server_callback is None:
        return "mcp\n\nRuntime MCP server attachment is not available."
    intent = _parse_mcp_server_name_argument(
        tokens,
        heading=heading,
        subcommand="reconnect",
    )
    if intent.error:
        return intent.error
    outcome = await mcp_runtime_handlers.handle_mcp_reconnect(
        ctx,
        manager=manager,
        agent_name=handler.current_agent_name,
        server_name=cast("str", intent.server_name),
    )
    await _refresh_acp_instruction_cache(handler)
    return handler._format_outcome_as_markdown(outcome, heading, io=io)


_MCP_COMMAND_HANDLERS: dict[str, "_McpCommandHandler"] = {
    "list": _handle_mcp_list_command,
    "disconnect": _handle_mcp_disconnect_command,
    "reconnect": _handle_mcp_reconnect_command,
}
if set(_MCP_COMMAND_HANDLERS) | {"connect"} != set(MCP_TOP_LEVEL_ACTIONS):
    raise RuntimeError("ACP MCP handlers do not match shared MCP actions")


async def handle_mcp(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    heading = "mcp"
    args = (arguments or "").strip() or "list"
    subcmd_text, remainder = split_subcommand_and_remainder(args)
    subcmd = strip_casefold(subcmd_text or "list")

    if is_help_flag(subcmd):
        return _mcp_usage_text(heading)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    manager = _AcpMcpRuntimeManager(handler)

    if subcmd == "connect":
        return await _handle_mcp_connect_command(
            handler,
            heading=heading,
            ctx=ctx,
            io=io,
            manager=manager,
            remainder=remainder,
        )

    try:
        tokens = split_commandline(args, syntax="posix")
    except ValueError as exc:
        return f"{heading}\n\nInvalid arguments: {exc}"

    if not is_mcp_top_level_action(subcmd):
        return _mcp_usage_text(heading)
    handler_func = _MCP_COMMAND_HANDLERS[subcmd]

    return await handler_func(
        handler,
        heading=heading,
        ctx=ctx,
        io=io,
        manager=manager,
        tokens=tokens,
    )
