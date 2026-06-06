"""Shared display command handlers."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, cast

from fast_agent.commands.command_discovery import (
    command_discovery_names,
    parse_commands_discovery_arguments,
    render_command_detail_markdown,
    render_commands_index_markdown,
    render_commands_json,
)
from fast_agent.commands.results import CommandOutcome
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.ui.usage_display import collect_agents_from_provider
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.markdown import markdown_code_span
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.types import PromptMessageExtended


def _last_assistant_message(
    message_history: Sequence["PromptMessageExtended"],
) -> "PromptMessageExtended | None":
    return next(
        (message for message in reversed(message_history) if message.role == "assistant"),
        None,
    )


def _last_assistant_text(message_history: Sequence["PromptMessageExtended"]) -> str | None:
    for message in reversed(message_history):
        if message.role != "assistant":
            continue
        text = message.last_text()
        if text:
            return text
    return None


def _decoded_process_output(output: bytes) -> str | None:
    return strip_to_none(output.decode("utf-8", errors="replace"))


async def handle_show_usage(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    del agent_name
    outcome = CommandOutcome()
    agents_to_show = collect_agents_from_provider(ctx.agent_provider)
    if not agents_to_show:
        outcome.add_message("No usage data available", channel="warning", right_info="usage")
        return outcome

    await ctx.io.display_usage_report(agents_to_show)
    return outcome


async def handle_show_system(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = cast("AgentProtocol", ctx.agent_provider._agent(agent_name))
    system_prompt = agent.instruction
    if not system_prompt:
        outcome.add_message("No system prompt available", channel="warning", right_info="system")
        return outcome

    server_count = 0
    if isinstance(agent, McpAgentProtocol):
        server_names = agent.aggregator.server_names
        server_count = len(server_names) if server_names else 0

    await ctx.io.display_system_prompt(
        agent_name,
        system_prompt,
        server_count=server_count,
    )

    return outcome


async def handle_show_markdown(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = ctx.agent_provider._agent(agent_name)
    if not agent.llm:
        outcome.add_message("No message history available", channel="warning")
        return outcome

    message_history = agent.message_history
    if not message_history:
        outcome.add_message("No messages in history", channel="warning")
        return outcome

    content = _last_assistant_text(message_history)
    if content is None:
        outcome.add_message("No assistant messages found", channel="warning")
        return outcome

    outcome.add_message(
        content,
        title="Last Assistant Response",
        right_info="display",
        agent_name=agent_name,
        render_markdown=True,
    )

    return outcome


async def handle_show_mcp_status(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    from fast_agent.ui.enhanced_prompt import show_mcp_status

    await show_mcp_status(agent_name, cast("AgentApp", ctx.agent_provider))
    return outcome


async def handle_check(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    del ctx, agent_name
    outcome = CommandOutcome()
    normalized_argument = strip_to_none(argument)
    command = [
        sys.executable,
        "-m",
        "fast_agent.cli.main",
        "--no-color",
        "check",
    ]
    if normalized_argument is not None:
        try:
            command.extend(split_commandline(normalized_argument, syntax="posix"))
        except ValueError as exc:
            outcome.add_message(f"Invalid check arguments: {exc}", channel="warning")
            return outcome

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    stdout_text = _decoded_process_output(stdout)
    stderr_text = _decoded_process_output(stderr)
    if stdout_text:
        outcome.add_message(stdout_text, right_info="check")
    if stderr_text:
        outcome.add_message(
            stderr_text,
            channel="error" if process.returncode else "warning",
            right_info="check",
        )
    if process.returncode != 0 and not stderr_text:
        outcome.add_message(
            f"fast-agent check exited with status code {process.returncode}.",
            channel="error",
            right_info="check",
        )
    return outcome


def _render_unknown_discovery_command(command_name: str) -> str:
    return (
        f"# commands\n\nUnknown command family: {markdown_code_span(command_name)}.\n"
        "Use `/commands` to list available commands."
    )


async def handle_commands(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    del ctx, agent_name
    outcome = CommandOutcome()
    command_args = argument or ""

    try:
        request = parse_commands_discovery_arguments(command_args)
    except ValueError as exc:
        outcome.add_message(f"# commands\n\n{exc}", render_markdown=True)
        return outcome

    command_names = set(command_discovery_names())
    if request.as_json:
        content = render_commands_json(
            command_name=request.command_name,
            action_name=request.action_name,
            command_names=command_names,
        )
    elif request.command_name is None:
        content = render_commands_index_markdown(command_names=command_names)
    elif request.command_name not in command_names:
        content = _render_unknown_discovery_command(request.command_name)
    else:
        content = render_command_detail_markdown(
            request.command_name,
            request.action_name,
        )
        if content is None:
            command = (
                f"/{request.command_name} {request.action_name}"
                if request.action_name is not None
                else request.command_name
            )
            content = f"# commands\n\nNo discovery metadata for {markdown_code_span(command)} yet."

    outcome.add_message(content, right_info="commands", render_markdown=True)
    return outcome
