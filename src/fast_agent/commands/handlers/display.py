"""Shared display command handlers."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from fast_agent.commands.command_discovery import (
    command_discovery_names,
    parse_commands_discovery_arguments,
    render_command_detail_markdown,
    render_commands_index_markdown,
    render_commands_json,
)
from fast_agent.commands.results import CommandOutcome
from fast_agent.config import get_settings
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.tools.durable_processes import (
    DurableProcessRecordError,
    DurableProcessSnapshot,
)
from fast_agent.ui.usage_display import collect_agents_from_provider
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.markdown import escape_markdown_table_cell, markdown_code_span
from fast_agent.utils.text import strip_to_none, summarize_command
from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.tools.execution_environment import ShellRuntimeInfo
    from fast_agent.tools.shell_runtime import ManagedProcessSnapshot, ProcessTerminationOutcome
    from fast_agent.types import PromptMessageExtended


@runtime_checkable
class _ShellRuntimeProvider(Protocol):
    @property
    def shell_runtime(self) -> object | None: ...


@runtime_checkable
class _ShellRuntimeInfoProvider(Protocol):
    def runtime_info(self) -> "ShellRuntimeInfo": ...


@runtime_checkable
class _ManagedProcessRuntimeProvider(Protocol):
    async def process_snapshots(self) -> tuple["ManagedProcessSnapshot", ...]: ...

    async def discover_durable_processes(self) -> tuple[DurableProcessSnapshot, ...]: ...

    async def attach_durable_process(
        self,
        process_id: str,
        *,
        session_id: str | None = None,
    ) -> DurableProcessSnapshot: ...


@runtime_checkable
class _ProcessTerminationRuntimeProvider(Protocol):
    async def terminate_interactive_process(
        self,
        process_id: str,
    ) -> "ProcessTerminationOutcome": ...


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
        verbatim=True,
    )

    return outcome


async def handle_show_mcp_status(
    ctx: CommandContext,
    *,
    agent_name: str,
) -> CommandOutcome:
    outcome = CommandOutcome()
    from fast_agent.ui.enhanced_prompt import show_mcp_status

    await show_mcp_status(
        agent_name,
        cast("AgentApp", ctx.agent_provider),
    )
    return outcome


def _active_environment_name(agent: object) -> str | None:
    if not isinstance(agent, _ShellRuntimeProvider):
        return None
    shell_runtime = agent.shell_runtime
    if not isinstance(shell_runtime, _ShellRuntimeInfoProvider):
        return None
    return shell_runtime.runtime_info().environment_name


def _active_runtime_detail(agent: object) -> str:
    if not isinstance(agent, _ShellRuntimeProvider):
        return ""
    shell_runtime = agent.shell_runtime
    if not isinstance(shell_runtime, _ShellRuntimeInfoProvider):
        return ""
    info = shell_runtime.runtime_info()
    parts = (
        info.kind,
        info.provider,
        info.name,
    )
    return " / ".join(part for part in parts if part)


async def handle_environment(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.settings or get_settings()
    active_agent = ctx.agent_provider._agent(agent_name)
    active_name = _active_environment_name(active_agent) or settings.default_environment
    runtime_detail = _active_runtime_detail(active_agent)

    lines = ["# environments", ""]
    if runtime_detail:
        lines.extend([f"Active runtime: `{runtime_detail}`", ""])
    lines.extend(["| Name | Type | Active |", "| --- | --- | --- |"])
    for name in sorted({"local", *settings.environments}):
        spec = settings.environments.get(name)
        environment_type = spec.type if spec is not None else "local"
        marker = "yes" if name == active_name else ""
        lines.append(f"| `{name}` | `{environment_type}` | {marker} |")

    outcome.add_message("\n".join(lines), right_info="environment", render_markdown=True)
    return outcome


async def handle_processes(
    ctx: CommandContext,
    *,
    agent_name: str,
    show_history: bool = False,
    attach_process_id: str | None = None,
    terminate_process_id: str | None = None,
) -> CommandOutcome:
    """Render active processes by default, or retained finished history."""
    outcome = CommandOutcome()
    agent = ctx.agent_provider._agent(agent_name)
    if not isinstance(agent, _ShellRuntimeProvider):
        outcome.add_message(
            "Managed shell processes are not available for this agent.",
            channel="warning",
            right_info="process",
        )
        return outcome

    shell_runtime = agent.shell_runtime
    if not isinstance(shell_runtime, _ManagedProcessRuntimeProvider):
        outcome.add_message(
            "Managed shell processes are not enabled for this agent.",
            channel="warning",
            right_info="process",
        )
        return outcome

    if attach_process_id is not None:
        session_id = (
            ctx.session_runtime.current_session_id() if ctx.session_runtime is not None else None
        ) or ctx.acp_session_id
        try:
            snapshot = await shell_runtime.attach_durable_process(
                attach_process_id,
                session_id=session_id,
            )
        except (DurableProcessRecordError, ValueError) as exc:
            outcome.add_message(
                f"Could not attach durable process `{attach_process_id}`: {exc}",
                channel="error",
                right_info="process attach",
                render_markdown=True,
            )
            return outcome
        status = _durable_process_display_status(snapshot)
        lines = [
            f"# attached `{snapshot.spec.process_id}`",
            "",
            f"- **status:** {status}",
            f"- **PID:** {snapshot.status.child_pid or '—'}",
            f"- **cwd:** `{snapshot.spec.cwd}`",
            f"- **command:** {markdown_code_span(summarize_command(snapshot.spec.command))}",
            "",
            (
                "Management and output observation are now attached to this runtime; "
                "terminal input is not attached."
            ),
        ]
        if session_id is not None:
            lines.append(f"Associated with session `{session_id}`.")
        outcome.add_message(
            "\n".join(lines),
            right_info="process attach",
            render_markdown=True,
        )
        return outcome

    if terminate_process_id is not None:
        if not isinstance(shell_runtime, _ProcessTerminationRuntimeProvider):
            outcome.add_message(
                "Process termination is not available for this runtime.",
                channel="warning",
                right_info="process terminate",
            )
            return outcome
        result = await shell_runtime.terminate_interactive_process(terminate_process_id)
        if result.state == "terminated":
            message = f"Terminated managed process `{terminate_process_id}`."
            channel = "system"
        elif result.state == "stop_requested":
            message = f"Termination requested for durable process `{terminate_process_id}`."
            channel = "system"
        elif result.state == "stop_already_requested":
            message = f"Termination was already requested for `{terminate_process_id}`."
            channel = "info"
        elif result.state == "already_exited":
            message = f"Process `{terminate_process_id}` has already exited."
            channel = "info"
        elif result.state == "unavailable":
            message = f"Process `{terminate_process_id}` is unavailable; no stop request was sent."
            channel = "error"
        elif result.state == "not_found":
            message = f"Managed process `{terminate_process_id}` was not found."
            channel = "error"
        else:
            detail = f": {result.error}" if result.error else "."
            message = f"Could not terminate process `{terminate_process_id}`{detail}"
            channel = "error"
        outcome.add_message(
            message,
            channel=channel,
            right_info="process terminate",
            render_markdown=True,
        )
        return outcome

    snapshots = list(await shell_runtime.process_snapshots())
    durable = list(await shell_runtime.discover_durable_processes())
    attached_ids = {snapshot.process_id for snapshot in snapshots}
    discoverable = [
        snapshot for snapshot in durable if snapshot.spec.process_id not in attached_ids
    ]
    if not snapshots and not discoverable:
        outcome.add_message(
            "No managed shell processes.",
            right_info="process",
        )
        return outcome

    active = [snapshot for snapshot in snapshots if snapshot.status == "running"]
    completed = [snapshot for snapshot in snapshots if snapshot.status != "running"]
    visible = completed if show_history else active
    visible_discoverable = [
        snapshot
        for snapshot in discoverable
        if (_durable_process_display_status(snapshot) != "running") == show_history
    ]
    if not visible and not visible_discoverable:
        message = (
            "No finished managed shell processes."
            if show_history
            else "No active managed shell processes. Use `/process --history` to show finished processes."
        )
        outcome.add_message(
            message,
            right_info="process history" if show_history else "process",
            render_markdown=not show_history,
        )
        return outcome

    if visible:
        lines = [
            "# finished managed processes" if show_history else "# active managed processes",
            "",
            (
                f"**{len(completed)} finished** · {len(snapshots)} retained"
                if show_history
                else f"↻ **{len(active)} active**"
            ),
            "",
            "| Process | Status | Elapsed | PID | Command |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for snapshot in visible:
            status = snapshot.status
            if snapshot.exit_code is not None:
                status = f"{status} ({snapshot.exit_code})"
            command = markdown_code_span(
                escape_markdown_table_cell(summarize_command(snapshot.command))
            )
            lines.append(
                "| "
                f"`{snapshot.process_id}` | {status} | "
                f"{format_duration(snapshot.elapsed_seconds)} | "
                f"{snapshot.os_process_id or '—'} | {command} |"
            )
        outcome.add_message(
            "\n".join(lines),
            right_info="process history" if show_history else "process",
            render_markdown=True,
        )
    if visible_discoverable:
        lines = [
            "# discoverable durable processes",
            "",
            "| Process | Status | PID | Session | Command |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for snapshot in visible_discoverable:
            sessions = ", ".join(snapshot.session_ids) or "—"
            lines.append(
                "| "
                f"`{snapshot.spec.process_id}` | "
                f"{_durable_process_display_status(snapshot)} | "
                f"{snapshot.status.child_pid or '—'} | "
                f"{escape_markdown_table_cell(sessions)} | "
                f"{markdown_code_span(escape_markdown_table_cell(summarize_command(snapshot.spec.command)))} |"
            )
        lines.extend(
            [
                "",
                "Use `/process attach <process-id>` to adopt management and output observation.",
                "Use `/process terminate <process-id>` to request termination without attaching.",
            ]
        )
        outcome.add_message(
            "\n".join(lines),
            right_info="durable processes",
            render_markdown=True,
        )
    return outcome


def _durable_process_display_status(snapshot: DurableProcessSnapshot) -> str:
    state = snapshot.status.state
    if state in {"created", "starting", "running", "stopping"}:
        return "running"
    if state == "stopped":
        return "terminated"
    if state == "unavailable":
        return "unavailable"
    return "completed" if snapshot.status.exit_code == 0 else "failed"


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
