"""Model-visible, non-interactive access to selected harness commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.commands.command_discovery import render_commands_index_markdown
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.results import CommandMessage
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.ui.usage_display import format_usage_markdown
from fast_agent.utils.slash_commands import parse_slash_command_line, split_subcommand_and_remainder
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.agents.tool_agent import ToolAgent
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.mcp_aggregator import ServerStatus

_COMMAND_HANDLERS = {
    "tools": tools_handlers.handle_list_tools,
    "prompts": prompt_handlers.handle_list_prompts,
    "usage": display_handlers.handle_show_usage,
    "system": display_handlers.handle_show_system,
    "markdown": display_handlers.handle_show_markdown,
}
_COMMAND_NAMES = ("commands", "tools", "prompts", "usage", "system", "markdown", "status")


@runtime_checkable
class _McpStatusProvider(Protocol):
    async def get_server_status(self) -> dict[str, "ServerStatus"]: ...


@dataclass(slots=True)
class _HarnessCommandIO(NonInteractiveCommandIOBase):
    messages: list[CommandMessage]

    async def emit(self, message: CommandMessage) -> None:
        self.messages.append(message)

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        await self.emit(CommandMessage(format_usage_markdown(agents), render_markdown=True))

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        await self.emit(
            CommandMessage(
                "\n".join(
                    (
                        f"Agent: `{agent_name}`",
                        f"Attached MCP servers: {server_count}",
                        "",
                        system_prompt,
                    )
                ),
                render_markdown=True,
            )
        )


def _agent_map(agent: ToolAgent) -> Mapping[str, AgentProtocol]:
    agents = dict(agent.agent_registry or {})
    agents.setdefault(agent.name, agent)
    return agents


def _command_context(agent: ToolAgent) -> tuple[CommandContext, _HarnessCommandIO]:
    io = _HarnessCommandIO(messages=[])
    context = agent.context
    return (
        CommandContext(
            agent_provider=StaticAgentProvider(_agent_map(agent)),
            current_agent_name=agent.name,
            io=io,
            settings=context.config if context else None,
            session_manager=context.session_manager if context else None,
        ),
        io,
    )


def _parse_command(command: str) -> tuple[str, str]:
    text = command.strip()
    if not text:
        raise AgentConfigError("Slash command is empty", "Pass a command like '/tools'.")
    parsed = (
        parse_slash_command_line(text)
        if text.startswith("/")
        else split_subcommand_and_remainder(text)
    )
    if parsed is None or not parsed[0]:
        raise AgentConfigError("Slash command is empty", "Pass a command like '/tools'.")
    return strip_casefold(parsed[0]), parsed[1]


def _markdown_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


async def _render_status(agent: ToolAgent) -> str:
    if not isinstance(agent, _McpStatusProvider):
        return "No MCP status is available for this agent."
    statuses = await agent.get_server_status()
    if not statuses:
        return "No MCP servers are attached."

    lines = [
        "| Server | Connected | Transport | Calls | Error |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for name, status in sorted(statuses.items()):
        connected = (
            "unknown" if status.is_connected is None else ("yes" if status.is_connected else "no")
        )
        server_name = _markdown_table_cell(name)
        transport = _markdown_table_cell(status.transport or "-")
        error = _markdown_table_cell(status.error_message or "")
        lines.append(
            f"| {server_name} | {connected} | "
            f"{transport} | {sum(status.call_counts.values())} | {error} |"
        )
    return "\n".join(lines)


async def execute_harness_command(agent: ToolAgent, command: str) -> str:
    """Execute an allow-listed read-only harness command."""
    command_name, arguments = _parse_command(command)
    if command_name in {"help", "?", "commands"}:
        if arguments.strip():
            raise AgentConfigError(
                "Unsupported /commands arguments",
                "The harness tool currently supports only `/commands`.",
            )
        return render_commands_index_markdown(command_names=_COMMAND_NAMES)

    if command_name == "status":
        if arguments.strip():
            raise AgentConfigError(
                f"Unsupported /{command_name} arguments",
                f"The harness tool currently supports only `/{command_name}`.",
            )
        return f"# {command_name}\n\n" + await _render_status(agent)

    handler = _COMMAND_HANDLERS.get(command_name)
    if handler is None:
        available = ", ".join(f"`/{name}`" for name in _COMMAND_NAMES)
        raise AgentConfigError(
            "Unsupported harness command",
            f"Command '/{command_name}' is unavailable. Supported commands: {available}.",
        )
    if arguments.strip():
        raise AgentConfigError(
            f"Unsupported /{command_name} arguments",
            f"The harness tool currently supports only `/{command_name}`.",
        )

    context, io = _command_context(agent)
    outcome = await handler(context, agent_name=agent.name)
    return render_command_outcome_markdown(
        outcome,
        heading=command_name,
        extra_messages=io.messages,
    )
