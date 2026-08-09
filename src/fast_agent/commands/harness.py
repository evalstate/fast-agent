"""Model-visible, non-interactive access to selected harness commands."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.commands.command_discovery import (
    parse_commands_discovery_arguments,
    render_command_detail_markdown,
    render_commands_index_markdown,
    render_commands_json,
)
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.mcp_command_intents import (
    is_mcp_server_name_action,
    is_mcp_top_level_action,
    parse_mcp_no_args_tokens,
    parse_mcp_server_name_tokens,
)
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.mcp.connect_targets import parse_connect_command_text
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.ui.usage_display import format_usage_markdown
from fast_agent.utils.action_normalization import is_help_flag
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.slash_commands import parse_slash_command_line, split_subcommand_and_remainder
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.agents.tool_agent import ToolAgent
    from fast_agent.config import MCPServerSettings
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.mcp_aggregator import (
        MCPAttachResult,
        MCPDetachResult,
        ServerStatus,
    )

_COMMAND_HANDLERS = {
    "prompts": prompt_handlers.handle_list_prompts,
    "usage": display_handlers.handle_show_usage,
    "system": display_handlers.handle_show_system,
    "markdown": display_handlers.handle_show_markdown,
}
_COMMAND_NAMES = (
    "commands",
    "tools",
    "prompts",
    "usage",
    "system",
    "markdown",
    "status",
    "mcp",
    "skills",
)


@runtime_checkable
class _McpStatusProvider(Protocol):
    async def get_server_status(self) -> dict[str, "ServerStatus"]: ...


@dataclass(slots=True)
class _HarnessMcpRuntimeManager:
    agent: McpAgentProtocol

    def _validate_agent_name(self, agent_name: str) -> None:
        if agent_name != self.agent.name:
            raise AgentConfigError(f"Unknown agent: {agent_name}")

    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: "MCPServerSettings | None" = None,
        options: "MCPAttachOptions | None" = None,
    ) -> "MCPAttachResult":
        self._validate_agent_name(agent_name)
        safe_options = replace(
            options or MCPAttachOptions(),
            trigger_oauth=False,
            oauth_event_handler=None,
            allow_oauth_paste_fallback=False,
        )
        result = await self.agent.attach_mcp_server(
            server_name=server_name,
            server_config=server_config,
            options=safe_options,
        )
        await rebuild_agent_instruction(self.agent)
        return result

    async def detach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
    ) -> "MCPDetachResult":
        self._validate_agent_name(agent_name)
        result = await self.agent.detach_mcp_server(server_name)
        await rebuild_agent_instruction(self.agent)
        return result

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]:
        self._validate_agent_name(agent_name)
        return self.agent.list_attached_mcp_servers()

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]:
        self._validate_agent_name(agent_name)
        return self.agent.aggregator.list_configured_detached_servers()


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


def _command_context(
    agent: ToolAgent,
    *,
    skill_source_overrides: dict[str, str] | None = None,
) -> tuple[CommandContext, _HarnessCommandIO]:
    io = _HarnessCommandIO(messages=[])
    context = agent.context
    return (
        CommandContext(
            agent_provider=StaticAgentProvider(_agent_map(agent)),
            current_agent_name=agent.name,
            io=io,
            settings=context.config if context else None,
            session_manager=context.session_manager if context else None,
            skill_source_overrides=(
                skill_source_overrides if skill_source_overrides is not None else {}
            ),
            persist_skill_source_overrides=False,
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


def _first_shell_token_end(text: str) -> int:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char.isspace():
            return index
    return len(text)


def _parse_family_action(
    arguments: str,
    *,
    command_name: str,
    default: str,
) -> tuple[str, str]:
    text = arguments.strip()
    if not text:
        return default, ""
    try:
        tokens = split_commandline(text, syntax="posix")
    except ValueError as exc:
        raise AgentConfigError(f"Invalid /{command_name} arguments", str(exc)) from exc
    if not tokens:
        return default, ""
    action_end = _first_shell_token_end(text)
    return strip_casefold(tokens[0]), text[action_end:].strip()


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


def _mcp_usage_text() -> str:
    return "\n".join(
        (
            "Usage:",
            "- /mcp list",
            "- /mcp status",
            "- /mcp attach <server_name>",
            (
                "- /mcp connect <target> [--name <server>] [--auth <token>] "
                "[--timeout <seconds>] [--protocol auto|modern|legacy] "
                "[--no-oauth] [--reconnect|--no-reconnect]"
            ),
            "  Model-initiated OAuth is unavailable; use a user-facing command.",
            "- /mcp disconnect <server_name>",
            "- /mcp reconnect <server_name>",
        )
    )


def _mcp_manager(agent: ToolAgent) -> _HarnessMcpRuntimeManager:
    if not isinstance(agent, McpAgentProtocol):
        raise AgentConfigError(
            "MCP management is unavailable",
            f"Agent '{agent.name}' does not support runtime MCP server management.",
        )
    return _HarnessMcpRuntimeManager(agent)


def _render_outcome(
    outcome: CommandOutcome,
    *,
    heading: str,
    io: _HarnessCommandIO,
) -> str:
    if any(message.channel == "error" for message in outcome.messages):
        rendered = render_command_outcome_markdown(
            outcome,
            heading=heading,
            extra_messages=io.messages,
        )
        raise AgentConfigError(f"/{heading} command failed", rendered)
    if outcome.direct_response is not None:
        return outcome.direct_response

    rendered = render_command_outcome_markdown(
        outcome,
        heading=heading,
        extra_messages=io.messages,
    )
    return rendered


def _auth_uses_environment_reference(auth_token: str | None) -> bool:
    token = strip_to_none(auth_token)
    if token is None:
        return False
    if strip_casefold(token).startswith("bearer "):
        token = strip_to_none(token[7:])
    return token is not None and token.startswith("$")


async def _execute_mcp_command(agent: ToolAgent, arguments: str) -> str:
    subcommand, remainder = _parse_family_action(
        arguments,
        command_name="mcp",
        default="status",
    )
    args = " ".join(part for part in (subcommand, remainder) if part)

    if is_help_flag(subcommand):
        return f"# mcp\n\n{_mcp_usage_text()}"

    if subcommand == "status":
        intent = parse_mcp_no_args_tokens(
            split_commandline(args, syntax="posix"),
            usage="Usage: /mcp status",
        )
        if intent.error:
            raise AgentConfigError("Invalid /mcp status arguments", intent.error)
        return f"# mcp status\n\n{await _render_status(agent)}"

    if not is_mcp_top_level_action(subcommand):
        raise AgentConfigError("Unsupported /mcp command", _mcp_usage_text())

    context, io = _command_context(agent)
    manager = _mcp_manager(agent)

    if subcommand == "connect":
        if not remainder:
            raise AgentConfigError(
                "Invalid /mcp connect arguments",
                "Usage: /mcp connect <target>",
            )
        try:
            request = parse_connect_command_text(remainder)
        except ValueError as exc:
            raise AgentConfigError("Invalid /mcp connect arguments", str(exc)) from exc
        if request.target.mode == "stdio" and not manager.agent.shell_runtime_enabled:
            raise AgentConfigError(
                "Shell access is required for ad-hoc stdio MCP servers",
                "Enable shell access with `shell: true` or `-xx`, attach a configured server, "
                "or connect to an MCP URL.",
            )
        if request.options.trigger_oauth is True:
            raise AgentConfigError(
                "Interactive MCP OAuth is unavailable to model tools",
                "Connect with `--no-oauth` or an explicit `--auth` token, or ask the user "
                "to complete `/mcp connect ... --oauth`.",
            )
        if _auth_uses_environment_reference(request.options.auth_token):
            raise AgentConfigError(
                "Environment-backed MCP auth is unavailable to model tools",
                "Do not pass `$ENV_VAR` through model-visible commands. Configure the server "
                "credential outside the model context or use a user-facing command.",
            )
        request = replace(
            request,
            options=replace(request.options, trigger_oauth=False),
        )
        outcome = await mcp_runtime_handlers.handle_mcp_connect(
            context,
            manager=manager,
            agent_name=agent.name,
            request=request,
        )
        return _render_outcome(outcome, heading="mcp", io=io)

    try:
        tokens = split_commandline(args, syntax="posix")
    except ValueError as exc:
        raise AgentConfigError(f"Invalid /mcp {subcommand} arguments", str(exc)) from exc

    if subcommand == "list":
        intent = parse_mcp_no_args_tokens(tokens, usage="Usage: /mcp list")
        if intent.error:
            raise AgentConfigError("Invalid /mcp list arguments", intent.error)
        outcome = await mcp_runtime_handlers.handle_mcp_list(
            manager=manager,
            agent_name=agent.name,
        )
    elif is_mcp_server_name_action(subcommand):
        intent = parse_mcp_server_name_tokens(
            tokens,
            usage=f"Usage: /mcp {subcommand} <server_name>",
        )
        if intent.error or intent.server_name is None:
            raise AgentConfigError(
                f"Invalid /mcp {subcommand} arguments",
                intent.error or f"Usage: /mcp {subcommand} <server_name>",
            )
        if subcommand == "attach":
            outcome = await mcp_runtime_handlers.handle_mcp_attach(
                context,
                manager=manager,
                agent_name=agent.name,
                server_name=intent.server_name,
            )
        elif subcommand == "disconnect":
            outcome = await mcp_runtime_handlers.handle_mcp_disconnect(
                context,
                manager=manager,
                agent_name=agent.name,
                server_name=intent.server_name,
            )
        else:
            outcome = await mcp_runtime_handlers.handle_mcp_reconnect(
                context,
                manager=manager,
                agent_name=agent.name,
                server_name=intent.server_name,
            )
    else:
        raise AgentConfigError("Unsupported /mcp command", _mcp_usage_text())

    return _render_outcome(outcome, heading="mcp", io=io)


async def _execute_skills_command(
    agent: ToolAgent,
    arguments: str,
    *,
    skill_source_overrides: dict[str, str],
) -> str:
    action, argument = _parse_family_action(
        arguments,
        command_name="skills",
        default="list",
    )
    context, io = _command_context(
        agent,
        skill_source_overrides=skill_source_overrides,
    )
    outcome = await skills_handlers.handle_skills_command(
        context,
        agent_name=agent.name,
        action=action,
        argument=strip_to_none(argument),
        interactive=False,
    )
    return _render_outcome(outcome, heading="skills", io=io)


def _execute_commands_command(arguments: str) -> str:
    try:
        request = parse_commands_discovery_arguments(arguments)
    except ValueError as exc:
        raise AgentConfigError("Invalid /commands arguments", str(exc)) from exc

    if request.as_json:
        return render_commands_json(
            command_name=request.command_name,
            action_name=request.action_name,
            command_names=_COMMAND_NAMES,
            model_facing=True,
        )
    if request.command_name is None:
        return render_commands_index_markdown(command_names=_COMMAND_NAMES)
    if request.command_name not in _COMMAND_NAMES:
        available = ", ".join(f"`/{name}`" for name in _COMMAND_NAMES)
        raise AgentConfigError(
            "Unsupported /commands target",
            f"Command '/{request.command_name}' is unavailable. Supported commands: {available}.",
        )

    rendered = render_command_detail_markdown(
        request.command_name,
        request.action_name,
        model_facing=True,
    )
    if rendered is not None:
        return rendered

    target = f"/{request.command_name}"
    if request.action_name is not None:
        target = f"{target} {request.action_name}"
    raise AgentConfigError(
        "Unknown command action",
        f"No discovery metadata is available for `{target}`.",
    )


def _parse_tools_argument(arguments: str) -> str | None:
    try:
        tokens = split_commandline(arguments, syntax="posix")
    except ValueError as exc:
        raise AgentConfigError("Invalid /tools arguments", str(exc)) from exc
    if len(tokens) > 1:
        raise AgentConfigError(
            "Invalid /tools arguments",
            "Usage: /tools [summary|<tool-name>]",
        )
    return tokens[0] if tokens else None


async def execute_harness_command(
    agent: ToolAgent,
    command: str,
    *,
    skill_source_overrides: dict[str, str] | None = None,
) -> str:
    """Execute an allow-listed model-visible harness command."""
    command_name, arguments = _parse_command(command)
    if command_name in {"help", "?", "commands"}:
        return _execute_commands_command(arguments)

    if command_name == "status":
        if arguments.strip():
            raise AgentConfigError(
                f"Unsupported /{command_name} arguments",
                f"The harness tool currently supports only `/{command_name}`.",
            )
        return f"# {command_name}\n\n" + await _render_status(agent)

    if command_name == "mcp":
        return await _execute_mcp_command(agent, arguments)

    if command_name == "skills":
        return await _execute_skills_command(
            agent,
            arguments,
            skill_source_overrides=(
                skill_source_overrides if skill_source_overrides is not None else {}
            ),
        )

    if command_name == "tools":
        context, io = _command_context(agent)
        outcome = await tools_handlers.handle_list_tools(
            context,
            agent_name=agent.name,
            argument=_parse_tools_argument(arguments),
        )
        return _render_outcome(outcome, heading=command_name, io=io)

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
    return _render_outcome(outcome, heading=command_name, io=io)
