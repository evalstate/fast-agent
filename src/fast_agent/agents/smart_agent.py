"""Smart agent implementation with built-in tools."""

from __future__ import annotations

import asyncio
import base64
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from mcp.types import BlobResourceContents, ReadResourceResult, TextResourceContents

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.workflow.agents_as_tools_agent import (
    AgentsAsToolsAgent,
    AgentsAsToolsOptions,
)
from fast_agent.commands.command_catalog import (
    command_action_names,
    command_alias_map,
    get_command_spec,
    suggest_command_name,
)
from fast_agent.commands.command_discovery import (
    command_discovery_names,
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
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import models_manager as models_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.agent_card_loader import load_agent_cards
from fast_agent.core.agent_card_validation import AgentCardScanResult, scan_agent_card_path
from fast_agent.core.default_agent import resolve_default_agent_name
from fast_agent.core.direct_factory import (
    create_basic_agents_in_dependency_order,
    get_model_factory,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.instruction_utils import apply_instruction_context
from fast_agent.core.internal_resources import (
    InternalResource,
    get_internal_resource,
    list_internal_resources,
    read_internal_resource,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.core.validation import validate_provider_keys_post_creation
from fast_agent.mcp.connect_targets import infer_server_name, parse_connect_command_text
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.mcp.ui_mixin import McpUIMixin
from fast_agent.paths import resolve_environment_paths
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.utils.slash_commands import split_subcommand_and_remainder

if TYPE_CHECKING:
    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult

logger = get_logger(__name__)

_INTERNAL_RESOURCE_SERVER = "internal"


@dataclass(frozen=True)
class _SmartCardBundle:
    agents_dict: dict[str, "AgentCardData | dict[str, Any]"]
    message_files: dict[str, list[Path]]


class _McpCapableAgent(Protocol):
    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult: ...

    async def detach_mcp_server(self, server_name: str) -> MCPDetachResult: ...

    def list_attached_mcp_servers(self) -> list[str]: ...


@dataclass(frozen=True)
class _SmartConnectSummary:
    connected: list[str]
    warnings: list[str]


class _SmartToolMcpManager:
    """Minimal MCP runtime manager adapter for temporary smart-tool agents."""

    def __init__(
        self,
        agents: Mapping[str, AgentProtocol],
        configured_server_names: set[str],
    ) -> None:
        self._agents = agents
        self._configured_server_names = configured_server_names

    def _agent(self, name: str) -> _McpCapableAgent:
        agent = self._agents.get(name)
        if agent is None:
            raise AgentConfigError("Unknown agent", f"Agent '{name}' is not loaded")

        required = ("attach_mcp_server", "detach_mcp_server", "list_attached_mcp_servers")
        if not all(callable(getattr(agent, attr, None)) for attr in required):
            raise AgentConfigError(
                "Agent does not support runtime MCP connection",
                f"Agent '{name}' cannot attach MCP servers",
            )

        return agent  # type: ignore[return-value]

    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        agent = self._agent(agent_name)
        return await agent.attach_mcp_server(
            server_name=server_name,
            server_config=server_config,
            options=options,
        )

    async def detach_mcp_server(self, agent_name: str, server_name: str) -> MCPDetachResult:
        agent = self._agent(agent_name)
        return await agent.detach_mcp_server(server_name)

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]:
        agent = self._agent(agent_name)
        return list(agent.list_attached_mcp_servers())

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]:
        attached = set(await self.list_attached_mcp_servers(agent_name))
        return sorted(self._configured_server_names - attached)


@dataclass(slots=True)
class _SmartToolCommandIO(NonInteractiveCommandIOBase):
    """Non-interactive command IO that buffers emitted messages."""

    messages: list[CommandMessage]

    async def emit(self, message: CommandMessage) -> None:
        self.messages.append(message)


def _resolve_default_agent_name(
    agents: Mapping[str, AgentProtocol],
    *,
    tool_only_agents: set[str],
) -> str:
    default_agent_name = resolve_default_agent_name(
        agents,
        is_default=lambda _name, agent: bool(getattr(agent.config, "default", False)),
        is_tool_only=lambda name, _agent: name in tool_only_agents,
    )
    if default_agent_name is None:
        raise AgentConfigError("Smart tool requires at least one agent")
    return default_agent_name


def _collect_outcome_messages(outcome: "CommandOutcome") -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for message in outcome.messages:
        text = str(message.text)
        if message.channel == "error":
            errors.append(text)
        elif message.channel == "warning":
            warnings.append(text)
    return errors, warnings


def _resolve_command_agent_map(agent: Any) -> dict[str, object]:
    registry = getattr(agent, "agent_registry", None)
    agents: dict[str, object] = {}
    if isinstance(registry, Mapping):
        agents.update({str(name): member for name, member in registry.items()})

    agent_name = str(getattr(agent, "name", ""))
    if agent_name:
        agents.setdefault(agent_name, agent)
    return agents


def _build_command_context(agent: Any) -> tuple[CommandContext, _SmartToolCommandIO]:
    context = getattr(agent, "context", None)
    settings = context.config if context else None
    agent_name = str(getattr(agent, "name", ""))
    if not agent_name:
        raise AgentConfigError("Command execution requires named agent", "Agent has no name")

    io = _SmartToolCommandIO(messages=[])
    provider = StaticAgentProvider(_resolve_command_agent_map(agent))
    return (
        CommandContext(
            agent_provider=provider,
            current_agent_name=agent_name,
            io=io,
            settings=settings,
        ),
        io,
    )


def _render_command_outcome(
    outcome: CommandOutcome,
    *,
    heading: str,
    io: _SmartToolCommandIO,
) -> str:
    return render_command_outcome_markdown(
        outcome,
        heading=heading,
        extra_messages=io.messages,
    )


async def _run_check_subprocess(argument: str | None) -> CommandOutcome:
    command = [
        sys.executable,
        "-m",
        "fast_agent.cli.main",
        "--no-color",
        "check",
    ]
    if argument and argument.strip():
        try:
            command.extend(shlex.split(argument))
        except ValueError as exc:
            raise AgentConfigError("Invalid check arguments", str(exc)) from exc

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    outcome = CommandOutcome()
    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if stdout_text:
        outcome.add_message(stdout_text)
    if stderr_text:
        channel = "error" if process.returncode else "warning"
        outcome.add_message(stderr_text, channel=channel)
    if process.returncode != 0 and not stderr_text:
        outcome.add_message(
            f"fast-agent check exited with status code {process.returncode}.",
            channel="error",
        )
    return outcome


async def _run_named_command_call(
    agent: Any,
    *,
    command_name: str,
    action: str | None = None,
    argument: str | None = None,
    heading: str | None = None,
) -> str:
    normalized_command = command_name.strip().lower()
    normalized_action = (action or "").strip().lower()

    if normalized_command == "check":
        outcome = await _run_check_subprocess(argument)
        return render_command_outcome_markdown(outcome, heading=heading or "check")

    context, io = _build_command_context(agent)
    agent_name = context.current_agent_name

    if normalized_command == "skills":
        selected_action = normalized_action or "list"
        outcome = await skills_handlers.handle_skills_command(
            context,
            agent_name=agent_name,
            action=selected_action,
            argument=argument,
        )
        final_heading = heading or f"skills.{selected_action}"
        return _render_command_outcome(outcome, heading=final_heading, io=io)

    if normalized_command == "cards":
        selected_action = normalized_action or "list"
        outcome = await cards_handlers.handle_cards_command(
            context,
            agent_name=agent_name,
            action=selected_action,
            argument=argument,
        )
        final_heading = heading or f"cards.{selected_action}"
        return _render_command_outcome(outcome, heading=final_heading, io=io)

    if normalized_command == "model":
        selected_action = normalized_action or "reasoning"
        management_actions = {"doctor", "references", "catalog", "help"}
        if selected_action in management_actions:
            outcome = await models_handlers.handle_models_command(
                context,
                agent_name=agent_name,
                action=selected_action,
                argument=argument,
            )
            final_heading = heading or f"model.{selected_action}"
            return _render_command_outcome(outcome, heading=final_heading, io=io)

        if selected_action == "verbosity":
            outcome = await model_handlers.handle_model_verbosity(
                context,
                agent_name=agent_name,
                value=argument,
            )
        elif selected_action == "fast":
            outcome = await model_handlers.handle_model_fast(
                context,
                agent_name=agent_name,
                value=argument,
            )
        elif selected_action == "web_search":
            outcome = await model_handlers.handle_model_web_search(
                context,
                agent_name=agent_name,
                value=argument,
            )
        elif selected_action == "web_fetch":
            outcome = await model_handlers.handle_model_web_fetch(
                context,
                agent_name=agent_name,
                value=argument,
            )
        elif selected_action == "switch":
            outcome = await model_handlers.handle_model_switch(
                context,
                agent_name=agent_name,
                value=argument,
            )
            if outcome.reset_session:
                if not context.noenv:
                    outcome.add_message(
                        "Model switch starts a new session to avoid mixing histories.",
                        channel="info",
                    )
                    session_outcome = await sessions_handlers.handle_create_session(
                        context,
                        session_name=None,
                    )
                    outcome.messages.extend(session_outcome.messages)
                else:
                    outcome.add_message(
                        "Model switch cleared in-memory history (--noenv disables session persistence).",
                        channel="info",
                    )
                cleared = clear_agent_histories(_resolve_command_agent_map(agent))
                if cleared:
                    outcome.add_message(
                        f"Cleared agent history: {', '.join(sorted(cleared))}",
                        channel="info",
                    )
        else:
            outcome = await model_handlers.handle_model_reasoning(
                context,
                agent_name=agent_name,
                value=argument,
            )
        final_heading = heading or f"model.{selected_action}"
        return _render_command_outcome(outcome, heading=final_heading, io=io)

    raise AgentConfigError(
        "Unsupported command",
        f"Command '{command_name}' is not supported by slash_command.",
    )


def _parse_family_command_action(
    command_name: str,
    arguments: str,
) -> tuple[str, str | None]:
    spec = get_command_spec(command_name)
    if spec is None:
        raise AgentConfigError(
            "Unsupported command",
            f"Command '{command_name}' is not supported by slash_command.",
        )

    trimmed = arguments.strip()
    if not trimmed:
        return spec.default_action, None

    try:
        tokens = shlex.split(trimmed)
    except ValueError as exc:
        raise AgentConfigError("Invalid command arguments", str(exc)) from exc

    if not tokens:
        return spec.default_action, None

    supported_actions = set(command_action_names(command_name))
    first = tokens[0].lower()

    if first in {"help", "--help", "-h"}:
        return "help", None

    aliases = command_alias_map(command_name)
    alias_resolved = aliases.get(first)
    if alias_resolved is not None:
        remainder = " ".join(tokens[1:]).strip() or None
        return alias_resolved, remainder

    if command_name == "skills":
        extra_actions = {
            "available",
            "search",
        }
        if first in extra_actions:
            remainder = " ".join(tokens[1:]).strip() or None
            return first, remainder

    if first in supported_actions:
        remainder = " ".join(tokens[1:]).strip() or None
        return first, remainder

    return spec.default_action, trimmed


def _parse_slash_command_text(command: str) -> tuple[str, str]:
    text = command.strip()
    if not text:
        raise AgentConfigError(
            "Slash command is empty",
            "Pass a command like '/mcp list' or '/skills list'.",
        )

    if text.startswith("/"):
        text = text[1:].lstrip()

    if not text:
        raise AgentConfigError(
            "Slash command is empty",
            "Pass a command like '/mcp list' or '/skills list'.",
        )

    command_name, _, arguments = text.partition(" ")
    return command_name.strip().lower(), arguments.strip()


def _smart_slash_usage() -> str:
    return render_commands_index_markdown(command_names=command_discovery_names())


def _render_unknown_slash_command(command_name: str) -> str:
    suggestions = suggest_command_name(command_name)
    suggestion_line = ""
    if suggestions:
        suggestion_line = "\nDid you mean: " + ", ".join(f"`/{name}`" for name in suggestions)

    return f"Unknown slash command '/{command_name}'.{suggestion_line}\n\n{_smart_slash_usage()}"


def _mcp_usage_text() -> str:
    return (
        "Usage: /mcp [list|connect|disconnect|reconnect|session|help] [args]\n"
        "- /mcp list\n"
        "- /mcp connect <target> [--name <server>] [--auth <token-value>] [--timeout <seconds>]\n"
        "- /mcp session [list|jar|new|use|clear] [args]"
    )


def _model_usage_text() -> str:
    return (
        "Usage: /model "
        "[reasoning|verbosity|fast|web_search|web_fetch|switch|doctor|references|catalog|help] [args]"
    )


def _run_commands_slash_command_call(arguments: str) -> str:
    try:
        request = parse_commands_discovery_arguments(arguments)
    except ValueError as exc:
        raise AgentConfigError("Invalid /commands arguments", str(exc)) from exc

    if request.as_json:
        return render_commands_json(command_name=request.command_name)

    if request.command_name is None:
        return render_commands_index_markdown(command_names=command_discovery_names())

    detail = render_command_detail_markdown(request.command_name)
    if detail is not None:
        return detail

    suggestions = suggest_command_name(request.command_name)
    suggestion_text = ""
    if suggestions:
        suggestion_text = "\nDid you mean: " + ", ".join(f"`/{name}`" for name in suggestions)
    return (
        f"Unknown command family: `{request.command_name}`.\n"
        f"Use `/commands` to list available commands.{suggestion_text}"
    )


def _render_smart_slash_outcome(
    outcome: CommandOutcome,
    *,
    heading: str,
    io: _SmartToolCommandIO,
) -> str:
    return _render_command_outcome(outcome, heading=heading, io=io)


def _parse_mcp_session_args(
    tokens: list[str],
) -> tuple[str, str | None, str | None, str | None, bool]:
    session_tokens = tokens[1:]
    action = "list"
    server_identity: str | None = None
    session_id: str | None = None
    title: str | None = None
    clear_all = False

    if not session_tokens:
        return action, server_identity, session_id, title, clear_all

    action = session_tokens[0].lower()
    args = session_tokens[1:]

    if action == "list":
        if len(args) > 1:
            raise AgentConfigError(
                "Invalid /mcp session arguments",
                "Usage: /mcp session list [<server_or_identity>]",
            )
        server_identity = args[0] if args else None
        return action, server_identity, session_id, title, clear_all

    if action == "jar":
        if len(args) > 1:
            raise AgentConfigError(
                "Invalid /mcp session arguments",
                "Usage: /mcp session jar [<server_or_identity>]",
            )
        server_identity = args[0] if args else None
        return action, server_identity, session_id, title, clear_all

    if action in {"new", "create"}:
        idx = 0
        while idx < len(args):
            token = args[idx]
            if token == "--title":
                idx += 1
                if idx >= len(args):
                    raise AgentConfigError(
                        "Invalid /mcp session arguments", "Missing value for --title"
                    )
                title = args[idx]
            elif token.startswith("--title="):
                title = token.split("=", 1)[1] or None
                if title is None:
                    raise AgentConfigError(
                        "Invalid /mcp session arguments",
                        "Missing value for --title",
                    )
            elif token.startswith("--"):
                raise AgentConfigError(
                    "Invalid /mcp session arguments",
                    f"Unknown flag: {token}",
                )
            elif server_identity is None:
                server_identity = token
            else:
                raise AgentConfigError(
                    "Invalid /mcp session arguments",
                    f"Unexpected argument: {token}",
                )
            idx += 1
        return "new", server_identity, session_id, title, clear_all

    if action in {"resume", "use"}:
        if len(args) != 2:
            raise AgentConfigError(
                "Invalid /mcp session arguments",
                "Usage: /mcp session use <server_or_identity> <session_id>",
            )
        server_identity, session_id = args
        return "use", server_identity, session_id, title, clear_all

    if action == "clear":
        for token in args:
            if token == "--all":
                clear_all = True
                continue
            if token.startswith("--"):
                raise AgentConfigError(
                    "Invalid /mcp session arguments",
                    f"Unknown flag: {token}",
                )
            if server_identity is None:
                server_identity = token
            else:
                raise AgentConfigError(
                    "Invalid /mcp session arguments",
                    f"Unexpected argument: {token}",
                )

        if clear_all and server_identity is not None:
            raise AgentConfigError(
                "Invalid /mcp session arguments",
                "Use either --all or a specific server, not both",
            )

        if not clear_all and server_identity is None:
            clear_all = True

        return action, server_identity, session_id, title, clear_all

    if args:
        raise AgentConfigError(
            "Invalid /mcp session arguments",
            (
                "Usage: /mcp session [list [server]|jar [server]|new [server] "
                "[--title <title>]|use <server> <session_id>|clear [server|--all]]"
            ),
        )

    server_identity = action
    return "list", server_identity, session_id, title, clear_all


async def _run_mcp_slash_command_call(agent: Any, arguments: str) -> str:
    context, io = _build_command_context(agent)
    agent_name = context.current_agent_name
    runtime_manager = _SmartToolMcpManager(
        {agent_name: agent},
        configured_server_names=_context_server_names(getattr(agent, "context", None)),
    )

    args = arguments.strip() or "list"
    subcmd_text, connect_remainder = split_subcommand_and_remainder(args)
    subcmd = (subcmd_text or "list").lower()

    if subcmd == "connect":
        if not connect_remainder:
            raise AgentConfigError(
                "Invalid /mcp connect arguments",
                (
                    "Usage: /mcp connect <target> [--name <server>] [--auth <token-value>] "
                    "[--timeout <seconds>] [--oauth|--no-oauth] [--reconnect|--no-reconnect]"
                ),
            )
        try:
            request = parse_connect_command_text(connect_remainder)
        except ValueError as exc:
            raise AgentConfigError("Invalid /mcp connect arguments", str(exc)) from exc
        outcome = await mcp_runtime_handlers.handle_mcp_connect(
            context,
            manager=runtime_manager,
            agent_name=agent_name,
            request=request,
        )
        return _render_smart_slash_outcome(outcome, heading="mcp", io=io)

    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        raise AgentConfigError("Invalid /mcp arguments", str(exc)) from exc

    if not tokens:
        tokens = ["list"]

    subcmd = tokens[0].lower()

    if subcmd in {"help", "--help", "-h"}:
        return _mcp_usage_text()

    if subcmd == "list":
        outcome = await mcp_runtime_handlers.handle_mcp_list(
            context,
            manager=runtime_manager,
            agent_name=agent_name,
        )
        return _render_smart_slash_outcome(outcome, heading="mcp", io=io)

    if subcmd == "disconnect":
        if len(tokens) != 2:
            raise AgentConfigError(
                "Invalid /mcp disconnect arguments",
                "Usage: /mcp disconnect <server_name>",
            )

        outcome = await mcp_runtime_handlers.handle_mcp_disconnect(
            context,
            manager=runtime_manager,
            agent_name=agent_name,
            server_name=tokens[1],
        )
        return _render_smart_slash_outcome(outcome, heading="mcp", io=io)

    if subcmd == "reconnect":
        if len(tokens) != 2:
            raise AgentConfigError(
                "Invalid /mcp reconnect arguments",
                "Usage: /mcp reconnect <server_name>",
            )

        outcome = await mcp_runtime_handlers.handle_mcp_reconnect(
            context,
            manager=runtime_manager,
            agent_name=agent_name,
            server_name=tokens[1],
        )
        return _render_smart_slash_outcome(outcome, heading="mcp", io=io)

    if subcmd == "session":
        action, server_identity, session_id, title, clear_all = _parse_mcp_session_args(tokens)
        outcome = await mcp_runtime_handlers.handle_mcp_session(
            context,
            agent_name=agent_name,
            action=cast("mcp_runtime_handlers.McpSessionAction", action),
            server_identity=server_identity,
            session_id=session_id,
            title=title,
            clear_all=clear_all,
        )
        return _render_smart_slash_outcome(outcome, heading="mcp", io=io)

    raise AgentConfigError(
        "Unsupported /mcp subcommand",
        _mcp_usage_text(),
    )


async def _run_slash_command_call(agent: Any, command: str) -> str:
    command_name, arguments = _parse_slash_command_text(command)

    if command_name in {"help", "?"}:
        return _smart_slash_usage()

    if command_name == "commands":
        return _run_commands_slash_command_call(arguments)

    if command_name in {"skills", "cards", "model"}:
        action, argument = _parse_family_command_action(command_name, arguments)
        normalized_action = action

        heading = f"{command_name}.{normalized_action}"
        return await _run_named_command_call(
            agent,
            command_name=command_name,
            action=normalized_action,
            argument=argument,
            heading=heading,
        )

    if command_name == "check":
        argument = arguments.strip() or None
        return await _run_named_command_call(
            agent,
            command_name="check",
            action="run",
            argument=argument,
            heading="check",
        )

    if command_name == "mcp":
        return await _run_mcp_slash_command_call(agent, arguments)

    context, io = _build_command_context(agent)
    agent_name = context.current_agent_name

    if command_name == "tools":
        outcome = await tools_handlers.handle_list_tools(context, agent_name=agent_name)
        return _render_smart_slash_outcome(outcome, heading="tools", io=io)

    if command_name == "prompts":
        outcome = await prompt_handlers.handle_list_prompts(context, agent_name=agent_name)
        return _render_smart_slash_outcome(outcome, heading="prompts", io=io)

    if command_name == "usage":
        outcome = await display_handlers.handle_show_usage(context, agent_name=agent_name)
        return _render_smart_slash_outcome(outcome, heading="usage", io=io)

    if command_name == "system":
        outcome = await display_handlers.handle_show_system(context, agent_name=agent_name)
        return _render_smart_slash_outcome(outcome, heading="system", io=io)

    if command_name == "markdown":
        outcome = await display_handlers.handle_show_markdown(context, agent_name=agent_name)
        return _render_smart_slash_outcome(outcome, heading="markdown", io=io)

    if command_name in {"mcpstatus", "status"}:
        outcome = await display_handlers.handle_show_mcp_status(context, agent_name=agent_name)
        return _render_smart_slash_outcome(outcome, heading="mcpstatus", io=io)

    return _render_unknown_slash_command(command_name)


def _context_server_names(context: "Context | None") -> set[str]:
    if context and context.config and context.config.mcp and context.config.mcp.servers:
        return set(context.config.mcp.servers.keys())
    return set()


async def _apply_runtime_mcp_connections(
    *,
    context: "Context | None",
    agents_map: Mapping[str, AgentProtocol],
    target_agent_name: str,
    mcp_connect: Sequence[str],
) -> _SmartConnectSummary:
    configured_names = _context_server_names(context)

    manager = _SmartToolMcpManager(agents_map, configured_server_names=configured_names)
    connected_names: list[str] = []
    warnings: list[str] = []

    for raw_target in mcp_connect:
        target = raw_target.strip()
        if not target:
            continue
        try:
            request = parse_connect_command_text(target)
        except ValueError as exc:
            raise AgentConfigError(
                "Failed to connect MCP server for smart tool call",
                str(exc),
            ) from exc

        outcome = await mcp_runtime_handlers.handle_mcp_connect(
            None,
            manager=manager,
            agent_name=target_agent_name,
            request=request,
        )
        errors, target_warnings = _collect_outcome_messages(outcome)
        warnings.extend(target_warnings)
        if errors:
            raise AgentConfigError(
                "Failed to connect MCP server for smart tool call",
                "\n".join(errors),
            )

        connected_names.append(infer_server_name(request.target))

    return _SmartConnectSummary(connected=connected_names, warnings=warnings)


def _resolve_agent_card_path(path_value: str, context: Context | None) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
    else:
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

    env_paths = resolve_environment_paths(
        settings=context.config if context else None, cwd=Path.cwd()
    )
    for base in (env_paths.agent_cards, env_paths.tool_cards):
        env_candidate = (base / candidate).resolve()
        if env_candidate.exists():
            return env_candidate

    raise AgentConfigError(
        "AgentCard path not found",
        f"Tried: {candidate} (cwd), {env_paths.agent_cards}, {env_paths.tool_cards}",
    )


def _ensure_basic_only_cards(cards: Sequence) -> _SmartCardBundle:
    agents_dict: dict[str, "AgentCardData | dict[str, Any]"] = {}
    message_files: dict[str, list[Path]] = {}
    for card in cards:
        agent_data = card.agent_data
        agent_type = agent_data.get("type")
        if agent_type != AgentType.BASIC.value:
            raise AgentConfigError(
                "Smart tool only supports 'agent' cards",
                f"Card '{card.name}' has unsupported type '{agent_type}'",
            )
        agents_dict[card.name] = agent_data
        if card.message_files:
            message_files[card.name] = list(card.message_files)
    return _SmartCardBundle(agents_dict=agents_dict, message_files=message_files)


def _apply_agent_card_histories(
    agents: dict[str, AgentProtocol],
    message_map: Mapping[str, list[Path]],
) -> None:
    for name, history_files in message_map.items():
        agent = agents.get(name)
        if agent is None:
            continue
        messages = []
        for history_file in history_files:
            messages.extend(load_prompt(history_file))
        agent.clear(clear_prompts=True)
        agent.message_history.extend(messages)


async def _apply_instruction_context(
    agents: dict[str, AgentProtocol],
    context_vars: Mapping[str, str],
) -> None:
    await apply_instruction_context(agents.values(), context_vars)


async def _shutdown_agents(agents: Mapping[str, AgentProtocol]) -> None:
    for agent in agents.values():
        try:
            await agent.shutdown()
        except Exception:
            pass


def _format_validation_results(results: Sequence[AgentCardScanResult]) -> str:
    if not results:
        return "No AgentCards found."

    lines: list[str] = []
    for entry in results:
        if entry.ignored_reason:
            status = f"ignored - {entry.ignored_reason}"
        elif entry.errors:
            status = "error"
        else:
            status = "ok"
        lines.append(f"{entry.name} ({entry.type}) - {status}")
        for error in entry.errors:
            lines.append(f"  - {error}")
    return "\n".join(lines)


async def _create_smart_tool_agents(
    context: "Context | None",
    agent_card_path: str,
) -> tuple[dict[str, AgentProtocol], _SmartCardBundle, str, set[str]]:
    if context is None:
        raise AgentConfigError("Smart tool requires an initialized context")

    resolved_path = _resolve_agent_card_path(agent_card_path, context)
    cards = load_agent_cards(resolved_path)
    bundle = _ensure_basic_only_cards(cards)

    def model_factory_func(model=None, request_params=None):
        return get_model_factory(context, model=model, request_params=request_params)

    agents_map = await create_basic_agents_in_dependency_order(
        context,
        bundle.agents_dict,
        model_factory_func,
    )
    validate_provider_keys_post_creation(agents_map)

    tool_only_agents = {
        name for name, data in bundle.agents_dict.items() if data.get("tool_only", False)
    }
    default_agent_name = _resolve_default_agent_name(
        agents_map,
        tool_only_agents=tool_only_agents,
    )

    return agents_map, bundle, default_agent_name, tool_only_agents


async def _hydrate_smart_agents_for_execution(
    agents_map: dict[str, AgentProtocol],
    bundle: _SmartCardBundle,
) -> None:
    if bundle.message_files:
        _apply_agent_card_histories(agents_map, bundle.message_files)

    context_vars: dict[str, str] = {}
    enrich_with_environment_context(
        context_vars,
        str(Path.cwd()),
        {"name": "fast-agent"},
    )
    await _apply_instruction_context(agents_map, context_vars)


def _extract_read_resource_text(result: ReadResourceResult, *, max_chars: int = 4000) -> str:
    lines: list[str] = []
    for idx, content in enumerate(result.contents, start=1):
        if isinstance(content, TextResourceContents):
            text = content.text
            lines.append(f"[{idx}] text ({content.mimeType or 'unknown'})")
            lines.append(text)
            continue

        if isinstance(content, BlobResourceContents):
            blob_len = len(content.blob)
            preview = ""
            try:
                decoded = base64.b64decode(content.blob)
                preview = decoded[:400].decode("utf-8", errors="replace")
            except Exception:
                preview = "<binary blob>"
            lines.append(f"[{idx}] blob ({content.mimeType or 'unknown'}, {blob_len} b64 chars)")
            if preview:
                lines.append(preview)
            continue

        text = get_text(content)
        if text:
            lines.append(f"[{idx}] content")
            lines.append(text)

    joined = "\n".join(lines).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 1] + "…\n[truncated]"


def _format_smart_resource_listing(
    resources: Mapping[str, list[str]],
    templates: Mapping[str, Sequence[Any]],
    *,
    mcp_server_names: Sequence[str] = (),
    internal_resources: Sequence[InternalResource] = (),
) -> str:
    lines: list[str] = []
    server_names = sorted(
        (set(resources.keys()) | set(templates.keys()) | set(mcp_server_names))
        - {_INTERNAL_RESOURCE_SERVER}
    )
    if not server_names and not internal_resources:
        return "No resources available."

    lines.append("server_names:")
    lines.append(f"  - {_INTERNAL_RESOURCE_SERVER} (always available)")
    if server_names:
        for server_name in server_names:
            lines.append(f"  - {server_name}")
    else:
        lines.append("  - (no attached MCP servers)")
    lines.append("")

    if internal_resources:
        lines.append(f"[{_INTERNAL_RESOURCE_SERVER}]")
        lines.append("resources:")
        for resource in internal_resources:
            lines.append(f"  - {resource.uri}")
            lines.append(f"    description: {resource.description}")
            lines.append(f"    why: {resource.why}")
        lines.append("templates: []")
        if server_names:
            lines.append("")

    for server_name in server_names:
        lines.append(f"[{server_name}]")
        server_resources = resources.get(server_name, [])
        if server_resources:
            lines.append("resources:")
            for uri in server_resources:
                lines.append(f"  - {uri}")
        else:
            lines.append("resources: []")

        server_templates = templates.get(server_name, [])
        if server_templates:
            lines.append("templates:")
            for template in server_templates:
                uri_template = getattr(template, "uriTemplate", "")
                name = getattr(template, "name", "")
                if name:
                    lines.append(f"  - {name}: {uri_template}")
                else:
                    lines.append(f"  - {uri_template}")
        else:
            lines.append("templates: []")

    return "\n".join(lines)


def _include_internal_resources(server_name: str | None) -> bool:
    return server_name is None or server_name == _INTERNAL_RESOURCE_SERVER


def _include_mcp_resources(server_name: str | None) -> bool:
    return server_name != _INTERNAL_RESOURCE_SERVER


def _is_internal_resource_uri(uri: str) -> bool:
    return uri.strip().startswith("internal://")


async def _run_internal_resource_read_call(uri: str) -> str:
    resource = get_internal_resource(uri)
    content = read_internal_resource(resource.uri)
    header = f"Resource: {resource.uri}\nTitle: {resource.title}"
    return f"{header}\n\n{content}" if content else header


async def _run_smart_call(
    context: "Context | None",
    agent_card_path: str,
    message: str,
    *,
    mcp_connect: Sequence[str] | None = None,
    disable_streaming: bool = False,
) -> str:
    agents_map, bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        app = AgentApp(agents_map, tool_only_agents=tool_only_agents)

        if mcp_connect:
            connect_summary = await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )
            if connect_summary.connected:
                logger.info(
                    "Connected runtime MCP servers for smart tool call",
                    data={
                        "agent": default_agent_name,
                        "servers": connect_summary.connected,
                    },
                )
            for warning in connect_summary.warnings:
                logger.warning(
                    "Runtime MCP connect warning in smart tool call",
                    data={"warning": warning, "agent": default_agent_name},
                )

        if disable_streaming:
            for agent in agents_map.values():
                setter = getattr(agent, "force_non_streaming_next_turn", None)
                if callable(setter):
                    setter(reason="parallel smart tool calls")
            logger.info(
                "Disabled streaming for smart tool child agents",
                data={"agent_count": len(agents_map)},
            )

        await _hydrate_smart_agents_for_execution(agents_map, bundle)

        return await app.send(message)
    finally:
        await _shutdown_agents(agents_map)


async def _run_validate_call(
    context: "Context | None",
    agent_card_path: str,
) -> str:
    resolved_path = _resolve_agent_card_path(agent_card_path, context)

    server_names = None
    if context and context.config and context.config.mcp and context.config.mcp.servers:
        server_names = set(context.config.mcp.servers.keys())

    results = scan_agent_card_path(resolved_path, server_names=server_names)
    return _format_validation_results(results)


async def _run_smart_list_resources_call(
    context: "Context | None",
    agent_card_path: str,
    *,
    server_name: str | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    resources: Mapping[str, list[str]] = {}
    templates: Mapping[str, Sequence[Any]] = {}
    mcp_server_names: list[str] = []

    if _include_mcp_resources(server_name):
        agents_map, _bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
            context,
            agent_card_path,
        )
        try:
            app = AgentApp(agents_map, tool_only_agents=tool_only_agents)
            if mcp_connect:
                await _apply_runtime_mcp_connections(
                    context=context,
                    agents_map=agents_map,
                    target_agent_name=default_agent_name,
                    mcp_connect=mcp_connect,
                )

            resources = await app.list_resources(
                server_name=server_name,
                agent_name=default_agent_name,
            )

            default_agent = agents_map.get(default_agent_name)
            aggregator = getattr(default_agent, "aggregator", None) if default_agent else None
            list_templates = getattr(aggregator, "list_resource_templates", None)
            if callable(list_templates):
                templates = await list_templates(server_name)

            list_attached = getattr(default_agent, "list_attached_mcp_servers", None)
            if callable(list_attached):
                try:
                    attached = list_attached()
                except Exception:
                    attached = []
                mcp_server_names = sorted(
                    set(attached) | set(resources.keys()) | set(templates.keys())
                )
        finally:
            await _shutdown_agents(agents_map)

    internal_resources: Sequence[InternalResource] = ()
    if _include_internal_resources(server_name):
        internal_resources = list_internal_resources()

    return _format_smart_resource_listing(
        resources,
        templates,
        mcp_server_names=mcp_server_names,
        internal_resources=internal_resources,
    )


async def _run_current_agent_get_resource_call(
    agent: Any,
    resource_uri: str,
    *,
    server_name: str | None = None,
) -> str:
    result = await agent.get_resource(resource_uri=resource_uri, namespace=server_name)
    body = _extract_read_resource_text(result)
    header = f"Resource: {resource_uri}"
    if server_name:
        header += f" (server={server_name})"
    return f"{header}\n\n{body}" if body else header


async def _run_smart_get_resource_call(
    context: "Context | None",
    agent_card_path: str,
    resource_uri: str,
    *,
    server_name: str | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    agents_map, _bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        app = AgentApp(agents_map, tool_only_agents=tool_only_agents)
        if mcp_connect:
            await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )

        result = await app.get_resource(
            resource_uri=resource_uri,
            server_name=server_name,
            agent_name=default_agent_name,
        )
        body = _extract_read_resource_text(result)
        header = f"Resource: {resource_uri}"
        if server_name:
            header += f" (server={server_name})"
        return f"{header}\n\n{body}" if body else header
    finally:
        await _shutdown_agents(agents_map)


async def _run_smart_with_resource_call(
    context: "Context | None",
    agent_card_path: str,
    message: str,
    resource_uri: str,
    *,
    server_name: str | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    agents_map, bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        app = AgentApp(agents_map, tool_only_agents=tool_only_agents)
        if mcp_connect:
            await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )

        await _hydrate_smart_agents_for_execution(agents_map, bundle)

        return await app.with_resource(
            prompt_content=message,
            resource_uri=resource_uri,
            server_name=server_name,
            agent_name=default_agent_name,
        )
    finally:
        await _shutdown_agents(agents_map)


def _resolve_template_server_name(
    template_uri: str,
    templates_by_server: Mapping[str, Sequence[Any]],
) -> str | None:
    matches: list[str] = []
    for server_name, templates in templates_by_server.items():
        for template in templates:
            if getattr(template, "uriTemplate", None) == template_uri:
                matches.append(server_name)
                break

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


async def _run_smart_complete_resource_argument_call(
    context: "Context | None",
    agent_card_path: str,
    template_uri: str,
    argument_name: str,
    value: str,
    *,
    server_name: str | None = None,
    context_args: Mapping[str, str] | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    agents_map, _bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        del tool_only_agents
        if mcp_connect:
            await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )

        default_agent = agents_map.get(default_agent_name)
        aggregator = getattr(default_agent, "aggregator", None) if default_agent else None
        if aggregator is None:
            raise AgentConfigError(
                "Smart resource completion requires MCP-capable card agent",
                f"Agent '{default_agent_name}' does not expose an MCP aggregator",
            )

        target_server = server_name
        if target_server is None:
            templates = await aggregator.list_resource_templates(None)
            target_server = _resolve_template_server_name(template_uri, templates)
            if target_server is None:
                raise AgentConfigError(
                    "Unable to resolve resource template server",
                    "Pass server_name explicitly when multiple servers expose the same template.",
                )

        completion = await aggregator.complete_resource_argument(
            server_name=target_server,
            template_uri=template_uri,
            argument_name=argument_name,
            value=value,
            context_args=dict(context_args) if context_args else None,
        )
        values = completion.values or []
        if not values:
            return "No completion values returned."
        return "\n".join(values)
    finally:
        await _shutdown_agents(agents_map)


def _slash_command_tool_description() -> str:
    return (
        "Execute a fast-agent slash command using native `/...` syntax. "
        "Use `/commands` or `/commands --json` to discover capabilities. "
        "Supports `/skills` (including available/search/registry/help), `/cards`, `/model`, `/mcp`, "
        "`/tools`, `/prompts`, "
        "`/usage`, `/system`, `/markdown`, and `/check`."
    )


def _enable_smart_tooling(agent: Any) -> None:
    """Register smart tool endpoints on a smart-capable agent."""
    setattr(agent, "_parallel_smart_tool_calls", False)
    smart_tool_names = {
        "smart",
        "slash_command",
        "get_resource",
    }
    setattr(agent, "_smart_tool_names", set(smart_tool_names))

    smart_tool = build_default_function_tool(
        agent.smart,
        name="smart",
        description=(
            "Run subagent tasks from a definition in a file or directory. Subagents are defined with "
            "AgentCards. Use action=`run` to load a subagent and send it a message. Optionally supply "
            "`mcp_connect` targets. Use action=`validate` to check card file validity without running them"
        ),
    )
    slash_command_tool = build_default_function_tool(
        agent.slash_command,
        name="slash_command",
        description=_slash_command_tool_description(),
    )
    resource_read_tool = build_default_function_tool(
        agent.read_resource,
        name="get_resource",
        description=(
            "Read a resource by URI. `internal://` URIs are read directly from bundled "
            "resources shown in the prompt; other URIs are fetched from attached MCP resources "
            "when available."
        ),
    )
    agent.add_tool(smart_tool)
    agent.add_tool(slash_command_tool)
    agent.add_tool(resource_read_tool)


async def _dispatch_smart_tool(
    agent: Any,
    agent_card_path: str,
    message: str | None = None,
    mcp_connect: list[str] | None = None,
    action: Literal["run", "validate"] = "run",
) -> str:
    context = getattr(agent, "context", None)
    if action == "validate":
        return await _run_validate_call(context, agent_card_path)
    if action != "run":
        raise AgentConfigError(
            "Invalid smart action",
            "Supported smart actions: run, validate.",
        )

    message_text = (message or "").strip()
    if not message_text:
        raise AgentConfigError(
            "Missing smart message",
            "Provide `message` when action=`run`.",
        )

    disable_streaming = bool(getattr(agent, "_parallel_smart_tool_calls", False))
    return await _run_smart_call(
        context,
        agent_card_path,
        message_text,
        mcp_connect=mcp_connect,
        disable_streaming=disable_streaming,
    )


async def _dispatch_slash_command_tool(agent: Any, command: str) -> str:
    return await _run_slash_command_call(agent, command)


async def _dispatch_resource_read_tool(
    agent: Any,
    uri: str,
    server_name: str | None = None,
) -> str:
    if _is_internal_resource_uri(uri):
        return await _run_internal_resource_read_call(uri)
    return await _run_current_agent_get_resource_call(
        agent,
        uri,
        server_name=server_name,
    )


async def _dispatch_smart_list_resources_tool(
    agent: Any,
    agent_card_path: str,
    mcp_connect: list[str] | None = None,
    server_name: str | None = None,
) -> str:
    context = getattr(agent, "context", None)
    return await _run_smart_list_resources_call(
        context,
        agent_card_path,
        server_name=server_name,
        mcp_connect=mcp_connect,
    )


async def _dispatch_smart_get_resource_tool(
    agent: Any,
    agent_card_path: str,
    resource_uri: str,
    server_name: str | None = None,
    mcp_connect: list[str] | None = None,
) -> str:
    if _is_internal_resource_uri(resource_uri):
        return await _run_internal_resource_read_call(resource_uri)

    context = getattr(agent, "context", None)
    return await _run_smart_get_resource_call(
        context,
        agent_card_path,
        resource_uri,
        server_name=server_name,
        mcp_connect=mcp_connect,
    )


async def _dispatch_smart_with_resource_tool(
    agent: Any,
    agent_card_path: str,
    message: str,
    resource_uri: str,
    server_name: str | None = None,
    mcp_connect: list[str] | None = None,
) -> str:
    context = getattr(agent, "context", None)
    return await _run_smart_with_resource_call(
        context,
        agent_card_path,
        message,
        resource_uri,
        server_name=server_name,
        mcp_connect=mcp_connect,
    )


async def _dispatch_smart_complete_resource_argument_tool(
    agent: Any,
    agent_card_path: str,
    template_uri: str,
    argument_name: str,
    value: str,
    server_name: str | None = None,
    context_args: dict[str, str] | None = None,
    mcp_connect: list[str] | None = None,
) -> str:
    context = getattr(agent, "context", None)
    return await _run_smart_complete_resource_argument_call(
        context,
        agent_card_path,
        template_uri,
        argument_name,
        value,
        server_name=server_name,
        context_args=context_args,
        mcp_connect=mcp_connect,
    )


class SmartAgent(McpAgent):
    """Smart agent with built-in tools for AgentCard execution and validation."""

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, context=context, **kwargs)
        _enable_smart_tooling(self)

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SMART

    async def smart(
        self,
        agent_card_path: str,
        message: str | None = None,
        mcp_connect: list[str] | None = None,
        action: Literal["run", "validate"] = "run",
    ) -> str:
        """Run or validate AgentCards."""
        return await _dispatch_smart_tool(
            self,
            agent_card_path,
            message,
            mcp_connect=mcp_connect,
            action=action,
        )

    async def slash_command(self, command: str) -> str:
        """Execute a slash command using `/...` syntax."""
        return await _dispatch_slash_command_tool(self, command)

    async def read_resource(self, uri: str, server_name: str | None = None) -> str:
        """Read a bundled internal resource or attached MCP resource by URI."""
        return await _dispatch_resource_read_tool(self, uri, server_name=server_name)

    async def smart_list_resources(
        self,
        agent_card_path: str,
        mcp_connect: list[str] | None = None,
        server_name: str | None = None,
    ) -> str:
        """List card-agent resources and bundled internal resources."""
        return await _dispatch_smart_list_resources_tool(
            self,
            agent_card_path,
            mcp_connect=mcp_connect,
            server_name=server_name,
        )

    async def smart_get_resource(
        self,
        agent_card_path: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        """Read a card-agent resource or bundled internal resource by URI."""
        return await _dispatch_smart_get_resource_tool(
            self,
            agent_card_path,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_with_resource(
        self,
        agent_card_path: str,
        message: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        """Run a card agent prompt with a resource attachment."""
        return await _dispatch_smart_with_resource_tool(
            self,
            agent_card_path,
            message,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_complete_resource_argument(
        self,
        agent_card_path: str,
        template_uri: str,
        argument_name: str,
        value: str,
        server_name: str | None = None,
        context_args: dict[str, str] | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        """Run MCP completion for a resource-template argument."""
        return await _dispatch_smart_complete_resource_argument_tool(
            self,
            agent_card_path,
            template_uri,
            argument_name,
            value,
            server_name=server_name,
            context_args=context_args,
            mcp_connect=mcp_connect,
        )


class SmartAgentsAsToolsAgent(AgentsAsToolsAgent):
    """Agents-as-tools wrapper with smart tools."""

    def __init__(
        self,
        config: AgentConfig,
        agents: list["LlmAgent"],
        options: AgentsAsToolsOptions | None = None,
        context: Any | None = None,
        child_message_files: dict[str, list[Path]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            agents=agents,
            options=options,
            context=context,
            child_message_files=child_message_files,
            **kwargs,
        )
        _enable_smart_tooling(self)

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SMART

    async def smart(
        self,
        agent_card_path: str,
        message: str | None = None,
        mcp_connect: list[str] | None = None,
        action: Literal["run", "validate"] = "run",
    ) -> str:
        return await _dispatch_smart_tool(
            self,
            agent_card_path,
            message,
            mcp_connect=mcp_connect,
            action=action,
        )

    async def slash_command(self, command: str) -> str:
        return await _dispatch_slash_command_tool(self, command)

    async def read_resource(self, uri: str, server_name: str | None = None) -> str:
        return await _dispatch_resource_read_tool(self, uri, server_name=server_name)

    async def smart_list_resources(
        self,
        agent_card_path: str,
        mcp_connect: list[str] | None = None,
        server_name: str | None = None,
    ) -> str:
        return await _dispatch_smart_list_resources_tool(
            self,
            agent_card_path,
            mcp_connect=mcp_connect,
            server_name=server_name,
        )

    async def smart_get_resource(
        self,
        agent_card_path: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        return await _dispatch_smart_get_resource_tool(
            self,
            agent_card_path,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_with_resource(
        self,
        agent_card_path: str,
        message: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        return await _dispatch_smart_with_resource_tool(
            self,
            agent_card_path,
            message,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_complete_resource_argument(
        self,
        agent_card_path: str,
        template_uri: str,
        argument_name: str,
        value: str,
        server_name: str | None = None,
        context_args: dict[str, str] | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        return await _dispatch_smart_complete_resource_argument_tool(
            self,
            agent_card_path,
            template_uri,
            argument_name,
            value,
            server_name=server_name,
            context_args=context_args,
            mcp_connect=mcp_connect,
        )


class SmartAgentWithUI(McpUIMixin, SmartAgent):
    """Smart agent with UI support."""

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
        ui_mode: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, context=context, ui_mode=ui_mode, **kwargs)
