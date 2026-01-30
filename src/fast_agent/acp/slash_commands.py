"""
Slash Commands for ACP

Provides slash command support for the ACP server, allowing clients to
discover and invoke special commands with the /command syntax.

Session commands (status, tools, skills, history, clear, session) are always available.
Agent-specific commands are queried from the current agent if it implements
ACPAwareProtocol.
"""

from __future__ import annotations

import inspect
import shlex
import time
import uuid
from importlib.metadata import version as get_version
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
    cast,
)

from acp.helpers import text_block, tool_content
from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    ToolCallProgress,
    ToolCallStart,
    UnstructuredCommandInput,
)

from fast_agent.acp.command_io import ACPCommandIO
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import agent_cards as agent_card_handlers
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.protocols import ACPCommandAllowlistProvider, InstructionAwareAgent
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.renderers.history_markdown import render_history_overview_markdown
from fast_agent.commands.renderers.session_markdown import render_session_list_markdown
from fast_agent.commands.renderers.skills_markdown import (
    render_marketplace_skills,
    render_skill_list,
    render_skills_by_directory,
    render_skills_registry_overview,
    render_skills_remove_list,
)
from fast_agent.commands.renderers.status_markdown import (
    render_permissions_markdown,
    render_status_markdown,
    render_system_prompt_markdown,
)
from fast_agent.commands.renderers.tools_markdown import render_tools_markdown
from fast_agent.commands.session_summaries import build_session_list_summary
from fast_agent.commands.status_summaries import (
    build_permissions_summary,
    build_status_summary,
    build_system_prompt_summary,
)
from fast_agent.commands.tool_summaries import build_tool_summaries
from fast_agent.config import get_settings
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.core.logging.logger import get_logger
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.interfaces import ACPAwareProtocol, AgentProtocol
from fast_agent.paths import resolve_environment_paths
from fast_agent.skills import SKILLS_DEFAULT
from fast_agent.skills.manager import (
    DEFAULT_SKILL_REGISTRIES,
    candidate_marketplace_urls,
    fetch_marketplace_skills,
    fetch_marketplace_skills_with_source,
    format_marketplace_display_url,
    get_manager_directory,
    get_marketplace_url,
    list_local_skills,
    reload_skill_manifests,
    resolve_skill_directories,
)
from fast_agent.skills.registry import SkillManifest, format_skills_for_prompt

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcp.types import ListToolsResult

    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.commands.context import AgentProvider
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.fastagent import AgentInstance


class _SimpleAgentProvider:
    def __init__(self, agents: "Mapping[str, object]") -> None:
        self._agents = agents

    def _agent(self, name: str):
        return self._agents[name]

    def agent_names(self) -> Iterable[str]:
        return list(self._agents.keys())

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None) -> object:
        return {}


class _ACPAgentCardManager:
    def __init__(self, handler: "SlashCommandHandler") -> None:
        self._handler = handler

    def can_load_agent_cards(self) -> bool:
        return self._handler._card_loader is not None

    def can_dump_agent_cards(self) -> bool:
        return self._handler._dump_agent_callback is not None

    def can_attach_agent_tools(self) -> bool:
        return self._handler._attach_agent_callback is not None

    def can_detach_agent_tools(self) -> bool:
        return self._handler._detach_agent_callback is not None

    def can_reload_agents(self) -> bool:
        return self._handler._reload_callback is not None

    async def load_agent_card(
        self, source: str, parent_agent: str | None = None
    ) -> tuple[list[str], list[str]]:
        if not self._handler._card_loader:
            raise RuntimeError("AgentCard loading is not available.")
        instance, loaded_names, attached_names = await self._handler._card_loader(
            source, parent_agent
        )
        self._handler.instance = instance
        return loaded_names, attached_names

    async def dump_agent_card(self, agent_name: str) -> str:
        if not self._handler._dump_agent_callback:
            raise RuntimeError("AgentCard dumping is not available.")
        return await self._handler._dump_agent_callback(agent_name)

    async def attach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]:
        if not self._handler._attach_agent_callback:
            raise RuntimeError("Agent tool attachment is not available.")
        instance, attached_names = await self._handler._attach_agent_callback(
            parent_agent, list(child_agents)
        )
        self._handler.instance = instance
        return attached_names

    async def detach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]:
        if not self._handler._detach_agent_callback:
            raise RuntimeError("Agent tool detachment is not available.")
        instance, removed_names = await self._handler._detach_agent_callback(
            parent_agent, list(child_agents)
        )
        self._handler.instance = instance
        return removed_names

    async def reload_agents(self) -> bool:
        if not self._handler._reload_callback:
            return False
        return await self._handler._reload_callback()

    def agent_names(self) -> Iterable[str]:
        return list(self._handler.instance.agents.keys())


class SlashCommandHandler:
    """Handles slash command execution for ACP sessions."""

    def __init__(
        self,
        session_id: str,
        instance: AgentInstance,
        primary_agent_name: str,
        *,
        history_exporter: type[HistoryExporter] | HistoryExporter | None = None,
        client_info: dict | None = None,
        client_capabilities: dict | None = None,
        protocol_version: int | None = None,
        session_instructions: dict[str, str] | None = None,
        card_loader: Callable[
            [str, str | None], Awaitable[tuple["AgentInstance", list[str], list[str]]]
        ]
        | None = None,
        attach_agent_callback: Callable[
            [str, Sequence[str]], Awaitable[tuple["AgentInstance", list[str]]]
        ]
        | None = None,
        detach_agent_callback: Callable[
            [str, Sequence[str]], Awaitable[tuple["AgentInstance", list[str]]]
        ]
        | None = None,
        dump_agent_callback: Callable[[str], Awaitable[str]] | None = None,
        reload_callback: Callable[[], Awaitable[bool]] | None = None,
        set_current_mode_callback: Callable[[str], Awaitable[None] | None] | None = None,
        instruction_resolver: Callable[[str], Awaitable[str | None]] | None = None,
    ):
        """
        Initialize the slash command handler.

        Args:
            session_id: The ACP session ID
            instance: The agent instance for this session
            primary_agent_name: Name of the primary agent
            history_exporter: Optional history exporter
            client_info: Client information from ACP initialize
            client_capabilities: Client capabilities from ACP initialize
            protocol_version: ACP protocol version
        """
        self.session_id = session_id
        self.instance = instance
        self.primary_agent_name = primary_agent_name
        self._logger = get_logger(__name__)
        # Track current agent (can change via setSessionMode). Ensure it exists.
        if primary_agent_name in instance.agents:
            self.current_agent_name = primary_agent_name
        else:
            # Fallback: pick the first registered agent to enable agent-specific commands.
            self.current_agent_name = next(iter(instance.agents.keys()), primary_agent_name)
        self.history_exporter = history_exporter or HistoryExporter
        self._created_at = time.time()
        self.client_info = client_info
        self.client_capabilities = client_capabilities
        self.protocol_version = protocol_version
        self._session_instructions = session_instructions or {}
        self._card_loader = card_loader
        self._attach_agent_callback = attach_agent_callback
        self._detach_agent_callback = detach_agent_callback
        self._dump_agent_callback = dump_agent_callback
        self._reload_callback = reload_callback
        self._set_current_mode_callback = set_current_mode_callback
        self._instruction_resolver = instruction_resolver
        self._acp_context: ACPContext | None = None

        # Session-level commands (always available, operate on current agent)
        self._session_commands: dict[str, AvailableCommand] = {
            "status": AvailableCommand(
                name="status",
                description="Show fast-agent diagnostics",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="[system|auth|authreset]")
                ),
            ),
            "tools": AvailableCommand(
                name="tools",
                description="List available tools",
                input=None,
            ),
            "skills": AvailableCommand(
                name="skills",
                description="List or manage local skills (add/remove/registry)",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="[add|remove|registry] [name|number|url]")
                ),
            ),
            "model": AvailableCommand(
                name="model",
                description="Update model settings",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="reasoning <value> | verbosity <value>")
                ),
            ),
            "history": AvailableCommand(
                name="history",
                description="Show or manage conversation history",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="[show|save|load] [args]")
                ),
            ),
            "clear": AvailableCommand(
                name="clear",
                description="Clear history (`last` for prev. turn)",
                input=AvailableCommandInput(root=UnstructuredCommandInput(hint="[last]")),
            ),
            "session": AvailableCommand(
                name="session",
                description="List or manage sessions",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(
                        hint="[list|new|resume|title|fork|clear] [args]"
                    )
                ),
            ),
            "card": AvailableCommand(
                name="card",
                description="Load an AgentCard from file or URL",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="<filename|url> [--tool [remove]]")
                ),
            ),
            "agent": AvailableCommand(
                name="agent",
                description="Attach an agent as a tool or dump its AgentCard",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="<@name> [--tool [remove]|--dump]")
                ),
            ),
        }
        if self._reload_callback is not None:
            self._session_commands["reload"] = AvailableCommand(
                name="reload",
                description="Reload AgentCards",
                input=None,
            )

    def get_available_commands(self) -> list[AvailableCommand]:
        """Get combined session commands and current agent's commands."""
        commands = list(self._get_allowed_session_commands().values())

        # Add agent-specific commands if current agent is ACP-aware
        agent = self._get_current_agent()
        if isinstance(agent, ACPAwareProtocol):
            for name, cmd in agent.acp_commands.items():
                # Convert ACPCommand to AvailableCommand
                cmd_input = None
                if cmd.input_hint:
                    cmd_input = AvailableCommandInput(
                        root=UnstructuredCommandInput(hint=cmd.input_hint)
                    )
                commands.append(
                    AvailableCommand(name=name, description=cmd.description, input=cmd_input)
                )

        return commands

    def set_acp_context(self, acp_context: ACPContext | None) -> None:
        """Set the ACP context for this handler."""
        self._acp_context = acp_context

    def _get_allowed_session_commands(self) -> dict[str, AvailableCommand]:
        """
        Return session-level commands filtered by the current agent's policy.

        By default, all session commands are available. ACP-aware agents can restrict
        session commands (e.g. Setup/wizard flows) by defining a
        `acp_session_commands_allowlist: set[str] | None` attribute.
        """
        agent = self._get_current_agent()
        if not isinstance(agent, ACPAwareProtocol):
            return self._session_commands

        allowlist = None
        if isinstance(agent, ACPCommandAllowlistProvider):
            allowlist = agent.acp_session_commands_allowlist

        if allowlist is None:
            return self._session_commands

        try:
            allowset = {str(name) for name in allowlist}
        except Exception:
            return self._session_commands

        return {name: cmd for name, cmd in self._session_commands.items() if name in allowset}

    def set_current_agent(self, agent_name: str) -> None:
        """
        Update the current agent for this session.

        This is called when the user switches modes via setSessionMode.

        Args:
            agent_name: Name of the agent to use for slash commands
        """
        self.current_agent_name = agent_name

    async def _switch_current_mode(self, agent_name: str) -> bool:
        """Switch current mode for ACP session state if available."""
        if agent_name not in self.instance.agents:
            return False
        self.set_current_agent(agent_name)
        if self._set_current_mode_callback:
            result = self._set_current_mode_callback(agent_name)
            if inspect.isawaitable(result):
                await result
        return True

    def update_session_instruction(self, agent_name: str, instruction: str | None) -> None:
        """
        Update the cached session instruction for an agent.

        Call this when an agent's system prompt has been rebuilt (e.g., after
        connecting new MCP servers) to keep the /system command output current.

        Args:
            agent_name: Name of the agent whose instruction was updated
            instruction: The new instruction (or None to remove from cache)
        """
        if instruction:
            self._session_instructions[agent_name] = instruction
        elif agent_name in self._session_instructions:
            del self._session_instructions[agent_name]

    def _get_current_agent(self) -> AgentProtocol | None:
        """Return the current agent or None if it does not exist."""
        return self.instance.agents.get(self.current_agent_name)

    def _get_current_agent_or_error(
        self,
        heading: str,
        missing_template: str | None = None,
    ) -> tuple[AgentProtocol | None, str | None]:
        """
        Return the current agent or an error response string if it is missing.

        Args:
            heading: Heading for the error message.
            missing_template: Optional custom missing-agent message.
        """
        agent = self._get_current_agent()
        if agent:
            return agent, None

        message = (
            missing_template or f"Agent '{self.current_agent_name}' not found for this session."
        )
        return None, "\n".join([heading, "", message])

    def _build_command_context(self) -> CommandContext:
        settings = get_settings()
        provider = getattr(self.instance, "app", None)
        if provider is None:
            provider = _SimpleAgentProvider(self.instance.agents)
        return CommandContext(
            agent_provider=cast("AgentProvider", provider),
            current_agent_name=self.current_agent_name,
            io=ACPCommandIO(),
            settings=settings,
        )

    def _format_outcome_as_markdown(
        self,
        outcome: CommandOutcome,
        heading: str,
        *,
        io: ACPCommandIO | None = None,
    ) -> str:
        extra_messages = io.messages if io else None
        return render_command_outcome_markdown(
            outcome,
            heading=heading,
            extra_messages=extra_messages,
        )

    async def _send_session_info_update(self) -> None:
        if self._acp_context is None:
            return
        from fast_agent.session import extract_session_title, get_session_manager

        manager = get_session_manager()
        session = manager.current_session
        if session is None:
            return

        title = extract_session_title(session.info.metadata)
        if title is None:
            return

        try:
            await self._acp_context.send_session_info_update(
                title=title,
                updated_at=session.info.last_activity.isoformat(),
            )
        except Exception as exc:
            self._logger.debug(
                "Failed to send ACP session info update",
                session_id=self.session_id,
                error=str(exc),
            )

    def is_slash_command(self, prompt_text: str) -> bool:
        """Check if the prompt text is a slash command."""
        return prompt_text.strip().startswith("/")

    def parse_command(self, prompt_text: str) -> tuple[str, str]:
        """
        Parse a slash command into command name and arguments.

        Args:
            prompt_text: The full prompt text starting with /

        Returns:
            Tuple of (command_name, arguments)
        """
        text = prompt_text.strip()
        if not text.startswith("/"):
            return "", text

        # Remove leading slash
        text = text[1:]

        # Split on first whitespace
        command_name, _, arguments = text.partition(" ")
        arguments = arguments.lstrip()

        return command_name, arguments

    async def execute_command(self, command_name: str, arguments: str) -> str:
        """
        Execute a slash command and return the response.

        Args:
            command_name: Name of the command to execute
            arguments: Arguments passed to the command

        Returns:
            The command response as a string
        """
        # Check session-level commands first (filtered by agent policy)
        allowed_session_commands = self._get_allowed_session_commands()
        if command_name in allowed_session_commands:
            if command_name == "status":
                return await self._handle_status(arguments)
            if command_name == "tools":
                return await self._handle_tools()
            if command_name == "skills":
                return await self._handle_skills(arguments)
            if command_name == "history":
                return await self._handle_history(arguments)
            if command_name == "clear":
                return await self._handle_clear(arguments)
            if command_name == "model":
                return await self._handle_model(arguments)
            if command_name == "session":
                return await self._handle_session(arguments)
            if command_name == "card":
                return await self._handle_card(arguments)
            if command_name == "agent":
                return await self._handle_agent(arguments)
            if command_name == "reload":
                return await self._handle_reload()

        # Check agent-specific commands
        agent = self._get_current_agent()
        if isinstance(agent, ACPAwareProtocol):
            agent_commands = agent.acp_commands
            if command_name in agent_commands:
                return await agent_commands[command_name].handler(arguments)

        # Unknown command
        available = self.get_available_commands()
        return f"Unknown command: /{command_name}\n\nAvailable commands:\n" + "\n".join(
            f"  /{cmd.name} - {cmd.description}" for cmd in available
        )

    async def _handle_history(self, arguments: str | None = None) -> str:
        """Handle the /history command."""
        remainder = (arguments or "").strip()
        if not remainder:
            return await self._render_history_overview()

        try:
            tokens = shlex.split(remainder)
        except ValueError:
            tokens = remainder.split(maxsplit=1)

        if not tokens:
            return await self._render_history_overview()

        subcmd = tokens[0].lower()
        argument = remainder[len(tokens[0]) :].strip()

        if subcmd in {"show", "list"}:
            return await self._render_history_overview()
        if subcmd == "save":
            return await self._handle_save(argument)
        if subcmd == "load":
            return await self._handle_load(argument)

        return "\n".join(
            [
                "# history",
                "",
                f"Unknown /history action: {subcmd}",
                "Usage: /history [show|save|load] [args]",
            ]
        )

    async def _handle_model(self, arguments: str | None = None) -> str:
        remainder = (arguments or "").strip()
        value = None
        return_value = "reasoning"
        if remainder:
            try:
                tokens = shlex.split(remainder)
            except ValueError:
                tokens = remainder.split(maxsplit=1)

            if tokens:
                subcmd = tokens[0].lower()
                argument = remainder[len(tokens[0]) :].strip()
                if subcmd == "verbosity":
                    return_value = "verbosity"
                    value = argument or None
                elif subcmd == "reasoning":
                    value = argument or None
                else:
                    return "Usage: /model reasoning <value> | /model verbosity <value>"

        io = ACPCommandIO()
        ctx = CommandContext(
            agent_provider=_SimpleAgentProvider(self.instance.agents),
            current_agent_name=self.current_agent_name,
            io=io,
        )
        if return_value == "verbosity":
            outcome = await model_handlers.handle_model_verbosity(
                ctx,
                agent_name=self.current_agent_name,
                value=value,
            )
        else:
            outcome = await model_handlers.handle_model_reasoning(
                ctx,
                agent_name=self.current_agent_name,
                value=value,
            )
        return render_command_outcome_markdown(outcome, heading="model")

    async def _handle_session(self, arguments: str | None = None) -> str:
        """Handle the /session command."""
        remainder = (arguments or "").strip()
        if not remainder:
            return self._render_session_list()

        try:
            tokens = shlex.split(remainder)
        except ValueError:
            tokens = remainder.split(maxsplit=1)

        if not tokens:
            return self._render_session_list()

        subcmd = tokens[0].lower()
        argument = remainder[len(tokens[0]) :].strip()

        if subcmd == "list":
            return self._render_session_list()
        if subcmd == "new":
            return await self._handle_session_new(argument)
        if subcmd == "resume":
            return await self._handle_session_resume(argument)
        if subcmd == "title":
            return await self._handle_session_title(argument)
        if subcmd == "fork":
            return await self._handle_session_fork(argument)
        if subcmd in {"delete", "clear"}:
            return await self._handle_session_delete(argument)
        if subcmd == "pin":
            return await self._handle_session_pin(argument)

        return "\n".join(
            [
                "# session",
                "",
                f"Unknown /session action: {subcmd}",
                "Usage: /session [list|new|resume|title|fork|delete|pin] [args]",
            ]
        )

    async def _handle_status(self, arguments: str | None = None) -> str:
        """Handle the /status command."""
        normalized = (arguments or "").strip().lower()
        if normalized == "system":
            return await self._handle_status_system()
        if normalized == "auth":
            return self._handle_status_auth()
        if normalized == "authreset":
            return self._handle_status_authreset()

        try:
            fa_version = get_version("fast-agent-mcp")
        except Exception:
            fa_version = "unknown"

        agent = self._get_current_agent()
        uptime_seconds = max(time.time() - self._created_at, 0.0)
        summary = build_status_summary(
            fast_agent_version=fa_version,
            agent=agent,
            client_info=self.client_info,
            client_capabilities=self.client_capabilities,
            protocol_version=str(self.protocol_version) if self.protocol_version is not None else None,
            uptime_seconds=uptime_seconds,
            instance=self.instance,
        )
        return render_status_markdown(summary, heading="fast-agent ACP status")

    async def _handle_status_system(self) -> str:
        """Handle the /status system command to show the system prompt."""
        heading = "# system prompt"

        agent, error = self._get_current_agent_or_error(heading)
        if error:
            return error

        if self._instruction_resolver:
            try:
                refreshed = await self._instruction_resolver(self.current_agent_name)
            except Exception as exc:
                self._logger.debug(
                    "Failed to refresh session instruction",
                    agent_name=self.current_agent_name,
                    error=str(exc),
                )
            else:
                if refreshed:
                    self.update_session_instruction(self.current_agent_name, refreshed)
                    if isinstance(agent, InstructionAwareAgent):
                        self.update_session_instruction(agent.name, refreshed)

        summary = build_system_prompt_summary(
            agent=agent,
            session_instructions=self._session_instructions,
            current_agent_name=self.current_agent_name,
        )
        return render_system_prompt_markdown(summary, heading="system prompt")

    async def _render_history_overview(self) -> str:
        """Render a lightweight conversation history overview."""
        heading = "# conversation history"
        agent, error = self._get_current_agent_or_error(heading)
        if error:
            return error
        assert agent is not None

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        await history_handlers.handle_show_history(
            ctx,
            agent_name=self.current_agent_name,
        )
        if not io.history_overview:
            return "\n".join([heading, "", "No messages yet."])

        return render_history_overview_markdown(
            io.history_overview,
            heading="conversation history",
        )

    def _render_session_list(self) -> str:
        """Render a list of recent sessions."""
        summary = build_session_list_summary()
        return render_session_list_markdown(summary, heading="sessions")

    async def _handle_session_resume(self, argument: str) -> str:
        session_id = argument or None
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await sessions_handlers.handle_resume_session(
            ctx,
            agent_name=self.current_agent_name,
            session_id=session_id,
        )
        if outcome.switch_agent:
            await self._switch_current_mode(outcome.switch_agent)
        return self._format_outcome_as_markdown(outcome, "session resume", io=io)

    async def _handle_session_title(self, argument: str) -> str:
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        title = argument.strip() or None
        outcome = await sessions_handlers.handle_title_session(
            ctx,
            title=title,
            session_id=self.session_id,
        )
        if title:
            await self._send_session_info_update()
        return self._format_outcome_as_markdown(outcome, "session title", io=io)

    async def _handle_session_fork(self, argument: str) -> str:
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await sessions_handlers.handle_fork_session(
            ctx,
            title=argument.strip() or None,
        )
        return self._format_outcome_as_markdown(outcome, "session fork", io=io)

    async def _handle_session_new(self, argument: str) -> str:
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await sessions_handlers.handle_create_session(
            ctx,
            session_name=argument.strip() or None,
        )
        cleared = clear_agent_histories(self.instance.agents, self._logger)
        if cleared:
            cleared_list = ", ".join(sorted(cleared))
            outcome.add_message(
                f"Cleared agent history: {cleared_list}",
                channel="info",
            )
        return self._format_outcome_as_markdown(outcome, "session new", io=io)

    async def _handle_session_delete(self, argument: str) -> str:
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await sessions_handlers.handle_clear_sessions(
            ctx,
            target=argument.strip() or None,
        )
        return self._format_outcome_as_markdown(outcome, "session delete", io=io)

    async def _handle_session_pin(self, argument: str) -> str:
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        pin_argument = argument.strip() if argument else ""
        value: str | None = None
        target: str | None = None
        if pin_argument:
            try:
                pin_tokens = shlex.split(pin_argument)
            except ValueError:
                pin_tokens = pin_argument.split(maxsplit=1)
            if pin_tokens:
                first = pin_tokens[0].lower()
                value_tokens = {
                    "on",
                    "off",
                    "toggle",
                    "true",
                    "false",
                    "yes",
                    "no",
                    "1",
                    "0",
                    "enable",
                    "enabled",
                    "disable",
                    "disabled",
                }
                if first in value_tokens:
                    value = first
                    target = " ".join(pin_tokens[1:]).strip() or None
                else:
                    target = pin_argument
        outcome = await sessions_handlers.handle_pin_session(
            ctx,
            value=value,
            target=target,
        )
        return self._format_outcome_as_markdown(outcome, "session pin", io=io)


    def _handle_status_auth(self) -> str:
        """Handle the /status auth command to show permissions from auths.md."""
        heading = "permissions"
        auths_path = resolve_environment_paths().permissions_file
        resolved_path = auths_path.resolve()

        if not auths_path.exists():
            summary = build_permissions_summary(
                heading=heading,
                message="No permissions set",
                path=str(resolved_path),
            )
            return render_permissions_markdown(summary)

        try:
            content = auths_path.read_text(encoding="utf-8")
            message = content.strip() if content.strip() else "No permissions set"
            summary = build_permissions_summary(
                heading=heading,
                message=message,
                path=str(resolved_path),
            )
            return render_permissions_markdown(summary)
        except Exception as exc:
            summary = build_permissions_summary(
                heading=heading,
                message=f"Failed to read permissions file: {exc}",
                path=str(resolved_path),
            )
            return render_permissions_markdown(summary)

    def _handle_status_authreset(self) -> str:
        """Handle the /status authreset command to remove the auths.md file."""
        heading = "reset permissions"
        auths_path = resolve_environment_paths().permissions_file
        resolved_path = auths_path.resolve()

        if not auths_path.exists():
            summary = build_permissions_summary(
                heading=heading,
                message="No permissions file exists.",
                path=str(resolved_path),
            )
            return render_permissions_markdown(summary)

        try:
            auths_path.unlink()
            summary = build_permissions_summary(
                heading=heading,
                message="Permissions file removed successfully.",
                path=str(resolved_path),
            )
            return render_permissions_markdown(summary)
        except Exception as exc:
            summary = build_permissions_summary(
                heading=heading,
                message=f"Failed to remove permissions file: {exc}",
                path=str(resolved_path),
            )
            return render_permissions_markdown(summary)

    async def _handle_tools(self) -> str:
        """List available tools for the current agent."""
        heading = "tools"

        agent, error = self._get_current_agent_or_error(f"# {heading}")
        if error:
            return error

        if not isinstance(agent, AgentProtocol):
            return "\n".join(
                [
                    f"# {heading}",
                    "",
                    "This agent does not support tool listing.",
                ]
            )

        try:
            tools_result: "ListToolsResult" = await agent.list_tools()
        except Exception as exc:  # noqa: BLE001
            return "\n".join(
                [
                    f"# {heading}",
                    "",
                    "Failed to fetch tools from the agent.",
                    f"Details: {exc}",
                ]
            )

        tools = tools_result.tools if tools_result else None
        if not tools:
            return "\n".join(
                [
                    f"# {heading}",
                    "",
                    "No MCP tools available for this agent.",
                ]
            )

        summaries = build_tool_summaries(agent, list(tools))
        return render_tools_markdown(summaries, heading=heading)

    async def _handle_skills(self, arguments: str | None = None) -> str:
        """Manage local skills (list/add/remove)."""
        tokens = (arguments or "").strip().split(maxsplit=1)
        action = tokens[0].lower() if tokens else "list"
        remainder = tokens[1] if len(tokens) > 1 else ""

        if action in {"list", ""}:
            return self._handle_skills_list()
        if action in {"add", "install"}:
            return await self._handle_skills_add(remainder)
        if action in {"registry", "marketplace", "source"}:
            return await self._handle_skills_registry(remainder)
        if action in {"remove", "rm", "delete", "uninstall"}:
            return await self._handle_skills_remove(remainder)

        return "Unknown /skills action. Use `/skills`, `/skills add`, or `/skills remove`."

    async def _handle_skills_registry(self, argument: str) -> str:
        heading = "# skills registry"
        argument = argument.strip()

        # Get configured registries from settings
        settings = get_settings()
        configured_urls = settings.skills.marketplace_urls or list(DEFAULT_SKILL_REGISTRIES)

        if not argument:
            current = get_marketplace_url(settings)
            display_current = format_marketplace_display_url(current)
            display_registries = [format_marketplace_display_url(url) for url in configured_urls]
            return render_skills_registry_overview(
                heading="skills registry",
                current_registry=display_current,
                configured_urls=display_registries,
            )

        # Check if argument is a number (select from configured registries)
        if argument.isdigit():
            index = int(argument)
            if not configured_urls:
                return f"{heading}\n\nNo registries configured."
            if 1 <= index <= len(configured_urls):
                url = configured_urls[index - 1]
            else:
                return f"{heading}\n\nInvalid registry number. Use 1-{len(configured_urls)}."
        else:
            url = argument

        candidates = candidate_marketplace_urls(url)
        try:
            marketplace, resolved_url = await fetch_marketplace_skills_with_source(url)
        except Exception as exc:  # noqa: BLE001
            display_url = format_marketplace_display_url(url)
            self._logger.warning(
                "Failed to load skills registry",
                data={
                    "registry": url,
                    "candidates": candidates,
                    "error": str(exc),
                },
            )
            return "\n".join(
                [
                    heading,
                    "",
                    f"Failed to load registry: {exc}",
                    f"Registry: {display_url}",
                ]
            )

        if not marketplace:
            display_url = format_marketplace_display_url(url)
            return "\n".join(
                [
                    heading,
                    "",
                    "No skills found in the registry; registry unchanged.",
                    f"Registry: {display_url}",
                ]
            )

        # Update only the active registry, preserve the configured list
        settings.skills.marketplace_url = resolved_url

        display_url = format_marketplace_display_url(resolved_url)
        if candidates:
            self._logger.debug(
                "Resolved skills registry",
                data={
                    "input": url,
                    "resolved": resolved_url,
                    "candidates": candidates,
                },
            )
        response_lines = [
            heading,
            "",
            f"Registry set to: `{display_url}`",
            "",
            f"Skills discovered: {len(marketplace)}",
        ]

        return "\n".join(response_lines)

    def _handle_skills_list(self) -> str:
        directories = resolve_skill_directories()
        all_manifests: dict[Path, list[SkillManifest]] = {}
        for directory in directories:
            all_manifests[directory] = list_local_skills(directory) if directory.exists() else []
        response = render_skills_by_directory(all_manifests, heading="skills", cwd=Path.cwd())
        override_section = self._skills_override_section()
        if override_section:
            return "\n".join([response, "", override_section])
        return response

    def _skills_override_section(self) -> str | None:
        agent = self._get_current_agent()
        if not agent:
            return None
        config = getattr(agent, "config", None)
        if not config:
            return None
        if getattr(config, "skills", SKILLS_DEFAULT) is SKILLS_DEFAULT:
            return None
        manifests = list(getattr(config, "skill_manifests", []) or [])
        sources: list[str] = []
        for manifest in manifests:
            path = getattr(manifest, "path", None)
            if not path:
                continue
            source_path = path.parent if Path(path).is_file() else Path(path)
            try:
                display_path = source_path.relative_to(Path.cwd())
            except ValueError:
                display_path = source_path
            sources.append(str(display_path))
        sources = sorted(set(sources))
        lines = [
            "## Active agent skills (override)",
            "",
            "Note: this agent has an explicit skills configuration. `/skills` lists global skills directories from settings, not per-agent overrides.",
            "Update settings.skills.directories or the --skills flag to change this list.",
        ]
        if sources:
            sources_list = ", ".join(f"`{source}`" for source in sources)
            lines.extend(["", f"Sources: {sources_list}"])
        lines.append("")
        if not manifests:
            lines.append("No skills configured for this agent.")
        else:
            lines.append("Configured skills:")
            lines.extend(render_skill_list(manifests, cwd=Path.cwd()))
        return "\n".join(lines)

    async def _handle_skills_add(self, argument: str) -> str:
        if argument.strip().lower() in {"q", "quit", "exit"}:
            return "Cancelled."

        agent, error = self._get_current_agent_or_error("# skills add")
        if error:
            return error
        assert agent is not None

        tool_call_id = self._build_tool_call_id()
        await self._send_skills_update(
            agent,
            tool_call_id,
            title="Install skill",
            status="in_progress",
            message="Fetching marketplace…",
            start=True,
        )

        argument_value = argument.strip() or None

        if not argument_value:
            marketplace_url = get_marketplace_url(get_settings())
            display_url = format_marketplace_display_url(marketplace_url)
            try:
                marketplace = await fetch_marketplace_skills(marketplace_url)
            except Exception as exc:  # noqa: BLE001
                return (
                    "# skills add\n\n"
                    f"Failed to load marketplace: {exc}\n\n"
                    f"Repository: `{display_url}`"
                )

            repository = display_url
            if marketplace:
                repo_url = marketplace[0].repo_url
                repo_ref = marketplace[0].repo_ref
                repository = f"{repo_url}@{repo_ref}" if repo_ref else repo_url

            return render_marketplace_skills(
                marketplace,
                heading="skills add",
                repository=repository,
            )

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        try:
            outcome = await skills_handlers.handle_add_skill(
                ctx,
                agent_name=self.current_agent_name,
                argument=argument_value,
                interactive=False,
            )
        except Exception as exc:  # noqa: BLE001
            await self._send_skills_update(
                agent,
                tool_call_id,
                title="Install failed",
                status="completed",
                message=str(exc),
            )
            return f"# skills add\n\nFailed to install skill: {exc}"

        if any(message.channel == "error" for message in outcome.messages):
            await self._send_skills_update(
                agent,
                tool_call_id,
                title="Install failed",
                status="completed",
                message="Failed to install skill",
            )
        else:
            await self._send_skills_update(
                agent,
                tool_call_id,
                title="Install complete",
                status="completed",
                message="Installed skill",
            )

        return self._format_outcome_as_markdown(outcome, "skills add", io=io)

    async def _handle_skills_remove(self, argument: str) -> str:
        if argument.strip().lower() in {"q", "quit", "exit"}:
            return "Cancelled."

        argument_value = argument.strip() or None
        if not argument_value:
            manager_dir = get_manager_directory()
            manifests = list_local_skills(manager_dir)
            return render_skills_remove_list(
                heading="skills remove",
                manager_dir=manager_dir,
                manifests=manifests,
                cwd=Path.cwd(),
            )

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        try:
            outcome = await skills_handlers.handle_remove_skill(
                ctx,
                agent_name=self.current_agent_name,
                argument=argument_value,
                interactive=False,
            )
        except Exception as exc:  # noqa: BLE001
            return f"# skills remove\n\nFailed to remove skill: {exc}"

        return self._format_outcome_as_markdown(outcome, "skills remove", io=io)

    async def _refresh_agent_skills(self, agent: AgentProtocol) -> None:
        override_dirs = resolve_skill_directories(get_settings())
        registry, manifests = reload_skill_manifests(
            base_dir=Path.cwd(), override_directories=override_dirs
        )
        instruction_context = None
        try:
            skills_text = format_skills_for_prompt(manifests, read_tool_name="read_text_file")
            instruction_context = {"agentSkills": skills_text}
        except Exception:
            instruction_context = None

        await rebuild_agent_instruction(
            agent,
            skill_manifests=manifests,
            context=instruction_context,
            skill_registry=registry,
        )

    def _build_tool_call_id(self) -> str:
        return str(uuid.uuid4())

    async def _send_skills_update(
        self,
        agent: AgentProtocol,
        tool_call_id: str,
        *,
        title: str,
        status: str,
        message: str | None = None,
        start: bool = False,
    ) -> None:
        if not isinstance(agent, ACPAwareProtocol):
            return
        acp = agent.acp
        if not acp:
            return
        try:
            if start:
                await acp.send_session_update(
                    ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=title,
                        kind="fetch",
                        status="in_progress",
                        session_update="tool_call",
                    )
                )
            content = [tool_content(text_block(message))] if message else None
            await acp.send_session_update(
                ToolCallProgress(
                    tool_call_id=tool_call_id,
                    title=title,
                    status=status,  # type: ignore[arg-type]
                    content=content,
                    session_update="tool_call_update",
                )
            )
        except Exception:
            return

    async def _handle_save(self, arguments: str | None = None) -> str:
        """Handle the /history save command by persisting conversation history."""
        heading = "# save conversation"

        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error
        assert agent is not None

        filename = arguments.strip() if arguments and arguments.strip() else None

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await history_handlers.handle_history_save(
            ctx,
            agent_name=self.current_agent_name,
            filename=filename,
            send_func=None,
            history_exporter=self.history_exporter,
        )
        return self._format_outcome_as_markdown(outcome, "save conversation", io=io)

    async def _handle_load(self, arguments: str | None = None) -> str:
        """Handle the /history load command by loading conversation history from a file."""
        heading = "# load conversation"

        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error
        assert agent is not None

        filename = arguments.strip() if arguments and arguments.strip() else None
        error_message = None
        if not filename:
            error_message = "Filename required for /history load."
        else:
            file_path = Path(filename)
            if not file_path.exists():
                error_message = f"File not found: {filename}"

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await history_handlers.handle_history_load(
            ctx,
            agent_name=self.current_agent_name,
            filename=filename,
            error=error_message,
        )
        return self._format_outcome_as_markdown(outcome, "load conversation", io=io)

    async def _handle_card(self, arguments: str | None = None) -> str:
        """Handle the /card command by loading an AgentCard and refreshing agents."""
        args = (arguments or "").strip()
        tokens: list[str] = []
        if args:
            try:
                tokens = shlex.split(args)
            except ValueError as exc:
                return f"Invalid arguments: {exc}"

        add_tool = False
        remove_tool = False
        filename = None
        for token in tokens:
            if token in {"tool", "--tool", "--as-tool", "-t"}:
                add_tool = True
                continue
            if token in {"remove", "--remove"}:
                add_tool = True
                remove_tool = True
                continue
            if filename is None:
                filename = token

        manager = _ACPAgentCardManager(self)
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await agent_card_handlers.handle_card_load(
            ctx,
            manager=manager,
            filename=filename,
            add_tool=add_tool,
            remove_tool=remove_tool,
            current_agent=self.current_agent_name or self.primary_agent_name,
        )
        return self._format_outcome_as_markdown(outcome, "card", io=io)

    async def _handle_agent(self, arguments: str | None = None) -> str:
        """Handle the /agent command for attach or dump actions."""
        args = (arguments or "").strip()
        if not args:
            return "Usage: /agent <name> --tool | /agent [name] --dump"

        try:
            tokens = shlex.split(args)
        except ValueError as exc:
            return f"Invalid arguments: {exc}"

        add_tool = False
        remove_tool = False
        dump = False
        agent_name = None
        unknown: list[str] = []
        for token in tokens:
            if token in {"tool", "--tool", "--as-tool", "-t"}:
                add_tool = True
                continue
            if token in {"remove", "--remove"}:
                add_tool = True
                remove_tool = True
                continue
            if token in {"dump", "--dump", "-d"}:
                dump = True
                continue
            if agent_name is None:
                agent_name = token[1:] if token.startswith("@") else token
                continue
            unknown.append(token)

        if unknown:
            return f"Unexpected arguments: {', '.join(unknown)}"
        if add_tool and dump:
            return "Use either --tool or --dump, not both."
        if not add_tool and not dump:
            return "Usage: /agent <name> --tool [remove] | /agent [name] --dump"

        target_agent = agent_name or self.current_agent_name or self.primary_agent_name
        if not target_agent:
            return "No agent available for this session."

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await agent_card_handlers.handle_agent_command(
            ctx,
            manager=_ACPAgentCardManager(self),
            current_agent=self.current_agent_name or self.primary_agent_name or target_agent,
            target_agent=agent_name,
            add_tool=add_tool,
            remove_tool=remove_tool,
            dump=dump,
        )
        return self._format_outcome_as_markdown(outcome, "agent", io=io)

    async def _handle_reload(self) -> str:
        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await agent_card_handlers.handle_reload_agents(
            ctx,
            manager=_ACPAgentCardManager(self),
        )
        return self._format_outcome_as_markdown(outcome, "reload", io=io)

    async def _handle_clear(self, arguments: str | None = None) -> str:
        """Handle /clear and /clear last commands."""
        normalized = (arguments or "").strip().lower()
        if normalized == "last":
            return await self._handle_clear_last()
        return await self._handle_clear_all()

    async def _handle_clear_all(self) -> str:
        """Clear the entire conversation history."""
        heading = "# clear conversation"
        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error
        assert agent is not None

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await history_handlers.handle_history_clear_all(
            ctx,
            agent_name=self.current_agent_name,
        )
        return self._format_outcome_as_markdown(outcome, "clear conversation", io=io)

    async def _handle_clear_last(self) -> str:
        """Remove the most recent conversation message."""
        heading = "# clear last conversation turn"
        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error
        assert agent is not None

        ctx = self._build_command_context()
        io = cast("ACPCommandIO", ctx.io)
        outcome = await history_handlers.handle_history_clear_last(
            ctx,
            agent_name=self.current_agent_name,
        )
        return self._format_outcome_as_markdown(outcome, "clear last conversation turn", io=io)
