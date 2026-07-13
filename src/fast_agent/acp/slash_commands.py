"""
Slash Commands for ACP

Provides slash command support for the ACP server, allowing clients to
discover and invoke special commands with the /command syntax.

Session commands (status, tools, skills, cards, history, clear, session) are always available.
Agent-specific commands are queried from the current agent if it implements
ACPAwareProtocol.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    cast,
)

from acp.helpers import update_agent_message_text
from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    UnstructuredCommandInput,
)

from fast_agent.acp.command_io import ACPCommandIO
from fast_agent.acp.slash.handlers import cards as cards_slash_handlers
from fast_agent.acp.slash.handlers import cards_manager as cards_manager_slash_handlers
from fast_agent.acp.slash.handlers import clear as clear_slash_handlers
from fast_agent.acp.slash.handlers import commands as commands_slash_handlers
from fast_agent.acp.slash.handlers import history as history_slash_handlers
from fast_agent.acp.slash.handlers import mcp as mcp_slash_handlers
from fast_agent.acp.slash.handlers import model as model_slash_handlers
from fast_agent.acp.slash.handlers import plugins as plugins_slash_handlers
from fast_agent.acp.slash.handlers import session as session_slash_handlers
from fast_agent.acp.slash.handlers import skills as skills_slash_handlers
from fast_agent.acp.slash.handlers import status as status_slash_handlers
from fast_agent.acp.slash.handlers import tools as tools_slash_handlers
from fast_agent.command_actions import (
    PluginCommandActionContext,
    PluginCommandActionRegistry,
    PluginRuntimeFacade,
)
from fast_agent.command_actions.accessors import (
    plugin_command_base_path_for_provider,
    plugin_commands_for_agent,
    plugin_commands_for_provider,
)
from fast_agent.commands.command_catalog import command_action_names
from fast_agent.commands.context import CommandContext, StaticAgentProvider
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.protocols import ACPCommandAllowlistProvider
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.renderers.history_markdown import render_history_overview_markdown
from fast_agent.commands.results import CommandOutcome
from fast_agent.config import get_settings
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.interfaces import ACPAwareProtocol, AgentProtocol, FastAgentLLMProtocol
from fast_agent.session.context import SessionContextCapable
from fast_agent.session.identity import SessionStoreScope, normalize_session_store_scope
from fast_agent.utils.slash_commands import parse_slash_command_line
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.command_actions.models import PluginCommandAgentProtocol
    from fast_agent.command_actions.runtime import AttachMcpServerCallback, DetachMcpServerCallback
    from fast_agent.commands.context import AgentProvider
    from fast_agent.config import MCPServerSettings
    from fast_agent.core.agent_app import AgentCardLoadResult
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions


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
    ) -> AgentCardLoadResult:
        if not self._handler._card_loader:
            raise RuntimeError("AgentCard loading is not available.")
        instance, loaded_card = await self._handler._card_loader(source, parent_agent)
        self._handler.instance = instance
        return loaded_card

    async def dump_agent_card(self, agent_name: str) -> str:
        if not self._handler._dump_agent_callback:
            raise RuntimeError("AgentCard dumping is not available.")
        return await self._handler._dump_agent_callback(agent_name)

    async def attach_agent_tools(self, parent_agent: str, child_agents: Sequence[str]) -> list[str]:
        if not self._handler._attach_agent_callback:
            raise RuntimeError("Agent tool attachment is not available.")
        instance, attached_names = await self._handler._attach_agent_callback(
            parent_agent, list(child_agents)
        )
        self._handler.instance = instance
        return attached_names

    async def detach_agent_tools(self, parent_agent: str, child_agents: Sequence[str]) -> list[str]:
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

    def registered_agent_names(self) -> Iterable[str]:
        return list(self._handler.instance.agents.keys())


def _command_input(input_hint: str | None) -> AvailableCommandInput | None:
    if not input_hint:
        return None
    return AvailableCommandInput(root=UnstructuredCommandInput(hint=input_hint))


def _append_unshadowed_commands(
    commands: list[AvailableCommand],
    command_specs: Iterable[tuple[str, str, str | None]],
) -> None:
    existing_names = {strip_casefold(command.name) for command in commands}
    for name, description, input_hint in command_specs:
        normalized_name = strip_casefold(name)
        if normalized_name in existing_names:
            continue
        commands.append(
            AvailableCommand(
                name=name,
                description=description,
                input=_command_input(input_hint),
            )
        )
        existing_names.add(normalized_name)


type _BuiltinSlashHandler = Callable[[str], Awaitable[str]]
type _BuiltinSlashInputHint = str | Callable[[], str | None] | None


@dataclass(frozen=True, slots=True)
class _BuiltinSlashCommandSpec:
    name: str
    description: str
    handler: _BuiltinSlashHandler
    input_hint: _BuiltinSlashInputHint = None
    available: Callable[[], bool] | None = None

    def is_available(self) -> bool:
        return True if self.available is None else self.available()

    def available_command(self) -> AvailableCommand:
        if self.input_hint is None or isinstance(self.input_hint, str):
            input_hint = self.input_hint
        else:
            input_hint = self.input_hint()
        return AvailableCommand(
            name=self.name,
            description=self.description,
            input=_command_input(input_hint),
        )


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
            [str, str | None], Awaitable[tuple["AgentInstance", AgentCardLoadResult]]
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
        attach_mcp_server_callback: Callable[
            [str, str, MCPServerSettings | None, MCPAttachOptions | None],
            Awaitable[object],
        ]
        | None = None,
        detach_mcp_server_callback: Callable[[str, str], Awaitable[object]] | None = None,
        list_attached_mcp_servers_callback: Callable[[str], Awaitable[list[str]]] | None = None,
        list_configured_detached_mcp_servers_callback: Callable[[str], Awaitable[list[str]]]
        | None = None,
        dump_agent_callback: Callable[[str], Awaitable[str]] | None = None,
        reload_callback: Callable[[], Awaitable[bool]] | None = None,
        set_current_mode_callback: Callable[[str], Awaitable[None] | None] | None = None,
        instruction_resolver: Callable[[str], Awaitable[str | None]] | None = None,
        no_home: bool = False,
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
        self._attach_mcp_server_callback = attach_mcp_server_callback
        self._detach_mcp_server_callback = detach_mcp_server_callback
        self._list_attached_mcp_servers_callback = list_attached_mcp_servers_callback
        self._list_configured_detached_mcp_servers_callback = (
            list_configured_detached_mcp_servers_callback
        )
        self._dump_agent_callback = dump_agent_callback
        self._reload_callback = reload_callback
        self._set_current_mode_callback = set_current_mode_callback
        self._instruction_resolver = instruction_resolver
        self._no_home = no_home
        self._acp_context: ACPContext | None = None

        cards_action_hint = (
            "|".join(
                action
                for action in command_action_names("cards")
                if action not in {"list", "readme", "help"}
            )
            or "add|remove|update|publish|registry"
        )

        # Session-level commands operate on the current agent.
        self._session_commands = self._build_builtin_session_commands(cards_action_hint)

    def _build_builtin_session_commands(
        self,
        cards_action_hint: str,
    ) -> dict[str, _BuiltinSlashCommandSpec]:
        specs = (
            _BuiltinSlashCommandSpec(
                name="status",
                description="Show fast-agent diagnostics",
                handler=self._handle_status,
                input_hint="[system|auth|authreset]",
            ),
            _BuiltinSlashCommandSpec(
                name="tools",
                description="List available tools",
                handler=self._handle_tools,
                input_hint="[summary|<tool-name>]",
            ),
            _BuiltinSlashCommandSpec(
                name="environment",
                description="List configured execution environments",
                handler=self._handle_environment,
            ),
            _BuiltinSlashCommandSpec(
                name="commands",
                description="Discover slash commands and usage",
                handler=self._handle_commands,
                input_hint="[<command>] [--json]",
            ),
            _BuiltinSlashCommandSpec(
                name="skills",
                description="List, browse, search, or manage local skills",
                handler=self._handle_skills,
                input_hint=(
                    "[list|available|search <query>|add <name|number>|"
                    "remove <name|number>|update <name|number|all> [--force] [--yes]|"
                    "registry [number|url|path]|help]"
                ),
            ),
            _BuiltinSlashCommandSpec(
                name="cards",
                description="List or manage card packs (add/remove/update/publish/registry)",
                handler=self._handle_cards,
                input_hint=(
                    f"[{cards_action_hint}] "
                    "[name|number|all|url] "
                    "[--force|--yes|--no-push|--message|--temp-dir|--keep-temp]"
                ),
            ),
            _BuiltinSlashCommandSpec(
                name="plugins",
                description="List or manage command plugins",
                handler=self._handle_plugins,
                input_hint=(
                    "[list|available|add <name|number>|remove <name|number>|"
                    "update <name|number|all> [--force] [--yes]|"
                    "registry [number|url|path]|help]"
                ),
            ),
            _BuiltinSlashCommandSpec(
                name="model",
                description="Inspect, switch, or update model settings",
                handler=self._handle_model,
                input_hint=self._model_command_hint,
            ),
            _BuiltinSlashCommandSpec(
                name="history",
                description="Show or manage conversation history",
                handler=self._handle_history,
                input_hint="[show|detail <turn>|save|load] [args]",
            ),
            _BuiltinSlashCommandSpec(
                name="clear",
                description="Clear history (`last` for prev. turn)",
                handler=self._handle_clear,
                input_hint="[last]",
            ),
            _BuiltinSlashCommandSpec(
                name="session",
                description="List or manage sessions",
                handler=self._handle_session,
                input_hint="[list|new|resume|title|fork|delete|pin|unpin|export] [args]",
            ),
            _BuiltinSlashCommandSpec(
                name="card",
                description="Load an AgentCard from file or URL",
                handler=self._handle_card,
                input_hint="<filename|url> [--tool [remove]]",
            ),
            _BuiltinSlashCommandSpec(
                name="agent",
                description="Attach an agent as a tool or dump its AgentCard",
                handler=self._handle_agent,
                input_hint="<@name> [--tool [remove]|--dump]",
            ),
            _BuiltinSlashCommandSpec(
                name="mcp",
                description="Manage runtime MCP servers and MCP data-layer sessions",
                handler=self._handle_mcp,
                input_hint=(
                    "list | connect <target> [--name <server>] [--auth <token>] "
                    "[--timeout <seconds>] [--oauth|--no-oauth] "
                    "[--reconnect|--no-reconnect] | session [list|jar|new|use|clear] | "
                    "disconnect <server>"
                ),
            ),
            _BuiltinSlashCommandSpec(
                name="reload",
                description="Reload AgentCards",
                handler=self._handle_reload,
                available=lambda: self._reload_callback is not None,
            ),
        )
        return {spec.name: spec for spec in specs}

    def get_available_commands(self) -> list[AvailableCommand]:
        """Get combined session commands and current agent's commands."""
        commands = [
            command.available_command() for command in self._get_allowed_session_commands().values()
        ]

        # Add agent-specific commands if current agent is ACP-aware
        agent = self._get_current_agent()
        if isinstance(agent, ACPAwareProtocol):
            _append_unshadowed_commands(
                commands,
                (
                    (name, command.description, command.input_hint)
                    for name, command in agent.acp_commands.items()
                ),
            )

        agent_commands = plugin_commands_for_agent(agent)
        if agent_commands:
            _append_unshadowed_commands(
                commands,
                (
                    (name, command.description, command.input_hint)
                    for name, command in agent_commands.items()
                ),
            )

        global_commands = plugin_commands_for_provider(self.instance.app)
        if global_commands:
            _append_unshadowed_commands(
                commands,
                (
                    (name, command.description, command.input_hint)
                    for name, command in global_commands.items()
                ),
            )

        return commands

    def _get_current_llm(self) -> FastAgentLLMProtocol | None:
        agent = self._get_current_agent()
        if agent is None:
            return None
        try:
            return agent.llm
        except Exception:
            return None

    def _model_command_hint(self) -> str:
        llm = self._get_current_llm()
        if llm is None:
            return (
                "reasoning <value> | task_budget <off|20k+ when supported> | "
                "verbosity <value> | fast <on|off|status|flex when supported> | "
                "web_search <on|off|default> | x_search <on|off|default> | "
                "web_fetch <on|off|default>"
            )

        options = ["reasoning <value>"]
        if model_handlers.model_supports_task_budget(llm):
            options.append("task_budget <off|20k+>")
        if model_handlers.model_supports_text_verbosity(llm):
            options.append("verbosity <value>")
        if model_handlers.model_supports_service_tier(llm):
            service_tier_values = "|".join(model_handlers.service_tier_command_values(llm))
            options.append(f"fast <{service_tier_values}>")
        if model_handlers.model_supports_web_search(llm):
            options.append("web_search <on|off|default>")
        if model_handlers.model_supports_x_search(llm):
            options.append("x_search <on|off|default>")
        if model_handlers.model_supports_web_fetch(llm):
            options.append("web_fetch <on|off|default>")
        options.extend(
            [
                "switch [<model>]",
                "doctor",
                "references [list|set|unset]",
                "catalog <provider> [--all]",
            ]
        )
        return " | ".join(options)

    def _model_usage_text(self) -> str:
        return f"Usage: /model {self._model_command_hint()}"

    def set_acp_context(self, acp_context: ACPContext | None) -> None:
        """Set the ACP context for this handler."""
        self._acp_context = acp_context

    def _get_allowed_session_commands(self) -> dict[str, _BuiltinSlashCommandSpec]:
        """
        Return session-level commands filtered by the current agent's policy.

        By default, all session commands are available. ACP-aware agents can restrict
        session commands (e.g. Setup/wizard flows) by defining a
        `acp_session_commands_allowlist: set[str] | None` attribute.
        """
        agent = self._get_current_agent()
        session_commands = {
            name: command
            for name, command in self._session_commands.items()
            if command.is_available()
        }
        if not isinstance(agent, ACPAwareProtocol):
            return session_commands

        allowlist = None
        if isinstance(agent, ACPCommandAllowlistProvider):
            allowlist = agent.acp_session_commands_allowlist

        if allowlist is None:
            return session_commands

        try:
            allowset = {str(name) for name in allowlist}
        except Exception:
            return session_commands

        return {name: cmd for name, cmd in session_commands.items() if name in allowset}

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

    def _build_card_manager(self) -> _ACPAgentCardManager:
        return _ACPAgentCardManager(self)

    def _agent_provider(self) -> "AgentProvider":
        return cast("AgentProvider", self.instance.app)

    def _resolve_acp_session_metadata(
        self,
    ) -> tuple[object | None, SessionStoreScope, object | None]:
        if self._acp_context is None:
            return None, "workspace", None

        session_cwd = self._acp_context.session_cwd
        return (
            session_cwd,
            normalize_session_store_scope(self._acp_context.session_store_scope),
            self._acp_context.session_store_cwd,
        )

    def _build_command_context(self) -> CommandContext:
        settings = get_settings()
        raw_session_cwd, session_store_scope, raw_session_store_cwd = (
            self._resolve_acp_session_metadata()
        )
        current_agent = self._get_current_agent()
        agent_context = (
            current_agent.context
            if current_agent and isinstance(current_agent, SessionContextCapable)
            else None
        )
        session_manager = agent_context.session_manager if agent_context else None
        session_runtime = None
        if not self._no_home and session_manager is None:
            from fast_agent.commands.session_runtime import SessionManagerCommandRuntime

            session_runtime = SessionManagerCommandRuntime(
                session_cwd=(
                    Path(str(raw_session_cwd)).expanduser().resolve() if raw_session_cwd else None
                ),
                session_store_scope=session_store_scope,
                session_store_cwd=(
                    Path(str(raw_session_store_cwd)).expanduser().resolve()
                    if raw_session_store_cwd
                    else None
                ),
                settings=settings,
            )
        return CommandContext(
            agent_provider=StaticAgentProvider(
                cast("dict[str, object]", dict(self.instance.agents))
            ),
            current_agent_name=self.current_agent_name,
            io=ACPCommandIO(),
            settings=settings,
            no_home=self._no_home,
            acp_session_id=self.session_id,
            session_cwd=(
                Path(str(raw_session_cwd)).expanduser().resolve() if raw_session_cwd else None
            ),
            session_store_scope=session_store_scope,
            session_store_cwd=(
                Path(str(raw_session_store_cwd)).expanduser().resolve()
                if raw_session_store_cwd
                else None
            ),
            session_manager=session_manager,
            session_runtime=session_runtime,
        )

    def _format_outcome_as_markdown(
        self,
        outcome: CommandOutcome,
        heading: str,
        *,
        io: ACPCommandIO | None = None,
    ) -> str:
        extra_messages = io.messages if io else None
        markdown = render_command_outcome_markdown(
            outcome,
            heading=heading,
            extra_messages=extra_messages,
        )
        if io and io.history_overview:
            history_markdown = render_history_overview_markdown(
                io.history_overview,
                heading="conversation history",
            )
            return "\n\n".join(part for part in (markdown, history_markdown) if part)
        return markdown

    async def _send_session_info_update(self) -> None:
        if self._acp_context is None:
            return
        if self._no_home:
            return
        from fast_agent.session import extract_session_title

        current_agent = self._get_current_agent()
        agent_context = current_agent.context if current_agent else None
        manager = agent_context.session_manager if agent_context else None
        if manager is None:
            raise RuntimeError("ACP slash command session update has no active session manager.")
        session = manager.current_session
        if session is None or session.info.name != self.session_id:
            session = manager.get_session(self.session_id)
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

    async def _send_progress_update(self, message: str) -> None:
        if self._acp_context is None:
            return
        try:
            await self._acp_context.send_session_update(update_agent_message_text(message))
        except Exception as exc:
            self._logger.debug(
                "Failed to send ACP progress update",
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
        return parse_slash_command_line(prompt_text) or ("", prompt_text.strip())

    async def execute_command(self, command_name: str, arguments: str) -> str:
        """
        Execute a slash command and return the response.

        Args:
            command_name: Name of the command to execute
            arguments: Arguments passed to the command

        Returns:
            The command response as a string
        """
        original_command_name = command_name
        normalized_command_name = strip_casefold(command_name)

        # Exact-case agent/plugin commands win before case-folded built-ins.
        agent = self._get_current_agent()
        if isinstance(agent, ACPAwareProtocol):
            agent_commands = agent.acp_commands
            if original_command_name in agent_commands:
                return await agent_commands[original_command_name].handler(arguments)

        agent_commands = plugin_commands_for_agent(agent)
        if agent is not None and agent_commands and original_command_name in agent_commands:
            spec = agent_commands[original_command_name]
            base_path = None
            if isinstance(agent, AgentProtocol) and agent.config.source_path:
                base_path = agent.config.source_path.parent
            return await self._execute_plugin_command_action(
                agent,
                original_command_name,
                arguments,
                spec=spec,
                base_path=base_path,
            )

        global_commands = plugin_commands_for_provider(self.instance.app)
        if agent is not None and global_commands and original_command_name in global_commands:
            return await self._execute_plugin_command_action(
                agent,
                original_command_name,
                arguments,
                spec=global_commands[original_command_name],
                base_path=plugin_command_base_path_for_provider(self.instance.app),
            )

        # Check session-level commands (filtered by agent policy).
        allowed_session_commands = self._get_allowed_session_commands()
        builtin_command = allowed_session_commands.get(normalized_command_name)
        if builtin_command is not None:
            return await builtin_command.handler(arguments)

        # Unknown command
        available = self.get_available_commands()
        return f"Unknown command: /{command_name}\n\nAvailable commands:\n" + "\n".join(
            f"  /{cmd.name} - {cmd.description}" for cmd in available
        )

    async def _execute_plugin_command_action(
        self,
        agent: AgentProtocol,
        command_name: str,
        arguments: str,
        spec,
        base_path: Path | None,
    ) -> str:
        command_context = self._build_command_context()
        try:
            registry = PluginCommandActionRegistry.from_specs(
                {command_name: spec},
                base_path=base_path,
            )
            result = await registry.execute(
                command_name,
                PluginCommandActionContext(
                    command_name=command_name,
                    arguments=arguments,
                    agent=cast("PluginCommandAgentProtocol", agent),
                    settings=command_context.settings,
                    session_cwd=command_context.session_cwd,
                    runtime=PluginRuntimeFacade(
                        current_agent_name=agent.name,
                        attach_mcp_server_callback=cast(
                            "AttachMcpServerCallback | None",
                            self._attach_mcp_server_callback,
                        ),
                        detach_mcp_server_callback=cast(
                            "DetachMcpServerCallback | None",
                            self._detach_mcp_server_callback,
                        ),
                        list_attached_mcp_servers_callback=(
                            self._list_attached_mcp_servers_callback
                        ),
                        list_configured_detached_mcp_servers_callback=(
                            self._list_configured_detached_mcp_servers_callback
                        ),
                    ),
                    is_acp=True,
                ),
            )
        except AgentConfigError as exc:
            return f"Command /{command_name} failed to load: {exc}"
        except Exception as exc:
            self._logger.exception("Plugin command action failed", command=command_name)
            return f"Command /{command_name} failed: {exc}"

        if result is None:
            return ""

        outcome = CommandOutcome(
            buffer_prefill=result.buffer_prefill,
            switch_agent=result.switch_agent,
            requires_refresh=result.refresh_agents,
        )
        if result.markdown:
            outcome.add_message(result.markdown, render_markdown=True)
        elif result.message:
            outcome.add_message(result.message)
        if result.images:
            lines = ["Images:"]
            for image in result.images:
                label = image.label or image.source
                lines.append(f"- [{label}]({image.source})")
            outcome.add_message("\n".join(lines), render_markdown=True)
        if result.buffer_prefill:
            outcome.add_message(
                f"Command produced draft text:\n\n```text\n{result.buffer_prefill}\n```",
                render_markdown=True,
            )
        if outcome.switch_agent is not None:
            switched = await self._switch_current_mode(outcome.switch_agent)
            if not switched:
                outcome.add_message(
                    f"Unknown agent: {outcome.switch_agent}",
                    channel="error",
                )
                outcome.switch_agent = None
        if outcome.requires_refresh and self._acp_context is not None:
            await self._acp_context.send_available_commands_update()
        return self._format_outcome_as_markdown(outcome, f"/{command_name}")

    async def _handle_history(self, arguments: str | None = None) -> str:
        return await history_slash_handlers.handle_history(self, arguments)

    async def _handle_model(self, arguments: str | None = None) -> str:
        return await model_slash_handlers.handle_model(self, arguments)

    async def _handle_session(self, arguments: str | None = None) -> str:
        return await session_slash_handlers.handle_session(self, arguments)

    async def _handle_status(self, arguments: str | None = None) -> str:
        return await status_slash_handlers.handle_status(self, arguments)

    async def _handle_tools(self, arguments: str | None = None) -> str:
        return await tools_slash_handlers.handle_tools(self, arguments)

    async def _handle_environment(self, arguments: str | None = None) -> str:
        del arguments
        outcome = await display_handlers.handle_environment(
            self._build_command_context(),
            agent_name=self.current_agent_name,
        )
        return self._format_outcome_as_markdown(outcome, "/environment")

    async def _handle_commands(self, arguments: str | None = None) -> str:
        return await commands_slash_handlers.handle_commands(self, arguments)

    async def _handle_skills(self, arguments: str | None = None) -> str:
        return await skills_slash_handlers.handle_skills(self, arguments)

    async def _handle_cards(self, arguments: str | None = None) -> str:
        return await cards_manager_slash_handlers.handle_cards(self, arguments)

    async def _handle_plugins(self, arguments: str | None = None) -> str:
        return await plugins_slash_handlers.handle_plugins(self, arguments)

    async def _refresh_agent_skills(self, agent: AgentProtocol) -> None:
        await skills_slash_handlers.refresh_agent_skills(agent)

    def _build_tool_call_id(self) -> str:
        return skills_slash_handlers.build_tool_call_id()

    async def _handle_card(self, arguments: str | None = None) -> str:
        return await cards_slash_handlers.handle_card(self, arguments)

    async def _handle_agent(self, arguments: str | None = None) -> str:
        return await cards_slash_handlers.handle_agent(self, arguments)

    async def _handle_mcp(self, arguments: str | None = None) -> str:
        return await mcp_slash_handlers.handle_mcp(self, arguments)

    async def _handle_reload(self, arguments: str | None = None) -> str:
        del arguments
        return await cards_slash_handlers.handle_reload(self)

    async def _handle_clear(self, arguments: str | None = None) -> str:
        return await clear_slash_handlers.handle_clear(self, arguments)
