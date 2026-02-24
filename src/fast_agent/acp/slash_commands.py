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
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
    cast,
)

from acp.helpers import update_agent_message_text
from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    UnstructuredCommandInput,
)

from fast_agent.acp.command_io import ACPCommandIO
from fast_agent.acp.slash import dispatch as slash_dispatch
from fast_agent.acp.slash.command_catalog import apply_dynamic_session_hints
from fast_agent.acp.slash.handlers import cards as cards_slash_handlers
from fast_agent.acp.slash.handlers import cards_manager as cards_manager_slash_handlers
from fast_agent.acp.slash.handlers import clear as clear_slash_handlers
from fast_agent.acp.slash.handlers import history as history_slash_handlers
from fast_agent.acp.slash.handlers import mcp as mcp_slash_handlers
from fast_agent.acp.slash.handlers import model as model_slash_handlers
from fast_agent.acp.slash.handlers import models_manager as models_manager_slash_handlers
from fast_agent.acp.slash.handlers import session as session_slash_handlers
from fast_agent.acp.slash.handlers import skills as skills_slash_handlers
from fast_agent.acp.slash.handlers import status as status_slash_handlers
from fast_agent.acp.slash.handlers import tools as tools_slash_handlers
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.protocols import ACPCommandAllowlistProvider
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.config import get_settings
from fast_agent.core.logging.logger import get_logger
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.interfaces import ACPAwareProtocol, AgentProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.commands.context import AgentProvider
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.config import MCPServerSettings
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions


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
        noenv: bool = False,
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
        self._noenv = noenv
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
                description="List or manage local skills (add/remove/update/registry)",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(
                        hint="[add|remove|update|registry] [name|number|all|url] [--force] [--yes]"
                    )
                ),
            ),
            "cards": AvailableCommand(
                name="cards",
                description="List or manage card packs (add/remove/update/publish/registry)",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(
                        hint=(
                            "[add|remove|update|publish|registry] "
                            "[name|number|all|url] "
                            "[--force|--yes|--no-push|--message|--temp-dir|--keep-temp]"
                        )
                    )
                ),
            ),
            "model": AvailableCommand(
                name="model",
                description="Update model settings",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(
                        hint=(
                            "reasoning <value> | verbosity <value> | "
                            "web_search <on|off|default> | web_fetch <on|off|default>"
                        )
                    )
                ),
            ),
            "models": AvailableCommand(
                name="models",
                description="Inspect model onboarding state (doctor/aliases/catalog)",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(
                        hint="[doctor|aliases|catalog <provider> [--all]]"
                    )
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
            "mcp": AvailableCommand(
                name="mcp",
                description="Manage runtime MCP servers and experimental sessions",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(
                        hint=(
                            "list | connect <target> [--name <server>] [--auth <token>] "
                            "[--timeout <seconds>] [--oauth|--no-oauth] "
                            "[--reconnect|--no-reconnect] | session [list|jar|new|use|clear] | "
                            "disconnect <server>"
                        )
                    )
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
        commands_by_name = dict(self._get_allowed_session_commands())
        commands = apply_dynamic_session_hints(commands_by_name, self._model_command_hint())

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

    def _apply_dynamic_session_command_hints(
        self, commands_by_name: dict[str, AvailableCommand]
    ) -> None:
        model_command = commands_by_name.get("model")
        if model_command is None:
            return

        commands_by_name["model"] = AvailableCommand(
            name=model_command.name,
            description=model_command.description,
            input=AvailableCommandInput(
                root=UnstructuredCommandInput(hint=self._model_command_hint())
            ),
        )

    def _get_current_llm(self) -> object | None:
        agent = self._get_current_agent()
        if agent is None:
            return None
        try:
            return getattr(agent, "llm", None) or getattr(agent, "_llm", None)
        except Exception:
            return None

    def _model_command_hint(self) -> str:
        llm = self._get_current_llm()
        if llm is None:
            return (
                "reasoning <value> | verbosity <value> | web_search <on|off|default> "
                "| web_fetch <on|off|default>"
            )

        options = ["reasoning <value>"]
        if model_handlers.model_supports_text_verbosity(llm):
            options.append("verbosity <value>")
        if model_handlers.model_supports_web_search(llm):
            options.append("web_search <on|off|default>")
        if model_handlers.model_supports_web_fetch(llm):
            options.append("web_fetch <on|off|default>")
        return " | ".join(options)

    def _model_usage_text(self) -> str:
        return f"Usage: /model {self._model_command_hint()}"

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

    def _build_card_manager(self) -> _ACPAgentCardManager:
        return _ACPAgentCardManager(self)

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
            noenv=self._noenv,
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
        if self._noenv:
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
            try:
                return await slash_dispatch.execute(self, command_name, arguments)
            except slash_dispatch.UnknownSlashCommandError:
                pass

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
        return await history_slash_handlers.handle_history(self, arguments)

    async def _handle_model(self, arguments: str | None = None) -> str:
        return await model_slash_handlers.handle_model(self, arguments)

    async def _handle_models(self, arguments: str | None = None) -> str:
        return await models_manager_slash_handlers.handle_models(self, arguments)

    async def _handle_session(self, arguments: str | None = None) -> str:
        return await session_slash_handlers.handle_session(self, arguments)

    async def _handle_status(self, arguments: str | None = None) -> str:
        return await status_slash_handlers.handle_status(self, arguments)

    async def _handle_status_system(self) -> str:
        return await status_slash_handlers.handle_status_system(self)

    async def _render_history_overview(self) -> str:
        return await history_slash_handlers.render_history_overview(self)

    def _render_session_list(self) -> str:
        return session_slash_handlers.render_session_list(self)

    async def _handle_session_resume(self, argument: str) -> str:
        return await session_slash_handlers.handle_session_resume(self, argument)

    async def _handle_session_title(self, argument: str) -> str:
        return await session_slash_handlers.handle_session_title(self, argument)

    async def _handle_session_fork(self, argument: str) -> str:
        return await session_slash_handlers.handle_session_fork(self, argument)

    async def _handle_session_new(self, argument: str) -> str:
        return await session_slash_handlers.handle_session_new(self, argument)

    async def _handle_session_delete(self, argument: str) -> str:
        return await session_slash_handlers.handle_session_delete(self, argument)

    async def _handle_session_pin(self, argument: str) -> str:
        return await session_slash_handlers.handle_session_pin(self, argument)

    def _handle_status_auth(self) -> str:
        return status_slash_handlers.handle_status_auth(self)

    def _handle_status_authreset(self) -> str:
        return status_slash_handlers.handle_status_authreset(self)

    async def _handle_tools(self) -> str:
        return await tools_slash_handlers.handle_tools(self)

    async def _handle_skills(self, arguments: str | None = None) -> str:
        return await skills_slash_handlers.handle_skills(self, arguments)

    async def _handle_cards(self, arguments: str | None = None) -> str:
        return await cards_manager_slash_handlers.handle_cards(self, arguments)

    async def _handle_skills_registry(self, argument: str) -> str:
        return await skills_slash_handlers.handle_skills_registry(self, argument)

    def _handle_skills_list(self) -> str:
        return skills_slash_handlers.handle_skills_list(self)

    def _skills_override_section(self) -> str | None:
        return skills_slash_handlers.skills_override_section(self)

    async def _handle_skills_add(self, argument: str) -> str:
        return await skills_slash_handlers.handle_skills_add(self, argument)

    async def _handle_skills_remove(self, argument: str) -> str:
        return await skills_slash_handlers.handle_skills_remove(self, argument)

    async def _handle_skills_update(self, argument: str) -> str:
        return await skills_slash_handlers.handle_skills_update(self, argument)

    async def _refresh_agent_skills(self, agent: AgentProtocol) -> None:
        await skills_slash_handlers.refresh_agent_skills(agent)

    def _build_tool_call_id(self) -> str:
        return skills_slash_handlers.build_tool_call_id()

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
        await skills_slash_handlers.send_skills_update(self, agent, tool_call_id, title=title, status=status, message=message, start=start)

    async def _handle_save(self, arguments: str | None = None) -> str:
        return await history_slash_handlers.handle_save(self, arguments)

    async def _handle_load(self, arguments: str | None = None) -> str:
        return await history_slash_handlers.handle_load(self, arguments)

    async def _handle_history_webclear(self) -> str:
        return await history_slash_handlers.handle_history_webclear(self)

    async def _handle_card(self, arguments: str | None = None) -> str:
        return await cards_slash_handlers.handle_card(self, arguments)

    async def _handle_agent(self, arguments: str | None = None) -> str:
        return await cards_slash_handlers.handle_agent(self, arguments)

    async def _handle_mcp(self, arguments: str | None = None) -> str:
        return await mcp_slash_handlers.handle_mcp(self, arguments)

    async def _handle_reload(self) -> str:
        return await cards_slash_handlers.handle_reload(self)

    async def _handle_clear(self, arguments: str | None = None) -> str:
        return await clear_slash_handlers.handle_clear(self, arguments)

    async def _handle_clear_all(self) -> str:
        return await clear_slash_handlers.handle_clear_all(self)

    async def _handle_clear_last(self) -> str:
        return await clear_slash_handlers.handle_clear_last(self)
