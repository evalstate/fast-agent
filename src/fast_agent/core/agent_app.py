"""
Direct AgentApp implementation for interacting with agents without proxies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from deprecated import deprecated
from rich import print as rich_print
from rich.markup import escape

from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.core.default_agent import agent_is_default, resolve_default_agent_name
from fast_agent.core.exceptions import AgentConfigError, ServerConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.usage_tracking import last_turn_usage
from fast_agent.ui.display_suppression import display_usage_enabled
from fast_agent.ui.progress_display import progress_display
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from mcp.types import GetPromptResult, PromptMessage

    from fast_agent.agents.agent_types import AgentType
    from fast_agent.cli.runtime.shell_cwd_policy import MissingShellCwdPolicy
    from fast_agent.command_actions import PluginCommandActionSpec
    from fast_agent.config import MCPServerSettings
    from fast_agent.core.harness import HarnessSession
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
    from fast_agent.session.session_manager import ResumeSessionAgentsResult, SessionManager
    from fast_agent.types import PromptMessageExtended, RequestParams
    from fast_agent.ui.interactive_prompt import PromptLoopResult

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class AgentRefreshResult:
    changed: bool
    active_agent: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class AgentCardLoadResult:
    loaded_names: list[str]
    attached_names: list[str] = field(default_factory=list)


def _format_interactive_final_error(error: Exception) -> str:
    detail_candidates = [str(arg) for arg in error.args]
    detail_candidates.extend([str(error), repr(error), type(error).__name__])

    detail = next(
        (
            normalized
            for candidate in detail_candidates
            if (normalized := strip_to_none(candidate)) is not None
        ),
        "",
    )
    error_type = type(error).__name__
    if detail and not detail.startswith(error_type):
        detail = f"{error_type}: {detail}"

    clean_detail = detail.replace("\n", " ")
    if len(clean_detail) > 300:
        clean_detail = clean_detail[:297] + "..."
    clean_detail = escape(clean_detail)
    return (
        f"▲ **System Error:** The agent failed after repeated attempts.\n"
        f"Error details: {clean_detail}\n"
        f"\n*Your context is preserved. You can try sending the message again.*"
    )


class AgentApp:
    """
    Container for active agents that provides a simple API for interacting with them.
    This implementation works directly with Agent instances without proxies.

    The DirectAgentApp provides both attribute-style access (app.agent_name)
    and dictionary-style access (app["agent_name"]) to agents.

    It also implements the AgentProtocol interface, automatically forwarding
    calls to the default agent (the first agent in the container).
    """

    def __init__(
        self,
        agents: dict[str, AgentProtocol],
        *,
        reload_callback: Callable[[], Awaitable[AgentRefreshResult]] | None = None,
        refresh_callback: Callable[[], Awaitable[AgentRefreshResult]] | None = None,
        load_card_callback: Callable[[str, str | None], Awaitable[AgentCardLoadResult]]
        | None = None,
        attach_agent_tools_callback: Callable[[str, Sequence[str]], Awaitable[list[str]]]
        | None = None,
        detach_agent_tools_callback: Callable[[str, Sequence[str]], Awaitable[list[str]]]
        | None = None,
        dump_agent_callback: Callable[[str], Awaitable[str]] | None = None,
        attach_mcp_server_callback: Callable[
            [str, str, "MCPServerSettings | None", "MCPAttachOptions | None"],
            Awaitable["MCPAttachResult"],
        ]
        | None = None,
        detach_mcp_server_callback: Callable[[str, str], Awaitable["MCPDetachResult"]]
        | None = None,
        list_attached_mcp_servers_callback: Callable[[str], Awaitable[list[str]]] | None = None,
        list_configured_detached_mcp_servers_callback: Callable[[str], Awaitable[list[str]]]
        | None = None,
        tool_only_agents: set[str] | None = None,
        card_collision_warnings: list[str] | None = None,
        noenv_mode: bool = False,
        plugin_commands: dict[str, PluginCommandActionSpec] | None = None,
        plugin_command_base_path: Path | None = None,
    ) -> None:
        """
        Initialize the DirectAgentApp.

        Args:
            agents: Dictionary of agent instances keyed by name
            reload_callback: Optional callback for manual AgentCard reloads
            refresh_callback: Optional callback for lazy instance refresh before requests
            load_card_callback: Optional callback for loading AgentCards at runtime
            attach_agent_tools_callback: Optional callback for attaching agent tools
            detach_agent_tools_callback: Optional callback for detaching agent tools
            dump_agent_callback: Optional callback for dumping AgentCards
            tool_only_agents: Optional set of agent names that are tool-only (hidden from listings)
            card_collision_warnings: Optional list of warnings from agent card name collisions
        """
        if not agents:
            raise ValueError("No agents provided!")
        self._agents = agents
        self._reload_callback = reload_callback
        self._refresh_callback = refresh_callback
        self._load_card_callback = load_card_callback
        self._attach_agent_tools_callback = attach_agent_tools_callback
        self._detach_agent_tools_callback = detach_agent_tools_callback
        self._dump_agent_callback = dump_agent_callback
        self._attach_mcp_server_callback = attach_mcp_server_callback
        self._detach_mcp_server_callback = detach_mcp_server_callback
        self._list_attached_mcp_servers_callback = list_attached_mcp_servers_callback
        self._list_configured_detached_mcp_servers_callback = (
            list_configured_detached_mcp_servers_callback
        )
        self._tool_only_agents: set[str] = tool_only_agents or set()
        self._card_collision_warnings: list[str] = card_collision_warnings or []
        self._plugin_commands = plugin_commands
        self._plugin_command_base_path = plugin_command_base_path
        self._last_refresh_result = AgentRefreshResult(changed=False)
        self._session_restore_result: ResumeSessionAgentsResult | None = None
        self._noenv_mode = noenv_mode
        self._missing_shell_cwd_policy_override: "MissingShellCwdPolicy | None" = None
        self._apply_agent_registry()

    def _apply_agent_registry(self) -> None:
        for agent in self._agents.values():
            registry_setter = getattr(agent, "set_agent_registry", None)
            if callable(registry_setter):
                registry_setter(self._agents)

    def __getitem__(self, key: str) -> AgentProtocol:
        """Allow access to agents using dictionary syntax."""
        if key not in self._agents:
            raise KeyError(f"Agent '{key}' not found")
        return self._agents[key]

    def get_agent(self, name: str) -> AgentProtocol | None:
        """Return the named agent if available, else None."""
        return self._agents.get(name)

    def resolve_agent(self, name: str | None = None) -> AgentProtocol:
        """Return the resolved target agent, raising when no target is available."""
        resolved_agent_name = self.resolve_target_agent_name(name)
        if resolved_agent_name is None:
            raise ValueError("No agents provided!")
        return self._agents[resolved_agent_name]

    @property
    def plugin_commands(self) -> dict[str, PluginCommandActionSpec] | None:
        return self._plugin_commands

    @property
    def plugin_command_base_path(self) -> Path | None:
        return self._plugin_command_base_path

    def set_plugin_commands(
        self,
        commands: dict[str, PluginCommandActionSpec] | None,
        *,
        base_path: Path | None,
    ) -> None:
        self._plugin_commands = commands
        self._plugin_command_base_path = base_path

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        if agent_name is None:
            return self.get_default_agent_name()
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        return agent_name

    def __getattr__(self, name: str) -> AgentProtocol:
        """Allow access to agents using attribute syntax."""
        if name in self._agents:
            return self._agents[name]
        raise AttributeError(f"Agent '{name}' not found")

    async def __call__(
        self,
        message: str | PromptMessage | PromptMessageExtended | None = None,
        agent_name: str | None = None,
        default_prompt: str = "",
        request_params: RequestParams | None = None,
    ) -> PromptLoopResult:
        """
        Make the object callable to send messages or start interactive prompt.
        This mirrors the FastAgent implementation that allowed agent("message").

        Args:
            message: Message content in various formats:
                - String: Converted to a user PromptMessageExtended
                - PromptMessage: Converted to PromptMessageExtended
                - PromptMessageExtended: Used directly
            agent_name: Optional name of the agent to send to (defaults to first agent)
            default_prompt: Default message to use in interactive prompt mode
            request_params: Optional request parameters including MCP metadata

        Returns:
            The agent's response as a string or the result of the interactive session
        """
        if message:
            await self._refresh_if_needed()
            return await self._agent(agent_name).send(message, request_params)

        return await self.interactive(
            agent_name=agent_name, default_prompt=default_prompt, request_params=request_params
        )

    async def send(
        self,
        message: str | PromptMessage | PromptMessageExtended,
        agent_name: str | None = None,
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Send a message to the specified agent (or to all agents).

        Args:
            message: Message content in various formats:
                - String: Converted to a user PromptMessageExtended
                - PromptMessage: Converted to PromptMessageExtended
                - PromptMessageExtended: Used directly
            agent_name: Optional name of the agent to send to
            request_params: Optional request parameters including MCP metadata

        Returns:
            The agent's response as a string
        """
        await self._refresh_if_needed()
        return await self._agent(agent_name).send(message, request_params)

    def get_default_agent_name(self) -> str | None:
        return resolve_default_agent_name(
            self._agents,
            is_default=lambda _name, agent: agent_is_default(agent),
            is_tool_only=lambda name, _agent: name in self._tool_only_agents,
        )

    def _agent(self, agent_name: str | None) -> AgentProtocol:
        return self.resolve_agent(agent_name)

    async def apply_prompt(
        self,
        prompt: str | GetPromptResult,
        arguments: dict[str, str] | None = None,
        agent_name: str | None = None,
        as_template: bool = False,
    ) -> str:
        """
        Apply a prompt template to an agent (default agent if not specified).

        Args:
            prompt: Name of the prompt template to apply OR a GetPromptResult object
            arguments: Optional arguments for the prompt template
            agent_name: Name of the agent to send to
            as_template: If True, store as persistent template (always included in context)

        Returns:
            The agent's response as a string
        """
        await self._refresh_if_needed()
        return await self._agent(agent_name).apply_prompt(
            prompt, arguments, as_template=as_template
        )

    async def list_prompts(self, namespace: str | None = None, agent_name: str | None = None):
        """
        List available prompts for an agent.

        Args:
            server_name: Optional name of the server to list prompts from
            agent_name: Name of the agent to list prompts for

        Returns:
            Dictionary mapping server names to lists of available prompts
        """
        await self._refresh_if_needed()
        if not agent_name:
            results = {}
            for agent in self._agents.values():
                curr_prompts = await agent.list_prompts(namespace=namespace)
                results.update(curr_prompts)
            return results
        return await self._agent(agent_name).list_prompts(namespace=namespace)

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
        server_name: str | None = None,
        agent_name: str | None = None,
    ):
        """
        Get a prompt from a server.

        Args:
            prompt_name: Name of the prompt, optionally namespaced
            arguments: Optional dictionary of arguments to pass to the prompt template
            server_name: Optional name of the server to get the prompt from
            agent_name: Name of the agent to use

        Returns:
            GetPromptResult containing the prompt information
        """
        await self._refresh_if_needed()
        return await self._agent(agent_name).get_prompt(
            prompt_name=prompt_name, arguments=arguments, namespace=server_name
        )

    async def with_resource(
        self,
        prompt_content: str | PromptMessage | PromptMessageExtended,
        resource_uri: str,
        server_name: str | None = None,
        agent_name: str | None = None,
    ) -> str:
        """
        Send a message with an attached MCP resource.

        Args:
            prompt_content: Content in various formats (String, PromptMessage, or PromptMessageExtended)
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from
            agent_name: Name of the agent to use

        Returns:
            The agent's response as a string
        """
        await self._refresh_if_needed()
        return await self._agent(agent_name).with_resource(
            prompt_content=prompt_content, resource_uri=resource_uri, namespace=server_name
        )

    async def list_resources(
        self,
        server_name: str | None = None,
        agent_name: str | None = None,
    ) -> Mapping[str, list[str]]:
        """
        List available resources from one or all servers.

        Args:
            server_name: Optional server name to list resources from
            agent_name: Name of the agent to use

        Returns:
            Dictionary mapping server names to lists of resource URIs
        """
        await self._refresh_if_needed()
        return await self._agent(agent_name).list_resources(namespace=server_name)

    async def get_resource(
        self,
        resource_uri: str,
        server_name: str | None = None,
        agent_name: str | None = None,
    ):
        """
        Get a resource from an MCP server.

        Args:
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from
            agent_name: Name of the agent to use

        Returns:
            ReadResourceResult object containing the resource content
        """
        await self._refresh_if_needed()
        return await self._agent(agent_name).get_resource(
            resource_uri=resource_uri, namespace=server_name
        )

    async def reload_agents(self) -> bool:
        """Reload AgentCards and refresh active instances when available."""
        if not self._reload_callback:
            self._last_refresh_result = AgentRefreshResult(changed=False)
            return False
        result = await self._reload_callback()
        self._last_refresh_result = result
        return result.changed

    def can_reload_agents(self) -> bool:
        """Return True if manual reload is available."""
        return self._reload_callback is not None

    def can_load_agent_cards(self) -> bool:
        """Return True if agent card loading is available."""
        return self._load_card_callback is not None

    def can_attach_agent_tools(self) -> bool:
        """Return True if agent tool attachment is available."""
        return self._attach_agent_tools_callback is not None

    def can_dump_agent_cards(self) -> bool:
        """Return True if agent card dumping is available."""
        return self._dump_agent_callback is not None

    async def load_agent_card(
        self, source: str, parent_agent: str | None = None
    ) -> AgentCardLoadResult:
        """Load an AgentCard source and refresh active instances when available."""
        if not self._load_card_callback:
            raise RuntimeError("Agent card loading is not available.")
        return await self._load_card_callback(source, parent_agent)

    async def attach_agent_tools(self, parent_agent: str, child_agents: Sequence[str]) -> list[str]:
        """Attach agents as tools to a parent agent."""
        if not self._attach_agent_tools_callback:
            raise RuntimeError("Agent tool attachment is not available.")
        return await self._attach_agent_tools_callback(parent_agent, child_agents)

    async def detach_agent_tools(self, parent_agent: str, child_agents: Sequence[str]) -> list[str]:
        """Detach agents-as-tools from a parent agent."""
        if not self._detach_agent_tools_callback:
            raise RuntimeError("Agent tool detachment is not available.")
        return await self._detach_agent_tools_callback(parent_agent, child_agents)

    async def dump_agent_card(self, agent_name: str) -> str:
        """Dump an AgentCard for the requested agent."""
        if not self._dump_agent_callback:
            raise RuntimeError("Agent card dumping is not available.")
        return await self._dump_agent_callback(agent_name)

    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: "MCPServerSettings | None" = None,
        options: "MCPAttachOptions | None" = None,
    ) -> "MCPAttachResult":
        """Attach an MCP server to a running MCP agent."""
        if not self._attach_mcp_server_callback:
            raise RuntimeError("Runtime MCP server attachment is not available.")
        return await self._attach_mcp_server_callback(
            agent_name, server_name, server_config, options
        )

    async def detach_mcp_server(self, agent_name: str, server_name: str) -> "MCPDetachResult":
        """Detach an MCP server from a running MCP agent."""
        if not self._detach_mcp_server_callback:
            raise RuntimeError("Runtime MCP server detachment is not available.")
        return await self._detach_mcp_server_callback(agent_name, server_name)

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]:
        """List MCP servers attached to a running MCP agent."""
        if not self._list_attached_mcp_servers_callback:
            raise RuntimeError("Runtime MCP server listing is not available.")
        return await self._list_attached_mcp_servers_callback(agent_name)

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]:
        """List configured MCP servers that are not currently attached."""
        if not self._list_configured_detached_mcp_servers_callback:
            raise RuntimeError("Configured MCP server listing is not available.")
        return await self._list_configured_detached_mcp_servers_callback(agent_name)

    def set_agents(
        self,
        agents: dict[str, AgentProtocol],
        tool_only_agents: set[str] | None = None,
        card_collision_warnings: list[str] | None = None,
    ) -> None:
        """Replace the active agent map (used after reload)."""
        if not agents:
            raise ValueError("No agents provided!")
        self._agents = agents
        self._apply_agent_registry()
        if tool_only_agents is not None:
            self._tool_only_agents = tool_only_agents
        if card_collision_warnings is not None:
            self._card_collision_warnings = card_collision_warnings

    @property
    def card_collision_warnings(self) -> list[str]:
        """Return warnings from agent card name collisions."""
        return list(self._card_collision_warnings)

    @property
    def noenv_mode(self) -> bool:
        return self._noenv_mode

    @noenv_mode.setter
    def noenv_mode(self, value: bool) -> None:
        self._noenv_mode = value

    @property
    def missing_shell_cwd_policy_override(self) -> "MissingShellCwdPolicy | None":
        return self._missing_shell_cwd_policy_override

    @missing_shell_cwd_policy_override.setter
    def missing_shell_cwd_policy_override(self, value: "MissingShellCwdPolicy | None") -> None:
        self._missing_shell_cwd_policy_override = value

    def latest_refresh_result(self) -> AgentRefreshResult:
        return self._last_refresh_result

    def set_refresh_result(self, result: AgentRefreshResult) -> None:
        self._last_refresh_result = result

    def latest_session_restore_result(self) -> "ResumeSessionAgentsResult | None":
        return self._session_restore_result

    def set_session_restore_result(self, result: "ResumeSessionAgentsResult | None") -> None:
        self._session_restore_result = result

    def set_reload_callback(
        self, callback: Callable[[], Awaitable[AgentRefreshResult]] | None
    ) -> None:
        """Update the reload callback for manual AgentCard refresh."""
        self._reload_callback = callback

    def set_refresh_callback(
        self, callback: Callable[[], Awaitable[AgentRefreshResult]] | None
    ) -> None:
        """Update the refresh callback for lazy instance swaps."""
        self._refresh_callback = callback

    def set_load_card_callback(
        self,
        callback: Callable[[str, str | None], Awaitable[AgentCardLoadResult]] | None,
    ) -> None:
        """Update the callback for loading agent cards at runtime."""
        self._load_card_callback = callback

    def set_attach_agent_tools_callback(
        self, callback: Callable[[str, Sequence[str]], Awaitable[list[str]]] | None
    ) -> None:
        """Update the callback for attaching agent tools."""
        self._attach_agent_tools_callback = callback

    def set_detach_agent_tools_callback(
        self, callback: Callable[[str, Sequence[str]], Awaitable[list[str]]] | None
    ) -> None:
        """Update the callback for detaching agent tools."""
        self._detach_agent_tools_callback = callback

    def set_dump_agent_callback(self, callback: Callable[[str], Awaitable[str]] | None) -> None:
        """Update the callback for dumping agent cards."""
        self._dump_agent_callback = callback

    def set_attach_mcp_server_callback(
        self,
        callback: Callable[
            [str, str, "MCPServerSettings | None", "MCPAttachOptions | None"],
            Awaitable["MCPAttachResult"],
        ]
        | None,
    ) -> None:
        """Update callback for attaching MCP servers at runtime."""
        self._attach_mcp_server_callback = callback

    def set_detach_mcp_server_callback(
        self,
        callback: Callable[[str, str], Awaitable["MCPDetachResult"]] | None,
    ) -> None:
        """Update callback for detaching MCP servers at runtime."""
        self._detach_mcp_server_callback = callback

    def set_list_attached_mcp_servers_callback(
        self,
        callback: Callable[[str], Awaitable[list[str]]] | None,
    ) -> None:
        """Update callback for listing attached MCP servers."""
        self._list_attached_mcp_servers_callback = callback

    def set_list_configured_detached_mcp_servers_callback(
        self,
        callback: Callable[[str], Awaitable[list[str]]] | None,
    ) -> None:
        """Update callback for listing configured detached MCP servers."""
        self._list_configured_detached_mcp_servers_callback = callback

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        names = [name for name in self._agents if name not in self._tool_only_agents]
        if force_include and force_include in self._agents and force_include not in names:
            return [force_include, *names]
        return names

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        visible_names = set(self.visible_agent_names(force_include=force_include))
        return {
            name: agent.agent_type for name, agent in self._agents.items() if name in visible_names
        }

    def registered_agents(self) -> Mapping[str, AgentProtocol]:
        return self._agents

    def registered_agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def registered_agent_types(self) -> dict[str, AgentType]:
        return {name: agent.agent_type for name, agent in self._agents.items()}

    def can_detach_agent_tools(self) -> bool:
        """Return True if agent tool detachment is available."""
        return self._detach_agent_tools_callback is not None

    async def refresh_if_needed(self) -> bool:
        """Refresh agent instances if the registry has changed."""
        result = await self._refresh_if_needed()
        self._last_refresh_result = result
        return result.changed

    async def _refresh_if_needed(self) -> AgentRefreshResult:
        if self._refresh_callback:
            return await self._refresh_callback()
        return AgentRefreshResult(changed=False)

    @deprecated
    async def prompt(
        self,
        agent_name: str | None = None,
        default_prompt: str = "",
        request_params: RequestParams | None = None,
    ) -> PromptLoopResult:
        """
        Deprecated - use interactive() instead.
        """
        return await self.interactive(
            agent_name=agent_name, default_prompt=default_prompt, request_params=request_params
        )

    async def interactive(
        self,
        agent_name: str | None = None,
        default_prompt: str = "",
        pretty_print_parallel: bool = False,
        request_params: RequestParams | None = None,
        session_manager: "SessionManager | None" = None,
        harness_session: "HarnessSession | None" = None,
    ) -> PromptLoopResult:
        """
        Interactive prompt for sending messages with advanced features.

        Args:
            agent_name: Optional target agent name (uses default if not specified)
            default: Default message to use when user presses enter
            pretty_print_parallel: Enable clean parallel results display for parallel agents
            request_params: Optional request parameters including MCP metadata
            session_manager: Optional session manager for session-backed interactive runs
            harness_session: Optional harness session that owns this interactive run

        Returns:
            The result of the interactive session
        """
        target_name = self.resolve_target_agent_name(agent_name)
        if target_name is None:
            raise ValueError("No agents provided!")

        # Don't delegate to the agent's own prompt method - use our implementation
        # The agent's prompt method doesn't fully support switching between agents

        available_names = self.visible_agent_names(force_include=agent_name)
        agent_types = self.visible_agent_types(force_include=agent_name)

        # Create the interactive prompt
        from fast_agent.ui.interactive_prompt import InteractivePrompt

        prompt = InteractivePrompt(agent_types=agent_types)

        async def send_with_error_handling(message, agent_name, *, show_usage: bool) -> str:
            if harness_session is not None:
                try:
                    turn_start_indices = self._capture_turn_start_indices(agent_name)
                    result = await harness_session.send(
                        message,
                        agent_name=agent_name,
                        request_params=request_params,
                    )
                    if show_usage and display_usage_enabled():
                        self._show_turn_usage(agent_name, turn_start_indices)
                    return result
                except Exception as e:
                    if isinstance(e, (KeyboardInterrupt, AgentConfigError, ServerConfigError)):
                        raise

                    logger.exception(
                        "Agent failed after repeated attempts",
                        agent_name=agent_name,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    return _format_interactive_final_error(e)

            return await self._send_interactive_message(
                message,
                agent_name,
                request_params=request_params,
                show_usage=show_usage,
            )

        async def send_wrapper(message, agent_name) -> str:
            return await send_with_error_handling(message, agent_name, show_usage=True)

        async def quiet_send_wrapper(message, agent_name) -> str:
            return await send_with_error_handling(message, agent_name, show_usage=False)

        return await prompt.prompt_loop(
            send_func=send_wrapper,
            quiet_send_func=quiet_send_wrapper,
            default_agent=target_name,  # Pass the agent name, not the agent object
            available_agents=available_names,
            prompt_provider=self,  # Pass self as the prompt provider
            pinned_agent=agent_name,
            default=default_prompt,
            session_manager=session_manager,
        )

    async def _send_interactive_message(
        self,
        message: str | PromptMessage | PromptMessageExtended,
        agent_name: str | None,
        *,
        request_params: RequestParams | None,
        show_usage: bool,
    ) -> str:
        try:
            # The LLM layer will handle the 10s/20s/30s retries internally.
            turn_start_indices = self._capture_turn_start_indices(agent_name)
            result = await self.send(message, agent_name, request_params)
            if show_usage and display_usage_enabled():
                self._show_turn_usage(agent_name, turn_start_indices)
            return result
        except Exception as e:
            # If we catch an exception here, it means all retries failed.
            if isinstance(e, (KeyboardInterrupt, AgentConfigError, ServerConfigError)):
                raise

            logger.exception(
                "Agent failed after repeated attempts",
                agent_name=agent_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            return _format_interactive_final_error(e)

    def _show_turn_usage(
        self, agent_name: str | None, turn_start_indices: dict[str, int] | None = None
    ) -> None:
        """Show subtle usage information after each turn."""
        if agent_name is None:
            return
        agent = self._agents.get(agent_name)
        if not agent:
            return

        if isinstance(agent, ParallelAgent):
            self._show_parallel_agent_usage(agent, turn_start_indices or {})
        else:
            self._show_regular_agent_usage(agent, (turn_start_indices or {}).get(agent.name))

    def _capture_turn_start_indices(self, agent_name: str | None) -> dict[str, int]:
        """Capture usage accumulator turn indices for a user-initiated turn."""
        if agent_name is None:
            return {}
        agent = self._agents.get(agent_name)
        if not agent:
            return {}

        indices: dict[str, int] = {}

        def record(target: AgentProtocol) -> None:
            accumulator = getattr(target, "usage_accumulator", None)
            if accumulator is not None:
                indices[target.name] = len(accumulator.turns)

        if isinstance(agent, ParallelAgent):
            for child_agent in agent.fan_out_agents:
                record(child_agent)
            record(agent.fan_in_agent)
        else:
            record(agent)

        return indices

    def _show_regular_agent_usage(self, agent, turn_start_index: int | None) -> None:
        """Show usage for a regular (non-parallel) agent."""
        usage_info = self._format_agent_usage(agent, turn_start_index)
        if usage_info:
            with progress_display.paused():
                rich_print()
                rich_print(
                    f"[dim]Last turn: {usage_info['display_text']}[/dim]{usage_info['cache_suffix']}"
                )

    def _show_parallel_agent_usage(
        self, parallel_agent: ParallelAgent, turn_start_indices: dict[str, int]
    ) -> None:
        """Show usage for a parallel agent and its children."""
        child_usage_data = []
        total_input = 0
        total_output = 0
        total_tool_calls = 0

        for child_agent in parallel_agent.fan_out_agents:
            usage_info = self._format_agent_usage(
                child_agent, turn_start_indices.get(child_agent.name)
            )
            if usage_info:
                child_usage_data.append({**usage_info, "name": child_agent.name})
                total_input += usage_info["input_tokens"]
                total_output += usage_info["output_tokens"]
                total_tool_calls += usage_info["tool_calls"]

        if parallel_agent.fan_in_agent:
            usage_info = self._format_agent_usage(
                parallel_agent.fan_in_agent,
                turn_start_indices.get(parallel_agent.fan_in_agent.name),
            )
            if usage_info:
                child_usage_data.append({**usage_info, "name": parallel_agent.fan_in_agent.name})
                total_input += usage_info["input_tokens"]
                total_output += usage_info["output_tokens"]
                total_tool_calls += usage_info["tool_calls"]

        if not child_usage_data:
            return

        # Show aggregated usage for parallel agent (no context percentage)
        with progress_display.paused():
            tool_info = f", {total_tool_calls} tool calls" if total_tool_calls > 0 else ""
            rich_print(
                f"[dim]Last turn (parallel): {total_input:,} Input, {total_output:,} Output{tool_info}[/dim]"
            )

            # Show individual child agent usage
            for i, usage_data in enumerate(child_usage_data):
                is_last = i == len(child_usage_data) - 1
                prefix = "└─" if is_last else "├─"
                rich_print(
                    f"[dim]  {prefix} {usage_data['name']}: {usage_data['display_text']}[/dim]{usage_data['cache_suffix']}"
                )

    def _format_agent_usage(self, agent, turn_start_index: int | None) -> dict | None:
        """Format usage information for a single agent."""
        if not agent or not agent.usage_accumulator:
            return None

        # Get the last turn's usage (if any)
        turns = agent.usage_accumulator.turns
        if not turns:
            return None

        usage_totals, turn_slice = self._usage_totals_and_turns(agent, turn_start_index)
        input_tokens = usage_totals["input_tokens"]
        output_tokens = usage_totals["output_tokens"]
        tool_calls = usage_totals["tool_calls"]
        cache_indicators = self._cache_indicators(turn_slice)
        cache_expiry_text = self._cache_expiry_text(agent, cache_indicators)
        context_percentage = agent.usage_accumulator.context_usage_percentage
        display_text = self._usage_display_text(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            context_percentage=context_percentage,
        )
        cache_suffix = f" {cache_indicators}{cache_expiry_text}" if cache_indicators else ""

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tool_calls": tool_calls,
            "context_percentage": context_percentage,
            "display_text": display_text,
            "cache_suffix": cache_suffix,
        }

    def _usage_totals_and_turns(self, agent, turn_start_index: int | None):
        turns = agent.usage_accumulator.turns
        totals = last_turn_usage(agent.usage_accumulator, turn_start_index)
        if totals:
            turn_slice = turns[turn_start_index:] if turn_start_index is not None else [turns[-1]]
            return totals, turn_slice

        last_turn = turns[-1]
        return (
            {
                "input_tokens": last_turn.display_input_tokens,
                "output_tokens": last_turn.output_tokens,
                "tool_calls": last_turn.tool_calls,
            },
            [last_turn],
        )

    @staticmethod
    def _cache_indicators(turn_slice) -> str:
        indicators = ""
        if any(turn.cache_usage.cache_write_tokens > 0 for turn in turn_slice):
            indicators += "[bright_yellow]^[/bright_yellow]"
        if any(
            turn.cache_usage.cache_read_tokens > 0 or turn.cache_usage.cache_hit_tokens > 0
            for turn in turn_slice
        ):
            indicators += "[bright_green]*[/bright_green]"
        return indicators

    def _cache_expiry_text(self, agent, cache_indicators: str) -> str:
        last_cache_activity = agent.usage_accumulator.last_cache_activity_time
        if not cache_indicators or not last_cache_activity:
            return ""

        cache_ttl = self._cache_ttl(agent)
        if not cache_ttl:
            return ""

        ttl_minutes = 60 if cache_ttl == "1h" else 5
        expiry_timestamp = last_cache_activity + (ttl_minutes * 60)
        if expiry_timestamp <= time.time():
            return ""

        expiry_time = datetime.fromtimestamp(expiry_timestamp).strftime("%H:%M")
        return f" [dim]({expiry_time})[/dim]"

    @staticmethod
    def _cache_ttl(agent) -> str | None:
        llm = getattr(agent, "llm", None)
        resolved_model = getattr(llm, "resolved_model", None)
        cache_ttl = getattr(resolved_model, "cache_ttl", None)
        context = getattr(agent, "context", None)
        if context and context.config and context.config.anthropic:
            return context.config.anthropic.cache_ttl
        return cache_ttl

    @staticmethod
    def _usage_display_text(
        *,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int,
        context_percentage: float | None,
    ) -> str:
        tool_info = f", {tool_calls} tool calls" if tool_calls > 0 else ""
        context_info = f" ({context_percentage:.1f}%)" if context_percentage is not None else ""
        return f"{input_tokens:,} Input, {output_tokens:,} Output{tool_info}{context_info}"
