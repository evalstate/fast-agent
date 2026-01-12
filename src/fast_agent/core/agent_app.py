"""
Direct AgentApp implementation for interacting with agents without proxies.
"""

from typing import Awaitable, Callable, Mapping, Sequence, Union

from deprecated import deprecated
from mcp.types import GetPromptResult, PromptMessage
from rich import print as rich_print

from fast_agent.agents.agent_types import AgentType
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.core.exceptions import AgentConfigError, ServerConfigError
from fast_agent.interfaces import AgentProtocol
from fast_agent.llm.usage_tracking import last_turn_usage
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.ui.interactive_prompt import InteractivePrompt
from fast_agent.ui.progress_display import progress_display


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
        reload_callback: Callable[[], Awaitable[bool]] | None = None,
        refresh_callback: Callable[[], Awaitable[bool]] | None = None,
        load_card_callback: Callable[
            [str, str | None], Awaitable[tuple[list[str], list[str]]]
        ]
        | None = None,
        attach_agent_tools_callback: Callable[
            [str, Sequence[str]], Awaitable[list[str]]
        ]
        | None = None,
        detach_agent_tools_callback: Callable[
            [str, Sequence[str]], Awaitable[list[str]]
        ]
        | None = None,
        dump_agent_callback: Callable[[str], Awaitable[str]] | None = None,
        tool_only_agents: set[str] | None = None,
        card_collision_warnings: list[str] | None = None,
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
        if len(agents) == 0:
            raise ValueError("No agents provided!")
        self._agents = agents
        self._reload_callback = reload_callback
        self._refresh_callback = refresh_callback
        self._load_card_callback = load_card_callback
        self._attach_agent_tools_callback = attach_agent_tools_callback
        self._detach_agent_tools_callback = detach_agent_tools_callback
        self._dump_agent_callback = dump_agent_callback
        self._tool_only_agents: set[str] = tool_only_agents or set()
        self._card_collision_warnings: list[str] = card_collision_warnings or []

    def __getitem__(self, key: str) -> AgentProtocol:
        """Allow access to agents using dictionary syntax."""
        if key not in self._agents:
            raise KeyError(f"Agent '{key}' not found")
        return self._agents[key]

    def __getattr__(self, name: str) -> AgentProtocol:
        """Allow access to agents using attribute syntax."""
        if name in self._agents:
            return self._agents[name]
        raise AttributeError(f"Agent '{name}' not found")

    async def __call__(
        self,
        message: Union[str, PromptMessage, PromptMessageExtended] | None = None,
        agent_name: str | None = None,
        default_prompt: str = "",
        request_params: RequestParams | None = None,
    ) -> str:
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
        message: Union[str, PromptMessage, PromptMessageExtended],
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

    def _agent(self, agent_name: str | None) -> AgentProtocol:
        if agent_name:
            if agent_name not in self._agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            return self._agents[agent_name]

        # Skip tool_only agents when selecting default
        for agent in self._agents.values():
            if agent.config.default and agent.name not in self._tool_only_agents:
                return agent

        # Fall back to first non-tool_only agent
        for name, agent in self._agents.items():
            if name not in self._tool_only_agents:
                return agent

        # If all agents are tool_only, return the first one anyway
        return next(iter(self._agents.values()))

    async def apply_prompt(
        self,
        prompt: Union[str, GetPromptResult],
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
        prompt_content: Union[str, PromptMessage, PromptMessageExtended],
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
            return False
        return await self._reload_callback()

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
    ) -> tuple[list[str], list[str]]:
        """Load an AgentCard source and refresh active instances when available."""
        if not self._load_card_callback:
            raise RuntimeError("Agent card loading is not available.")
        return await self._load_card_callback(source, parent_agent)

    async def attach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]:
        """Attach agents as tools to a parent agent."""
        if not self._attach_agent_tools_callback:
            raise RuntimeError("Agent tool attachment is not available.")
        return await self._attach_agent_tools_callback(parent_agent, child_agents)

    async def detach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]:
        """Detach agents-as-tools from a parent agent."""
        if not self._detach_agent_tools_callback:
            raise RuntimeError("Agent tool detachment is not available.")
        return await self._detach_agent_tools_callback(parent_agent, child_agents)

    async def dump_agent_card(self, agent_name: str) -> str:
        """Dump an AgentCard for the requested agent."""
        if not self._dump_agent_callback:
            raise RuntimeError("Agent card dumping is not available.")
        return await self._dump_agent_callback(agent_name)

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
        if tool_only_agents is not None:
            self._tool_only_agents = tool_only_agents
        if card_collision_warnings is not None:
            self._card_collision_warnings = card_collision_warnings

    @property
    def card_collision_warnings(self) -> list[str]:
        """Return warnings from agent card name collisions."""
        return list(self._card_collision_warnings)

    def set_reload_callback(self, callback: Callable[[], Awaitable[bool]] | None) -> None:
        """Update the reload callback for manual AgentCard refresh."""
        self._reload_callback = callback

    def set_refresh_callback(self, callback: Callable[[], Awaitable[bool]] | None) -> None:
        """Update the refresh callback for lazy instance swaps."""
        self._refresh_callback = callback

    def set_load_card_callback(
        self,
        callback: Callable[[str, str | None], Awaitable[tuple[list[str], list[str]]]]
        | None,
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

    def set_dump_agent_callback(
        self, callback: Callable[[str], Awaitable[str]] | None
    ) -> None:
        """Update the callback for dumping agent cards."""
        self._dump_agent_callback = callback

    def agent_names(self) -> list[str]:
        """Return available agent names (excluding tool_only agents)."""
        return [name for name in self._agents.keys() if name not in self._tool_only_agents]

    def can_detach_agent_tools(self) -> bool:
        """Return True if agent tool detachment is available."""
        return self._detach_agent_tools_callback is not None

    def agent_types(self) -> dict[str, AgentType]:
        """Return mapping of agent names to agent types."""
        return {name: agent.agent_type for name, agent in self._agents.items()}

    async def refresh_if_needed(self) -> bool:
        """Refresh agent instances if the registry has changed."""
        return await self._refresh_if_needed()

    async def _refresh_if_needed(self) -> bool:
        if self._refresh_callback:
            return await self._refresh_callback()
        return False

    @deprecated
    async def prompt(
        self,
        agent_name: str | None = None,
        default_prompt: str = "",
        request_params: RequestParams | None = None,
    ) -> str:
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
    ) -> str:
        """
        Interactive prompt for sending messages with advanced features.

        Args:
            agent_name: Optional target agent name (uses default if not specified)
            default: Default message to use when user presses enter
            pretty_print_parallel: Enable clean parallel results display for parallel agents
            request_params: Optional request parameters including MCP metadata

        Returns:
            The result of the interactive session
        """
        # Get the default agent name if none specified
        if agent_name:
            # Validate that this agent exists
            if agent_name not in self._agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            target_name = agent_name
        else:
            target_name = None
            for agent in self._agents.values():
                if agent.config.default:
                    target_name = agent.name
                    break

            if not target_name:
                # Use the first agent's name as default
                target_name = next(iter(self._agents.keys()))

        # Don't delegate to the agent's own prompt method - use our implementation
        # The agent's prompt method doesn't fully support switching between agents

        # Create agent_types dictionary mapping agent names to their types (excluding tool_only)
        visible_names = set(self.agent_names())
        agent_types = {
            name: agent.agent_type
            for name, agent in self._agents.items()
            if name in visible_names
        }

        # Create the interactive prompt
        prompt = InteractivePrompt(agent_types=agent_types)
        
        # Helper for pretty formatting the FINAL error    
        def _format_final_error(error: Exception) -> str:
            detail = getattr(error, "message", None) or str(error)
            detail = detail.strip() if isinstance(detail, str) else ""
            clean_detail = detail.replace("\n", " ")
            if len(clean_detail) > 300:
                clean_detail = clean_detail[:297] + "..."
            
            return (
                f"⚠️ **System Error:** The agent failed after repeated attempts.\n"
                f"Error details: {clean_detail}\n"
                f"\n*Your context is preserved. You can try sending the message again.*"
            )

        async def send_wrapper(message, agent_name):
            try:
                # The LLM layer will handle the 10s/20s/30s retries internally.
                turn_start_indices = self._capture_turn_start_indices(agent_name)
                result = await self.send(message, agent_name, request_params)
                # Show usage info after each turn
                self._show_turn_usage(agent_name, turn_start_indices)
                return result

            except Exception as e:
                # If we catch an exception here, it means all retries FAILED.
                if isinstance(e, (KeyboardInterrupt, AgentConfigError, ServerConfigError)):
                    raise e

                # Return pretty text for API failures (keeps session alive)
                return _format_final_error(e)

        return await prompt.prompt_loop(
            send_func=send_wrapper,
            default_agent=target_name,  # Pass the agent name, not the agent object
            available_agents=self.agent_names(),  # Excludes tool_only agents
            prompt_provider=self,  # Pass self as the prompt provider
            default=default_prompt,
        )

    def _show_turn_usage(
        self, agent_name: str, turn_start_indices: dict[str, int] | None = None
    ) -> None:
        """Show subtle usage information after each turn."""
        agent = self._agents.get(agent_name)
        if not agent:
            return

        # Check if this is a parallel agent
        if agent.agent_type == AgentType.PARALLEL:
            self._show_parallel_agent_usage(agent, turn_start_indices or {})
        else:
            self._show_regular_agent_usage(
                agent, (turn_start_indices or {}).get(agent.name)
            )

    def _capture_turn_start_indices(self, agent_name: str) -> dict[str, int]:
        """Capture usage accumulator turn indices for a user-initiated turn."""
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
                rich_print(
                    f"[dim]Last turn: {usage_info['display_text']}[/dim]{usage_info['cache_suffix']}"
                )

    def _show_parallel_agent_usage(
        self, parallel_agent, turn_start_indices: dict[str, int]
    ) -> None:
        """Show usage for a parallel agent and its children."""
        # Collect usage from all child agents
        child_usage_data = []
        total_input = 0
        total_output = 0
        total_tool_calls = 0

        # Get usage from fan-out agents
        if hasattr(parallel_agent, "fan_out_agents") and parallel_agent.fan_out_agents:
            for child_agent in parallel_agent.fan_out_agents:
                usage_info = self._format_agent_usage(
                    child_agent, turn_start_indices.get(child_agent.name)
                )
                if usage_info:
                    child_usage_data.append({**usage_info, "name": child_agent.name})
                    total_input += usage_info["input_tokens"]
                    total_output += usage_info["output_tokens"]
                    total_tool_calls += usage_info["tool_calls"]

        # Get usage from fan-in agent
        if hasattr(parallel_agent, "fan_in_agent") and parallel_agent.fan_in_agent:
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

        last_turn = turns[-1]
        totals = last_turn_usage(agent.usage_accumulator, turn_start_index)
        if totals:
            input_tokens = totals["input_tokens"]
            output_tokens = totals["output_tokens"]
            tool_calls = totals["tool_calls"]
            turn_slice = turns[turn_start_index:] if turn_start_index is not None else [last_turn]
        else:
            input_tokens = last_turn.display_input_tokens
            output_tokens = last_turn.output_tokens
            tool_calls = last_turn.tool_calls
            turn_slice = [last_turn]

        # Build cache indicators with bright colors
        cache_indicators = ""
        if any(turn.cache_usage.cache_write_tokens > 0 for turn in turn_slice):
            cache_indicators += "[bright_yellow]^[/bright_yellow]"
        if any(
            turn.cache_usage.cache_read_tokens > 0 or turn.cache_usage.cache_hit_tokens > 0
            for turn in turn_slice
        ):
            cache_indicators += "[bright_green]*[/bright_green]"

        # Build context percentage - get from accumulator, not individual turn
        context_info = ""
        context_percentage = agent.usage_accumulator.context_usage_percentage
        if context_percentage is not None:
            context_info = f" ({context_percentage:.1f}%)"

        # Build tool call info
        tool_info = f", {tool_calls} tool calls" if tool_calls > 0 else ""

        # Build display text
        display_text = f"{input_tokens:,} Input, {output_tokens:,} Output{tool_info}{context_info}"
        cache_suffix = f" {cache_indicators}" if cache_indicators else ""

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tool_calls": tool_calls,
            "context_percentage": context_percentage,
            "display_text": display_text,
            "cache_suffix": cache_suffix,
        }
