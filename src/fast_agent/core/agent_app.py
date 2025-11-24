"""
Direct AgentApp implementation for interacting with agents without proxies.
"""

import asyncio
import os
from typing import Mapping, Union

from deprecated import deprecated
from mcp.types import GetPromptResult, PromptMessage
from rich import print as rich_print

from fast_agent.agents.agent_types import AgentType
from fast_agent.interfaces import AgentProtocol
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.ui.interactive_prompt import InteractivePrompt
from fast_agent.ui.progress_display import progress_display
from fast_agent.core.exceptions import ProviderKeyError, AgentConfigError, ServerConfigError


class AgentApp:
    """
    Container for active agents that provides a simple API for interacting with them.
    This implementation works directly with Agent instances without proxies.

    The DirectAgentApp provides both attribute-style access (app.agent_name)
    and dictionary-style access (app["agent_name"]) to agents.

    It also implements the AgentProtocol interface, automatically forwarding
    calls to the default agent (the first agent in the container).
    """

    def __init__(self, agents: dict[str, AgentProtocol]) -> None:
        """
        Initialize the DirectAgentApp.

        Args:
            agents: Dictionary of agent instances keyed by name
        """
        if len(agents) == 0:
            raise ValueError("No agents provided!")
        self._agents = agents

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
        return await self._agent(agent_name).send(message, request_params)

    def _agent(self, agent_name: str | None) -> AgentProtocol:
        if agent_name:
            if agent_name not in self._agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            return self._agents[agent_name]

        for agent in self._agents.values():
            if agent.config.default:
                return agent

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
        return await self._agent(agent_name).get_resource(
            resource_uri=resource_uri, namespace=server_name
        )

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
        retries: int | None = None,
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
        # Get the number of retries
        if retries is None:
            retries = int(os.getenv("FAST_AGENT_RETRIES", "3"))
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

        # Create agent_types dictionary mapping agent names to their types
        agent_types = {name: agent.agent_type for name, agent in self._agents.items()}

        # Create the interactive prompt
        prompt = InteractivePrompt(agent_types=agent_types)

        def _is_fatal_error(e: Exception) -> bool:
            if isinstance(e, (KeyboardInterrupt, AgentConfigError, ServerConfigError)):
                return True
            # Google 429s look like Auth errors but are actually retriable
            if isinstance(e, ProviderKeyError):
                msg = str(e).lower()
                retry_keywords = ["429", "503", "quota", "exhausted", "overloaded", "unavailable", "timeout"]
                if any(k in msg for k in retry_keywords):
                    return False
                return True
            return False
        
        # Define the wrapper for send function
        def _format_final_error(error: Exception, attempts: int) -> str:
            """Recreates the polite error message style from the providers."""
            
            detail = getattr(error, "message", None) or str(error)
            detail = detail.strip() if isinstance(detail, str) else ""
            
            parts = [f"Request failed after {attempts} retries"]
            code = getattr(error, "code", None)
            status = getattr(error, "status_code", None)
            
            if code: parts.append(f"(code: {code})")
            if status: parts.append(f"(status={status})")
            
            message_line = " ".join(parts)
            if detail:
                # Remove excessive newlines from raw JSON
                clean_detail = detail.replace("\n", " ")
                # Limit length to avoid flooding chat, but keep enough to read
                if len(clean_detail) > 300:
                    clean_detail = clean_detail[:297] + "..."
                message_line = f"{message_line}: {clean_detail}"

            return (
                f"⚠️ **System Error:** {message_line}\n"
                f"\n*Your context is preserved. You can try sending the message again.*"
            )

        # --- 3. The Retry Logic ---
        async def send_wrapper(message, agent_name):
            last_error = None
            
            for attempt in range(retries + 1):
                try:
                    return await self.send(message, agent_name, request_params)
                
                except Exception as e:
                    # Fatal errors crash immediately
                    if _is_fatal_error(e):
                        raise e
                    
                    last_error = e

                    if attempt < retries:
                        wait_time = 10 * (attempt + 1) # 10s, 20s, 30s...
                        
                        with progress_display.paused():
                            error_preview = str(e).replace("\n", " ")[:300]
                            rich_print(f"\n[yellow]⚠ Provider Error: {error_preview}...[/yellow]")
                            rich_print(f"[dim]⟳ Retrying in {wait_time}s... (Attempt {attempt+1}/{retries})[/dim]")
                        
                        await asyncio.sleep(wait_time)
                    else:
                        # Final failure
                        with progress_display.paused():
                            rich_print(f"\n[red]❌ All {retries} retries failed.[/red]")

            # Check if last_error exists before using it
            if last_error is None:
                return "⚠️ **System Error:** Operation failed with no captured exception."

            return _format_final_error(last_error, retries)

        return await prompt.prompt_loop(
            send_func=send_wrapper,
            default_agent=target_name,  # Pass the agent name, not the agent object
            available_agents=list(self._agents.keys()),
            prompt_provider=self,  # Pass self as the prompt provider
            default=default_prompt,
        )

    def _show_turn_usage(self, agent_name: str) -> None:
        """Show subtle usage information after each turn."""
        agent = self._agents.get(agent_name)
        if not agent:
            return

        # Check if this is a parallel agent
        if agent.agent_type == AgentType.PARALLEL:
            self._show_parallel_agent_usage(agent)
        else:
            self._show_regular_agent_usage(agent)

    def _show_regular_agent_usage(self, agent) -> None:
        """Show usage for a regular (non-parallel) agent."""
        usage_info = self._format_agent_usage(agent)
        if usage_info:
            with progress_display.paused():
                rich_print(
                    f"[dim]Last turn: {usage_info['display_text']}[/dim]{usage_info['cache_suffix']}"
                )

    def _show_parallel_agent_usage(self, parallel_agent) -> None:
        """Show usage for a parallel agent and its children."""
        # Collect usage from all child agents
        child_usage_data = []
        total_input = 0
        total_output = 0
        total_tool_calls = 0

        # Get usage from fan-out agents
        if hasattr(parallel_agent, "fan_out_agents") and parallel_agent.fan_out_agents:
            for child_agent in parallel_agent.fan_out_agents:
                usage_info = self._format_agent_usage(child_agent)
                if usage_info:
                    child_usage_data.append({**usage_info, "name": child_agent.name})
                    total_input += usage_info["input_tokens"]
                    total_output += usage_info["output_tokens"]
                    total_tool_calls += usage_info["tool_calls"]

        # Get usage from fan-in agent
        if hasattr(parallel_agent, "fan_in_agent") and parallel_agent.fan_in_agent:
            usage_info = self._format_agent_usage(parallel_agent.fan_in_agent)
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

    def _format_agent_usage(self, agent) -> dict | None:
        """Format usage information for a single agent."""
        if not agent or not agent.usage_accumulator:
            return None

        # Get the last turn's usage (if any)
        turns = agent.usage_accumulator.turns
        if not turns:
            return None

        last_turn = turns[-1]
        input_tokens = last_turn.display_input_tokens
        output_tokens = last_turn.output_tokens

        # Build cache indicators with bright colors
        cache_indicators = ""
        if last_turn.cache_usage.cache_write_tokens > 0:
            cache_indicators += "[bright_yellow]^[/bright_yellow]"
        if (
            last_turn.cache_usage.cache_read_tokens > 0
            or last_turn.cache_usage.cache_hit_tokens > 0
        ):
            cache_indicators += "[bright_green]*[/bright_green]"

        # Build context percentage - get from accumulator, not individual turn
        context_info = ""
        context_percentage = agent.usage_accumulator.context_usage_percentage
        if context_percentage is not None:
            context_info = f" ({context_percentage:.1f}%)"

        # Build tool call info
        tool_info = f", {last_turn.tool_calls} tool calls" if last_turn.tool_calls > 0 else ""

        # Build display text
        display_text = f"{input_tokens:,} Input, {output_tokens:,} Output{tool_info}{context_info}"
        cache_suffix = f" {cache_indicators}" if cache_indicators else ""

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tool_calls": last_turn.tool_calls,
            "context_percentage": context_percentage,
            "display_text": display_text,
            "cache_suffix": cache_suffix,
        }
