"""
Direct AgentApp implementation for interacting with agents without proxies.
"""

from typing import Dict, List, Optional, Union

from deprecated import deprecated
from mcp.types import PromptMessage

from mcp_agent.agents.agent import Agent
from mcp_agent.core.interactive_prompt import InteractivePrompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class AgentApp:
    """
    Container for active agents that provides a simple API for interacting with them.
    This implementation works directly with Agent instances without proxies.

    The DirectAgentApp provides both attribute-style access (app.agent_name)
    and dictionary-style access (app["agent_name"]) to agents.

    It also implements the AgentProtocol interface, automatically forwarding
    calls to the default agent (the first agent in the container).
    """

    def __init__(self, agents: Dict[str, Agent]) -> None:
        """
        Initialize the DirectAgentApp.

        Args:
            agents: Dictionary of agent instances keyed by name
        """
        self._agents = agents

    def __getitem__(self, key: str) -> Agent:
        """Allow access to agents using dictionary syntax."""
        if key not in self._agents:
            raise KeyError(f"Agent '{key}' not found")
        return self._agents[key]

    def __getattr__(self, name: str) -> Agent:
        """Allow access to agents using attribute syntax."""
        if name in self._agents:
            return self._agents[name]
        raise AttributeError(f"Agent '{name}' not found")

    async def __call__(
        self,
        message: Union[str, PromptMessage, PromptMessageMultipart] | None = None,
        agent_name: str | None = None,
        default_prompt: str = "",
    ) -> str:
        """
        Make the object callable to send messages or start interactive prompt.
        This mirrors the FastAgent implementation that allowed agent("message").

        Args:
            message: Message content in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
            agent_name: Optional name of the agent to send to (defaults to first agent)
            default_prompt: Default message to use in interactive prompt mode

        Returns:
            The agent's response as a string or the result of the interactive session
        """
        if message:
            return await self._agent(agent_name).send(message)

        return await self.interactive(agent=agent_name, default_prompt=default_prompt)

    async def send(
        self,
        message: Union[str, PromptMessage, PromptMessageMultipart],
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Send a message to the specified agent (or to all agents).

        Args:
            message: Message content in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
            agent_name: Optional name of the agent to send to

        Returns:
            The agent's response as a string
        """
        return await self._agent(agent_name).send(message)

    def _agent(self, agent_name: str | None) -> Agent:
        if agent_name:
            if agent_name not in self._agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            return self._agents[agent_name]

        return next(iter(self._agents.values()))

    async def apply_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        agent_name: str | None = None,
    ) -> str:
        """
        Apply a prompt template to an agent (default agent if not specified).

        Args:
            prompt_name: Name of the prompt template to apply
            arguments: Optional arguments for the prompt template
            agent_name: Name of the agent to send to

        Returns:
            The agent's response as a string
        """
        return await self._agent(agent_name).apply_prompt(prompt_name, arguments)

    async def list_prompts(self, server_name: str | None = None, agent_name: str | None = None):
        """
        List available prompts for an agent.

        Args:
            server_name: Optional name of the server to list prompts from
            agent_name: Name of the agent to list prompts for

        Returns:
            Dictionary mapping server names to lists of available prompts
        """
        return await self._agent(agent_name).list_prompts(server_name=server_name)

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
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
            prompt_name=prompt_name, arguments=arguments, server_name=server_name
        )

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageMultipart],
        resource_uri: str,
        server_name: str | None = None,
        agent_name: str | None = None,
    ) -> str:
        """
        Send a message with an attached MCP resource.

        Args:
            prompt_content: Content in various formats (String, PromptMessage, or PromptMessageMultipart)
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from
            agent_name: Name of the agent to use

        Returns:
            The agent's response as a string
        """
        return await self._agent(agent_name).with_resource(
            prompt_content=prompt_content, resource_uri=resource_uri, server_name=server_name
        )

    async def list_resources(
        self,
        server_name: str | None = None,
        agent_name: str | None = None,
    ) -> Dict[str, List[str]]:
        """
        List available resources from one or all servers.

        Args:
            server_name: Optional server name to list resources from
            agent_name: Name of the agent to use

        Returns:
            Dictionary mapping server names to lists of resource URIs
        """
        return await self._agent(agent_name).list_resources(server_name=server_name)

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
            resource_uri=resource_uri, server_name=server_name
        )

    @deprecated
    async def prompt(self, agent_name: str | None = None, default_prompt: str = "") -> str:
        """
        Deprecated - use interactive() instead.
        """
        return await self.interactive(agent=agent_name, default_prompt=default_prompt)

    async def interactive(self, agent: str | None = None, default_prompt: str = "") -> str:
        """
        Interactive prompt for sending messages with advanced features.

        Args:
            agent_name: Optional target agent name (uses default if not specified)
            default: Default message to use when user presses enter

        Returns:
            The result of the interactive session
        """

        # Get the default agent name if none specified
        if agent:
            # Validate that this agent exists
            if agent not in self._agents:
                raise ValueError(f"Agent '{agent}' not found")
            target_name = agent
        else:
            # Use the first agent's name as default
            target_name = next(iter(self._agents.keys()))

        # Don't delegate to the agent's own prompt method - use our implementation
        # The agent's prompt method doesn't fully support switching between agents

        # Create agent_types dictionary mapping agent names to their types
        agent_types = {name: agent.agent_type for name, agent in self._agents.items()}

        # Create the interactive prompt
        prompt = InteractivePrompt(agent_types=agent_types)

        # Define the wrapper for send function
        async def send_wrapper(message, agent_name):
            return await self.send(message, agent_name)

        # Start the prompt loop with the agent name (not the agent object)
        return await prompt.prompt_loop(
            send_func=send_wrapper,
            default_agent=target_name,  # Pass the agent name, not the agent object
            available_agents=list(self._agents.keys()),
            apply_prompt_func=self.apply_prompt,
            list_prompts_func=self.list_prompts,
            default=default_prompt,
        )
