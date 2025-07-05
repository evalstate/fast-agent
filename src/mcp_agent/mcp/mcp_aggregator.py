from asyncio import Lock, gather
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    TypeVar,
)

from mcp import GetPromptResult, ReadResourceResult
from mcp.client.session import ClientSession
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Prompt,
    TextContent,
    Tool,
)
from opentelemetry import trace
from pydantic import AnyUrl, BaseModel, ConfigDict

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.event_progress import ProgressAction
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.common import SEP, create_namespaced_name, is_namespaced_name
from mcp_agent.mcp.gen_client import gen_client
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

if TYPE_CHECKING:
    from mcp_agent.context import Context


logger = get_logger(__name__)  # This will be replaced per-instance when agent_name is available

# Define type variables for the generalized method
T = TypeVar("T")
R = TypeVar("R")


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.
    """

    tool: Tool
    server_name: str
    namespaced_tool_name: str


class MCPAggregator(ContextDependent):
    """
    Aggregates multiple MCP servers. When a developer calls, e.g. call_tool(...),
    the aggregator searches all servers in its list for a server that provides that tool.
    """

    initialized: bool = False
    """Whether the aggregator has been initialized with tools and resources from all servers."""

    connection_persistence: bool = False
    """Whether to maintain a persistent connection to the server."""

    server_names: List[str]
    """A list of server names to connect to."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    async def __aenter__(self):
        if self.initialized:
            return self

        # Keep a connection manager to manage persistent connections for this aggregator
        if self.connection_persistence:
            # Try to get existing connection manager from context
            if not hasattr(self.context, "_connection_manager"):
                self.context._connection_manager = MCPConnectionManager(
                    self.context.server_registry
                )
                await self.context._connection_manager.__aenter__()
            self._persistent_connection_manager = self.context._connection_manager

        # Import the display component here to avoid circular imports
        from mcp_agent.ui.console_display import ConsoleDisplay

        # Initialize the display component
        self.display = ConsoleDisplay(config=self.context.config)

        await self.load_servers()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __init__(
        self,
        server_names: List[str],
        connection_persistence: bool = True,  # Default to True for better stability
        context: Optional["Context"] = None,
        name: str = None,
        **kwargs,
    ) -> None:
        """
        :param server_names: A list of server names to connect to.
        :param connection_persistence: Whether to maintain persistent connections to servers (default: True).
        Note: The server names must be resolvable by the gen_client function, and specified in the server registry.
        """
        super().__init__(
            context=context,
            **kwargs,
        )

        self.server_names = server_names
        self.connection_persistence = connection_persistence
        self.agent_name = name
        self._persistent_connection_manager: MCPConnectionManager = None

        # Set up logger with agent name in namespace if available
        global logger
        logger_name = f"{__name__}.{name}" if name else __name__
        logger = get_logger(logger_name)

        # Maps namespaced_tool_name -> namespaced tool info
        self._namespaced_tool_map: Dict[str, NamespacedTool] = {}
        # Maps server_name -> list of tools
        self._server_to_tool_map: Dict[str, List[NamespacedTool]] = {}
        self._tool_map_lock = Lock()

        # Cache for prompt objects, maps server_name -> list of prompt objects
        self._prompt_cache: Dict[str, List[Prompt]] = {}
        self._prompt_cache_lock = Lock()

        # Lock for refreshing tools from a server
        self._refresh_lock = Lock()

    async def close(self) -> None:
        """
        Close all persistent connections when the aggregator is deleted.
        """
        if self.connection_persistence and self._persistent_connection_manager:
            try:
                # Only attempt cleanup if we own the connection manager
                if (
                    hasattr(self.context, "_connection_manager")
                    and self.context._connection_manager == self._persistent_connection_manager
                ):
                    logger.info("Shutting down all persistent connections...")
                    await self._persistent_connection_manager.disconnect_all()
                    await self._persistent_connection_manager.__aexit__(None, None, None)
                    delattr(self.context, "_connection_manager")
                self.initialized = False
            except Exception as e:
                logger.error(f"Error during connection manager cleanup: {e}")

    @classmethod
    async def create(
        cls,
        server_names: List[str],
        connection_persistence: bool = False,
    ) -> "MCPAggregator":
        """
        Factory method to create and initialize an MCPAggregator.
        Use this instead of constructor since we need async initialization.
        If connection_persistence is True, the aggregator will maintain a
        persistent connection to the servers for as long as this aggregator is around.
        By default we do not maintain a persistent connection.
        """

        logger.info(f"Creating MCPAggregator with servers: {server_names}")

        instance = cls(
            server_names=server_names,
            connection_persistence=connection_persistence,
        )

        try:
            await instance.__aenter__()

            logger.debug("Loading servers...")
            await instance.load_servers()

            logger.debug("MCPAggregator created and initialized.")
            return instance
        except Exception as e:
            logger.error(f"Error creating MCPAggregator: {e}")
            await instance.__aexit__(None, None, None)

    async def load_servers(self) -> None:
        """
        Discover tools from each server in parallel and build an index of namespaced tool names.
        Also populate the prompt cache.
        """
        if self.initialized:
            logger.debug("MCPAggregator already initialized.")
            return

        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()

        async with self._prompt_cache_lock:
            self._prompt_cache.clear()

        for server_name in self.server_names:
            if self.connection_persistence:
                logger.info(
                    f"Creating persistent connection to server: {server_name}",
                    data={
                        "progress_action": ProgressAction.STARTING,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )

                # Create a wrapper to capture the parameters for the client session
                def session_factory(read_stream, write_stream, read_timeout, **kwargs):
                    # Get agent's model and name if this aggregator is part of an agent
                    agent_model: str | None = None
                    agent_name: str | None = None
                    elicitation_handler = None

                    # Check if this aggregator is part of an Agent (which has config)
                    # Import here to avoid circular dependency
                    from mcp_agent.agents.base_agent import BaseAgent

                    if isinstance(self, BaseAgent):
                        agent_model = self.config.model
                        agent_name = self.config.name
                        elicitation_handler = self.config.elicitation_handler

                    return MCPAgentClientSession(
                        read_stream,
                        write_stream,
                        read_timeout,
                        server_name=server_name,
                        agent_model=agent_model,
                        agent_name=agent_name,
                        elicitation_handler=elicitation_handler,
                        tool_list_changed_callback=self._handle_tool_list_changed,
                        **kwargs,  # Pass through any additional kwargs like server_config
                    )

                await self._persistent_connection_manager.get_server(
                    server_name, client_session_factory=session_factory
                )

            logger.info(
                f"MCP Servers initialized for agent '{self.agent_name}'",
                data={
                    "progress_action": ProgressAction.INITIALIZED,
                    "agent_name": self.agent_name,
                },
            )

        async def fetch_tools(client: ClientSession):
            try:
                result: ListToolsResult = await client.list_tools()
                return result.tools or []
            except Exception as e:
                logger.error(f"Error loading tools from server '{server_name}'", data=e)
                return []

        async def fetch_prompts(client: ClientSession, server_name: str) -> List[Prompt]:
            # Only fetch prompts if the server supports them
            if not await self.server_supports_feature(server_name, "prompts"):
                logger.debug(f"Server '{server_name}' does not support prompts")
                return []

            try:
                result = await client.list_prompts()
                return getattr(result, "prompts", [])
            except Exception as e:
                logger.debug(f"Error loading prompts from server '{server_name}': {e}")
                return []

        async def load_server_data(server_name: str):
            tools: List[Tool] = []
            prompts: List[Prompt] = []

            if self.connection_persistence:
                server_connection = await self._persistent_connection_manager.get_server(
                    server_name, client_session_factory=MCPAgentClientSession
                )
                tools = await fetch_tools(server_connection.session)
                prompts = await fetch_prompts(server_connection.session, server_name)
            else:
                # Create a factory function for the client session
                def create_session(read_stream, write_stream, read_timeout, **kwargs):
                    # Get agent's model and name if this aggregator is part of an agent
                    agent_model: str | None = None
                    agent_name: str | None = None

                    # Check if this aggregator is part of an Agent (which has config)
                    # Import here to avoid circular dependency
                    from mcp_agent.agents.base_agent import BaseAgent

                    if isinstance(self, BaseAgent):
                        agent_model = self.config.model
                        agent_name = self.config.name

                    return MCPAgentClientSession(
                        read_stream,
                        write_stream,
                        read_timeout,
                        server_name=server_name,
                        agent_model=agent_model,
                        agent_name=agent_name,
                        tool_list_changed_callback=self._handle_tool_list_changed,
                        **kwargs,  # Pass through any additional kwargs like server_config
                    )

                async with gen_client(
                    server_name,
                    server_registry=self.context.server_registry,
                    client_session_factory=create_session,
                ) as client:
                    tools = await fetch_tools(client)
                    prompts = await fetch_prompts(client, server_name)

            return server_name, tools, prompts

        # Gather data from all servers concurrently
        results = await gather(
            *(load_server_data(server_name) for server_name in self.server_names),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Error loading server data: {result}")
                continue

            server_name, tools, prompts = result

            # Process tools
            self._server_to_tool_map[server_name] = []
            for tool in tools:
                namespaced_tool_name = create_namespaced_name(server_name, tool.name)
                namespaced_tool = NamespacedTool(
                    tool=tool,
                    server_name=server_name,
                    namespaced_tool_name=namespaced_tool_name,
                )

                self._namespaced_tool_map[namespaced_tool_name] = namespaced_tool
                self._server_to_tool_map[server_name].append(namespaced_tool)

            # Process prompts
            async with self._prompt_cache_lock:
                self._prompt_cache[server_name] = prompts

            logger.debug(
                f"MCP Aggregator initialized for server '{server_name}'",
                data={
                    "progress_action": ProgressAction.INITIALIZED,
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                    "tool_count": len(tools),
                    "prompt_count": len(prompts),
                },
            )

        self.initialized = True

    async def get_capabilities(self, server_name: str):
        """Get server capabilities if available."""
        if not self.connection_persistence:
            # For non-persistent connections, we can't easily check capabilities
            return None

        try:
            server_conn = await self._persistent_connection_manager.get_server(
                server_name, client_session_factory=MCPAgentClientSession
            )
            # server_capabilities is a property, not a coroutine
            return server_conn.server_capabilities
        except Exception as e:
            logger.debug(f"Error getting capabilities for server '{server_name}': {e}")
            return None

    async def validate_server(self, server_name: str) -> bool:
        """
        Validate that a server exists in our server list.

        Args:
            server_name: Name of the server to validate

        Returns:
            True if the server exists, False otherwise
        """
        valid = server_name in self.server_names
        if not valid:
            logger.debug(f"Server '{server_name}' not found")
        return valid

    async def server_supports_feature(self, server_name: str, feature: str) -> bool:
        """
        Check if a server supports a specific feature.

        Args:
            server_name: Name of the server to check
            feature: Feature to check for (e.g., "prompts", "resources")

        Returns:
            True if the server supports the feature, False otherwise
        """
        if not await self.validate_server(server_name):
            return False

        capabilities = await self.get_capabilities(server_name)
        if not capabilities:
            return False

        return getattr(capabilities, feature, False)

    async def list_servers(self) -> List[str]:
        """Return the list of server names aggregated by this agent."""
        if not self.initialized:
            await self.load_servers()

        return self.server_names

    async def list_tools(self) -> ListToolsResult:
        """
        :return: Tools from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        if not self.initialized:
            await self.load_servers()

        return ListToolsResult(
            tools=[
                namespaced_tool.tool.model_copy(update={"name": namespaced_tool_name})
                for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items()
            ]
        )

    async def refresh_all_tools(self) -> None:
        """
        Refresh the tools for all servers.
        This is useful when you know tools have changed but haven't received notifications.
        """
        logger.info("Refreshing tools for all servers")
        for server_name in self.server_names:
            await self._refresh_server_tools(server_name)

    async def _execute_on_server(
        self,
        server_name: str,
        operation_type: str,
        operation_name: str,
        method_name: str,
        method_args: Dict[str, Any] = None,
        error_factory: Callable[[str], R] = None,
    ) -> R:
        """
        Generic method to execute operations on a specific server.

        Args:
            server_name: Name of the server to execute the operation on
            operation_type: Type of operation (for logging) e.g., "tool", "prompt"
            operation_name: Name of the specific operation being called (for logging)
            method_name: Name of the method to call on the client session
            method_args: Arguments to pass to the method
            error_factory: Function to create an error return value if the operation fails

        Returns:
            Result from the operation or an error result
        """

        async def try_execute(client: ClientSession):
            try:
                method = getattr(client, method_name)
                return await method(**method_args)
            except Exception as e:
                error_msg = (
                    f"Failed to {method_name} '{operation_name}' on server '{server_name}': {e}"
                )
                logger.error(error_msg)
                if error_factory:
                    return error_factory(error_msg)
                else:
                    # Re-raise the original exception to propagate it
                    raise e

        if self.connection_persistence:
            server_connection = await self._persistent_connection_manager.get_server(
                server_name, client_session_factory=MCPAgentClientSession
            )
            return await try_execute(server_connection.session)
        else:
            logger.debug(
                f"Creating temporary connection to server: {server_name}",
                data={
                    "progress_action": ProgressAction.STARTING,
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                },
            )
            async with gen_client(
                server_name, server_registry=self.context.server_registry
            ) as client:
                result = await try_execute(client)
                logger.debug(
                    f"Closing temporary connection to server: {server_name}",
                    data={
                        "progress_action": ProgressAction.SHUTDOWN,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )
                return result

    async def _parse_resource_name(self, name: str, resource_type: str) -> tuple[str, str]:
        """
        Parse a possibly namespaced resource name into server name and local resource name.

        Args:
            name: The resource name, possibly namespaced
            resource_type: Type of resource (for error messages), e.g. "tool", "prompt"

        Returns:
            Tuple of (server_name, local_resource_name)
        """
        # First, check if this is a direct hit in our namespaced tool map
        # This handles both namespaced and non-namespaced direct lookups
        if resource_type == "tool" and name in self._namespaced_tool_map:
            namespaced_tool = self._namespaced_tool_map[name]
            return namespaced_tool.server_name, namespaced_tool.tool.name

        # Next, attempt to interpret as a namespaced name
        if is_namespaced_name(name):
            parts = name.split(SEP, 1)
            server_name, local_name = parts[0], parts[1]

            # Validate that the parsed server actually exists
            if server_name in self.server_names:
                return server_name, local_name

            # If the server name doesn't exist, it might be a tool with a hyphen in its name
            # Fall through to the next checks

        # For tools, search all servers for the tool by exact name match
        if resource_type == "tool":
            for server_name, tools in self._server_to_tool_map.items():
                for namespaced_tool in tools:
                    if namespaced_tool.tool.name == name:
                        return server_name, name

        # For all other resource types, use the first server
        return (self.server_names[0] if self.server_names else None, name)

    async def call_tool(self, name: str, arguments: dict | None = None) -> CallToolResult:
        """
        Call a namespaced tool, e.g., 'server_name-tool_name'.
        """
        if not self.initialized:
            await self.load_servers()

        # Use the common parser to get server and tool name
        server_name, local_tool_name = await self._parse_resource_name(name, "tool")

        if server_name is None:
            logger.error(f"Error: Tool '{name}' not found")
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Tool '{name}' not found")],
            )

        logger.info(
            "Requesting tool call",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "tool_name": local_tool_name,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"MCP Tool: {server_name}/{local_tool_name}"):
            trace.get_current_span().set_attribute("tool_name", local_tool_name)
            trace.get_current_span().set_attribute("server_name", server_name)
            return await self._execute_on_server(
                server_name=server_name,
                operation_type="tool",
                operation_name=local_tool_name,
                method_name="call_tool",
                method_args={
                    "name": local_tool_name,
                    "arguments": arguments,
                },
                error_factory=lambda msg: CallToolResult(
                    isError=True, content=[TextContent(type="text", text=msg)]
                ),
            )

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
        server_name: str | None = None,
    ) -> GetPromptResult:
        """
        Get a prompt from a server.

        :param prompt_name: Name of the prompt, optionally namespaced with server name
                           using the format 'server_name-prompt_name'
        :param arguments: Optional dictionary of string arguments to pass to the prompt template
                         for templating
        :param server_name: Optional name of the server to get the prompt from. If not provided
                          and prompt_name is not namespaced, will search all servers.
        :return: GetPromptResult containing the prompt description and messages
                 with a namespaced_name property for display purposes
        """
        if not self.initialized:
            await self.load_servers()

        # If server_name is explicitly provided, use it
        if server_name:
            local_prompt_name = prompt_name
        # Otherwise, check if prompt_name is namespaced and validate the server exists
        elif is_namespaced_name(prompt_name):
            parts = prompt_name.split(SEP, 1)
            potential_server = parts[0]

            # Only treat as namespaced if the server part is valid
            if potential_server in self.server_names:
                server_name = potential_server
                local_prompt_name = parts[1]
            else:
                # The hyphen is part of the prompt name, not a namespace separator
                local_prompt_name = prompt_name
        # Otherwise, use prompt_name as-is for searching
        else:
            local_prompt_name = prompt_name
            # We'll search all servers below

        # If we have a specific server to check
        if server_name:
            if not await self.validate_server(server_name):
                logger.error(f"Error: Server '{server_name}' not found")
                return GetPromptResult(
                    description=f"Error: Server '{server_name}' not found",
                    messages=[],
                )

            # Check if server supports prompts
            if not await self.server_supports_feature(server_name, "prompts"):
                logger.debug(f"Server '{server_name}' does not support prompts")
                return GetPromptResult(
                    description=f"Server '{server_name}' does not support prompts",
                    messages=[],
                )

            # Check the prompt cache to avoid unnecessary errors
            if local_prompt_name:
                async with self._prompt_cache_lock:
                    if server_name in self._prompt_cache:
                        # Check if any prompt in the cache has this name
                        prompt_names = [prompt.name for prompt in self._prompt_cache[server_name]]
                        if local_prompt_name not in prompt_names:
                            logger.debug(
                                f"Prompt '{local_prompt_name}' not found in cache for server '{server_name}'"
                            )
                            return GetPromptResult(
                                description=f"Prompt '{local_prompt_name}' not found on server '{server_name}'",
                                messages=[],
                            )

            # Try to get the prompt from the specified server
            method_args = {"name": local_prompt_name} if local_prompt_name else {}
            if arguments:
                method_args["arguments"] = arguments

            result = await self._execute_on_server(
                server_name=server_name,
                operation_type="prompt",
                operation_name=local_prompt_name or "default",
                method_name="get_prompt",
                method_args=method_args,
                error_factory=lambda msg: GetPromptResult(description=msg, messages=[]),
            )

            # Add namespaced name and source server to the result
            if result and result.messages:
                result.namespaced_name = create_namespaced_name(server_name, local_prompt_name)

                # Store the arguments in the result for display purposes
                if arguments:
                    result.arguments = arguments

            return result

        # No specific server - use the cache to find servers that have this prompt
        logger.debug(f"Searching for prompt '{local_prompt_name}' using cache")

        # Find potential servers from the cache
        potential_servers = []
        async with self._prompt_cache_lock:
            for s_name, prompt_list in self._prompt_cache.items():
                prompt_names = [prompt.name for prompt in prompt_list]
                if local_prompt_name in prompt_names:
                    potential_servers.append(s_name)

        if potential_servers:
            logger.debug(
                f"Found prompt '{local_prompt_name}' in cache for servers: {potential_servers}"
            )

            # Try each server from the cache
            for s_name in potential_servers:
                # Check if this server supports prompts
                capabilities = await self.get_capabilities(s_name)
                if not capabilities or not capabilities.prompts:
                    logger.debug(f"Server '{s_name}' does not support prompts, skipping")
                    continue

                try:
                    method_args = {"name": local_prompt_name}
                    if arguments:
                        method_args["arguments"] = arguments

                    result = await self._execute_on_server(
                        server_name=s_name,
                        operation_type="prompt",
                        operation_name=local_prompt_name,
                        method_name="get_prompt",
                        method_args=method_args,
                        error_factory=lambda _: None,  # Return None instead of an error
                    )

                    # If we got a successful result with messages, return it
                    if result and result.messages:
                        logger.debug(
                            f"Successfully retrieved prompt '{local_prompt_name}' from server '{s_name}'"
                        )
                        # Add namespaced name using the actual server where found
                        result.namespaced_name = create_namespaced_name(s_name, local_prompt_name)

                        # Store the arguments in the result for display purposes
                        if arguments:
                            result.arguments = arguments

                        return result

                except Exception as e:
                    logger.debug(f"Error retrieving prompt from server '{s_name}': {e}")
        else:
            logger.debug(f"Prompt '{local_prompt_name}' not found in any server's cache")

            # If not in cache, perform a full search as fallback (cache might be outdated)
            # First identify servers that support prompts
            supported_servers = []
            for s_name in self.server_names:
                capabilities = await self.get_capabilities(s_name)
                if capabilities and capabilities.prompts:
                    supported_servers.append(s_name)
                else:
                    logger.debug(
                        f"Server '{s_name}' does not support prompts, skipping from fallback search"
                    )

            # Try all supported servers in order
            for s_name in supported_servers:
                try:
                    # Use a quiet approach - don't log errors if not found
                    method_args = {"name": local_prompt_name}
                    if arguments:
                        method_args["arguments"] = arguments

                    result = await self._execute_on_server(
                        server_name=s_name,
                        operation_type="prompt",
                        operation_name=local_prompt_name,
                        method_name="get_prompt",
                        method_args=method_args,
                        error_factory=lambda _: None,  # Return None instead of an error
                    )

                    # If we got a successful result with messages, return it
                    if result and result.messages:
                        logger.debug(
                            f"Found prompt '{local_prompt_name}' on server '{s_name}' (not in cache)"
                        )
                        # Add namespaced name using the actual server where found
                        result.namespaced_name = create_namespaced_name(s_name, local_prompt_name)

                        # Store the arguments in the result for display purposes
                        if arguments:
                            result.arguments = arguments

                        # Update the cache - need to fetch the prompt object to store in cache
                        try:
                            prompt_list_result = await self._execute_on_server(
                                server_name=s_name,
                                operation_type="prompts-list",
                                operation_name="",
                                method_name="list_prompts",
                                error_factory=lambda _: None,
                            )

                            prompts = getattr(prompt_list_result, "prompts", [])
                            matching_prompts = [p for p in prompts if p.name == local_prompt_name]
                            if matching_prompts:
                                async with self._prompt_cache_lock:
                                    if s_name not in self._prompt_cache:
                                        self._prompt_cache[s_name] = []
                                    # Add if not already in the cache
                                    prompt_names_in_cache = [
                                        p.name for p in self._prompt_cache[s_name]
                                    ]
                                    if local_prompt_name not in prompt_names_in_cache:
                                        self._prompt_cache[s_name].append(matching_prompts[0])
                        except Exception:
                            # Ignore errors when updating cache
                            pass

                        return result

                except Exception:
                    # Don't log errors during fallback search
                    pass

        # If we get here, we couldn't find the prompt on any server
        logger.info(f"Prompt '{local_prompt_name}' not found on any server")
        return GetPromptResult(
            description=f"Prompt '{local_prompt_name}' not found on any server",
            messages=[],
        )

    async def list_prompts(
        self, server_name: str | None = None, agent_name: str | None = None
    ) -> Mapping[str, List[Prompt]]:
        """
        List available prompts from one or all servers.

        :param server_name: Optional server name to list prompts from. If not provided,
                           lists prompts from all servers.
        :param agent_name: Optional agent name (ignored at this level, used by multi-agent apps)
        :return: Dictionary mapping server names to lists of Prompt objects
        """
        if not self.initialized:
            await self.load_servers()

        results: Dict[str, List[Prompt]] = {}

        # If specific server requested
        if server_name:
            if server_name not in self.server_names:
                logger.error(f"Server '{server_name}' not found")
                return results

            # Check cache first
            async with self._prompt_cache_lock:
                if server_name in self._prompt_cache:
                    results[server_name] = self._prompt_cache[server_name]
                    logger.debug(f"Returning cached prompts for server '{server_name}'")
                    return results

            # Check if server supports prompts
            capabilities = await self.get_capabilities(server_name)
            if not capabilities or not capabilities.prompts:
                logger.debug(f"Server '{server_name}' does not support prompts")
                results[server_name] = []
                return results

            # Fetch from server
            result = await self._execute_on_server(
                server_name=server_name,
                operation_type="prompts-list",
                operation_name="",
                method_name="list_prompts",
                error_factory=lambda _: None,
            )

            # Get prompts from result
            prompts = getattr(result, "prompts", [])

            # Update cache
            async with self._prompt_cache_lock:
                self._prompt_cache[server_name] = prompts

            results[server_name] = prompts
            return results

        # No specific server - check if we can use the cache for all servers
        async with self._prompt_cache_lock:
            if all(s_name in self._prompt_cache for s_name in self.server_names):
                for s_name, prompt_list in self._prompt_cache.items():
                    results[s_name] = prompt_list
                logger.debug("Returning cached prompts for all servers")
                return results

        # Identify servers that support prompts
        supported_servers = []
        for s_name in self.server_names:
            capabilities = await self.get_capabilities(s_name)
            if capabilities and capabilities.prompts:
                supported_servers.append(s_name)
            else:
                logger.debug(f"Server '{s_name}' does not support prompts, skipping")
                results[s_name] = []

        # Fetch prompts from supported servers
        for s_name in supported_servers:
            try:
                result = await self._execute_on_server(
                    server_name=s_name,
                    operation_type="prompts-list",
                    operation_name="",
                    method_name="list_prompts",
                    error_factory=lambda _: None,
                )

                prompts = getattr(result, "prompts", [])

                # Update cache and results
                async with self._prompt_cache_lock:
                    self._prompt_cache[s_name] = prompts

                results[s_name] = prompts
            except Exception as e:
                logger.debug(f"Error fetching prompts from {s_name}: {e}")
                results[s_name] = []

        logger.debug(f"Available prompts across servers: {results}")
        return results

    async def _handle_tool_list_changed(self, server_name: str) -> None:
        """
        Callback handler for ToolListChangedNotification.
        This will refresh the tools for the specified server.

        Args:
            server_name: The name of the server whose tools have changed
        """
        logger.info(f"Tool list changed for server '{server_name}', refreshing tools")

        # Refresh the tools for this server
        await self._refresh_server_tools(server_name)

    async def _refresh_server_tools(self, server_name: str) -> None:
        """
        Refresh the tools for a specific server.

        Args:
            server_name: The name of the server to refresh tools for
        """
        if not await self.validate_server(server_name):
            logger.error(f"Cannot refresh tools for unknown server '{server_name}'")
            return

        await self.display.show_tool_update(aggregator=self, updated_server=server_name)

        async with self._refresh_lock:
            try:
                # Fetch new tools from the server
                if self.connection_persistence:
                    # Create a factory function that will include our parameters
                    def create_session(read_stream, write_stream, read_timeout):
                        # Get agent name if available
                        agent_name: str | None = None

                        # Import here to avoid circular dependency
                        from mcp_agent.agents.base_agent import BaseAgent

                        if isinstance(self, BaseAgent):
                            agent_name = self.config.name
                            elicitation_handler = self.config.elicitation_handler

                        return MCPAgentClientSession(
                            read_stream,
                            write_stream,
                            read_timeout,
                            server_name=server_name,
                            agent_name=agent_name,
                            elicitation_handler=elicitation_handler,
                            tool_list_changed_callback=self._handle_tool_list_changed,
                        )

                    server_connection = await self._persistent_connection_manager.get_server(
                        server_name, client_session_factory=create_session
                    )
                    tools_result = await server_connection.session.list_tools()
                    new_tools = tools_result.tools or []
                else:
                    # Create a factory function for the client session
                    def create_session(read_stream, write_stream, read_timeout):
                        # Get agent name if available
                        agent_name: str | None = None

                        # Import here to avoid circular dependency
                        from mcp_agent.agents.base_agent import BaseAgent

                        if isinstance(self, BaseAgent):
                            agent_name = self.config.name
                            elicitation_handler = self.config.elicitation_handler

                        return MCPAgentClientSession(
                            read_stream,
                            write_stream,
                            read_timeout,
                            server_name=server_name,
                            agent_name=agent_name,
                            elicitation_handler=elicitation_handler,
                            tool_list_changed_callback=self._handle_tool_list_changed,
                        )

                    async with gen_client(
                        server_name,
                        server_registry=self.context.server_registry,
                        client_session_factory=create_session,
                    ) as client:
                        tools_result = await client.list_tools()
                        new_tools = tools_result.tools or []

                # Update tool maps
                async with self._tool_map_lock:
                    # Remove old tools for this server
                    old_tools = self._server_to_tool_map.get(server_name, [])
                    for old_tool in old_tools:
                        if old_tool.namespaced_tool_name in self._namespaced_tool_map:
                            del self._namespaced_tool_map[old_tool.namespaced_tool_name]

                    # Add new tools
                    self._server_to_tool_map[server_name] = []
                    for tool in new_tools:
                        namespaced_tool_name = create_namespaced_name(server_name, tool.name)
                        namespaced_tool = NamespacedTool(
                            tool=tool,
                            server_name=server_name,
                            namespaced_tool_name=namespaced_tool_name,
                        )

                        self._namespaced_tool_map[namespaced_tool_name] = namespaced_tool
                        self._server_to_tool_map[server_name].append(namespaced_tool)

                logger.info(
                    f"Successfully refreshed tools for server '{server_name}'",
                    data={
                        "progress_action": ProgressAction.UPDATED,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                        "tool_count": len(new_tools),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to refresh tools for server '{server_name}': {e}")

    async def get_resource(
        self, resource_uri: str, server_name: str | None = None
    ) -> ReadResourceResult:
        """
        Get a resource directly from an MCP server by URI.
        If server_name is None, will search all available servers.

        Args:
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from

        Returns:
            ReadResourceResult object containing the resource content

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        if not self.initialized:
            await self.load_servers()

        # If specific server requested, use only that server
        if server_name is not None:
            if server_name not in self.server_names:
                raise ValueError(f"Server '{server_name}' not found")

            # Get the resource from the specified server
            return await self._get_resource_from_server(server_name, resource_uri)

        # If no server specified, search all servers
        if not self.server_names:
            raise ValueError("No servers available to get resource from")

        # Try each server in order - simply attempt to get the resource
        for s_name in self.server_names:
            try:
                return await self._get_resource_from_server(s_name, resource_uri)
            except Exception:
                # Continue to next server if not found
                continue

        # If we reach here, we couldn't find the resource on any server
        raise ValueError(f"Resource '{resource_uri}' not found on any server")

    async def _get_resource_from_server(
        self, server_name: str, resource_uri: str
    ) -> ReadResourceResult:
        """
        Internal helper method to get a resource from a specific server.

        Args:
            server_name: Name of the server to get the resource from
            resource_uri: URI of the resource to retrieve

        Returns:
            ReadResourceResult containing the resource

        Raises:
            Exception: If the resource couldn't be found or other error occurs
        """
        logger.info(
            "Requesting resource",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "resource_uri": resource_uri,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )

        try:
            uri = AnyUrl(resource_uri)
        except Exception as e:
            raise ValueError(f"Invalid resource URI: {resource_uri}. Error: {e}")

        # Use the _execute_on_server method to call read_resource on the server
        result = await self._execute_on_server(
            server_name=server_name,
            operation_type="resource",
            operation_name=resource_uri,
            method_name="read_resource",
            method_args={"uri": uri},
            # Don't create ValueError, just return None on error so we can catch it
            #            error_factory=lambda _: None,
        )

        # If result is None, the resource was not found
        if result is None:
            raise ValueError(f"Resource '{resource_uri}' not found on server '{server_name}'")

        return result

    async def list_resources(self, server_name: str | None = None) -> Dict[str, List[str]]:
        """
        List available resources from one or all servers.

        Args:
            server_name: Optional server name to list resources from. If not provided,
                        lists resources from all servers.

        Returns:
            Dictionary mapping server names to lists of resource URIs
        """
        if not self.initialized:
            await self.load_servers()

        results: Dict[str, List[str]] = {}

        # Get the list of servers to check
        servers_to_check = [server_name] if server_name else self.server_names

        # For each server, try to list its resources
        for s_name in servers_to_check:
            if s_name not in self.server_names:
                logger.error(f"Server '{s_name}' not found")
                continue

            # Initialize empty list for this server
            results[s_name] = []

            try:
                # Use the _execute_on_server method to call list_resources on the server
                result = await self._execute_on_server(
                    server_name=s_name,
                    operation_type="resources-list",
                    operation_name="",
                    method_name="list_resources",
                    method_args={},  # Empty dictionary instead of None
                    # No error_factory to allow exceptions to propagate
                )

                # Get resources from result
                resources = getattr(result, "resources", [])
                results[s_name] = [str(r.uri) for r in resources]

            except Exception as e:
                logger.error(f"Error fetching resources from {s_name}: {e}")

        return results
