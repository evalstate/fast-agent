from asyncio import Lock, gather
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    TypeVar,
    cast,
)

from mcp import GetPromptResult, ReadResourceResult
from mcp.client.session import ClientSession
from mcp.shared.session import ProgressFnT
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Prompt,
    ServerCapabilities,
    TextContent,
    Tool,
)
from opentelemetry import trace
from pydantic import AnyUrl, BaseModel, ConfigDict, Field

from fast_agent.context_dependent import ContextDependent
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.common import SEP, create_namespaced_name, is_namespaced_name
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager
from fast_agent.mcp.skybridge import (
    SKYBRIDGE_MIME_TYPE,
    SkybridgeResourceConfig,
    SkybridgeServerConfig,
    SkybridgeToolConfig,
)
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler, ToolExecutionHandler
from fast_agent.mcp.transport_tracking import TransportSnapshot

if TYPE_CHECKING:
    from fast_agent.context import Context


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


@dataclass
class ServerStats:
    call_counts: Counter = field(default_factory=Counter)
    last_call_at: datetime | None = None
    last_error_at: datetime | None = None

    def record(self, operation_type: str, success: bool) -> None:
        self.call_counts[operation_type] += 1
        now = datetime.now(timezone.utc)
        self.last_call_at = now
        if not success:
            self.last_error_at = now


class ServerStatus(BaseModel):
    server_name: str
    implementation_name: str | None = None
    implementation_version: str | None = None
    server_capabilities: ServerCapabilities | None = None
    client_capabilities: Mapping[str, Any] | None = None
    client_info_name: str | None = None
    client_info_version: str | None = None
    transport: str | None = None
    is_connected: bool | None = None
    last_call_at: datetime | None = None
    last_error_at: datetime | None = None
    staleness_seconds: float | None = None
    call_counts: Dict[str, int] = Field(default_factory=dict)
    error_message: str | None = None
    instructions_available: bool | None = None
    instructions_enabled: bool | None = None
    instructions_included: bool | None = None
    roots_configured: bool | None = None
    roots_count: int | None = None
    elicitation_mode: str | None = None
    sampling_mode: str | None = None
    spoofing_enabled: bool | None = None
    session_id: str | None = None
    transport_channels: TransportSnapshot | None = None
    skybridge: SkybridgeServerConfig | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
            context = self.context
            if not hasattr(context, "_connection_manager") or context._connection_manager is None:
                server_registry = context.server_registry
                if server_registry is None:
                    raise RuntimeError("Context is missing server registry for MCP connections")
                manager = MCPConnectionManager(server_registry, context=context)
                await manager.__aenter__()
                context._connection_manager = manager
            self._persistent_connection_manager = cast(
                "MCPConnectionManager", context._connection_manager
            )

        # Import the display component here to avoid circular imports
        from fast_agent.ui.console_display import ConsoleDisplay

        # Initialize the display component
        self.display = ConsoleDisplay(config=self.context.config)

        await self.load_servers()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __init__(
        self,
        server_names: List[str],
        connection_persistence: bool = True,
        context: Optional["Context"] = None,
        name: str | None = None,
        config: Optional[Any] = None,  # Accept the agent config for elicitation_handler access
        tool_handler: ToolExecutionHandler | None = None,
        **kwargs,
    ) -> None:
        """
        :param server_names: A list of server names to connect to.
        :param connection_persistence: Whether to maintain persistent connections to servers (default: True).
        :param config: Optional agent config containing elicitation_handler and other settings.
        :param tool_handler: Optional handler for tool execution lifecycle events (e.g., for ACP notifications).
        Note: The server names must be resolvable by the gen_client function, and specified in the server registry.
        """
        super().__init__(
            context=context,
            **kwargs,
        )

        self.server_names = server_names
        self.connection_persistence = connection_persistence
        self.agent_name = name
        self.config = config  # Store the config for access in session factory

        # Store tool execution handler for integration with ACP or other protocols
        # Default to NoOpToolExecutionHandler if none provided
        self._tool_handler = tool_handler or NoOpToolExecutionHandler()

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

        # Track runtime stats per server
        self._server_stats: Dict[str, ServerStats] = {}
        self._stats_lock = Lock()

        # Track discovered Skybridge configurations per server
        self._skybridge_configs: Dict[str, SkybridgeServerConfig] = {}

    def _create_progress_callback(
        self, server_name: str, tool_name: str, tool_call_id: str
    ) -> "ProgressFnT":
        """Create a progress callback function for tool execution."""

        async def progress_callback(
            progress: float, total: float | None, message: str | None
        ) -> None:
            """Handle progress notifications from MCP tool execution."""
            logger.info(
                "Tool progress update",
                data={
                    "progress_action": ProgressAction.TOOL_PROGRESS,
                    "tool_name": tool_name,
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                    "progress": progress,
                    "total": total,
                    "details": message or "",  # Put the message in details column
                },
            )

            # Forward progress to tool handler (e.g., for ACP notifications)
            try:
                await self._tool_handler.on_tool_progress(tool_call_id, progress, total, message)
            except Exception as e:
                logger.error(f"Error in tool progress handler: {e}", exc_info=True)

        return progress_callback

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

    def _create_session_factory(self, server_name: str):
        """
        Create a session factory function for the given server.
        This centralizes the logic for creating MCPAgentClientSession instances.

        Args:
            server_name: The name of the server to create a session for

        Returns:
            A factory function that creates MCPAgentClientSession instances
        """

        def session_factory(read_stream, write_stream, read_timeout, **kwargs):
            # Get agent's model and name from config if available
            agent_model: str | None = None
            agent_name: str | None = None
            elicitation_handler = None
            api_key: str | None = None

            # Access config directly if it was passed from BaseAgent
            if self.config:
                agent_model = self.config.model
                agent_name = self.config.name
                elicitation_handler = self.config.elicitation_handler
                api_key = self.config.api_key

            return MCPAgentClientSession(
                read_stream,
                write_stream,
                read_timeout,
                server_name=server_name,
                agent_model=agent_model,
                agent_name=agent_name,
                api_key=api_key,
                elicitation_handler=elicitation_handler,
                tool_list_changed_callback=self._handle_tool_list_changed,
                **kwargs,  # Pass through any additional kwargs like server_config
            )

        return session_factory

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

        self._skybridge_configs.clear()

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

                await self._persistent_connection_manager.get_server(
                    server_name, client_session_factory=self._create_session_factory(server_name)
                )

                # Record the initialize call that happened during connection setup
                await self._record_server_call(server_name, "initialize", True)

            logger.info(
                f"MCP Servers initialized for agent '{self.agent_name}'",
                data={
                    "progress_action": ProgressAction.INITIALIZED,
                    "agent_name": self.agent_name,
                },
            )

        async def fetch_tools(server_name: str) -> List[Tool]:
            # Only fetch tools if the server supports them
            if not await self.server_supports_feature(server_name, "tools"):
                logger.debug(f"Server '{server_name}' does not support tools")
                return []

            try:
                result: ListToolsResult = await self._execute_on_server(
                    server_name=server_name,
                    operation_type="tools/list",
                    operation_name="",
                    method_name="list_tools",
                    method_args={},
                )
                return result.tools or []
            except Exception as e:
                logger.error(f"Error loading tools from server '{server_name}'", data=e)
                return []

        async def fetch_prompts(server_name: str) -> List[Prompt]:
            # Only fetch prompts if the server supports them
            if not await self.server_supports_feature(server_name, "prompts"):
                logger.debug(f"Server '{server_name}' does not support prompts")
                return []

            try:
                result = await self._execute_on_server(
                    server_name=server_name,
                    operation_type="prompts/list",
                    operation_name="",
                    method_name="list_prompts",
                    method_args={},
                )
                return getattr(result, "prompts", [])
            except Exception as e:
                logger.debug(f"Error loading prompts from server '{server_name}': {e}")
                return []

        async def load_server_data(server_name: str):
            tools: List[Tool] = []
            prompts: List[Prompt] = []

            # Use _execute_on_server for consistent tracking regardless of connection mode
            tools = await fetch_tools(server_name)
            prompts = await fetch_prompts(server_name)

            return server_name, tools, prompts

        # Gather data from all servers concurrently
        results = await gather(
            *(load_server_data(server_name) for server_name in self.server_names),
            return_exceptions=True,
        )

        total_tool_count = 0
        total_prompt_count = 0

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

            total_tool_count += len(tools)

            # Process prompts
            async with self._prompt_cache_lock:
                self._prompt_cache[server_name] = prompts

            total_prompt_count += len(prompts)

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

        await self._initialize_skybridge_configs()

        self._display_startup_state(total_tool_count, total_prompt_count)

        self.initialized = True

    async def _initialize_skybridge_configs(self) -> None:
        """Discover Skybridge resources across servers."""
        if not self.server_names:
            return

        tasks = [
            self._evaluate_skybridge_for_server(server_name) for server_name in self.server_names
        ]
        results = await gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                logger.debug("Skybridge discovery failed: %s", str(result))
                continue

            server_name, config = result
            self._skybridge_configs[server_name] = config

    async def _evaluate_skybridge_for_server(
        self, server_name: str
    ) -> tuple[str, SkybridgeServerConfig]:
        """Inspect a single server for Skybridge-compatible resources."""
        config = SkybridgeServerConfig(server_name=server_name)

        tool_entries = self._server_to_tool_map.get(server_name, [])
        tool_configs: List[SkybridgeToolConfig] = []

        for namespaced_tool in tool_entries:
            tool_meta = getattr(namespaced_tool.tool, "meta", None) or {}
            template_value = tool_meta.get("openai/outputTemplate")
            if not template_value:
                continue

            try:
                template_uri = AnyUrl(template_value)
            except Exception as exc:
                warning = (
                    f"Tool '{namespaced_tool.namespaced_tool_name}' outputTemplate "
                    f"'{template_value}' is invalid: {exc}"
                )
                config.warnings.append(warning)
                logger.error(warning)
                tool_configs.append(
                    SkybridgeToolConfig(
                        tool_name=namespaced_tool.tool.name,
                        namespaced_tool_name=namespaced_tool.namespaced_tool_name,
                        warning=warning,
                    )
                )
                continue

            tool_configs.append(
                SkybridgeToolConfig(
                    tool_name=namespaced_tool.tool.name,
                    namespaced_tool_name=namespaced_tool.namespaced_tool_name,
                    template_uri=template_uri,
                )
            )

        raw_resources_capability = await self.server_supports_feature(server_name, "resources")
        supports_resources = bool(raw_resources_capability)
        config.supports_resources = supports_resources
        config.tools = tool_configs

        if not supports_resources:
            return server_name, config

        try:
            resources = await self._list_resources_from_server(server_name, check_support=False)
        except Exception as exc:  # noqa: BLE001 - logging and surfacing gracefully
            config.warnings.append(f"Failed to list resources: {exc}")
            return server_name, config

        for resource_entry in resources:
            uri = getattr(resource_entry, "uri", None)
            if not uri:
                continue

            uri_str = str(uri)
            if not uri_str.startswith("ui://"):
                continue

            try:
                uri_value = AnyUrl(uri_str)
            except Exception as exc:  # noqa: BLE001
                warning = f"Ignoring Skybridge candidate '{uri_str}': invalid URI ({exc})"
                config.warnings.append(warning)
                logger.debug(warning)
                continue

            sky_resource = SkybridgeResourceConfig(uri=uri_value)
            config.ui_resources.append(sky_resource)

            try:
                read_result: ReadResourceResult = await self._get_resource_from_server(
                    server_name, uri_str
                )
            except Exception as exc:  # noqa: BLE001
                warning = f"Failed to read resource '{uri_str}': {exc}"
                sky_resource.warning = warning
                config.warnings.append(warning)
                continue

            contents = getattr(read_result, "contents", []) or []
            seen_mime_types: List[str] = []

            for content in contents:
                mime_type = getattr(content, "mimeType", None)
                if mime_type:
                    seen_mime_types.append(mime_type)
                if mime_type == SKYBRIDGE_MIME_TYPE:
                    sky_resource.mime_type = mime_type
                    sky_resource.is_skybridge = True
                    break

            if sky_resource.mime_type is None and seen_mime_types:
                sky_resource.mime_type = seen_mime_types[0]

            if not sky_resource.is_skybridge:
                observed_type = sky_resource.mime_type or "unknown MIME type"
                warning = (
                    f"served as '{observed_type}' instead of '{SKYBRIDGE_MIME_TYPE}'"
                )
                sky_resource.warning = warning
                config.warnings.append(f"{uri_str}: {warning}")

        resource_lookup = {str(resource.uri): resource for resource in config.ui_resources}
        for tool_config in tool_configs:
            if tool_config.template_uri is None:
                continue

            resource_match = resource_lookup.get(str(tool_config.template_uri))
            if not resource_match:
                warning = (
                    f"Tool '{tool_config.namespaced_tool_name}' references missing "
                    f"Skybridge resource '{tool_config.template_uri}'"
                )
                tool_config.warning = warning
                config.warnings.append(warning)
                logger.error(warning)
                continue

            tool_config.resource_uri = resource_match.uri
            tool_config.is_valid = resource_match.is_skybridge

            if not resource_match.is_skybridge:
                warning = (
                    f"Tool '{tool_config.namespaced_tool_name}' references resource "
                    f"'{resource_match.uri}' served as '{resource_match.mime_type or 'unknown'}' "
                    f"instead of '{SKYBRIDGE_MIME_TYPE}'"
                )
                tool_config.warning = warning
                config.warnings.append(warning)
                logger.warning(warning)

        config.tools = tool_configs

        valid_tool_count = sum(1 for tool in tool_configs if tool.is_valid)
        if config.enabled and valid_tool_count == 0:
            warning = (
                f"Skybridge resources detected on server '{server_name}' but no tools expose them"
            )
            config.warnings.append(warning)
            logger.warning(warning)

        return server_name, config

    def _display_startup_state(self, total_tool_count: int, total_prompt_count: int) -> None:
        """Display startup summary and Skybridge status information."""
        # In interactive contexts the UI helper will render both the agent summary and the
        # Skybridge status. For non-interactive contexts, the warnings collected during
        # discovery are emitted through the logger, so we don't need to duplicate output here.
        if not self._skybridge_configs:
            return

        logger.debug(
            "Skybridge discovery completed",
            data={
                "agent_name": self.agent_name,
                "server_count": len(self._skybridge_configs),
            },
        )

    async def get_capabilities(self, server_name: str):
        """Get server capabilities if available."""
        if not self.connection_persistence:
            # For non-persistent connections, we can't easily check capabilities
            return None

        try:
            server_conn = await self._persistent_connection_manager.get_server(
                server_name, client_session_factory=self._create_session_factory(server_name)
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

        feature_value = getattr(capabilities, feature, False)
        if isinstance(feature_value, bool):
            return feature_value
        if feature_value is None:
            return False
        try:
            return bool(feature_value)
        except Exception:  # noqa: BLE001
            return True

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

        tools: List[Tool] = []

        for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items():
            tool_copy = namespaced_tool.tool.model_copy(
                deep=True, update={"name": namespaced_tool_name}
            )
            skybridge_config = self._skybridge_configs.get(namespaced_tool.server_name)
            if skybridge_config:
                matching_tool = next(
                    (
                        tool
                        for tool in skybridge_config.tools
                        if tool.namespaced_tool_name == namespaced_tool_name and tool.is_valid
                    ),
                    None,
                )
                if matching_tool:
                    meta = dict(tool_copy.meta or {})
                    meta["openai/skybridgeEnabled"] = True
                    meta["openai/skybridgeTemplate"] = str(matching_tool.template_uri)
                    tool_copy.meta = meta
            tools.append(tool_copy)

        return ListToolsResult(tools=tools)

    async def refresh_all_tools(self) -> None:
        """
        Refresh the tools for all servers.
        This is useful when you know tools have changed but haven't received notifications.
        """
        logger.info("Refreshing tools for all servers")
        for server_name in self.server_names:
            await self._refresh_server_tools(server_name)

    async def _record_server_call(
        self, server_name: str, operation_type: str, success: bool
    ) -> None:
        async with self._stats_lock:
            stats = self._server_stats.setdefault(server_name, ServerStats())
            stats.record(operation_type, success)

            # For stdio servers, also emit synthetic transport events to create activity timeline
            await self._notify_stdio_transport_activity(server_name, operation_type, success)

    async def _notify_stdio_transport_activity(
        self, server_name: str, operation_type: str, success: bool
    ) -> None:
        """Notify transport metrics of activity for stdio servers to create activity timeline."""
        if not self._persistent_connection_manager:
            return

        try:
            # Get the server connection and check if it's stdio transport
            server_conn = self._persistent_connection_manager.running_servers.get(server_name)
            if not server_conn:
                return

            server_config = getattr(server_conn, "server_config", None)
            if not server_config or server_config.transport != "stdio":
                return

            # Get transport metrics and emit synthetic message event
            transport_metrics = getattr(server_conn, "transport_metrics", None)
            if transport_metrics:
                # Import here to avoid circular imports
                from fast_agent.mcp.transport_tracking import ChannelEvent

                # Create a synthetic message event to represent the MCP operation
                event = ChannelEvent(
                    channel="stdio",
                    event_type="message",
                    detail=f"{operation_type} ({'success' if success else 'error'})",
                )
                transport_metrics.record_event(event)
        except Exception:
            # Don't let transport tracking errors break normal operation
            logger.debug(
                "Failed to notify stdio transport activity for %s", server_name, exc_info=True
            )

    async def get_server_instructions(self) -> Dict[str, tuple[str, List[str]]]:
        """
        Get instructions from all connected servers along with their tool names.

        Returns:
            Dict mapping server name to tuple of (instructions, list of tool names)
        """
        instructions = {}

        if self.connection_persistence and hasattr(self, "_persistent_connection_manager"):
            # Get instructions from persistent connections
            for server_name in self.server_names:
                try:
                    server_conn = await self._persistent_connection_manager.get_server(
                        server_name,
                        client_session_factory=self._create_session_factory(server_name),
                    )
                    # Always include server, even if no instructions
                    # Get tool names for this server
                    tool_names = [
                        namespaced_tool.tool.name
                        for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items()
                        if namespaced_tool.server_name == server_name
                    ]
                    # Include server even if instructions is None
                    instructions[server_name] = (server_conn.server_instructions, tool_names)
                except Exception as e:
                    logger.debug(f"Failed to get instructions from server {server_name}: {e}")

        return instructions

    async def collect_server_status(self) -> Dict[str, ServerStatus]:
        """Return aggregated status information for each configured server."""
        if not self.initialized:
            await self.load_servers()

        now = datetime.now(timezone.utc)
        status_map: Dict[str, ServerStatus] = {}

        for server_name in self.server_names:
            stats = self._server_stats.get(server_name)
            last_call = stats.last_call_at if stats else None
            last_error = stats.last_error_at if stats else None
            staleness = (now - last_call).total_seconds() if last_call else None
            call_counts = dict(stats.call_counts) if stats else {}

            implementation_name = None
            implementation_version = None
            capabilities: ServerCapabilities | None = None
            client_capabilities: Mapping[str, Any] | None = None
            client_info_name = None
            client_info_version = None
            is_connected = None
            error_message = None
            instructions_available = None
            instructions_enabled = None
            instructions_included = None
            roots_configured = None
            roots_count = None
            elicitation_mode = None
            sampling_mode = None
            spoofing_enabled = None
            server_cfg = None
            session_id = None
            server_conn = None
            transport: str | None = None
            transport_snapshot: TransportSnapshot | None = None

            manager = getattr(self, "_persistent_connection_manager", None)
            if self.connection_persistence and manager is not None:
                try:
                    server_conn = await manager.get_server(
                        server_name,
                        client_session_factory=self._create_session_factory(server_name),
                    )
                    implementation = getattr(server_conn, "server_implementation", None)
                    if implementation:
                        implementation_name = getattr(implementation, "name", None)
                        implementation_version = getattr(implementation, "version", None)
                    capabilities = getattr(server_conn, "server_capabilities", None)
                    client_capabilities = getattr(server_conn, "client_capabilities", None)
                    session = server_conn.session
                    client_info = getattr(session, "client_info", None) if session else None
                    if client_info:
                        client_info_name = getattr(client_info, "name", None)
                        client_info_version = getattr(client_info, "version", None)
                    is_connected = server_conn.is_healthy()
                    error_message = getattr(server_conn, "_error_message", None)
                    instructions_available = getattr(
                        server_conn, "server_instructions_available", None
                    )
                    instructions_enabled = getattr(server_conn, "server_instructions_enabled", None)
                    instructions_included = bool(getattr(server_conn, "server_instructions", None))
                    server_cfg = getattr(server_conn, "server_config", None)
                    if session:
                        elicitation_mode = getattr(
                            session, "effective_elicitation_mode", elicitation_mode
                        )
                        session_id = getattr(server_conn, "session_id", None)
                        if not session_id and getattr(server_conn, "_get_session_id_cb", None):
                            try:
                                session_id = server_conn._get_session_id_cb()  # type: ignore[attr-defined]
                            except Exception:
                                session_id = None
                    metrics = getattr(server_conn, "transport_metrics", None)
                    if metrics is not None:
                        try:
                            transport_snapshot = metrics.snapshot()
                        except Exception:
                            logger.debug(
                                "Failed to snapshot transport metrics for server '%s'",
                                server_name,
                                exc_info=True,
                            )
                except Exception as exc:
                    logger.debug(
                        f"Failed to collect status for server '{server_name}'",
                        data={"error": str(exc)},
                    )

            if (
                server_cfg is None
                and self.context
                and getattr(self.context, "server_registry", None)
            ):
                try:
                    server_cfg = self.context.server_registry.get_server_config(server_name)
                except Exception:
                    server_cfg = None

            if server_cfg is not None:
                instructions_enabled = (
                    instructions_enabled
                    if instructions_enabled is not None
                    else server_cfg.include_instructions
                )
                roots = getattr(server_cfg, "roots", None)
                roots_configured = bool(roots)
                roots_count = len(roots) if roots else 0
                transport = getattr(server_cfg, "transport", transport)
                elicitation = getattr(server_cfg, "elicitation", None)
                elicitation_mode = (
                    getattr(elicitation, "mode", None) if elicitation else elicitation_mode
                )
                sampling_cfg = getattr(server_cfg, "sampling", None)
                spoofing_enabled = bool(getattr(server_cfg, "implementation", None))
                if implementation_name is None and getattr(server_cfg, "implementation", None):
                    implementation_name = server_cfg.implementation.name
                    implementation_version = getattr(server_cfg.implementation, "version", None)
                if session_id is None:
                    if server_cfg.transport == "stdio":
                        session_id = "local"
                    elif server_conn and getattr(server_conn, "_get_session_id_cb", None):
                        try:
                            session_id = server_conn._get_session_id_cb()  # type: ignore[attr-defined]
                        except Exception:
                            session_id = None

                if sampling_cfg is not None:
                    sampling_mode = "configured"
                else:
                    auto_sampling = True
                    if self.context and getattr(self.context, "config", None):
                        auto_sampling = getattr(self.context.config, "auto_sampling", True)
                    sampling_mode = "auto" if auto_sampling else "off"
            else:
                # Fall back to defaults when config missing
                auto_sampling = True
                if self.context and getattr(self.context, "config", None):
                    auto_sampling = getattr(self.context.config, "auto_sampling", True)
                sampling_mode = sampling_mode or ("auto" if auto_sampling else "off")

            status_map[server_name] = ServerStatus(
                server_name=server_name,
                implementation_name=implementation_name,
                implementation_version=implementation_version,
                server_capabilities=capabilities,
                client_capabilities=client_capabilities,
                client_info_name=client_info_name,
                client_info_version=client_info_version,
                transport=transport,
                is_connected=is_connected,
                last_call_at=last_call,
                last_error_at=last_error,
                staleness_seconds=staleness,
                call_counts=call_counts,
                error_message=error_message,
                instructions_available=instructions_available,
                instructions_enabled=instructions_enabled,
                instructions_included=instructions_included,
                roots_configured=roots_configured,
                roots_count=roots_count,
                elicitation_mode=elicitation_mode,
                sampling_mode=sampling_mode,
                spoofing_enabled=spoofing_enabled,
                session_id=session_id,
                transport_channels=transport_snapshot,
                skybridge=self._skybridge_configs.get(server_name),
            )

        return status_map

    async def get_skybridge_configs(self) -> Dict[str, SkybridgeServerConfig]:
        """Expose discovered Skybridge configurations keyed by server."""
        if not self.initialized:
            await self.load_servers()
        return dict(self._skybridge_configs)

    async def get_skybridge_config(self, server_name: str) -> SkybridgeServerConfig | None:
        """Return the Skybridge configuration for a specific server, loading if necessary."""
        if not self.initialized:
            await self.load_servers()
        return self._skybridge_configs.get(server_name)

    async def _execute_on_server(
        self,
        server_name: str,
        operation_type: str,
        operation_name: str,
        method_name: str,
        method_args: Dict[str, Any] = None,
        error_factory: Callable[[str], R] | None = None,
        progress_callback: ProgressFnT | None = None,
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
            progress_callback: Optional progress callback for operations that support it

        Returns:
            Result from the operation or an error result
        """

        async def try_execute(client: ClientSession):
            try:
                method = getattr(client, method_name)

                # Get metadata from context for tool, resource, and prompt calls
                metadata = None
                if method_name in ["call_tool", "read_resource", "get_prompt"]:
                    from fast_agent.llm.fastagent_llm import _mcp_metadata_var

                    metadata = _mcp_metadata_var.get()

                # Prepare kwargs
                kwargs = method_args or {}
                if metadata:
                    kwargs["_meta"] = metadata

                # For call_tool method, check if we need to add progress_callback
                if method_name == "call_tool" and progress_callback:
                    # The call_tool method signature includes progress_callback parameter
                    return await method(progress_callback=progress_callback, **kwargs)
                else:
                    return await method(**(kwargs or {}))
            except ConnectionError:
                # Let ConnectionError pass through for reconnection logic
                raise
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

        success_flag: bool | None = None
        result: R | None = None

        # Try initial execution
        try:
            if self.connection_persistence:
                server_connection = await self._persistent_connection_manager.get_server(
                    server_name, client_session_factory=self._create_session_factory(server_name)
                )
                result = await try_execute(server_connection.session)
                success_flag = True
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
                    success_flag = True
        except ConnectionError:
            # Server offline - attempt reconnection
            from fast_agent.ui import console

            console.console.print(
                f"[dim yellow]MCP server {server_name} reconnecting...[/dim yellow]"
            )

            try:
                if self.connection_persistence:
                    # Force disconnect and create fresh connection
                    await self._persistent_connection_manager.disconnect_server(server_name)
                    import asyncio

                    await asyncio.sleep(0.1)

                    server_connection = await self._persistent_connection_manager.get_server(
                        server_name,
                        client_session_factory=self._create_session_factory(server_name),
                    )
                    result = await try_execute(server_connection.session)
                else:
                    # For non-persistent connections, just try again
                    async with gen_client(
                        server_name, server_registry=self.context.server_registry
                    ) as client:
                        result = await try_execute(client)

                # Success!
                console.console.print(f"[dim green]MCP server {server_name} online[/dim green]")
                success_flag = True

            except Exception:
                # Reconnection failed
                console.console.print(
                    f"[dim red]MCP server {server_name} offline - failed to reconnect[/dim red]"
                )
                error_msg = f"MCP server {server_name} offline - failed to reconnect"
                success_flag = False
                if error_factory:
                    result = error_factory(error_msg)
                else:
                    raise Exception(error_msg)
        except Exception:
            success_flag = False
            raise
        finally:
            if success_flag is not None:
                await self._record_server_call(server_name, operation_type, success_flag)

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
            # Try to match against known server names, handling server names with hyphens
            for server_name in self.server_names:
                if name.startswith(f"{server_name}{SEP}"):
                    local_name = name[len(server_name) + len(SEP) :]
                    return server_name, local_name

            # If no server name matched, it might be a tool with a hyphen in its name
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
        Call a namespaced tool, e.g., 'server_name__tool_name'.
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

        # Notify tool handler that execution is starting
        try:
            tool_call_id = await self._tool_handler.on_tool_start(
                local_tool_name, server_name, arguments
            )
        except Exception as e:
            logger.error(f"Error in tool start handler: {e}", exc_info=True)
            # Generate fallback ID if handler fails
            import uuid
            tool_call_id = str(uuid.uuid4())

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"MCP Tool: {server_name}/{local_tool_name}"):
            trace.get_current_span().set_attribute("tool_name", local_tool_name)
            trace.get_current_span().set_attribute("server_name", server_name)

            # Create progress callback for this tool execution
            progress_callback = self._create_progress_callback(
                server_name, local_tool_name, tool_call_id
            )

            try:
                result = await self._execute_on_server(
                    server_name=server_name,
                    operation_type="tools/call",
                    operation_name=local_tool_name,
                    method_name="call_tool",
                    method_args={
                        "name": local_tool_name,
                        "arguments": arguments,
                    },
                    error_factory=lambda msg: CallToolResult(
                        isError=True, content=[TextContent(type="text", text=msg)]
                    ),
                    progress_callback=progress_callback,
                )

                # Notify tool handler of completion
                try:
                    # Extract result text from the result
                    result_text = None
                    if result.content:
                        text_contents = [
                            c.text for c in result.content if hasattr(c, "text") and c.text
                        ]
                        if text_contents:
                            result_text = "\n".join(text_contents)

                    error_text = None
                    if result.isError and result_text:
                        error_text = result_text
                        result_text = None

                    await self._tool_handler.on_tool_complete(
                        tool_call_id, not result.isError, result_text, error_text
                    )
                except Exception as e:
                    logger.error(f"Error in tool complete handler: {e}", exc_info=True)

                return result

            except Exception as e:
                # Notify tool handler of error
                try:
                    await self._tool_handler.on_tool_complete(tool_call_id, False, None, str(e))
                except Exception as handler_error:
                    logger.error(f"Error in tool complete handler: {handler_error}", exc_info=True)
                raise

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
                operation_type="prompts/get",
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
                        operation_type="prompts/get",
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
                        operation_type="prompts/get",
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
                                operation_type="prompts/list",
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
                operation_type="prompts/list",
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
                    operation_type="prompts/list",
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

        # Check if server supports tools capability
        if not await self.server_supports_feature(server_name, "tools"):
            logger.debug(f"Server '{server_name}' does not support tools")
            return

        await self.display.show_tool_update(
            updated_server=server_name, agent_name="Tool List Change Notification"
        )

        async with self._refresh_lock:
            try:
                # Fetch new tools from the server using _execute_on_server to properly record stats
                tools_result = await self._execute_on_server(
                    server_name=server_name,
                    operation_type="tools/list",
                    operation_name="",
                    method_name="list_tools",
                    method_args={},
                )
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
        # Check if server supports resources capability
        if not await self.server_supports_feature(server_name, "resources"):
            raise ValueError(f"Server '{server_name}' does not support resources")

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
            operation_type="resources/read",
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

    async def _list_resources_from_server(
        self, server_name: str, *, check_support: bool = True
    ) -> List[Any]:
        """
        Internal helper method to list resources from a specific server.

        Args:
            server_name: Name of the server whose resources to list
            check_support: Whether to verify the server supports resources before listing

        Returns:
            A list of resources as returned by the MCP server
        """
        if check_support and not await self.server_supports_feature(server_name, "resources"):
            return []

        result = await self._execute_on_server(
            server_name=server_name,
            operation_type="resources/list",
            operation_name="",
            method_name="list_resources",
            method_args={},
        )

        return getattr(result, "resources", []) or []

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

            # Check if server supports resources capability
            if not await self.server_supports_feature(s_name, "resources"):
                logger.debug(f"Server '{s_name}' does not support resources")
                continue

            try:
                resources = await self._list_resources_from_server(s_name, check_support=False)
                formatted_resources: List[str] = []
                for resource in resources:
                    uri = getattr(resource, "uri", None)
                    if uri is not None:
                        formatted_resources.append(str(uri))
                results[s_name] = formatted_resources
            except Exception as e:
                logger.error(f"Error fetching resources from {s_name}: {e}")

        return results

    async def list_mcp_tools(self, server_name: str | None = None) -> Dict[str, List[Tool]]:
        """
        List available tools from one or all servers, grouped by server name.

        Args:
            server_name: Optional server name to list tools from. If not provided,
                        lists tools from all servers.

        Returns:
            Dictionary mapping server names to lists of Tool objects (with original names, not namespaced)
        """
        if not self.initialized:
            await self.load_servers()

        results: Dict[str, List[Tool]] = {}

        # Get the list of servers to check
        servers_to_check = [server_name] if server_name else self.server_names

        # For each server, try to list its tools
        for s_name in servers_to_check:
            if s_name not in self.server_names:
                logger.error(f"Server '{s_name}' not found")
                continue

            # Initialize empty list for this server
            results[s_name] = []

            # Check if server supports tools capability
            if not await self.server_supports_feature(s_name, "tools"):
                logger.debug(f"Server '{s_name}' does not support tools")
                continue

            try:
                # Use the _execute_on_server method to call list_tools on the server
                result = await self._execute_on_server(
                    server_name=s_name,
                    operation_type="tools/list",
                    operation_name="",
                    method_name="list_tools",
                    method_args={},
                )

                # Get tools from result (these have original names, not namespaced)
                tools = getattr(result, "tools", [])
                results[s_name] = tools

            except Exception as e:
                logger.error(f"Error fetching tools from {s_name}: {e}")

        return results
