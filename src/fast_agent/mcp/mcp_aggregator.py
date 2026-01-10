from asyncio import Lock
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    TypeVar,
    Union,
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
from fast_agent.core.exceptions import ServerSessionTerminatedError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.model_resolution import HARDCODED_DEFAULT_MODEL, resolve_model_spec
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.common import SEP, create_namespaced_name, is_namespaced_name
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.interfaces import ServerRegistryProtocol
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager
from fast_agent.mcp.skybridge import (
    SKYBRIDGE_MIME_TYPE,
    SkybridgeResourceConfig,
    SkybridgeServerConfig,
    SkybridgeToolConfig,
)
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler, ToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import NoOpToolPermissionHandler, ToolPermissionHandler
from fast_agent.mcp.transport_tracking import TransportSnapshot
from fast_agent.utils.async_utils import gather_with_cancel

if TYPE_CHECKING:
    from fast_agent.context import Context
    from fast_agent.mcp_server_registry import ServerRegistry


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
    reconnect_count: int = 0

    def record(self, operation_type: str, success: bool) -> None:
        self.call_counts[operation_type] += 1
        now = datetime.now(timezone.utc)
        self.last_call_at = now
        if not success:
            self.last_error_at = now

    def record_reconnect(self) -> None:
        """Record a successful reconnection."""
        self.reconnect_count += 1


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
    call_counts: dict[str, int] = Field(default_factory=dict)
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
    reconnect_count: int = 0
    ping_interval_seconds: int | None = None
    ping_max_missed: int | None = None
    ping_ok_count: int | None = None
    ping_fail_count: int | None = None
    ping_consecutive_failures: int | None = None
    ping_last_ok_at: datetime | None = None
    ping_last_fail_at: datetime | None = None
    ping_last_error: str | None = None
    ping_activity_buckets: list[str] | None = None
    ping_activity_bucket_seconds: int | None = None
    ping_activity_bucket_count: int | None = None

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

    server_names: list[str]
    """A list of server names to connect to."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    async def __aenter__(self):
        if self.initialized:
            return self

        # Keep a connection manager to manage persistent connections for this aggregator
        if self.connection_persistence:
            context = self._require_context()
            # Try to get existing connection manager from context
            if not hasattr(context, "_connection_manager") or context._connection_manager is None:
                server_registry = cast("ServerRegistry", self._require_server_registry())
                manager = MCPConnectionManager(server_registry, context=context)
                await manager.__aenter__()
                context._connection_manager = manager
                self._owns_connection_manager = True
            self._persistent_connection_manager = context._connection_manager
        else:
            self._persistent_connection_manager = None

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
        server_names: list[str],
        connection_persistence: bool = True,
        context: Union["Context", None] = None,
        name: str | None = None,
        config: Any | None = None,  # Accept the agent config for elicitation_handler access
        tool_handler: ToolExecutionHandler | None = None,
        permission_handler: ToolPermissionHandler | None = None,
        **kwargs,
    ) -> None:
        """
        :param server_names: A list of server names to connect to.
        :param connection_persistence: Whether to maintain persistent connections to servers (default: True).
        :param config: Optional agent config containing elicitation_handler and other settings.
        :param tool_handler: Optional handler for tool execution lifecycle events (e.g., for ACP notifications).
        :param permission_handler: Optional handler for tool permission checks (e.g., for ACP permissions).
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
        self._persistent_connection_manager: MCPConnectionManager | None = None
        self._owns_connection_manager = False

        # Store tool execution handler for integration with ACP or other protocols
        # Default to NoOpToolExecutionHandler if none provided
        self._tool_handler = tool_handler or NoOpToolExecutionHandler()

        # Store tool permission handler for ACP or other permission systems
        # Default to NoOpToolPermissionHandler if none provided (allows all)
        self._permission_handler = permission_handler or NoOpToolPermissionHandler()

        # Set up logger with agent name in namespace if available
        global logger
        logger_name = f"{__name__}.{name}" if name else __name__
        logger = get_logger(logger_name)

        # Maps namespaced_tool_name -> namespaced tool info
        self._namespaced_tool_map: dict[str, NamespacedTool] = {}
        # Maps server_name -> list of tools
        self._server_to_tool_map: dict[str, list[NamespacedTool]] = {}
        self._tool_map_lock = Lock()

        # Cache for prompt objects, maps server_name -> list of prompt objects
        self._prompt_cache: dict[str, list[Prompt]] = {}
        self._prompt_cache_lock = Lock()

        # Lock for refreshing tools from a server
        self._refresh_lock = Lock()

        # Track runtime stats per server
        self._server_stats: dict[str, ServerStats] = {}
        self._stats_lock = Lock()

        # Track discovered Skybridge configurations per server
        self._skybridge_configs: dict[str, SkybridgeServerConfig] = {}

    def _require_context(self) -> "Context":
        if self.context is None:
            raise RuntimeError("MCPAggregator requires a context")
        return self.context

    def _require_server_registry(self) -> ServerRegistryProtocol:
        context = self._require_context()
        server_registry = getattr(context, "server_registry", None)
        if server_registry is None:
            raise RuntimeError("Context is missing server registry for MCP connections")
        return server_registry

    def _require_connection_manager(self) -> MCPConnectionManager:
        if self._persistent_connection_manager is None:
            raise RuntimeError("Persistent connection manager is not initialized")
        return self._persistent_connection_manager

    def _create_progress_callback(
        self,
        server_name: str,
        tool_name: str,
        tool_call_id: str,
        request_tool_handler: ToolExecutionHandler | None = None,
    ) -> "ProgressFnT":
        """Create a progress callback function for tool execution."""
        handler_for_request = request_tool_handler or self._tool_handler

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
                await handler_for_request.on_tool_progress(
                    tool_call_id, progress, total, message
                )
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
                if self._owns_connection_manager and (
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
        server_names: list[str],
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
            raise

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
                cli_model_override = None
                if self.context and getattr(self.context, "config", None):
                    cli_model_override = getattr(
                        self.context.config, "cli_model_override", None
                    )
                agent_model, model_source = resolve_model_spec(
                    self.context,
                    model=self.config.model,
                    cli_model=cli_model_override,
                    hardcoded_default=HARDCODED_DEFAULT_MODEL,
                )
                if model_source:
                    logger.info(
                        f"Resolved MCP agent model '{agent_model}' via {model_source}",
                        model=agent_model,
                        source=model_source,
                    )
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

    async def load_servers(self, *, force_connect: bool = False) -> None:
        """
        Discover tools from each server in parallel and build an index of namespaced tool names.
        Also populate the prompt cache.

        Set force_connect=True to override load_on_start guards (e.g., when a user issues /connect).
        """
        if self.initialized and not force_connect:
            logger.debug("MCPAggregator already initialized.")
            return

        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()

        async with self._prompt_cache_lock:
            self._prompt_cache.clear()

        self._skybridge_configs.clear()

        servers_to_load: list[str] = []
        skipped_servers: list[str] = []

        for server_name in self.server_names:
            # Check if server should be loaded on start
            server_registry = self.context.server_registry if self.context else None
            if server_registry is not None:
                server_config = server_registry.get_server_config(server_name)
                if (
                    server_config
                    and not getattr(server_config, "load_on_start", True)
                    and not force_connect
                ):
                    logger.debug(f"Skipping server '{server_name}' - load_on_start=False")
                    skipped_servers.append(server_name)
                    continue

            servers_to_load.append(server_name)

            if self.connection_persistence:
                logger.info(
                    f"Creating persistent connection to server: {server_name}",
                    data={
                        "progress_action": ProgressAction.STARTING,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )

                manager = self._require_connection_manager()
                await manager.get_server(
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

        if skipped_servers:
            logger.debug(
                "Deferred MCP servers due to load_on_start=False",
                data={
                    "agent_name": self.agent_name,
                    "servers": skipped_servers,
                },
            )

        async def fetch_tools(server_name: str) -> list[Tool]:
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

        async def fetch_prompts(server_name: str) -> list[Prompt]:
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
            tools: list[Tool] = []
            prompts: list[Prompt] = []

            # Use _execute_on_server for consistent tracking regardless of connection mode
            tools = await fetch_tools(server_name)
            prompts = await fetch_prompts(server_name)

            return server_name, tools, prompts

        if not servers_to_load:
            self.initialized = True
            return

        # Gather data from all servers concurrently
        results = await gather_with_cancel(
            load_server_data(server_name) for server_name in servers_to_load
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

        await self._initialize_skybridge_configs(servers_to_load)

        self._display_startup_state(total_tool_count, total_prompt_count)

        self.initialized = True

    async def _initialize_skybridge_configs(self, server_names: list[str] | None = None) -> None:
        """Discover Skybridge resources across servers."""
        target_servers = server_names if server_names is not None else self.server_names
        if not target_servers:
            return

        tasks = [
            self._evaluate_skybridge_for_server(server_name) for server_name in target_servers
        ]
        results = await gather_with_cancel(tasks)

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
        tool_configs: list[SkybridgeToolConfig] = []

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
            seen_mime_types: list[str] = []

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
                warning = f"served as '{observed_type}' instead of '{SKYBRIDGE_MIME_TYPE}'"
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
            manager = self._require_connection_manager()
            server_conn = await manager.get_server(
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

    async def list_servers(self) -> list[str]:
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

        tools: list[Tool] = []

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

    async def _record_reconnect(self, server_name: str) -> None:
        """Record a successful server reconnection."""
        async with self._stats_lock:
            stats = self._server_stats.setdefault(server_name, ServerStats())
            stats.record_reconnect()

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

            server_config = server_conn.server_config
            if server_config.transport != "stdio":
                return

            # Get transport metrics and emit synthetic message event
            transport_metrics = server_conn.transport_metrics
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

    async def get_server_instructions(self) -> dict[str, tuple[str | None, list[str]]]:
        """
        Get instructions from currently-connected servers along with their tool names.

        Returns:
            Dict mapping server name to tuple of (instructions, list of tool names).

        Notes:
            This method must not implicitly connect to servers. Connection is controlled
            by `load_servers()` (and its `load_on_start` / `force_connect` behavior).
            This ensures optional MCP servers don't get launched just because an agent
            prompt contains the `{{serverInstructions}}` placeholder.
        """
        instructions: dict[str, tuple[str | None, list[str]]] = {}

        if not self.connection_persistence:
            return instructions

        manager = getattr(self, "_persistent_connection_manager", None)
        if manager is None:
            return instructions

        # Only read from already-running server connections to avoid implicit connects.
        running_servers = getattr(manager, "running_servers", {}) or {}
        for server_name in self.server_names:
            server_conn = running_servers.get(server_name)
            if not server_conn:
                continue

            try:
                if not server_conn.is_healthy():
                    continue
            except Exception:
                continue

            tool_names = [
                namespaced_tool.tool.name
                for _, namespaced_tool in self._namespaced_tool_map.items()
                if namespaced_tool.server_name == server_name
            ]

            try:
                instructions[server_name] = (server_conn.server_instructions, tool_names)
            except Exception as e:
                logger.debug(f"Failed to get instructions from server {server_name}: {e}")

        return instructions

    async def collect_server_status(self) -> dict[str, ServerStatus]:
        """Return aggregated status information for each configured server."""
        if not self.initialized:
            await self.load_servers()

        now = datetime.now(timezone.utc)
        status_map: dict[str, ServerStatus] = {}

        for server_name in self.server_names:
            stats = self._server_stats.get(server_name)
            last_call = stats.last_call_at if stats else None
            last_error = stats.last_error_at if stats else None
            staleness = (now - last_call).total_seconds() if last_call else None
            call_counts = dict(stats.call_counts) if stats else {}
            reconnect_count = stats.reconnect_count if stats else 0

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
            ping_interval_seconds: int | None = None
            ping_max_missed: int | None = None
            ping_ok_count: int | None = None
            ping_fail_count: int | None = None
            ping_consecutive_failures: int | None = None
            ping_last_ok_at: datetime | None = None
            ping_last_fail_at: datetime | None = None
            ping_last_error: str | None = None
            ping_activity_buckets: list[str] | None = None
            ping_activity_bucket_seconds: int | None = None
            ping_activity_bucket_count: int | None = None

            manager = getattr(self, "_persistent_connection_manager", None)
            if self.connection_persistence and manager is not None:
                try:
                    async with manager._lock:
                        server_conn = manager.running_servers.get(server_name)
                    if server_conn is None:
                        is_connected = False
                    else:
                        implementation = server_conn.server_implementation
                        if implementation is not None:
                            implementation_name = implementation.name
                            implementation_version = implementation.version
                        capabilities = server_conn.server_capabilities
                        client_capabilities = server_conn.client_capabilities
                        session = server_conn.session
                        client_info = getattr(session, "client_info", None) if session else None
                        if client_info:
                            client_info_name = getattr(client_info, "name", None)
                            client_info_version = getattr(client_info, "version", None)
                        if server_conn._initialized_event.is_set():
                            is_connected = server_conn.is_healthy()
                        else:
                            is_connected = False
                            error_message = error_message or "initializing..."
                        error_message = error_message or server_conn._error_message
                        instructions_available = server_conn.server_instructions_available
                        instructions_enabled = server_conn.server_instructions_enabled
                        instructions_included = bool(server_conn.server_instructions)
                        server_cfg = server_conn.server_config
                        ping_interval_seconds = server_cfg.ping_interval_seconds
                        ping_max_missed = server_cfg.max_missed_pings
                        ping_ok_count = server_conn._ping_ok_count
                        ping_fail_count = server_conn._ping_fail_count
                        ping_consecutive_failures = server_conn._ping_consecutive_failures
                        ping_last_ok_at = server_conn._ping_last_ok_at
                        ping_last_fail_at = server_conn._ping_last_fail_at
                        ping_last_error = server_conn._ping_last_error
                        if session:
                            elicitation_mode = session.effective_elicitation_mode
                            session_id = server_conn.session_id
                            if not session_id and server_conn._get_session_id_cb:
                                try:
                                    session_id = server_conn._get_session_id_cb()  # type: ignore[attr-defined]
                                except Exception:
                                    session_id = None
                        metrics = server_conn.transport_metrics
                        if metrics is not None:
                            try:
                                transport_snapshot = metrics.snapshot()
                            except Exception:
                                logger.debug(
                                    "Failed to snapshot transport metrics for server '%s'",
                                    server_name,
                                    exc_info=True,
                                )
                        bucket_seconds = (
                            transport_snapshot.activity_bucket_seconds
                            if transport_snapshot and transport_snapshot.activity_bucket_seconds
                            else 30
                        )
                        bucket_count = (
                            transport_snapshot.activity_bucket_count
                            if transport_snapshot and transport_snapshot.activity_bucket_count
                            else 20
                        )
                        ping_activity_buckets = server_conn.build_ping_activity_buckets(
                            bucket_seconds, bucket_count
                        )
                        ping_activity_bucket_seconds = bucket_seconds
                        ping_activity_bucket_count = bucket_count
                except Exception as exc:
                    logger.debug(
                        f"Failed to collect status for server '{server_name}'",
                        data={"error": str(exc)},
                    )

            if server_cfg is None:
                server_registry = self.context.server_registry if self.context else None
                if server_registry is not None:
                    try:
                        server_cfg = server_registry.get_server_config(server_name)
                    except Exception:
                        server_cfg = None

            if server_cfg is not None:
                instructions_enabled = (
                    instructions_enabled
                    if instructions_enabled is not None
                    else server_cfg.include_instructions
                )
                roots = server_cfg.roots
                roots_configured = bool(roots)
                roots_count = len(roots) if roots else 0
                transport = server_cfg.transport or transport
                elicitation = server_cfg.elicitation
                elicitation_mode = (
                    getattr(elicitation, "mode", None) if elicitation else elicitation_mode
                )
                ping_interval_seconds = ping_interval_seconds or server_cfg.ping_interval_seconds
                ping_max_missed = ping_max_missed or server_cfg.max_missed_pings
                sampling_cfg = server_cfg.sampling
                spoofing_enabled = server_cfg.implementation is not None
                if implementation_name is None and server_cfg.implementation is not None:
                    implementation_name = server_cfg.implementation.name
                    implementation_version = server_cfg.implementation.version
                if session_id is None:
                    if server_cfg.transport == "stdio":
                        session_id = "local"
                    elif server_conn and server_conn._get_session_id_cb:
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
                reconnect_count=reconnect_count,
                ping_interval_seconds=ping_interval_seconds,
                ping_max_missed=ping_max_missed,
                ping_ok_count=ping_ok_count,
                ping_fail_count=ping_fail_count,
                ping_consecutive_failures=ping_consecutive_failures,
                ping_last_ok_at=ping_last_ok_at,
                ping_last_fail_at=ping_last_fail_at,
                ping_last_error=ping_last_error,
                ping_activity_buckets=ping_activity_buckets,
                ping_activity_bucket_seconds=ping_activity_bucket_seconds,
                ping_activity_bucket_count=ping_activity_bucket_count,
            )

        return status_map

    async def get_skybridge_configs(self) -> dict[str, SkybridgeServerConfig]:
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
        method_args: dict[str, Any] | None = None,
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
            except ServerSessionTerminatedError:
                # Let ServerSessionTerminatedError pass through for reconnection logic
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
                manager = self._require_connection_manager()
                server_connection = await manager.get_server(
                    server_name, client_session_factory=self._create_session_factory(server_name)
                )
                session = server_connection.session
                if session is None:
                    raise RuntimeError(f"Server session not initialized for '{server_name}'")
                result = await try_execute(session)
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
                server_registry = self._require_server_registry()
                async with gen_client(
                    server_name, server_registry=server_registry
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
            result, success_flag = await self._handle_connection_error(
                server_name, try_execute, error_factory
            )
        except ServerSessionTerminatedError as exc:
            # Session terminated (e.g., 404 from restarted server)
            result, success_flag = await self._handle_session_terminated(
                server_name, try_execute, error_factory, exc
            )
        except Exception:
            success_flag = False
            raise
        finally:
            if success_flag is not None:
                await self._record_server_call(server_name, operation_type, success_flag)

        if result is None:
            error_msg = (
                f"Failed to {method_name} '{operation_name}' on server '{server_name}'"
            )
            if error_factory:
                return error_factory(error_msg)
            raise RuntimeError(error_msg)
        return result

    async def _handle_connection_error(
        self,
        server_name: str,
        try_execute: Callable,
        error_factory: Callable[[str], R] | None,
    ) -> tuple[R | None, bool]:
        """Handle ConnectionError by attempting to reconnect to the server."""
        from fast_agent.ui import console

        console.console.print(f"[dim yellow]MCP server {server_name} reconnecting...[/dim yellow]")

        try:
            if self.connection_persistence:
                # Force disconnect and create fresh connection
                manager = self._require_connection_manager()
                server_connection = await manager.reconnect_server(
                    server_name,
                    client_session_factory=self._create_session_factory(server_name),
                )
                session = server_connection.session
                if session is None:
                    raise RuntimeError(f"Server session not initialized for '{server_name}'")
                result = await try_execute(session)
            else:
                # For non-persistent connections, just try again
                server_registry = self._require_server_registry()
                async with gen_client(
                    server_name, server_registry=server_registry
                ) as client:
                    result = await try_execute(client)

            # Success!
            console.console.print(f"[dim green]MCP server {server_name} online[/dim green]")
            return result, True

        except ServerSessionTerminatedError:
            # After reconnecting for connection error, we got session terminated
            # Don't loop - just report the error
            console.console.print(
                f"[dim red]MCP server {server_name} session terminated after reconnect[/dim red]"
            )
            error_msg = (
                f"MCP server {server_name} reconnected but session was immediately terminated. "
                "Please check server status."
            )
            if error_factory:
                return error_factory(error_msg), False
            else:
                raise Exception(error_msg)

        except Exception as e:
            # Reconnection failed
            console.console.print(
                f"[dim red]MCP server {server_name} offline - failed to reconnect: {e}[/dim red]"
            )
            error_msg = f"MCP server {server_name} offline - failed to reconnect"
            if error_factory:
                return error_factory(error_msg), False
            else:
                raise Exception(error_msg)

    async def _handle_session_terminated(
        self,
        server_name: str,
        try_execute: Callable,
        error_factory: Callable[[str], R] | None,
        exc: ServerSessionTerminatedError,
    ) -> tuple[R | None, bool]:
        """Handle ServerSessionTerminatedError by attempting to reconnect if configured."""
        from fast_agent.ui import console

        # Check if reconnect_on_disconnect is enabled for this server
        server_config = None
        server_registry = self.context.server_registry if self.context else None
        if server_registry is not None:
            server_config = server_registry.get_server_config(server_name)

        reconnect_enabled = server_config and server_config.reconnect_on_disconnect

        if not reconnect_enabled:
            # Reconnection not enabled - inform user and fail
            console.console.print(
                f"[dim red]MCP server {server_name} session terminated (404)[/dim red]"
            )
            console.console.print(
                "[dim]Tip: Enable 'reconnect_on_disconnect: true' in config to auto-reconnect[/dim]"
            )
            error_msg = f"MCP server {server_name} session terminated - reconnection not enabled"
            if error_factory:
                return error_factory(error_msg), False
            else:
                raise exc

        # Attempt reconnection
        console.console.print(
            f"[dim yellow]MCP server {server_name} session terminated - reconnecting...[/dim yellow]"
        )

        try:
            if self.connection_persistence:
                manager = self._require_connection_manager()
                server_connection = await manager.reconnect_server(
                    server_name,
                    client_session_factory=self._create_session_factory(server_name),
                )
                session = server_connection.session
                if session is None:
                    raise RuntimeError(f"Server session not initialized for '{server_name}'")
                result = await try_execute(session)
            else:
                # For non-persistent connections, just try again
                server_registry = self._require_server_registry()
                async with gen_client(
                    server_name, server_registry=server_registry
                ) as client:
                    result = await try_execute(client)

            # Success! Record the reconnection
            await self._record_reconnect(server_name)
            console.console.print(
                f"[dim green]MCP server {server_name} reconnected successfully[/dim green]"
            )
            return result, True

        except ServerSessionTerminatedError:
            # Retry after reconnection ALSO failed with session terminated
            # Do NOT attempt another reconnection - this would cause an infinite loop
            console.console.print(
                f"[dim red]MCP server {server_name} session terminated again after reconnect[/dim red]"
            )
            error_msg = (
                f"MCP server {server_name} session terminated even after reconnection. "
                "The server may be persistently rejecting this session. "
                "Please check server status or try again later."
            )
            if error_factory:
                return error_factory(error_msg), False
            else:
                raise Exception(error_msg)

        except Exception as e:
            # Other reconnection failure
            console.console.print(
                f"[dim red]MCP server {server_name} failed to reconnect: {e}[/dim red]"
            )
            error_msg = f"MCP server {server_name} failed to reconnect: {e}"
            if error_factory:
                return error_factory(error_msg), False
            else:
                raise Exception(error_msg)

    async def _parse_resource_name(
        self, name: str, resource_type: str
    ) -> tuple[str | None, str]:
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

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
        tool_use_id: str | None = None,
        *,
        request_tool_handler: ToolExecutionHandler | None = None,
    ) -> CallToolResult:
        """
        Call a namespaced tool, e.g., 'server_name__tool_name'.

        Args:
            name: Tool name (possibly namespaced)
            arguments: Tool arguments
            tool_use_id: LLM's tool use ID (for matching with stream events)
            request_tool_handler: Optional per-request handler for tool execution events
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

        namespaced_tool_name = create_namespaced_name(server_name, local_tool_name)

        active_tool_handler = request_tool_handler or self._tool_handler

        # Check tool permission before execution
        try:
            permission_result = await self._permission_handler.check_permission(
                tool_name=local_tool_name,
                server_name=server_name,
                arguments=arguments,
                tool_use_id=tool_use_id,
            )
            if not permission_result.allowed:
                error_msg = permission_result.error_message
                if error_msg is None:
                    if permission_result.remember:
                        error_msg = (
                            f"The user has permanently declined permission to use this tool: "
                            f"{namespaced_tool_name}"
                        )
                    else:
                        error_msg = f"The user has declined permission to use this tool: {namespaced_tool_name}"

                # Notify tool handler so ACP clients can reflect the cancellation/denial
                if hasattr(active_tool_handler, "on_tool_permission_denied"):
                    try:
                        await active_tool_handler.on_tool_permission_denied(
                            local_tool_name, server_name, tool_use_id, error_msg
                        )
                    except Exception as e:
                        logger.error(f"Error notifying permission denial: {e}", exc_info=True)
                logger.info(
                    "Tool execution denied by permission handler",
                    data={
                        "tool_name": local_tool_name,
                        "server_name": server_name,
                        "cancelled": permission_result.is_cancelled,
                    },
                )
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=error_msg)],
                )
        except Exception as e:
            logger.error(f"Error checking tool permission: {e}", exc_info=True)
            # Fail-safe: deny on permission check error
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Permission check failed: {e}")],
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
            tool_call_id = await active_tool_handler.on_tool_start(
                local_tool_name, server_name, arguments, tool_use_id
            )
        except Exception as e:
            logger.error(f"Error in tool start handler: {e}", exc_info=True)
            # Generate fallback ID if handler fails
            import uuid

            tool_call_id = str(uuid.uuid4())

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"MCP Tool: {namespaced_tool_name}"):
            trace.get_current_span().set_attribute("tool_name", local_tool_name)
            trace.get_current_span().set_attribute("server_name", server_name)
            trace.get_current_span().set_attribute("namespaced_tool_name", namespaced_tool_name)

            # Create progress callback for this tool execution
            progress_callback = self._create_progress_callback(
                server_name,
                local_tool_name,
                tool_call_id,
                active_tool_handler,
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
                    # Pass the full content blocks to the handler
                    content = result.content if result.content else None

                    logger.debug(
                        f"Tool execution completed, notifying handler: {tool_call_id}",
                        name="mcp_tool_complete_notify",
                        tool_call_id=tool_call_id,
                        has_content=content is not None,
                        content_count=len(content) if content else 0,
                        is_error=result.isError,
                    )

                    # If there's an error, extract error text
                    error_text = None
                    if result.isError and content:
                        # Extract text from content for error message
                        text_parts = [c.text for c in content if hasattr(c, "text") and c.text]
                        error_text = "\n".join(text_parts) if text_parts else None
                        content = None  # Don't send content when there's an error

                    await active_tool_handler.on_tool_complete(
                        tool_call_id, not result.isError, content, error_text
                    )

                    logger.debug(
                        f"Tool handler notified successfully: {tool_call_id}",
                        name="mcp_tool_complete_done",
                    )
                except Exception as e:
                    logger.error(f"Error in tool complete handler: {e}", exc_info=True)

                return result

            except Exception as e:
                # Notify tool handler of error
                try:
                    await active_tool_handler.on_tool_complete(
                        tool_call_id, False, None, str(e)
                    )
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
    ) -> Mapping[str, list[Prompt]]:
        """
        List available prompts from one or all servers.

        :param server_name: Optional server name to list prompts from. If not provided,
                           lists prompts from all servers.
        :param agent_name: Optional agent name (ignored at this level, used by multi-agent apps)
        :return: Dictionary mapping server names to lists of Prompt objects
        """
        if not self.initialized:
            await self.load_servers()

        results: dict[str, list[Prompt]] = {}

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
    ) -> list[Any]:
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

    async def list_resources(self, server_name: str | None = None) -> dict[str, list[str]]:
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

        results: dict[str, list[str]] = {}

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
                formatted_resources: list[str] = []
                for resource in resources:
                    uri = getattr(resource, "uri", None)
                    if uri is not None:
                        formatted_resources.append(str(uri))
                results[s_name] = formatted_resources
            except Exception as e:
                logger.error(f"Error fetching resources from {s_name}: {e}")

        return results

    async def list_mcp_tools(self, server_name: str | None = None) -> dict[str, list[Tool]]:
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

        results: dict[str, list[Tool]] = {}

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
