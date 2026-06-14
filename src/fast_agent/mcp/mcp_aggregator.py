import sys
from asyncio import Lock
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from mcp import GetPromptResult, ReadResourceResult
from mcp.client.session import ClientSession
from mcp.shared.exceptions import McpError
from mcp.shared.session import ProgressFnT
from mcp.types import (
    CallToolResult,
    CompleteResult,
    Completion,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    Prompt,
    Resource,
    ResourceTemplate,
    ResourceTemplateReference,
    ServerCapabilities,
    TextContent,
    Tool,
)
from opentelemetry import trace
from pydantic import AnyUrl, BaseModel, ConfigDict, Field

from fast_agent.config import MCPServerSettings
from fast_agent.context_dependent import ContextDependent
from fast_agent.core.exceptions import ServerSessionTerminatedError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.logging.progress_payloads import build_progress_payload
from fast_agent.core.model_resolution import (
    HARDCODED_DEFAULT_MODEL,
    get_context_cli_model_override,
    resolve_model_spec,
)
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.common import SEP, create_namespaced_name, is_namespaced_name
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.interfaces import ServerRegistryProtocol
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.mcp.mcp_connection_manager import (
    MCPConnectionManager,
    ServerConnection,
    _is_http_auth_challenge_error,
    _resolve_oauth_mode,
)
from fast_agent.mcp.prompt_metadata import with_prompt_metadata
from fast_agent.mcp.skybridge import (
    MCP_APP_MIME_TYPE,
    SKYBRIDGE_MIME_TYPE,
    AppIntegrationKind,
    SkybridgeResourceConfig,
    SkybridgeServerConfig,
    SkybridgeToolConfig,
    extract_app_tool_metadata,
)
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler, ToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import (
    NoOpToolPermissionHandler,
    ToolPermissionHandler,
    ToolPermissionResult,
)
from fast_agent.mcp.tool_result_metadata import set_url_elicitation_required_payload
from fast_agent.mcp.transport_tracking import TransportSnapshot
from fast_agent.skills.mcp_registry import (
    McpSkillRegistry,
    scan_mcp_skill_registry,
    server_supports_mcp_skills,
)
from fast_agent.ui.tool_call_ids import format_tool_call_id
from fast_agent.utils.async_utils import gather_with_cancel
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.env import env_flag
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from fast_agent.context import Context
    from fast_agent.mcp.oauth_client import OAuthEvent
    from fast_agent.mcp_server_registry import ServerRegistry


logger = get_logger(__name__)  # This will be replaced per-instance when agent_name is available


def _display_tool_id(tool_id: str | None) -> str:
    return format_tool_call_id(tool_id) or "unknown"


def _progress_trace_enabled() -> bool:
    return env_flag("FAST_AGENT_TRACE_MCP_PROGRESS")


def _progress_trace(message: str) -> None:
    if not _progress_trace_enabled():
        return
    print(f"[mcp-progress-trace] {message}", file=sys.stderr, flush=True)


# Define type variables for the generalized method
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class _ServerOperationRecovery(Generic[R]):
    result: R | None
    success: bool


@dataclass(frozen=True, slots=True)
class _ResourceNameResolution:
    server_name: str | None
    local_name: str


@dataclass(frozen=True, slots=True)
class _PromptNameResolution:
    server_name: str | None
    local_name: str


@dataclass(frozen=True, slots=True)
class _AttachedRegistryScanClient:
    aggregator: "MCPAggregator"

    async def get_capabilities(self, server_name: str) -> ServerCapabilities | None:
        return await self.aggregator.get_capabilities(server_name)

    async def get_resource(
        self,
        resource_uri: str,
        *,
        server_name: str | None = None,
    ) -> ReadResourceResult:
        if server_name is None:
            raise ValueError("server_name is required for attached registry scans")
        return await self.aggregator._get_resource_from_server(server_name, resource_uri)


METHOD_NOT_FOUND_ERROR_CODE = -32601
METHOD_NOT_FOUND_MESSAGE = "method not found"


@runtime_checkable
class ElicitationModeCapable(Protocol):
    effective_elicitation_mode: str | None


@runtime_checkable
class ClientInfoLike(Protocol):
    name: str | None
    version: str | None


@runtime_checkable
class SessionClientInfoCapable(Protocol):
    client_info: ClientInfoLike | None


def _is_capability_probe_error(exc: Exception) -> bool:
    """Return True when exc indicates a server does not support a probed method."""
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, McpError):
        code = exc.error.code
        if code == METHOD_NOT_FOUND_ERROR_CODE:
            return True
        # Only fall back to message matching when the server omitted the error code;
        # if a different code is set, trust the code over the message text.
        if code is None:
            message = exc.error.message
            if isinstance(message, str) and METHOD_NOT_FOUND_MESSAGE in strip_casefold(message):
                return True
    return False


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
    mcp_skills_enabled: bool | None = None
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


@dataclass(frozen=True, slots=True)
class MCPAttachOptions:
    startup_timeout_seconds: float = 10.0
    trigger_oauth: bool | None = None
    force_reconnect: bool = False
    reconnect_on_disconnect: bool | None = None
    oauth_event_handler: Callable[["OAuthEvent"], Awaitable[None]] | None = None
    allow_oauth_paste_fallback: bool = True


@dataclass(frozen=True, slots=True)
class MCPAttachResult:
    server_name: str
    transport: str
    attached: bool
    already_attached: bool
    tools_added: list[str]
    prompts_added: list[str]
    warnings: list[str]
    tools_total: int | None = None
    prompts_total: int | None = None
    skills_total: int | None = None


@dataclass(frozen=True, slots=True)
class MCPDetachResult:
    server_name: str
    detached: bool
    tools_removed: list[str]
    prompts_removed: list[str]


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

    @staticmethod
    def _unique_preserving_order(items: Iterable[str]) -> list[str]:
        return unique_preserve_order(items)

    async def __aenter__(self):
        if self.initialized:
            return self

        # Keep a connection manager to manage persistent connections for this aggregator
        if self.connection_persistence:
            context = self._require_context()
            # Try to get existing connection manager from context
            if context._connection_manager is None:
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

        self._configured_server_names = list(server_names)
        self.server_names = list(server_names)
        self._attached_server_names: list[str] = []
        self._supplemental_attached_server_names: list[str] = []
        self.connection_persistence = connection_persistence
        self.agent_name = name
        self.config = config  # Store the config for access in session factory
        self._persistent_connection_manager: MCPConnectionManager | None = None
        self._owns_connection_manager = False

        # Store tool execution handler for integration with ACP or other protocols.
        #
        # In ACP server contexts we attach an ACPContext to `Context` objects and store
        # a per-session progress manager there. Agent-as-tools workflows can spawn
        # detached agent instances (and thus new MCPAggregators) at runtime; those
        # aggregators must pick up the same progress manager so nested tool calls
        # are visible to ACP clients.
        resolved_tool_handler = tool_handler
        if resolved_tool_handler is None and context is not None and context.acp is not None:
            resolved_tool_handler = context.acp.progress_manager or None

        # Default to NoOpToolExecutionHandler if none provided.
        self._tool_handler = resolved_tool_handler or NoOpToolExecutionHandler()

        # Store tool permission handler for ACP or other permission systems.
        resolved_permission_handler = permission_handler
        if resolved_permission_handler is None and context is not None and context.acp is not None:
            resolved_permission_handler = context.acp.permission_handler or None

        # Default to NoOpToolPermissionHandler if none provided (allows all).
        self._permission_handler = resolved_permission_handler or NoOpToolPermissionHandler()

        # Server notification callback: async (server_name, notification) -> None
        # Set this to receive MCP server notifications (log messages, resource updates, etc.)
        self.server_notification_callback = None

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
        self._mcp_skill_registries: dict[str, McpSkillRegistry] = {}

        # Cache for server capabilities in non-persistent mode
        self._capabilities_cache: dict[str, ServerCapabilities] = {}
        self._capabilities_cache_lock = Lock()

    @property
    def tool_execution_handler(self) -> ToolExecutionHandler:
        return self._tool_handler

    def set_tool_execution_handler(self, handler: ToolExecutionHandler) -> None:
        self._tool_handler = handler

    @property
    def permission_handler(self) -> ToolPermissionHandler:
        return self._permission_handler

    def set_permission_handler(self, handler: ToolPermissionHandler) -> None:
        self._permission_handler = handler

    def _require_context(self) -> "Context":
        if self.context is None:
            raise RuntimeError("MCPAggregator requires a context")
        return self.context

    def _require_server_registry(self) -> ServerRegistryProtocol:
        context = self._require_context()
        server_registry = context.server_registry
        if server_registry is None:
            raise RuntimeError("Context is missing server registry for MCP connections")
        return cast("ServerRegistryProtocol", server_registry)

    def _require_connection_manager(self) -> MCPConnectionManager:
        if self._persistent_connection_manager is None:
            raise RuntimeError("Persistent connection manager is not initialized")
        return self._persistent_connection_manager

    def _create_progress_callback(
        self,
        server_name: str,
        tool_name: str,
        tool_call_id: str,
        tool_use_id: str | None = None,
        request_tool_handler: ToolExecutionHandler | None = None,
    ) -> "ProgressFnT":
        """Create a progress callback function for tool execution."""
        handler_for_request = request_tool_handler or self._tool_handler

        async def progress_callback(
            progress: float, total: float | None, message: str | None
        ) -> None:
            """Handle progress notifications from MCP tool execution."""
            _progress_trace(
                "callback-progress "
                f"server={server_name} "
                f"tool={tool_name} "
                f"tool_call_id={_display_tool_id(tool_call_id)} "
                f"progress={progress!r} "
                f"total={total!r} "
                f"message={message!r}"
            )

            logger.info(
                "Tool progress update",
                data=build_progress_payload(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_name=tool_name,
                    server_name=server_name,
                    agent_name=self.agent_name,
                    tool_call_id=tool_call_id,
                    tool_use_id=tool_use_id,
                    progress=progress,
                    total=total,
                    details=message or "",  # Put the message in details column
                ),
            )

            # Forward progress to tool handler (e.g., for ACP notifications)
            try:
                await handler_for_request.on_tool_progress(tool_call_id, progress, total, message)
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
                    self.context is not None
                    and self.context._connection_manager == self._persistent_connection_manager
                ):
                    logger.info("Shutting down all persistent connections...")
                    await self._persistent_connection_manager.disconnect_all()
                    await self._persistent_connection_manager.__aexit__(None, None, None)
                    self.context._connection_manager = None
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
                resolved_model = resolve_model_spec(
                    self.context,
                    model=self.config.model,
                    cli_model=get_context_cli_model_override(self.context),
                    hardcoded_default=HARDCODED_DEFAULT_MODEL,
                )
                agent_model = resolved_model.model
                if resolved_model.source:
                    logger.info(
                        f"Resolved MCP agent model '{agent_model}' via {resolved_model.source}",
                        model=agent_model,
                        source=resolved_model.source,
                    )
                agent_name = self.config.name
                elicitation_handler = self.config.elicitation_handler
                api_key = self.config.api_key

            session = MCPAgentClientSession(
                read_stream,
                write_stream,
                read_timeout,
                server_name=server_name,
                agent_model=agent_model,
                agent_name=agent_name,
                api_key=api_key,
                elicitation_handler=elicitation_handler,
                tool_list_changed_callback=self._handle_tool_list_changed,
                aggregator=self,
                **kwargs,  # Pass through any additional kwargs like server_config
            )

            return session

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

        await self._reset_runtime_indexes()

        skipped_servers: list[str] = []
        attached_results: list[MCPAttachResult] = []

        servers_to_load = list(self._configured_server_names)

        for server_name in servers_to_load:
            # Check if server should be loaded on start
            server_registry = self.context.server_registry if self.context else None
            if server_registry is not None:
                server_config = server_registry.get_server_config(server_name)
                if server_config and not server_config.load_on_start and not force_connect:
                    logger.debug(f"Skipping server '{server_name}' - load_on_start=False")
                    skipped_servers.append(server_name)
                    continue

            attached_results.append(
                await self.attach_server(
                    server_name=server_name,
                    options=MCPAttachOptions(),
                )
            )

        if skipped_servers:
            logger.debug(
                "Deferred MCP servers due to load_on_start=False",
                data={
                    "agent_name": self.agent_name,
                    "servers": skipped_servers,
                },
            )

        if not attached_results:
            self.initialized = True
            return

        self._display_startup_state()

        self.initialized = True

    async def _reset_runtime_indexes(self) -> None:
        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()

        async with self._prompt_cache_lock:
            self._prompt_cache.clear()

        async with self._capabilities_cache_lock:
            self._capabilities_cache.clear()

        self._skybridge_configs.clear()
        self._mcp_skill_registries.clear()
        self._attached_server_names = []

    async def _fetch_server_tools(self, server_name: str) -> list[Tool]:
        supports_tools = await self.server_supports_feature(server_name, "tools")
        if not supports_tools:
            logger.debug(
                f"Server '{server_name}' did not advertise tools; attempting optimistic list_tools call"
            )

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
            if supports_tools:
                raise
            if not _is_capability_probe_error(e):
                raise
            logger.debug(f"Server '{server_name}' does not provide tools (list_tools failed): {e}")
            return []

    async def _fetch_server_prompts(self, server_name: str) -> list[Prompt]:
        if not await self.server_supports_feature(server_name, "prompts"):
            logger.debug(f"Server '{server_name}' does not support prompts")
            return []

        try:
            result: ListPromptsResult = await self._execute_on_server(
                server_name=server_name,
                operation_type="prompts/list",
                operation_name="",
                method_name="list_prompts",
                method_args={},
            )
            return result.prompts
        except Exception as e:
            logger.debug(f"Error loading prompts from server '{server_name}': {e}")
            return []

    async def attach_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        attach_options = options or MCPAttachOptions()
        server_registry = self._require_server_registry()

        resolved_config = self._resolve_attach_server_config(
            server_name,
            server_config,
            attach_options,
            server_registry,
        )
        existing_tool_names = self._attached_tool_names(server_name)
        existing_prompt_names = self._attached_prompt_names(server_name)

        already_attached = server_name in self._attached_server_names
        if already_attached and not attach_options.force_reconnect:
            return self._already_attached_result(
                server_name,
                resolved_config,
                existing_tool_names,
                existing_prompt_names,
            )

        await self._clear_capabilities_for_forced_reconnect(server_name, attach_options)

        if self.connection_persistence:
            await self._connect_persistent_server(server_name, attach_options)

        # Ensure capability-gated discovery can validate newly attached or reattached servers.
        if server_name not in self.server_names:
            self.server_names.append(server_name)

        skybridge_config = await self._refresh_attached_server_cache(server_name)

        if server_name not in self._attached_server_names:
            self._attached_server_names.append(server_name)

        self._log_server_initialized()
        return await self._attached_result(
            server_name=server_name,
            resolved_config=resolved_config,
            already_attached=already_attached,
            existing_tool_names=existing_tool_names,
            existing_prompt_names=existing_prompt_names,
            skybridge_config=skybridge_config,
        )

    def _resolve_attach_server_config(
        self,
        server_name: str,
        server_config: MCPServerSettings | None,
        attach_options: MCPAttachOptions,
        server_registry: Any,
    ) -> MCPServerSettings:
        if server_config is not None:
            server_registry.registry[server_name] = server_config
            if server_name not in self._configured_server_names:
                self._configured_server_names.append(server_name)

        resolved_config = server_registry.get_server_config(server_name)
        if resolved_config is None:
            raise ValueError(f"Server '{server_name}' not found in registry")

        if attach_options.reconnect_on_disconnect is None:
            return resolved_config

        updated_config = resolved_config.model_copy(
            update={"reconnect_on_disconnect": attach_options.reconnect_on_disconnect}
        )
        server_registry.registry[server_name] = updated_config
        return updated_config

    def _attached_tool_names(self, server_name: str) -> set[str]:
        return {tool.namespaced_tool_name for tool in self._server_to_tool_map.get(server_name, [])}

    def _attached_prompt_names(self, server_name: str) -> set[str]:
        return {prompt.name for prompt in self._prompt_cache.get(server_name, [])}

    @staticmethod
    def _already_attached_result(
        server_name: str,
        resolved_config: MCPServerSettings,
        existing_tool_names: set[str],
        existing_prompt_names: set[str],
    ) -> MCPAttachResult:
        return MCPAttachResult(
            server_name=server_name,
            transport=resolved_config.transport,
            attached=True,
            already_attached=True,
            tools_added=[],
            prompts_added=[],
            warnings=[],
            tools_total=len(existing_tool_names),
            prompts_total=len(existing_prompt_names),
            skills_total=None,
        )

    async def _clear_capabilities_for_forced_reconnect(
        self,
        server_name: str,
        attach_options: MCPAttachOptions,
    ) -> None:
        if not attach_options.force_reconnect:
            return
        async with self._capabilities_cache_lock:
            self._capabilities_cache.pop(server_name, None)

    async def _connect_persistent_server(
        self,
        server_name: str,
        attach_options: MCPAttachOptions,
    ) -> None:
        logger.info(
            f"Creating persistent connection to server: {server_name}",
            data={
                "progress_action": ProgressAction.CONNECTING,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )

        manager = self._require_connection_manager()
        connect = manager.reconnect_server if attach_options.force_reconnect else manager.get_server
        await connect(
            server_name,
            client_session_factory=self._create_session_factory(server_name),
            startup_timeout_seconds=attach_options.startup_timeout_seconds,
            trigger_oauth=attach_options.trigger_oauth,
            oauth_event_handler=attach_options.oauth_event_handler,
            allow_oauth_paste_fallback=attach_options.allow_oauth_paste_fallback,
        )
        await self._record_server_call(server_name, "initialize", True)

    async def _refresh_attached_server_cache(self, server_name: str) -> SkybridgeServerConfig:
        tools = await self._fetch_server_tools(server_name)
        prompts = await self._fetch_server_prompts(server_name)
        mcp_skill_registry = await self._scan_mcp_skill_registry(server_name)

        async with self._tool_map_lock:
            for namespaced in self._server_to_tool_map.get(server_name, []):
                self._namespaced_tool_map.pop(namespaced.namespaced_tool_name, None)

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

        async with self._prompt_cache_lock:
            self._prompt_cache[server_name] = prompts

        if mcp_skill_registry is None:
            self._mcp_skill_registries.pop(server_name, None)
        else:
            self._mcp_skill_registries[server_name] = mcp_skill_registry

        _, skybridge_config = await self._evaluate_skybridge_for_server(server_name)
        self._skybridge_configs[server_name] = skybridge_config
        return skybridge_config

    def _log_server_initialized(self) -> None:
        logger.info(
            f"MCP Servers initialized for agent '{self.agent_name}'",
            data={
                "progress_action": ProgressAction.INITIALIZED,
                "agent_name": self.agent_name,
            },
        )

    async def _attached_result(
        self,
        *,
        server_name: str,
        resolved_config: MCPServerSettings,
        already_attached: bool,
        existing_tool_names: set[str],
        existing_prompt_names: set[str],
        skybridge_config: SkybridgeServerConfig,
    ) -> MCPAttachResult:
        tool_names = self._attached_tool_names(server_name)
        prompt_names = self._attached_prompt_names(server_name)
        skills_total = await self._mcp_skills_total(server_name)
        return MCPAttachResult(
            server_name=server_name,
            transport=resolved_config.transport,
            attached=True,
            already_attached=already_attached,
            tools_added=sorted(tool_names - existing_tool_names),
            prompts_added=sorted(prompt_names - existing_prompt_names),
            warnings=list(skybridge_config.warnings),
            tools_total=len(tool_names),
            prompts_total=len(prompt_names),
            skills_total=skills_total,
        )

    async def _mcp_skills_total(self, server_name: str) -> int | None:
        registry = self._mcp_skill_registries.get(server_name)
        if registry is None:
            return None
        return len(registry.skills)

    async def detach_server(self, server_name: str) -> MCPDetachResult:
        existing_tools = self._server_to_tool_map.get(server_name, [])
        existing_prompts = self._prompt_cache.get(server_name, [])
        tools_removed = sorted(tool.namespaced_tool_name for tool in existing_tools)
        prompts_removed = sorted(prompt.name for prompt in existing_prompts)

        if server_name not in self._attached_server_names:
            return MCPDetachResult(
                server_name=server_name,
                detached=False,
                tools_removed=[],
                prompts_removed=[],
            )

        if self.connection_persistence and self._persistent_connection_manager is not None:
            await self._persistent_connection_manager.disconnect_server(server_name)

        async with self._tool_map_lock:
            for namespaced_tool in self._server_to_tool_map.pop(server_name, []):
                self._namespaced_tool_map.pop(namespaced_tool.namespaced_tool_name, None)

        async with self._prompt_cache_lock:
            self._prompt_cache.pop(server_name, None)

        async with self._capabilities_cache_lock:
            self._capabilities_cache.pop(server_name, None)

        self._skybridge_configs.pop(server_name, None)
        self._mcp_skill_registries.pop(server_name, None)
        self._attached_server_names = [
            name for name in self._attached_server_names if name != server_name
        ]
        self.server_names = [name for name in self.server_names if name != server_name]

        return MCPDetachResult(
            server_name=server_name,
            detached=True,
            tools_removed=tools_removed,
            prompts_removed=prompts_removed,
        )

    def list_attached_servers(self) -> list[str]:
        return self._unique_preserving_order(
            [*self._attached_server_names, *self._supplemental_attached_server_names]
        )

    def set_supplemental_attached_servers(self, server_names: Iterable[str]) -> None:
        self._supplemental_attached_server_names = self._unique_preserving_order(server_names)

    def list_configured_detached_servers(self) -> list[str]:
        configured = set(self._configured_server_names)
        server_registry = self.context.server_registry if self.context else None
        if server_registry is not None:
            configured.update(server_registry.registry.keys())
        return sorted(configured - set(self.list_attached_servers()))

    async def _initialize_skybridge_configs(self, server_names: list[str] | None = None) -> None:
        """Discover Skybridge resources across servers."""
        target_servers = server_names if server_names is not None else self.server_names
        if not target_servers:
            return

        tasks = [self._evaluate_skybridge_for_server(server_name) for server_name in target_servers]
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
        tool_configs = self._skybridge_tool_configs_for_server(server_name, config)

        raw_resources_capability = await self.server_supports_feature(server_name, "resources")
        supports_resources = bool(raw_resources_capability)
        config.supports_resources = supports_resources
        config.tools = tool_configs

        if not supports_resources:
            return server_name, config

        await self._collect_skybridge_resources(server_name, config, tool_configs)
        self._link_skybridge_tools_to_resources(config, tool_configs)
        self._warn_if_app_resources_are_unexposed(server_name, config, tool_configs)
        config.tools = tool_configs
        return server_name, config

    def _skybridge_tool_configs_for_server(
        self,
        server_name: str,
        config: SkybridgeServerConfig,
    ) -> list[SkybridgeToolConfig]:
        tool_configs: list[SkybridgeToolConfig] = []
        for namespaced_tool in self._server_to_tool_map.get(server_name, []):
            tool_config = self._skybridge_tool_config(namespaced_tool, config)
            if tool_config is not None:
                tool_configs.append(tool_config)
        return tool_configs

    @staticmethod
    def _metadata_error_tool_config(
        namespaced_tool: NamespacedTool,
        warning: str,
    ) -> SkybridgeToolConfig:
        return SkybridgeToolConfig(
            tool_name=namespaced_tool.tool.name,
            namespaced_tool_name=namespaced_tool.namespaced_tool_name,
            warning=warning,
        )

    def _skybridge_tool_config(
        self,
        namespaced_tool: NamespacedTool,
        config: SkybridgeServerConfig,
    ) -> SkybridgeToolConfig | None:
        tool_meta = namespaced_tool.tool.meta or {}
        try:
            app_metadata = extract_app_tool_metadata(
                tool_meta,
                namespaced_tool_name=namespaced_tool.namespaced_tool_name,
            )
        except ValueError as exc:
            warning = str(exc)
            config.warnings.append(warning)
            logger.error(warning)
            return self._metadata_error_tool_config(namespaced_tool, warning)

        if app_metadata is None:
            return None

        for metadata_warning in app_metadata.warnings:
            warning = f"Tool '{namespaced_tool.namespaced_tool_name}' {metadata_warning}"
            config.warnings.append(warning)
            logger.warning(warning)

        return SkybridgeToolConfig(
            tool_name=namespaced_tool.tool.name,
            namespaced_tool_name=namespaced_tool.namespaced_tool_name,
            template_uri=app_metadata.resource_uri,
            kind=app_metadata.kind,
            visibility=app_metadata.visibility,
        )

    async def _collect_skybridge_resources(
        self,
        server_name: str,
        config: SkybridgeServerConfig,
        tool_configs: list[SkybridgeToolConfig],
    ) -> None:
        try:
            resources = await self._list_resources_from_server(server_name, check_support=False)
        except Exception as exc:
            config.warnings.append(f"Failed to list resources: {exc}")
            return

        expected_mime_by_uri = {
            str(tool.template_uri): tool.kind.expected_mime_type
            for tool in tool_configs
            if tool.template_uri is not None
        }

        for resource_entry in resources:
            uri_str, sky_resource = self._skybridge_resource_candidate(resource_entry, config)
            if sky_resource is None:
                continue

            config.ui_resources.append(sky_resource)
            await self._read_skybridge_resource(
                server_name,
                uri_str,
                sky_resource,
                config,
                expected_mime_by_uri,
            )

    @staticmethod
    def _skybridge_resource_candidate(
        resource_entry: Resource,
        config: SkybridgeServerConfig,
    ) -> tuple[str, SkybridgeResourceConfig | None]:
        uri = resource_entry.uri
        if not uri:
            return "", None

        uri_str = str(uri)
        if not uri_str.startswith("ui://"):
            return uri_str, None

        try:
            uri_value = AnyUrl(uri_str)
        except Exception as exc:
            warning = f"Ignoring Skybridge candidate '{uri_str}': invalid URI ({exc})"
            config.warnings.append(warning)
            logger.debug(warning)
            return uri_str, None

        entry_meta = getattr(resource_entry, "meta", None)
        return uri_str, SkybridgeResourceConfig(
            uri=uri_value,
            meta=dict(entry_meta) if isinstance(entry_meta, dict) else {},
        )

    async def _read_skybridge_resource(
        self,
        server_name: str,
        uri_str: str,
        sky_resource: SkybridgeResourceConfig,
        config: SkybridgeServerConfig,
        expected_mime_by_uri: dict[str, str],
    ) -> None:
        try:
            read_result: ReadResourceResult = await self._get_resource_from_server(
                server_name,
                uri_str,
            )
        except Exception as exc:
            warning = f"Failed to read resource '{uri_str}': {exc}"
            sky_resource.warning = warning
            config.warnings.append(warning)
            return

        self._apply_skybridge_resource_contents(sky_resource, read_result)
        if not sky_resource.is_valid_app_resource:
            self._warn_invalid_skybridge_resource(
                uri_str,
                sky_resource,
                config,
                expected_mime_by_uri,
            )

    @staticmethod
    def _apply_skybridge_resource_contents(
        sky_resource: SkybridgeResourceConfig,
        read_result: ReadResourceResult,
    ) -> None:
        seen_mime_types: list[str] = []
        for content in read_result.contents:
            mime_type = content.mimeType
            if mime_type:
                seen_mime_types.append(mime_type)
            if mime_type == SKYBRIDGE_MIME_TYPE:
                sky_resource.mime_type = mime_type
                sky_resource.kind = AppIntegrationKind.SKYBRIDGE
                sky_resource.is_skybridge = True
            elif mime_type == MCP_APP_MIME_TYPE:
                sky_resource.mime_type = mime_type
                sky_resource.kind = AppIntegrationKind.MCP_APP
                sky_resource.is_mcp_app = True

            content_meta = getattr(content, "meta", None)
            if isinstance(content_meta, dict):
                sky_resource.meta.update(content_meta)

        if sky_resource.mime_type is None and seen_mime_types:
            sky_resource.mime_type = seen_mime_types[0]

    @staticmethod
    def _warn_invalid_skybridge_resource(
        uri_str: str,
        sky_resource: SkybridgeResourceConfig,
        config: SkybridgeServerConfig,
        expected_mime_by_uri: dict[str, str],
    ) -> None:
        observed_type = sky_resource.mime_type or "unknown MIME type"
        expected_mime_type = expected_mime_by_uri.get(uri_str)
        expected_label = (
            f"'{expected_mime_type}'"
            if expected_mime_type
            else f"'{SKYBRIDGE_MIME_TYPE}' or '{MCP_APP_MIME_TYPE}'"
        )
        warning = f"served as '{observed_type}' instead of {expected_label}"
        sky_resource.warning = warning
        config.warnings.append(f"{uri_str}: {warning}")

    def _link_skybridge_tools_to_resources(
        self,
        config: SkybridgeServerConfig,
        tool_configs: list[SkybridgeToolConfig],
    ) -> None:
        resource_lookup = {str(resource.uri): resource for resource in config.ui_resources}
        for tool_config in tool_configs:
            if tool_config.template_uri is None:
                continue

            resource_match = resource_lookup.get(str(tool_config.template_uri))
            if not resource_match:
                self._warn_missing_skybridge_resource(tool_config, config)
                continue

            self._apply_skybridge_tool_resource_match(tool_config, resource_match, config)

    @staticmethod
    def _warn_missing_skybridge_resource(
        tool_config: SkybridgeToolConfig,
        config: SkybridgeServerConfig,
    ) -> None:
        resource_label = (
            "Skybridge"
            if tool_config.kind is AppIntegrationKind.SKYBRIDGE
            else tool_config.kind.display_name
        )
        warning = (
            f"Tool '{tool_config.namespaced_tool_name}' references missing "
            f"{resource_label} resource '{tool_config.template_uri}'"
        )
        tool_config.warning = warning
        config.warnings.append(warning)
        logger.error(warning)

    @staticmethod
    def _apply_skybridge_tool_resource_match(
        tool_config: SkybridgeToolConfig,
        resource_match: SkybridgeResourceConfig,
        config: SkybridgeServerConfig,
    ) -> None:
        tool_config.resource_uri = resource_match.uri
        expected_mime_type = tool_config.kind.expected_mime_type
        tool_config.is_valid = (
            resource_match.is_skybridge
            if tool_config.kind is AppIntegrationKind.SKYBRIDGE
            else resource_match.is_mcp_app
        )

        if tool_config.is_valid:
            return

        warning = (
            f"Tool '{tool_config.namespaced_tool_name}' references resource "
            f"'{resource_match.uri}' served as '{resource_match.mime_type or 'unknown'}' "
            f"instead of '{expected_mime_type}'"
        )
        tool_config.warning = warning
        config.warnings.append(warning)
        logger.warning(warning)

    @staticmethod
    def _warn_if_app_resources_are_unexposed(
        server_name: str,
        config: SkybridgeServerConfig,
        tool_configs: list[SkybridgeToolConfig],
    ) -> None:
        valid_tool_count = sum(1 for tool in tool_configs if tool.is_valid)
        if config.enabled and valid_tool_count == 0:
            warning = f"App resources detected on server '{server_name}' but no tools expose them"
            config.warnings.append(warning)
            logger.warning(warning)

    def _display_startup_state(self) -> None:
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

    async def get_capabilities(self, server_name: str) -> ServerCapabilities | None:
        """Get server capabilities if available."""
        if not self.connection_persistence:
            # Check cache under lock (fast path)
            async with self._capabilities_cache_lock:
                cached = self._capabilities_cache.get(server_name)
                if cached is not None:
                    return cached

            # I/O without holding lock — allows concurrent probes for different servers
            try:
                server_registry = self._require_server_registry()
                async with server_registry.initialize_server(
                    server_name=server_name,
                ) as _session:
                    capabilities = server_registry.get_server_capabilities(server_name)

                if capabilities is not None:
                    async with self._capabilities_cache_lock:
                        self._capabilities_cache[server_name] = capabilities
                return capabilities
            except Exception as e:
                logger.debug(f"Error getting capabilities for server '{server_name}': {e}")
                return None

        try:
            manager = self._require_connection_manager()
            server_conn = await manager.get_server(
                server_name,
                client_session_factory=self._create_session_factory(server_name),
            )
            return server_conn.server_capabilities
        except Exception as e:
            logger.debug(f"Error getting capabilities for server '{server_name}': {e}")
            return None

    async def _scan_mcp_skill_registry(self, server_name: str) -> McpSkillRegistry | None:
        client = _AttachedRegistryScanClient(self)
        return await scan_mcp_skill_registry(
            client,
            server_name,
            server_version=await self._mcp_server_version(server_name),
        )

    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]:
        if not self.initialized:
            await self.load_servers()
        registries: list[McpSkillRegistry] = []
        for server_name in self.list_attached_servers():
            registry = self._mcp_skill_registries.get(server_name)
            if registry is None:
                registry = await self._scan_mcp_skill_registry(server_name)
                if registry is None:
                    continue
                self._mcp_skill_registries[server_name] = registry
            registries.append(registry)
        return registries

    def cached_mcp_skill_registries(self) -> list[McpSkillRegistry]:
        return sorted(
            self._mcp_skill_registries.values(),
            key=lambda registry: registry.server_name.lower(),
        )

    async def _mcp_server_version(self, server_name: str) -> str | None:
        manager = self._persistent_connection_manager
        if self.connection_persistence and manager is not None:
            with suppress(Exception):
                async with manager._lock:
                    server_conn = manager.running_servers.get(server_name)
                implementation = server_conn.server_implementation if server_conn else None
                if implementation is not None:
                    return implementation.version
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

    async def server_supports_feature(
        self,
        server_name: str,
        feature: Literal["prompts", "resources", "tools", "completions", "tasks"],
    ) -> bool:
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

        feature_value = {
            "prompts": capabilities.prompts,
            "resources": capabilities.resources,
            "tools": capabilities.tools,
            "completions": capabilities.completions,
            "tasks": capabilities.tasks,
        }[feature]
        if isinstance(feature_value, bool):
            return feature_value
        if feature_value is None:
            return False
        try:
            return bool(feature_value)
        except Exception:
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
            skybridge_config = self._skybridge_configs.get(namespaced_tool.server_name)
            discovered_tool = None
            matching_tool = None
            if skybridge_config:
                discovered_tool = next(
                    (
                        tool
                        for tool in skybridge_config.tools
                        if tool.namespaced_tool_name == namespaced_tool_name
                    ),
                    None,
                )
                if discovered_tool and discovered_tool.is_valid:
                    matching_tool = discovered_tool

            if discovered_tool and discovered_tool.is_app_only:
                continue

            tool_copy = namespaced_tool.tool.model_copy(
                deep=True, update={"name": namespaced_tool_name}
            )
            if matching_tool:
                meta = dict(tool_copy.meta or {})
                if matching_tool.kind is AppIntegrationKind.MCP_APP:
                    ui_meta = meta.get("ui")
                    ui_meta_dict = dict(ui_meta) if isinstance(ui_meta, dict) else {}
                    ui_meta_dict["resourceUri"] = str(matching_tool.template_uri)
                    ui_meta_dict["visibility"] = list(matching_tool.visibility)
                    meta["ui"] = ui_meta_dict
                    meta["ui/appEnabled"] = True
                    meta["ui/appTemplate"] = str(matching_tool.template_uri)
                else:
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

        manager = self._persistent_connection_manager
        if manager is None:
            return instructions

        # Only read from already-running server connections to avoid implicit connects.
        running_servers = manager.running_servers
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
            status = self._server_status_from_stats(server_name, now)
            server_cfg, server_conn = await self._collect_persistent_server_status(
                server_name,
                status,
            )
            if server_cfg is None:
                server_cfg = self._server_config_for_status(server_name)

            self._apply_config_status(status, server_cfg, server_conn)
            if status.server_capabilities is None:
                status.server_capabilities = await self._capabilities_for_status(server_name)
            status.mcp_skills_enabled = server_supports_mcp_skills(status.server_capabilities)
            status_map[server_name] = status

        return status_map

    async def _capabilities_for_status(self, server_name: str) -> ServerCapabilities | None:
        async with self._capabilities_cache_lock:
            cached = self._capabilities_cache.get(server_name)
        if cached is not None:
            return cached

        manager = self._persistent_connection_manager
        if self.connection_persistence and manager is not None:
            with suppress(Exception):
                async with manager._lock:
                    server_conn = manager.running_servers.get(server_name)
                return server_conn.server_capabilities if server_conn else None
        return None

    def _server_status_from_stats(self, server_name: str, now: datetime) -> ServerStatus:
        stats = self._server_stats.get(server_name)
        last_call = stats.last_call_at if stats else None
        return ServerStatus(
            server_name=server_name,
            last_call_at=last_call,
            last_error_at=stats.last_error_at if stats else None,
            staleness_seconds=(now - last_call).total_seconds() if last_call else None,
            call_counts=dict(stats.call_counts) if stats else {},
            reconnect_count=stats.reconnect_count if stats else 0,
            skybridge=self._skybridge_configs.get(server_name),
        )

    async def _collect_persistent_server_status(
        self,
        server_name: str,
        status: ServerStatus,
    ) -> tuple[MCPServerSettings | None, ServerConnection | None]:
        manager = self._persistent_connection_manager
        if not self.connection_persistence or manager is None:
            return None, None

        server_conn: ServerConnection | None = None
        server_cfg: MCPServerSettings | None = None
        try:
            async with manager._lock:
                server_conn = manager.running_servers.get(server_name)
            if server_conn is None:
                status.is_connected = False
                return None, None

            server_cfg = server_conn.server_config
            self._apply_connection_status(status, server_conn)
        except Exception as exc:
            logger.debug(
                f"Failed to collect status for server '{server_name}'",
                data={"error": str(exc)},
            )
        return server_cfg, server_conn

    def _apply_connection_status(
        self,
        status: ServerStatus,
        server_conn: ServerConnection,
    ) -> None:
        implementation = server_conn.server_implementation
        if implementation is not None:
            status.implementation_name = implementation.name
            status.implementation_version = implementation.version

        status.server_capabilities = server_conn.server_capabilities
        status.mcp_skills_enabled = server_supports_mcp_skills(server_conn.server_capabilities)
        status.client_capabilities = server_conn.client_capabilities
        self._apply_client_info_status(status, server_conn.session)

        if server_conn._initialized_event.is_set():
            status.is_connected = server_conn.is_healthy()
        else:
            status.is_connected = False
            status.error_message = status.error_message or "initializing..."

        status.error_message = status.error_message or server_conn._error_message
        status.instructions_available = server_conn.server_instructions_available
        status.instructions_enabled = server_conn.server_instructions_enabled
        status.instructions_included = bool(server_conn.server_instructions)

        self._apply_ping_status(status, server_conn)
        self._apply_session_status(status, server_conn)
        self._apply_transport_status(status, server_conn)

    @staticmethod
    def _apply_client_info_status(
        status: ServerStatus,
        session: ClientSession | None,
    ) -> None:
        if not isinstance(session, SessionClientInfoCapable):
            return

        client_info = session.client_info
        if client_info:
            status.client_info_name = client_info.name
            status.client_info_version = client_info.version

    @staticmethod
    def _apply_ping_status(status: ServerStatus, server_conn: ServerConnection) -> None:
        server_cfg = server_conn.server_config
        status.ping_interval_seconds = server_cfg.ping_interval_seconds
        status.ping_max_missed = server_cfg.max_missed_pings
        status.ping_ok_count = server_conn._ping_ok_count
        status.ping_fail_count = server_conn._ping_fail_count
        status.ping_consecutive_failures = server_conn._ping_consecutive_failures
        status.ping_last_ok_at = server_conn._ping_last_ok_at
        status.ping_last_fail_at = server_conn._ping_last_fail_at
        status.ping_last_error = server_conn._ping_last_error

    def _apply_session_status(
        self,
        status: ServerStatus,
        server_conn: ServerConnection,
    ) -> None:
        session = server_conn.session
        if isinstance(session, ElicitationModeCapable):
            status.elicitation_mode = session.effective_elicitation_mode

        # Mcp-Session-Id from the transport, when the server assigns one.
        status.session_id = server_conn.session_id or self._session_id_from_callback(server_conn)

    @staticmethod
    def _session_id_from_callback(server_conn: ServerConnection) -> str | None:
        if not server_conn._get_session_id_cb:
            return None
        try:
            return server_conn._get_session_id_cb()
        except Exception:
            return None

    def _apply_transport_status(
        self,
        status: ServerStatus,
        server_conn: ServerConnection,
    ) -> None:
        transport_snapshot = self._transport_snapshot_for_status(server_conn)
        status.transport_channels = transport_snapshot

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
        status.ping_activity_buckets = server_conn.build_ping_activity_buckets(
            bucket_seconds,
            bucket_count,
        )
        status.ping_activity_bucket_seconds = bucket_seconds
        status.ping_activity_bucket_count = bucket_count

    @staticmethod
    def _transport_snapshot_for_status(
        server_conn: ServerConnection,
    ) -> TransportSnapshot | None:
        metrics = server_conn.transport_metrics
        if metrics is None:
            return None
        try:
            return metrics.snapshot()
        except Exception:
            logger.debug(
                "Failed to snapshot transport metrics for server '%s'",
                server_conn.server_name,
                exc_info=True,
            )
            return None

    def _server_config_for_status(self, server_name: str) -> MCPServerSettings | None:
        server_registry = self.context.server_registry if self.context else None
        if server_registry is None:
            return None
        try:
            return server_registry.get_server_config(server_name)
        except Exception:
            return None

    def _apply_config_status(
        self,
        status: ServerStatus,
        server_cfg: MCPServerSettings | None,
        server_conn: ServerConnection | None,
    ) -> None:
        if server_cfg is None:
            status.sampling_mode = status.sampling_mode or self._auto_sampling_mode()
            return

        if status.instructions_enabled is None:
            status.instructions_enabled = server_cfg.include_instructions
        roots = server_cfg.roots
        status.roots_configured = bool(roots)
        status.roots_count = len(roots) if roots else 0
        status.transport = server_cfg.transport or status.transport
        elicitation = server_cfg.elicitation
        if elicitation:
            status.elicitation_mode = elicitation.mode
        status.ping_interval_seconds = (
            status.ping_interval_seconds or server_cfg.ping_interval_seconds
        )
        status.ping_max_missed = status.ping_max_missed or server_cfg.max_missed_pings
        status.spoofing_enabled = server_cfg.implementation is not None
        if status.implementation_name is None and server_cfg.implementation is not None:
            status.implementation_name = server_cfg.implementation.name
            status.implementation_version = server_cfg.implementation.version
        self._apply_config_session_id(status, server_cfg, server_conn)
        status.sampling_mode = (
            "configured" if server_cfg.sampling is not None else self._auto_sampling_mode()
        )

    def _apply_config_session_id(
        self,
        status: ServerStatus,
        server_cfg: MCPServerSettings,
        server_conn: ServerConnection | None,
    ) -> None:
        if status.session_id is not None:
            return
        if server_cfg.transport == "stdio":
            status.session_id = "local"
        elif server_conn is not None:
            status.session_id = self._session_id_from_callback(server_conn)

    def _auto_sampling_mode(self) -> Literal["auto", "off"]:
        auto_sampling = True
        if self.context and self.context.config is not None:
            auto_sampling = self.context.config.auto_sampling
        return "auto" if auto_sampling else "off"

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

        async def try_execute(client: ClientSession) -> R:
            return await self._execute_session_method(
                client,
                server_name=server_name,
                operation_name=operation_name,
                method_name=method_name,
                method_args=method_args,
                error_factory=error_factory,
                progress_callback=progress_callback,
            )

        success_flag: bool | None = None
        result: R | None = None

        try:
            result = await self._execute_initial_server_operation(server_name, try_execute)
            success_flag = True
        except ConnectionError:
            recovery = await self._handle_connection_error(server_name, try_execute, error_factory)
            result = recovery.result
            success_flag = recovery.success
        except ServerSessionTerminatedError as exc:
            recovery = await self._handle_session_terminated(
                server_name, try_execute, error_factory, exc
            )
            result = recovery.result
            success_flag = recovery.success
        except Exception as exc:
            if self._should_retry_with_oauth(server_name, exc):
                recovery = await self._handle_auth_challenge(
                    server_name, try_execute, error_factory
                )
                result = recovery.result
                success_flag = recovery.success
            else:
                success_flag = False
                raise
        finally:
            if success_flag is not None:
                await self._record_server_call(server_name, operation_type, success_flag)

        return self._resolved_server_operation_result(
            result,
            server_name=server_name,
            operation_name=operation_name,
            method_name=method_name,
            error_factory=error_factory,
        )

    async def _execute_session_method(
        self,
        client: ClientSession,
        *,
        server_name: str,
        operation_name: str,
        method_name: str,
        method_args: dict[str, Any] | None,
        error_factory: Callable[[str], R] | None,
        progress_callback: ProgressFnT | None,
    ) -> R:
        try:
            method = getattr(client, method_name)
            kwargs = self._server_method_kwargs(method_name, method_args)
            if method_name == "call_tool" and progress_callback:
                result = await method(progress_callback=progress_callback, **kwargs)
            else:
                result = await method(**kwargs)

            return result
        except (ConnectionError, ServerSessionTerminatedError):
            raise
        except Exception as e:
            return self._handle_session_method_error(
                exc=e,
                server_name=server_name,
                operation_name=operation_name,
                method_name=method_name,
                error_factory=error_factory,
            )

    @staticmethod
    def _server_method_kwargs(
        method_name: str,
        method_args: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs = dict(method_args or {})
        if method_name not in {"call_tool", "read_resource", "get_prompt"}:
            return kwargs

        from fast_agent.llm.fastagent_llm import _mcp_metadata_var

        metadata = _mcp_metadata_var.get()
        if metadata:
            kwargs["meta"] = metadata
        return kwargs

    def _handle_session_method_error(
        self,
        *,
        exc: Exception,
        server_name: str,
        operation_name: str,
        method_name: str,
        error_factory: Callable[[str], R] | None,
    ) -> R:
        error_msg = f"Failed to {method_name} '{operation_name}' on server '{server_name}': {exc}"
        logger.error(error_msg)
        if error_factory is None:
            raise exc

        error_result = error_factory(error_msg)
        payload = MCPAgentClientSession.get_url_elicitation_required_payload(exc)
        if payload is not None:
            with suppress(Exception):
                set_url_elicitation_required_payload(error_result, payload)
        return error_result

    async def _execute_initial_server_operation(
        self,
        server_name: str,
        try_execute: Callable[[ClientSession], Awaitable[R]],
    ) -> R:
        if self.connection_persistence:
            return await self._execute_persistent_server_operation(server_name, try_execute)
        return await self._execute_temporary_server_operation(server_name, try_execute)

    async def _execute_persistent_server_operation(
        self,
        server_name: str,
        try_execute: Callable[[ClientSession], Awaitable[R]],
    ) -> R:
        manager = self._require_connection_manager()
        server_connection = await manager.get_server(
            server_name,
            client_session_factory=self._create_session_factory(server_name),
        )
        session = server_connection.session
        if session is None:
            raise RuntimeError(f"Server session not initialized for '{server_name}'")
        return await try_execute(session)

    async def _execute_temporary_server_operation(
        self,
        server_name: str,
        try_execute: Callable[[ClientSession], Awaitable[R]],
    ) -> R:
        logger.debug(
            f"Creating temporary connection to server: {server_name}",
            data={
                "progress_action": ProgressAction.CONNECTING,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )
        server_registry = self._require_server_registry()
        async with gen_client(server_name, server_registry=server_registry) as client:
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

    @staticmethod
    def _resolved_server_operation_result(
        result: R | None,
        *,
        server_name: str,
        operation_name: str,
        method_name: str,
        error_factory: Callable[[str], R] | None,
    ) -> R:
        if result is None:
            error_msg = f"Failed to {method_name} '{operation_name}' on server '{server_name}'"
            if error_factory:
                return error_factory(error_msg)
            raise RuntimeError(error_msg)
        return result

    def _should_retry_with_oauth(self, server_name: str, exc: Exception) -> bool:
        if self.connection_persistence:
            manager = self._require_connection_manager()
            return manager.should_retry_server_with_oauth(server_name, exc)

        server_registry = self._require_server_registry()
        config = server_registry.get_server_config(server_name)
        if config is None:
            return False
        return _resolve_oauth_mode(
            config, trigger_oauth=None
        ) == "auto" and _is_http_auth_challenge_error(exc)

    async def _handle_auth_challenge(
        self,
        server_name: str,
        try_execute: Callable,
        error_factory: Callable[[str], R] | None,
    ) -> _ServerOperationRecovery[R]:
        from fast_agent.ui import console

        console.console.print(
            f"[dim yellow]MCP server {server_name} requested authorization - reconnecting with OAuth...[/dim yellow]"
        )

        try:
            if self.connection_persistence:
                manager = self._require_connection_manager()
                server_connection = await manager.reconnect_server(
                    server_name,
                    client_session_factory=self._create_session_factory(server_name),
                    trigger_oauth=True,
                )
                session = server_connection.session
                if session is None:
                    raise RuntimeError(f"Server session not initialized for '{server_name}'")
                result = await try_execute(session)
            else:
                server_registry = self._require_server_registry()
                async with gen_client(
                    server_name,
                    server_registry=server_registry,
                    trigger_oauth=True,
                ) as client:
                    result = await try_execute(client)
            console.console.print(
                f"[dim green]MCP server {server_name} reconnected with OAuth successfully[/dim green]"
            )
            return _ServerOperationRecovery(result=result, success=True)
        except Exception as retry_exc:
            if error_factory:
                return _ServerOperationRecovery(
                    result=error_factory(str(retry_exc)),
                    success=False,
                )
            raise

    async def _handle_connection_error(
        self,
        server_name: str,
        try_execute: Callable,
        error_factory: Callable[[str], R] | None,
    ) -> _ServerOperationRecovery[R]:
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
                async with gen_client(server_name, server_registry=server_registry) as client:
                    result = await try_execute(client)

            # Success!
            console.console.print(f"[dim green]MCP server {server_name} online[/dim green]")
            return _ServerOperationRecovery(result=result, success=True)

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
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
            raise RuntimeError(error_msg) from None

        except Exception as e:
            # Reconnection failed
            console.console.print(
                f"[dim red]MCP server {server_name} offline - failed to reconnect: {e}[/dim red]"
            )
            error_msg = f"MCP server {server_name} offline - failed to reconnect"
            if error_factory:
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
            raise RuntimeError(error_msg) from e

    async def _handle_session_terminated(
        self,
        server_name: str,
        try_execute: Callable,
        error_factory: Callable[[str], R] | None,
        exc: ServerSessionTerminatedError,
    ) -> _ServerOperationRecovery[R]:
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
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
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
                async with gen_client(server_name, server_registry=server_registry) as client:
                    result = await try_execute(client)

            # Success! Record the reconnection
            await self._record_reconnect(server_name)
            console.console.print(
                f"[dim green]MCP server {server_name} reconnected successfully[/dim green]"
            )
            return _ServerOperationRecovery(result=result, success=True)

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
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
            raise RuntimeError(error_msg) from None

        except Exception as e:
            # Other reconnection failure
            console.console.print(
                f"[dim red]MCP server {server_name} failed to reconnect: {e}[/dim red]"
            )
            error_msg = f"MCP server {server_name} failed to reconnect: {e}"
            if error_factory:
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
            raise RuntimeError(error_msg) from e

    async def _parse_resource_name(
        self,
        name: str,
        resource_type: str,
    ) -> _ResourceNameResolution:
        """
        Parse a possibly namespaced resource name into server name and local resource name.

        Args:
            name: The resource name, possibly namespaced
            resource_type: Type of resource (for error messages), e.g. "tool", "prompt"

        Returns:
            Server name plus local resource name.
        """
        # First, check if this is a direct hit in our namespaced tool map
        # This handles both namespaced and non-namespaced direct lookups
        if resource_type == "tool" and name in self._namespaced_tool_map:
            namespaced_tool = self._namespaced_tool_map[name]
            return _ResourceNameResolution(
                server_name=namespaced_tool.server_name,
                local_name=namespaced_tool.tool.name,
            )

        # Next, attempt to interpret as a namespaced name
        if is_namespaced_name(name):
            # Try to match against known server names, handling server names with hyphens
            for server_name in self.server_names:
                if name.startswith(f"{server_name}{SEP}"):
                    local_name = name[len(server_name) + len(SEP) :]
                    return _ResourceNameResolution(server_name=server_name, local_name=local_name)

            # If no server name matched, it might be a tool with a hyphen in its name
            # Fall through to the next checks

        # For tools, search all servers for the tool by exact name match
        if resource_type == "tool":
            for server_name, tools in self._server_to_tool_map.items():
                for namespaced_tool in tools:
                    if namespaced_tool.tool.name == name:
                        return _ResourceNameResolution(server_name=server_name, local_name=name)

        # For all other resource types, use the first server
        return _ResourceNameResolution(
            server_name=self.server_names[0] if self.server_names else None,
            local_name=name,
        )

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
        tool_name_resolution = await self._parse_resource_name(name, "tool")
        server_name = tool_name_resolution.server_name
        local_tool_name = tool_name_resolution.local_name

        if server_name is None:
            logger.error(f"Error: Tool '{name}' not found")
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Tool '{name}' not found")],
            )

        namespaced_tool_name = create_namespaced_name(server_name, local_tool_name)
        active_tool_handler = request_tool_handler or self._tool_handler

        permission_error = await self._tool_permission_error_result(
            local_tool_name=local_tool_name,
            server_name=server_name,
            namespaced_tool_name=namespaced_tool_name,
            arguments=arguments,
            tool_use_id=tool_use_id,
            active_tool_handler=active_tool_handler,
        )
        if permission_error is not None:
            return permission_error

        tool_call_id = await self._start_tool_execution(
            active_tool_handler,
            local_tool_name=local_tool_name,
            server_name=server_name,
            arguments=arguments,
            tool_use_id=tool_use_id,
        )

        logger.info(
            "Requesting tool call",
            data=build_progress_payload(
                action=ProgressAction.CALLING_TOOL,
                tool_name=local_tool_name,
                server_name=server_name,
                agent_name=self.agent_name,
                tool_call_id=tool_call_id,
                tool_use_id=tool_use_id,
            ),
        )

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
                tool_use_id,
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

                await self._complete_tool_execution(
                    active_tool_handler,
                    result=result,
                    local_tool_name=local_tool_name,
                    server_name=server_name,
                    tool_call_id=tool_call_id,
                    tool_use_id=tool_use_id,
                )
                return result

            except Exception as e:
                await self._fail_tool_execution(
                    active_tool_handler,
                    exc=e,
                    local_tool_name=local_tool_name,
                    server_name=server_name,
                    tool_call_id=tool_call_id,
                    tool_use_id=tool_use_id,
                )
                raise

    async def _tool_permission_error_result(
        self,
        *,
        local_tool_name: str,
        server_name: str,
        namespaced_tool_name: str,
        arguments: dict | None,
        tool_use_id: str | None,
        active_tool_handler: ToolExecutionHandler,
    ) -> CallToolResult | None:
        try:
            permission_result = await self._permission_handler.check_permission(
                tool_name=local_tool_name,
                server_name=server_name,
                arguments=arguments,
                tool_use_id=tool_use_id,
            )
        except Exception as e:
            logger.error(f"Error checking tool permission: {e}", exc_info=True)
            return self._tool_error_result(f"Permission check failed: {e}")

        if permission_result.allowed:
            return None

        error_msg = self._permission_denied_message(permission_result, namespaced_tool_name)
        await self._notify_tool_permission_denied(
            active_tool_handler,
            local_tool_name=local_tool_name,
            server_name=server_name,
            tool_use_id=tool_use_id,
            error_msg=error_msg,
        )
        logger.info(
            "Tool execution denied by permission handler",
            data={
                "tool_name": local_tool_name,
                "server_name": server_name,
                "cancelled": permission_result.is_cancelled,
            },
        )
        return self._tool_error_result(error_msg)

    @staticmethod
    def _permission_denied_message(
        permission_result: ToolPermissionResult,
        namespaced_tool_name: str,
    ) -> str:
        if permission_result.error_message is not None:
            return permission_result.error_message
        if permission_result.remember:
            return (
                "The user has permanently declined permission to use this tool: "
                f"{namespaced_tool_name}"
            )
        return f"The user has declined permission to use this tool: {namespaced_tool_name}"

    @staticmethod
    def _tool_error_result(message: str) -> CallToolResult:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=message)],
        )

    @staticmethod
    async def _notify_tool_permission_denied(
        active_tool_handler: ToolExecutionHandler,
        *,
        local_tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error_msg: str,
    ) -> None:
        try:
            await active_tool_handler.on_tool_permission_denied(
                local_tool_name,
                server_name,
                tool_use_id,
                error_msg,
            )
        except Exception as e:
            logger.error(f"Error notifying permission denial: {e}", exc_info=True)

    @staticmethod
    async def _start_tool_execution(
        active_tool_handler: ToolExecutionHandler,
        *,
        local_tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None,
    ) -> str:
        try:
            return await active_tool_handler.on_tool_start(
                local_tool_name,
                server_name,
                arguments,
                tool_use_id,
            )
        except Exception as e:
            logger.error(f"Error in tool start handler: {e}", exc_info=True)
            import uuid

            return str(uuid.uuid4())

    async def _complete_tool_execution(
        self,
        active_tool_handler: ToolExecutionHandler,
        *,
        result: CallToolResult,
        local_tool_name: str,
        server_name: str,
        tool_call_id: str,
        tool_use_id: str | None,
    ) -> None:
        completion_state = "completed" if not result.isError else "failed"
        logger.info(
            "Tool call completed",
            data=build_progress_payload(
                action=ProgressAction.TOOL_PROGRESS,
                tool_name=local_tool_name,
                server_name=server_name,
                agent_name=self.agent_name,
                tool_call_id=tool_call_id,
                tool_use_id=tool_use_id,
                details=completion_state,
                tool_state=completion_state,
                tool_terminal=True,
            ),
        )
        await self._notify_tool_complete(active_tool_handler, tool_call_id, result)

    @staticmethod
    async def _notify_tool_complete(
        active_tool_handler: ToolExecutionHandler,
        tool_call_id: str,
        result: CallToolResult,
    ) -> None:
        try:
            content = result.content if result.content else None
            logger.debug(
                f"Tool execution completed, notifying handler: {_display_tool_id(tool_call_id)}",
                name="mcp_tool_complete_notify",
                tool_call_id=tool_call_id,
                has_content=content is not None,
                content_count=len(content) if content else 0,
                is_error=result.isError,
            )

            error_text = None
            if result.isError and content:
                text_parts = [text for c in content if (text := get_text(c))]
                error_text = "\n".join(text_parts) if text_parts else None
                content = None

            await active_tool_handler.on_tool_complete(
                tool_call_id,
                not result.isError,
                content,
                error_text,
            )
            logger.debug(
                f"Tool handler notified successfully: {_display_tool_id(tool_call_id)}",
                name="mcp_tool_complete_done",
            )
        except Exception as e:
            logger.error(f"Error in tool complete handler: {e}", exc_info=True)

    async def _fail_tool_execution(
        self,
        active_tool_handler: ToolExecutionHandler,
        *,
        exc: Exception,
        local_tool_name: str,
        server_name: str,
        tool_call_id: str,
        tool_use_id: str | None,
    ) -> None:
        logger.info(
            "Tool call failed",
            data=build_progress_payload(
                action=ProgressAction.TOOL_PROGRESS,
                tool_name=local_tool_name,
                server_name=server_name,
                agent_name=self.agent_name,
                tool_call_id=tool_call_id,
                tool_use_id=tool_use_id,
                details=f"failed: {exc}",
                tool_state="failed",
                tool_terminal=True,
            ),
        )
        try:
            await active_tool_handler.on_tool_complete(tool_call_id, False, None, str(exc))
        except Exception as handler_error:
            logger.error(f"Error in tool complete handler: {handler_error}", exc_info=True)

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
        :return: GetPromptResult containing the prompt description and messages, with
                 fast-agent display metadata in ``meta``
        """
        if not self.initialized:
            await self.load_servers()

        prompt = self._resolve_prompt_name(prompt_name, server_name)
        if prompt.server_name:
            return await self._get_prompt_from_specific_server(prompt, arguments)

        # No specific server - use the cache to find servers that have this prompt
        logger.debug(f"Searching for prompt '{prompt.local_name}' using cache")
        cached_result = await self._search_cached_prompt_servers(prompt.local_name, arguments)
        if cached_result is not None:
            return cached_result

        fallback_result = await self._search_all_prompt_servers(prompt.local_name, arguments)
        if fallback_result is not None:
            return fallback_result

        # If we get here, we couldn't find the prompt on any server
        logger.info(f"Prompt '{prompt.local_name}' not found on any server")
        return GetPromptResult(
            description=f"Prompt '{prompt.local_name}' not found on any server",
            messages=[],
        )

    def _resolve_prompt_name(
        self,
        prompt_name: str,
        server_name: str | None,
    ) -> _PromptNameResolution:
        if server_name:
            return _PromptNameResolution(server_name=server_name, local_name=prompt_name)
        if not is_namespaced_name(prompt_name):
            return _PromptNameResolution(server_name=None, local_name=prompt_name)

        potential_server, local_name = prompt_name.split(SEP, 1)
        if potential_server in self.server_names:
            return _PromptNameResolution(server_name=potential_server, local_name=local_name)

        return _PromptNameResolution(server_name=None, local_name=prompt_name)

    async def _get_prompt_from_specific_server(
        self,
        prompt: _PromptNameResolution,
        arguments: dict[str, str] | None,
    ) -> GetPromptResult:
        server_name = prompt.server_name
        if server_name is None:
            raise ValueError("Expected resolved prompt server")

        unavailable = await self._prompt_server_unavailable_result(server_name)
        if unavailable is not None:
            return unavailable

        if await self._cached_prompt_missing(server_name, prompt.local_name):
            logger.debug(
                f"Prompt '{prompt.local_name}' not found in cache for server '{server_name}'"
            )
            return GetPromptResult(
                description=f"Prompt '{prompt.local_name}' not found on server '{server_name}'",
                messages=[],
            )

        result = await self._fetch_prompt_from_server(
            server_name,
            prompt.local_name,
            arguments,
            error_factory=lambda msg: GetPromptResult(description=msg, messages=[]),
        )
        if result and result.messages:
            return self._prompt_result_with_metadata(
                result, server_name, prompt.local_name, arguments
            )
        return result or GetPromptResult(
            description=f"Prompt '{prompt.local_name}' not found on server '{server_name}'",
            messages=[],
        )

    async def _prompt_server_unavailable_result(
        self,
        server_name: str,
    ) -> GetPromptResult | None:
        if not await self.validate_server(server_name):
            logger.error(f"Error: Server '{server_name}' not found")
            return GetPromptResult(
                description=f"Error: Server '{server_name}' not found",
                messages=[],
            )

        if await self.server_supports_feature(server_name, "prompts"):
            return None

        logger.debug(f"Server '{server_name}' does not support prompts")
        return GetPromptResult(
            description=f"Server '{server_name}' does not support prompts",
            messages=[],
        )

    async def _cached_prompt_missing(self, server_name: str, prompt_name: str) -> bool:
        if not prompt_name:
            return False
        async with self._prompt_cache_lock:
            if server_name not in self._prompt_cache:
                return False
            prompt_names = {prompt.name for prompt in self._prompt_cache[server_name]}
            return prompt_name not in prompt_names

    async def _search_cached_prompt_servers(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None,
    ) -> GetPromptResult | None:
        potential_servers = await self._servers_with_cached_prompt(prompt_name)
        if not potential_servers:
            logger.debug(f"Prompt '{prompt_name}' not found in any server's cache")
            return None

        logger.debug(f"Found prompt '{prompt_name}' in cache for servers: {potential_servers}")
        return await self._search_prompt_servers(
            potential_servers,
            prompt_name,
            arguments,
            update_cache_on_hit=False,
        )

    async def _servers_with_cached_prompt(self, prompt_name: str) -> list[str]:
        potential_servers = []
        async with self._prompt_cache_lock:
            for s_name, prompt_list in self._prompt_cache.items():
                if any(prompt.name == prompt_name for prompt in prompt_list):
                    potential_servers.append(s_name)
        return potential_servers

    async def _search_all_prompt_servers(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None,
    ) -> GetPromptResult | None:
        supported_servers = []
        for s_name in self.server_names:
            if await self._server_supports_prompts(s_name):
                supported_servers.append(s_name)
            else:
                logger.debug(
                    f"Server '{s_name}' does not support prompts, skipping from fallback search"
                )

        return await self._search_prompt_servers(
            supported_servers,
            prompt_name,
            arguments,
            update_cache_on_hit=True,
        )

    async def _search_prompt_servers(
        self,
        server_names: list[str],
        prompt_name: str,
        arguments: dict[str, str] | None,
        *,
        update_cache_on_hit: bool,
    ) -> GetPromptResult | None:
        for s_name in server_names:
            if not await self._server_supports_prompts(s_name):
                logger.debug(f"Server '{s_name}' does not support prompts, skipping")
                continue

            result = await self._fetch_prompt_quietly(s_name, prompt_name, arguments)
            if not result or not result.messages:
                continue

            logger.debug(f"Successfully retrieved prompt '{prompt_name}' from server '{s_name}'")
            if update_cache_on_hit:
                await self._cache_prompt_from_server(s_name, prompt_name)
            return self._prompt_result_with_metadata(result, s_name, prompt_name, arguments)
        return None

    async def _fetch_prompt_quietly(
        self,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, str] | None,
    ) -> GetPromptResult | None:
        try:
            return await self._fetch_prompt_from_server(
                server_name,
                prompt_name,
                arguments,
                error_factory=lambda _: None,
            )
        except Exception as e:
            logger.debug(f"Error retrieving prompt from server '{server_name}': {e}")
            return None

    async def _fetch_prompt_from_server(
        self,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, str] | None,
        *,
        error_factory: Callable[[str], GetPromptResult | None],
    ) -> GetPromptResult | None:
        return await self._execute_on_server(
            server_name=server_name,
            operation_type="prompts/get",
            operation_name=prompt_name or "default",
            method_name="get_prompt",
            method_args=self._prompt_method_args(prompt_name, arguments),
            error_factory=error_factory,
        )

    @staticmethod
    def _prompt_method_args(
        prompt_name: str,
        arguments: dict[str, str] | None,
    ) -> dict[str, Any]:
        method_args: dict[str, Any] = {"name": prompt_name} if prompt_name else {}
        if arguments:
            method_args["arguments"] = arguments
        return method_args

    @staticmethod
    def _prompt_result_with_metadata(
        result: GetPromptResult,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, str] | None,
    ) -> GetPromptResult:
        return with_prompt_metadata(
            result,
            namespaced_name=create_namespaced_name(server_name, prompt_name),
            arguments=arguments,
        )

    async def _cache_prompt_from_server(self, server_name: str, prompt_name: str) -> None:
        with suppress(Exception):
            prompt_list_result: ListPromptsResult | None = await self._execute_on_server(
                server_name=server_name,
                operation_type="prompts/list",
                operation_name="",
                method_name="list_prompts",
                error_factory=lambda _: None,
            )
            if prompt_list_result is None:
                return

            matching_prompt = next(
                (prompt for prompt in prompt_list_result.prompts if prompt.name == prompt_name),
                None,
            )
            if matching_prompt is None:
                return

            async with self._prompt_cache_lock:
                cached_prompts = self._prompt_cache.setdefault(server_name, [])
                if all(prompt.name != prompt_name for prompt in cached_prompts):
                    cached_prompts.append(matching_prompt)

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

        if server_name:
            return await self._list_prompts_for_server(server_name)

        cached_results = await self._cached_prompts_for_all_servers()
        if cached_results is not None:
            return cached_results

        results: dict[str, list[Prompt]] = {}
        supported_servers: list[str] = []
        for s_name in self.server_names:
            if await self._server_supports_prompts(s_name):
                supported_servers.append(s_name)
            else:
                logger.debug(f"Server '{s_name}' does not support prompts, skipping")
                results[s_name] = []

        for s_name in supported_servers:
            results[s_name] = await self._fetch_and_cache_prompts(s_name)

        logger.debug(f"Available prompts across servers: {results}")
        return results

    async def _list_prompts_for_server(self, server_name: str) -> dict[str, list[Prompt]]:
        if server_name not in self.server_names:
            logger.error(f"Server '{server_name}' not found")
            return {}

        cached_prompts = await self._cached_prompts_for_server(server_name)
        if cached_prompts is not None:
            return {server_name: cached_prompts}

        if not await self._server_supports_prompts(server_name):
            logger.debug(f"Server '{server_name}' does not support prompts")
            return {server_name: []}

        return {server_name: await self._fetch_and_cache_prompts(server_name)}

    async def _cached_prompts_for_server(self, server_name: str) -> list[Prompt] | None:
        async with self._prompt_cache_lock:
            if server_name not in self._prompt_cache:
                return None
            logger.debug(f"Returning cached prompts for server '{server_name}'")
            return self._prompt_cache[server_name]

    async def _cached_prompts_for_all_servers(self) -> dict[str, list[Prompt]] | None:
        async with self._prompt_cache_lock:
            if not all(s_name in self._prompt_cache for s_name in self.server_names):
                return None
            logger.debug("Returning cached prompts for all servers")
            return dict(self._prompt_cache)

    async def _server_supports_prompts(self, server_name: str) -> bool:
        capabilities = await self.get_capabilities(server_name)
        return bool(capabilities and capabilities.prompts)

    async def _fetch_and_cache_prompts(self, server_name: str) -> list[Prompt]:
        try:
            result: ListPromptsResult | None = await self._execute_on_server(
                server_name=server_name,
                operation_type="prompts/list",
                operation_name="",
                method_name="list_prompts",
                error_factory=lambda _: None,
            )
            if result is None:
                return []

            prompts = result.prompts
            async with self._prompt_cache_lock:
                self._prompt_cache[server_name] = prompts
            return prompts
        except Exception as e:
            logger.debug(f"Error fetching prompts from {server_name}: {e}")
            return []

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
            data=build_progress_payload(
                action=ProgressAction.READING_RESOURCE,
                server_name=server_name,
                agent_name=self.agent_name,
                details=resource_uri,
                extra={"resource_uri": resource_uri},
            ),
        )

        try:
            uri = AnyUrl(resource_uri)
        except Exception as e:
            raise ValueError(f"Invalid resource URI: {resource_uri}. Error: {e}") from e

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

    async def read_directory(
        self,
        uri: str,
        *,
        server_name: str | None = None,
        cursor: str | None = None,
    ) -> ListResourcesResult:
        """List the direct children of a directory resource via SEP-2640.

        Routes ``resources/directory/read`` to the named server. Callers should
        only invoke this against servers that declared ``directoryRead`` for the
        ``io.modelcontextprotocol/skills`` extension.
        """
        if not self.initialized:
            await self.load_servers()

        if server_name is not None:
            if server_name not in self.server_names:
                raise ValueError(f"Server '{server_name}' not found")
            return await self._read_directory_from_server(server_name, uri, cursor=cursor)

        if not self.server_names:
            raise ValueError("No servers available to read directory from")

        for s_name in self.server_names:
            try:
                return await self._read_directory_from_server(s_name, uri, cursor=cursor)
            except Exception:
                continue

        raise ValueError(f"Directory '{uri}' not found on any server")

    async def _read_directory_from_server(
        self, server_name: str, uri: str, *, cursor: str | None = None
    ) -> ListResourcesResult:
        """Internal helper to call ``resources/directory/read`` on a server."""
        if not await self.server_supports_feature(server_name, "resources"):
            raise ValueError(f"Server '{server_name}' does not support resources")

        try:
            uri_obj = AnyUrl(uri)
        except Exception as e:
            raise ValueError(f"Invalid directory URI: {uri}. Error: {e}") from e

        method_args: dict[str, Any] = {"uri": uri_obj}
        if cursor is not None:
            method_args["cursor"] = cursor

        result = await self._execute_on_server(
            server_name=server_name,
            operation_type="resources/directory/read",
            operation_name=uri,
            method_name="read_directory",
            method_args=method_args,
        )

        if result is None:
            raise ValueError(f"Directory '{uri}' not found on server '{server_name}'")

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

        result: ListResourcesResult = await self._execute_on_server(
            server_name=server_name,
            operation_type="resources/list",
            operation_name="",
            method_name="list_resources",
            method_args={},
        )

        return result.resources

    async def _list_resource_templates_from_server(
        self, server_name: str, *, check_support: bool = True
    ) -> list[ResourceTemplate]:
        """Internal helper to list resource templates from a specific server."""
        if check_support and not await self.server_supports_feature(server_name, "resources"):
            return []

        result: ListResourceTemplatesResult = await self._execute_on_server(
            server_name=server_name,
            operation_type="resources/templates/list",
            operation_name="",
            method_name="list_resource_templates",
            method_args={},
            error_factory=lambda _: ListResourceTemplatesResult(resourceTemplates=[]),
        )

        return result.resourceTemplates

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
                resources: list[Resource] = await self._list_resources_from_server(
                    s_name, check_support=False
                )
                formatted_resources: list[str] = []
                for resource in resources:
                    uri = resource.uri
                    if uri is not None:
                        formatted_resources.append(str(uri))
                results[s_name] = formatted_resources
            except Exception as e:
                logger.error(f"Error fetching resources from {s_name}: {e}")

        return results

    async def list_resource_templates(
        self, server_name: str | None = None
    ) -> dict[str, list[ResourceTemplate]]:
        """List available resource templates from one or all servers."""
        if not self.initialized:
            await self.load_servers()

        results: dict[str, list[ResourceTemplate]] = {}
        servers_to_check = [server_name] if server_name else self.server_names

        for s_name in servers_to_check:
            if s_name not in self.server_names:
                logger.error(f"Server '{s_name}' not found")
                continue

            results[s_name] = []

            if not await self.server_supports_feature(s_name, "resources"):
                logger.debug(f"Server '{s_name}' does not support resources")
                continue

            try:
                templates = await self._list_resource_templates_from_server(
                    s_name, check_support=False
                )
                results[s_name] = list(templates)
            except Exception as e:
                logger.error(f"Error fetching resource templates from {s_name}: {e}")

        return results

    async def complete_resource_argument(
        self,
        server_name: str,
        template_uri: str,
        argument_name: str,
        value: str,
        context_args: dict[str, str] | None = None,
    ) -> Completion:
        """Request MCP completion for resource template argument values."""
        if not await self.validate_server(server_name):
            return Completion(values=[])

        if not await self.server_supports_feature(server_name, "completions"):
            return Completion(values=[])

        result: CompleteResult = await self._execute_on_server(
            server_name=server_name,
            operation_type="completion/complete",
            operation_name=template_uri,
            method_name="complete",
            method_args={
                "ref": ResourceTemplateReference(type="ref/resource", uri=template_uri),
                "argument": {"name": argument_name, "value": value},
                "context_arguments": context_args,
            },
            error_factory=lambda _msg: CompleteResult(completion=Completion(values=[])),
        )

        return result.completion

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
                result: ListToolsResult = await self._execute_on_server(
                    server_name=s_name,
                    operation_type="tools/list",
                    operation_name="",
                    method_name="list_tools",
                    method_args={},
                )

                # Get tools from result (these have original names, not namespaced)
                tools = result.tools
                results[s_name] = tools

            except Exception as e:
                logger.error(f"Error fetching tools from {s_name}: {e}")

        return results
