import sys
from asyncio import Lock
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
)

from mcp import GetPromptResult, ReadResourceResult
from mcp.client import CacheMode
from mcp.client.subscriptions import ServerEvent
from mcp.shared.dispatcher import ProgressFnT
from mcp.shared.exceptions import MCPError
from mcp_types import (
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
from fast_agent.core.model_resolution import get_context_cli_model_override, resolve_model_spec
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.app_integrations import (
    AppResourceConfig,
    AppServerConfig,
    AppToolConfig,
    expected_mime_type,
    extract_app_tool_metadata,
    integration_kind_for_mime_type,
    mark_tool_metadata,
    supported_mime_types,
)
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_connection import MCPClientConnection
from fast_agent.mcp.client_gateway import (
    is_http_auth_challenge,
    resolve_oauth_mode,
)
from fast_agent.mcp.common import SEP, create_namespaced_name, is_namespaced_name
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.interfaces import ServerRegistryProtocol
from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager, ServerConnection
from fast_agent.mcp.prompt_metadata import with_prompt_metadata
from fast_agent.mcp.skills_extension import GetSkillResult, ListSkillsResult
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler, ToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import (
    NoOpToolPermissionHandler,
    ToolPermissionHandler,
    ToolPermissionResult,
)
from fast_agent.mcp.tool_result_metadata import (
    set_url_elicitation_required_payload,
    url_elicitation_required_payload,
)
from fast_agent.mcp.transport_tracking import TransportSnapshot
from fast_agent.skills.mcp_registry import (
    McpSkillRegistry,
    scan_mcp_skill_registry,
    server_supports_mcp_skills,
)
from fast_agent.ui.tool_call_ids import format_tool_call_id
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.env import env_flag
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from fast_agent.context import Context
    from fast_agent.mcp.oauth_client import OAuthEvent
    from fast_agent.mcp_server_registry import ServerRegistry


logger = get_logger(__name__)  # This will be replaced per-instance when agent_name is available

type MCPOperationClient = MCPClientConnection

_CONNECTION_ERROR_REPLAY_SAFE_METHODS = frozenset(
    {
        "complete",
        "get_prompt",
        "get_skill",
        "list_skills",
        "list_prompts",
        "list_resource_templates",
        "list_resources",
        "list_tools",
        "read_directory",
        "read_resource",
    }
)


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

    def __iter__(self):
        yield self.result
        yield self.success


@dataclass(frozen=True, slots=True)
class _PromptNameResolution:
    server_name: str | None
    local_name: str


@dataclass(frozen=True, slots=True)
class _AttachedRegistryScanClient:
    aggregator: "MCPAggregator"
    cache_mode: CacheMode = "use"

    async def get_capabilities(self, server_name: str) -> ServerCapabilities | None:
        return await self.aggregator.get_capabilities(server_name)

    async def list_skills(
        self,
        server_name: str,
        cursor: str | None,
    ) -> ListSkillsResult:
        return await self.aggregator._list_skills_from_server(
            server_name,
            cursor=cursor,
        )

    async def get_skill(self, uri: str, server_name: str) -> GetSkillResult:
        return await self.aggregator._get_skill_from_server(server_name, uri)


METHOD_NOT_FOUND_ERROR_CODE = -32601
METHOD_NOT_FOUND_MESSAGE = "method not found"


def _is_capability_probe_error(exc: Exception) -> bool:
    """Return True when exc indicates a server does not support a probed method."""
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, MCPError):
        code = exc.code
        if code == METHOD_NOT_FOUND_ERROR_CODE:
            return True
        # Only fall back to message matching when the server omitted the error code;
        # if a different code is set, trust the code over the message text.
        if code is None:
            if METHOD_NOT_FOUND_MESSAGE in strip_casefold(exc.message):
                return True
    return False


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.
    """

    tool: Tool
    server_name: str
    namespaced_tool_name: str


@dataclass(frozen=True, slots=True)
class ToolNameResolution:
    server_name: str | None
    local_name: str


@dataclass(frozen=True, slots=True)
class MCPToolCatalog:
    """Read-only snapshot of the aggregator's discovered MCP tools."""

    _by_namespaced_name: Mapping[str, NamespacedTool]
    _by_server: Mapping[str, tuple[NamespacedTool, ...]]
    _server_names: tuple[str, ...]

    @classmethod
    def snapshot(
        cls,
        *,
        by_namespaced_name: Mapping[str, NamespacedTool],
        by_server: Mapping[str, Iterable[NamespacedTool]],
        server_names: Iterable[str],
    ) -> "MCPToolCatalog":
        return cls(
            _by_namespaced_name=MappingProxyType(dict(by_namespaced_name)),
            _by_server=MappingProxyType(
                {server_name: tuple(tools) for server_name, tools in by_server.items()}
            ),
            _server_names=tuple(server_names),
        )

    def namespaced_tool(self, name: str) -> NamespacedTool | None:
        return self._by_namespaced_name.get(name)

    def first_tool_named(self, local_name: str) -> NamespacedTool | None:
        return next(
            (
                namespaced_tool
                for namespaced_tool in self._by_namespaced_name.values()
                if namespaced_tool.tool.name == local_name
            ),
            None,
        )

    def routable_tool_names(self) -> frozenset[str]:
        return frozenset(self._by_namespaced_name) | frozenset(
            tool.tool.name for tool in self._by_namespaced_name.values()
        )

    def server_tool_names(self, server_name: str) -> tuple[str, ...]:
        return tuple(
            namespaced_tool.tool.name for namespaced_tool in self._by_server.get(server_name, ())
        )

    def resolve_tool_name(self, name: str) -> ToolNameResolution:
        if namespaced_tool := self.namespaced_tool(name):
            return ToolNameResolution(
                server_name=namespaced_tool.server_name,
                local_name=namespaced_tool.tool.name,
            )

        if is_namespaced_name(name):
            for server_name in self._server_names:
                if name.startswith(f"{server_name}{SEP}"):
                    return ToolNameResolution(
                        server_name=server_name,
                        local_name=name[len(server_name) + len(SEP) :],
                    )

        for server_name, tools in self._by_server.items():
            if any(namespaced_tool.tool.name == name for namespaced_tool in tools):
                return ToolNameResolution(server_name=server_name, local_name=name)

        return ToolNameResolution(
            server_name=self._server_names[0] if self._server_names else None,
            local_name=name,
        )


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
    protocol_mode: Literal["auto", "modern", "legacy"] = "auto"
    implementation_name: str | None = None
    implementation_version: str | None = None
    protocol_version: str | None = None
    protocol_era: str | None = None
    supported_protocol_versions: tuple[str, ...] = ()
    negotiation: str | None = None
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
    subscription_state: str | None = None
    transport_channels: TransportSnapshot | None = None
    app_integration_config: AppServerConfig | None = None
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


@dataclass(frozen=True, slots=True)
class _AttachmentDiscovery:
    tools: list[NamespacedTool]
    prompts: list[Prompt]
    skill_registry: McpSkillRegistry | None
    app_integration_config: AppServerConfig
    capabilities: ServerCapabilities | None


class MCPAggregator(ContextDependent):
    """
    Aggregates multiple MCP servers. When a developer calls, e.g. call_tool(...),
    the aggregator searches all servers in its list for a server that provides that tool.
    """

    initialized: bool = False
    """Whether the aggregator has been initialized with tools and resources from all servers."""

    connection_persistence: bool = False
    """Whether to retain an attached local client runtime for the server."""

    server_names: list[str]
    """A list of server names to connect to."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @staticmethod
    def _unique_preserving_order(items: Iterable[str]) -> list[str]:
        return unique_preserve_order(items)

    async def __aenter__(self):
        if self.initialized:
            return self

        # Keep a runtime manager for attached clients owned by this aggregator.
        if self.connection_persistence:
            context = self._require_context()
            server_registry = cast("ServerRegistry", self._require_server_registry())
            manager = MCPConnectionManager(server_registry, context=context)
            await manager.__aenter__()
            self._persistent_connection_manager = manager
            self._owns_connection_manager = True
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
        context: "Context | None" = None,
        name: str | None = None,
        config: Any | None = None,  # Accept the agent config for elicitation_handler access
        tool_handler: ToolExecutionHandler | None = None,
        permission_handler: ToolPermissionHandler | None = None,
        **kwargs,
    ) -> None:
        """
        :param server_names: A list of server names to connect to.
        :param connection_persistence: Whether to retain attached client runtimes (default: True).
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
        self.config = config  # Agent-specific callback configuration.
        self._persistent_connection_manager: MCPConnectionManager | None = None
        self._owns_connection_manager = False
        self._lifecycle_lock = Lock()
        self._closed = False

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

        self._app_integration_configs: dict[str, AppServerConfig] = {}
        self._mcp_skill_registries: dict[str, McpSkillRegistry] = {}

        # Cache for capabilities discovered by on-demand clients.
        self._capabilities_cache: dict[str, ServerCapabilities] = {}
        self._capabilities_cache_lock = Lock()
        self._attachment_configs: dict[str, MCPServerSettings] = {}
        self._attachment_locks: dict[str, Lock] = {}
        self._staged_discovery_tools: dict[str, list[NamespacedTool]] = {}
        self._attachment_owner = f"aggregator:{id(self)}"
        self._runtime_definition_owner = self._attachment_owner

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

    def _should_use_request_scoped_connection(self, server_name: str) -> bool:
        """Use a fresh MCP transport when auth.forward depends on request context."""
        token_present = bool(request_bearer_token.get())
        if not token_present:
            return False
        try:
            config = self._server_config(server_name)
        except Exception:
            return False
        return (
            config is not None and config.auth is not None and config.auth.forward == "huggingface"
        )

    def _require_connection_manager(self) -> MCPConnectionManager:
        if self._persistent_connection_manager is None:
            raise RuntimeError("MCP runtime manager is not initialized")
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
        Close all attached MCP client runtimes when the aggregator is deleted.
        """
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            try:
                if (
                    self.connection_persistence
                    and self._persistent_connection_manager
                    and self._owns_connection_manager
                ):
                    logger.info("Shutting down attached MCP client runtimes...")
                    await self._persistent_connection_manager.disconnect_all()
                    await self._persistent_connection_manager.__aexit__(None, None, None)
                self.initialized = False
            except Exception as e:
                logger.error(f"Error during connection manager cleanup: {e}")
            finally:
                await self._release_owned_runtime_definitions(disconnect=False)
                self._attachment_configs.clear()
                await self._clear_runtime_indexes()

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

    def _create_callback_runtime(self, server_name: str) -> MCPClientCallbackRuntime:
        """Build callbacks and agent context for an SDK high-level client."""
        agent_name: str | None = None
        elicitation_handler = None
        api_key: str | None = None

        if self.config:
            agent_name = self.config.name
            elicitation_handler = self.config.elicitation_handler
            api_key = self.config.api_key

        return MCPClientCallbackRuntime(
            server_name=server_name,
            server_config=self._server_config(server_name),
            agent_model=self._resolve_callback_agent_model(),
            agent_model_resolver=self._resolve_callback_agent_model,
            agent_name=agent_name,
            api_key=api_key,
            custom_elicitation_handler=elicitation_handler,
            aggregator=self,
            context=self.context,
            tool_list_changed_callback=self._handle_tool_list_changed,
        )

    def _resolve_callback_agent_model(self) -> str | None:
        if self.config is None:
            return None
        return resolve_model_spec(
            self.context,
            model=self.config.model,
            cli_model=get_context_cli_model_override(self.context),
        ).model

    def _server_config(self, server_name: str) -> MCPServerSettings | None:
        return self._attachment_configs.get(
            server_name
        ) or self._require_server_registry().get_server_config(server_name)

    def _attachment_client_kwargs(self, server_name: str) -> dict[str, Any]:
        config = self._attachment_configs.get(server_name)
        if config is None:
            return {}
        return {"server_config": config, "publish_capabilities": False}

    def _attachment_manager_kwargs(self, server_name: str) -> dict[str, Any]:
        config = self._attachment_configs.get(server_name)
        return {"server_config": config} if config is not None else {}

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

        try:
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
        except BaseException:
            for result in reversed(attached_results):
                with suppress(Exception):
                    await self.detach_server(result.server_name)
            registry = self._require_server_registry()
            for server_name in servers_to_load:
                if "cli-startup" in registry.get_runtime_owners(server_name):
                    registry.remove_runtime(server_name, owner="cli-startup")
            raise

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
        async with self._lifecycle_lock:
            await self._release_owned_runtime_definitions(disconnect=True)
            self._attachment_configs.clear()
            await self._clear_runtime_indexes()

    async def _clear_runtime_indexes(self) -> None:
        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()

        async with self._prompt_cache_lock:
            self._prompt_cache.clear()

        async with self._capabilities_cache_lock:
            self._capabilities_cache.clear()

        self._app_integration_configs.clear()
        self._mcp_skill_registries.clear()
        self._attached_server_names = []

    async def _release_owned_runtime_definitions(self, *, disconnect: bool) -> None:
        registry = self.context.server_registry if self.context else None
        if registry is None:
            return
        server_names = set(registry.registry)
        server_names.update(self._attached_server_names)
        server_names.update(self._attachment_configs)
        for server_name in server_names:
            attachment_owned = self._attachment_owner in registry.get_attachment_owners(server_name)
            if attachment_owned:
                registry.release_attachment(
                    server_name,
                    owner=self._attachment_owner,
                )
            if self._runtime_definition_owner in registry.get_runtime_owners(server_name):
                registry.remove_runtime(
                    server_name,
                    owner=self._runtime_definition_owner,
                )
            if attachment_owned and disconnect and self._persistent_connection_manager is not None:
                await self._persistent_connection_manager.disconnect_server(server_name)

    async def _fetch_server_tools(
        self,
        server_name: str,
        *,
        cache_mode: CacheMode = "use",
    ) -> list[Tool]:
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
                method_args={"cache_mode": cache_mode} if cache_mode != "use" else {},
            )
            return result.tools or []
        except Exception as e:
            if supports_tools:
                raise
            if not _is_capability_probe_error(e):
                raise
            logger.debug(f"Server '{server_name}' does not provide tools (list_tools failed): {e}")
            return []

    async def _fetch_server_prompts(
        self,
        server_name: str,
        *,
        strict: bool = False,
        cache_mode: CacheMode = "use",
    ) -> list[Prompt]:
        if not await self.server_supports_feature(server_name, "prompts"):
            logger.debug(f"Server '{server_name}' does not support prompts")
            return []

        try:
            result: ListPromptsResult = await self._execute_on_server(
                server_name=server_name,
                operation_type="prompts/list",
                operation_name="",
                method_name="list_prompts",
                method_args={"cache_mode": cache_mode} if cache_mode != "use" else {},
            )
            return result.prompts
        except Exception as e:
            if strict:
                raise
            logger.debug(f"Error loading prompts from server '{server_name}': {e}")
            return []

    async def attach_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        server_name = self._resolve_server_key(server_name)
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MCP aggregator is closed")
            async with self._attachment_locks.setdefault(server_name, Lock()):
                return await self._attach_server_locked(
                    server_name=server_name,
                    server_config=server_config,
                    options=options,
                )

    async def _attach_server_locked(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None,
        options: MCPAttachOptions | None,
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

        self._attachment_configs[server_name] = resolved_config
        server_registry.register_attachment(
            server_name,
            owner=self._attachment_owner,
        )
        callback_runtime: MCPClientCallbackRuntime | None = None
        try:
            await self._clear_capabilities_for_forced_reconnect(server_name, attach_options)
            if self.connection_persistence:
                callback_runtime = await self._connect_persistent_server(
                    server_name,
                    resolved_config,
                    attach_options,
                )
            discovery = await self._discover_server_attachment(server_name)
            await self._commit_server_attachment(
                server_name,
                discovery,
                runtime_config=server_config,
            )
            if callback_runtime is not None:
                callback_runtime.mark_subscription_ready()
        except BaseException:
            self._attachment_configs.pop(server_name, None)
            await self._rollback_server_attachment(
                server_name,
                clear_existing=already_attached and attach_options.force_reconnect,
            )
            if "cli-startup" in server_registry.get_runtime_owners(server_name):
                server_registry.remove_runtime(server_name, owner="cli-startup")
            raise

        self._log_server_initialized()
        return await self._attached_result(
            server_name=server_name,
            resolved_config=resolved_config,
            already_attached=already_attached,
            existing_tool_names=existing_tool_names,
            existing_prompt_names=existing_prompt_names,
            app_integration_config=discovery.app_integration_config,
        )

    def _resolve_attach_server_config(
        self,
        server_name: str,
        server_config: MCPServerSettings | None,
        attach_options: MCPAttachOptions,
        server_registry: ServerRegistryProtocol,
    ) -> MCPServerSettings:
        if server_config is not None:
            origin = server_registry.get_server_origin(server_name)
            if origin in {"central", "card"}:
                raise ValueError(
                    f"Runtime MCP server '{server_name}' collides with {origin} configuration"
                )
            if origin == "runtime":
                existing = server_registry.get_server_config(server_name)
                if existing != server_config:
                    raise ValueError(
                        f"Runtime MCP server '{server_name}' is already registered with different settings"
                    )
            resolved_config = server_config.model_copy(deep=True)
        else:
            resolved_config = server_registry.get_server_config(server_name)
        if resolved_config is None:
            raise ValueError(f"Server '{server_name}' not found in registry")

        if attach_options.reconnect_on_disconnect is None:
            return resolved_config

        return resolved_config.model_copy(
            update={"reconnect_on_disconnect": attach_options.reconnect_on_disconnect}
        )

    def _attached_tool_names(self, server_name: str) -> set[str]:
        return {tool.namespaced_tool_name for tool in self._server_to_tool_map.get(server_name, [])}

    def _attached_prompt_names(self, server_name: str) -> set[str]:
        return {prompt.name for prompt in self._prompt_cache.get(server_name, [])}

    def _already_attached_result(
        self,
        server_name: str,
        resolved_config: MCPServerSettings,
        existing_tool_names: set[str],
        existing_prompt_names: set[str],
    ) -> MCPAttachResult:
        return MCPAttachResult(
            server_name=self.server_display_name(server_name),
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
        server_config: MCPServerSettings,
        attach_options: MCPAttachOptions,
    ) -> MCPClientCallbackRuntime:
        logger.info(
            f"Creating attached MCP client runtime for server: {server_name}",
            data={
                "progress_action": ProgressAction.CONNECTING,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )

        manager = self._require_connection_manager()
        connect = manager.reconnect_server if attach_options.force_reconnect else manager.get_server
        callback_runtime = self._create_callback_runtime(server_name)
        server_conn = await connect(
            server_name,
            server_config=server_config,
            callback_runtime=callback_runtime,
            startup_timeout_seconds=attach_options.startup_timeout_seconds,
            trigger_oauth=attach_options.trigger_oauth,
            oauth_event_handler=attach_options.oauth_event_handler,
            allow_oauth_paste_fallback=attach_options.allow_oauth_paste_fallback,
        )
        await self._record_connection_negotiation(server_name, server_conn)
        return server_conn._callback_runtime

    async def _discover_server_attachment(
        self,
        server_name: str,
        *,
        cache_mode: CacheMode = "use",
    ) -> _AttachmentDiscovery:
        if cache_mode == "use":
            tools = await self._fetch_server_tools(server_name)
            prompts = await self._fetch_server_prompts(server_name, strict=True)
            mcp_skill_registry = await self._scan_mcp_skill_registry(server_name)
        else:
            tools = await self._fetch_server_tools(server_name, cache_mode=cache_mode)
            prompts = await self._fetch_server_prompts(
                server_name,
                strict=True,
                cache_mode=cache_mode,
            )
            mcp_skill_registry = await self._scan_mcp_skill_registry(
                server_name,
                cache_mode=cache_mode,
            )
        namespace = self.server_display_name(server_name)
        namespaced_tools = [
            NamespacedTool(
                tool=tool,
                server_name=server_name,
                namespaced_tool_name=create_namespaced_name(namespace, tool.name),
            )
            for tool in tools
        ]
        self._staged_discovery_tools[server_name] = namespaced_tools
        try:
            if cache_mode == "use":
                _, app_integration_config = await self._evaluate_app_integrations_for_server(
                    server_name
                )
            else:
                _, app_integration_config = await self._evaluate_app_integrations_for_server(
                    server_name,
                    cache_mode=cache_mode,
                )
        finally:
            self._staged_discovery_tools.pop(server_name, None)
        return _AttachmentDiscovery(
            tools=namespaced_tools,
            prompts=prompts,
            skill_registry=mcp_skill_registry,
            app_integration_config=app_integration_config,
            capabilities=await self.get_capabilities(server_name),
        )

    async def _commit_server_attachment(
        self,
        server_name: str,
        discovery: _AttachmentDiscovery,
        *,
        runtime_config: MCPServerSettings | None = None,
    ) -> None:
        async with self._tool_map_lock:
            async with self._prompt_cache_lock:
                registry = self._require_server_registry()
                if runtime_config is not None:
                    registry.register_runtime(
                        server_name,
                        runtime_config,
                        owner=self._runtime_definition_owner,
                    )
                elif registry.get_server_origin(server_name) == "runtime":
                    registered_config = registry.get_server_config(server_name)
                    if registered_config is None:
                        raise ValueError(f"Server '{server_name}' not found in registry")
                    registry.register_runtime(
                        server_name,
                        registered_config,
                        owner=self._runtime_definition_owner,
                    )
                if "cli-startup" in registry.get_runtime_owners(server_name):
                    registry.remove_runtime(server_name, owner="cli-startup")
                for namespaced in self._server_to_tool_map.get(server_name, []):
                    self._namespaced_tool_map.pop(namespaced.namespaced_tool_name, None)

                self._server_to_tool_map[server_name] = discovery.tools
                for tool in discovery.tools:
                    self._namespaced_tool_map[tool.namespaced_tool_name] = tool
                self._prompt_cache[server_name] = discovery.prompts

                if discovery.skill_registry is None:
                    self._mcp_skill_registries.pop(server_name, None)
                else:
                    self._mcp_skill_registries[server_name] = discovery.skill_registry

                self._app_integration_configs[server_name] = discovery.app_integration_config
                if discovery.capabilities is not None:
                    registry.set_server_capabilities(
                        server_name,
                        discovery.capabilities,
                    )
                if server_name not in self.server_names:
                    self.server_names.append(server_name)
                if server_name not in self._attached_server_names:
                    self._attached_server_names.append(server_name)

    async def _refresh_attached_server_cache(
        self,
        server_name: str,
        *,
        cache_mode: CacheMode = "use",
    ) -> AppServerConfig:
        discovery = await self._discover_server_attachment(server_name, cache_mode=cache_mode)
        await self._commit_server_attachment(server_name, discovery)
        return discovery.app_integration_config

    def selected_materialized_resource_uris(self, server_name: str) -> tuple[str, ...]:
        """Return canonical materialized UI resource URIs for modern updates."""
        config = self._app_integration_configs.get(self._resolve_server_key(server_name))
        if config is None:
            return ()
        return tuple(sorted({str(resource.uri) for resource in config.resources}))

    async def refresh_subscription_state(self, server_name: str) -> tuple[str, ...]:
        """Force and atomically commit authoritative state for an acknowledged listener."""
        server_name = self._resolve_server_key(server_name)
        async with self._lifecycle_lock:
            if self._closed:
                return ()
            async with self._attachment_locks.setdefault(server_name, Lock()):
                if server_name not in self._attached_server_names:
                    return ()
                await self._refresh_attached_server_cache(
                    server_name,
                    cache_mode="refresh",
                )
                return self.selected_materialized_resource_uris(server_name)

    async def handle_subscription_event(
        self,
        server_name: str,
        event: ServerEvent,
    ) -> None:
        """Converge all attached derived state for a modern level-triggered event."""
        logger.debug(
            "Refreshing authoritative MCP state after subscription event",
            data={"server_name": server_name, "event": type(event).__name__},
        )
        await self.refresh_subscription_state(server_name)

    async def _rollback_server_attachment(
        self,
        server_name: str,
        *,
        clear_existing: bool,
    ) -> None:
        registry = self._require_server_registry()
        registry.release_attachment(
            server_name,
            owner=self._attachment_owner,
        )
        if self._persistent_connection_manager is not None:
            with suppress(Exception):
                await self._persistent_connection_manager.disconnect_server(server_name)
        async with self._capabilities_cache_lock:
            self._capabilities_cache.pop(server_name, None)
        registry.clear_server_capabilities(server_name)
        if clear_existing:
            async with self._tool_map_lock:
                for namespaced in self._server_to_tool_map.pop(server_name, []):
                    self._namespaced_tool_map.pop(namespaced.namespaced_tool_name, None)
            async with self._prompt_cache_lock:
                self._prompt_cache.pop(server_name, None)
            self._mcp_skill_registries.pop(server_name, None)
            self._app_integration_configs.pop(server_name, None)
            self._attached_server_names = [
                name for name in self._attached_server_names if name != server_name
            ]
            self.server_names = [name for name in self.server_names if name != server_name]
            registry.remove_runtime(
                server_name,
                owner=self._runtime_definition_owner,
            )

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
        app_integration_config: AppServerConfig,
    ) -> MCPAttachResult:
        tool_names = self._attached_tool_names(server_name)
        prompt_names = self._attached_prompt_names(server_name)
        skills_total = await self._mcp_skills_total(server_name)
        return MCPAttachResult(
            server_name=self.server_display_name(server_name),
            transport=resolved_config.transport,
            attached=True,
            already_attached=already_attached,
            tools_added=sorted(tool_names - existing_tool_names),
            prompts_added=sorted(prompt_names - existing_prompt_names),
            warnings=list(app_integration_config.warnings),
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
        server_name = self._resolve_server_key(server_name)
        async with self._lifecycle_lock:
            async with self._attachment_locks.setdefault(server_name, Lock()):
                return await self._detach_server_locked(server_name)

    async def _detach_server_locked(self, server_name: str) -> MCPDetachResult:
        display_name = self.server_display_name(server_name)
        existing_tools = self._server_to_tool_map.get(server_name, [])
        existing_prompts = self._prompt_cache.get(server_name, [])
        tools_removed = sorted(tool.namespaced_tool_name for tool in existing_tools)
        prompts_removed = sorted(prompt.name for prompt in existing_prompts)

        if server_name not in self._attached_server_names:
            return MCPDetachResult(
                server_name=display_name,
                detached=False,
                tools_removed=[],
                prompts_removed=[],
            )

        registry = self._require_server_registry()
        registry.release_attachment(
            server_name,
            owner=self._attachment_owner,
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

        self._app_integration_configs.pop(server_name, None)
        self._mcp_skill_registries.pop(server_name, None)
        self._attachment_configs.pop(server_name, None)
        registry.clear_server_capabilities(server_name)
        registry.remove_runtime(
            server_name,
            owner=self._runtime_definition_owner,
        )
        self._attached_server_names = [
            name for name in self._attached_server_names if name != server_name
        ]
        self.server_names = [name for name in self.server_names if name != server_name]

        return MCPDetachResult(
            server_name=display_name,
            detached=True,
            tools_removed=tools_removed,
            prompts_removed=prompts_removed,
        )

    def list_attached_servers(self) -> list[str]:
        return self._unique_preserving_order(
            [
                *(self.server_display_name(name) for name in self._attached_server_names),
                *self._supplemental_attached_server_names,
            ]
        )

    def server_display_name(self, server_name: str) -> str:
        registry = self.context.server_registry if self.context else None
        config = registry.get_server_config(server_name) if registry is not None else None
        return config.name if config is not None and config.name else server_name

    def _resolve_server_key(self, server_name: str) -> str:
        registry = self.context.server_registry if self.context else None
        if registry is None:
            return server_name
        scoped_servers = set(self._configured_server_names)
        scoped_servers.update(self._attached_server_names)
        scoped_servers.update(self._attachment_configs)
        scoped_servers.update(self.server_names)
        if server_name in scoped_servers:
            return server_name
        matches = [key for key in scoped_servers if self.server_display_name(key) == server_name]
        if len(matches) == 1:
            return matches[0]
        return server_name

    def set_supplemental_attached_servers(self, server_names: Iterable[str]) -> None:
        self._supplemental_attached_server_names = self._unique_preserving_order(server_names)

    def list_configured_detached_servers(self) -> list[str]:
        configured = set(self._configured_server_names)
        server_registry = self.context.server_registry if self.context else None
        if server_registry is not None:
            configured.update(server_registry.registry.keys())
        attached = set(self._attached_server_names)
        supplemental = set(self._supplemental_attached_server_names)
        return sorted(
            {
                display_name
                for name in configured
                if name not in attached
                and (display_name := self.server_display_name(name)) not in supplemental
            }
        )

    async def _evaluate_app_integrations_for_server(
        self,
        server_name: str,
        *,
        cache_mode: CacheMode = "use",
    ) -> tuple[str, AppServerConfig]:
        """Inspect a single server for supported interactive app resources."""
        config = AppServerConfig(server_name=server_name)
        tool_configs = self._app_tool_configs_for_server(
            server_name,
            config,
            self._staged_discovery_tools.get(server_name),
        )

        raw_resources_capability = await self.server_supports_feature(server_name, "resources")
        supports_resources = bool(raw_resources_capability)
        config.supports_resources = supports_resources
        config.tools = tool_configs

        if not supports_resources:
            return server_name, config

        await self._collect_app_resources(
            server_name,
            config,
            tool_configs,
            cache_mode=cache_mode,
            strict=cache_mode == "refresh",
        )
        self._link_app_tools_to_resources(config, tool_configs)
        self._warn_if_app_resources_are_unexposed(server_name, config, tool_configs)
        config.tools = tool_configs
        return server_name, config

    def _app_tool_configs_for_server(
        self,
        server_name: str,
        config: AppServerConfig,
        tools: list[NamespacedTool] | None = None,
    ) -> list[AppToolConfig]:
        tool_configs: list[AppToolConfig] = []
        for namespaced_tool in (
            tools if tools is not None else self._server_to_tool_map.get(server_name, [])
        ):
            tool_config = self._app_tool_config(namespaced_tool, config)
            if tool_config is not None:
                tool_configs.append(tool_config)
        return tool_configs

    @staticmethod
    def _metadata_error_tool_config(
        namespaced_tool: NamespacedTool,
        warning: str,
    ) -> AppToolConfig:
        return AppToolConfig(
            tool_name=namespaced_tool.tool.name,
            namespaced_tool_name=namespaced_tool.namespaced_tool_name,
            warning=warning,
        )

    def _app_tool_config(
        self,
        namespaced_tool: NamespacedTool,
        config: AppServerConfig,
    ) -> AppToolConfig | None:
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

        return AppToolConfig(
            tool_name=namespaced_tool.tool.name,
            namespaced_tool_name=namespaced_tool.namespaced_tool_name,
            resource_uri=app_metadata.resource_uri,
            kind=app_metadata.kind,
            visibility=app_metadata.visibility,
        )

    async def _collect_app_resources(
        self,
        server_name: str,
        config: AppServerConfig,
        tool_configs: list[AppToolConfig],
        *,
        cache_mode: CacheMode = "use",
        strict: bool = False,
    ) -> None:
        logger.info(
            "Scanning MCP app resources",
            data=build_progress_payload(
                action=ProgressAction.READING_RESOURCE,
                server_name=server_name,
                agent_name=self.agent_name,
                details="Apps",
            ),
        )
        try:
            if cache_mode == "use":
                resources = await self._list_resources_from_server(
                    server_name,
                    check_support=False,
                )
            else:
                resources = await self._list_resources_from_server(
                    server_name,
                    check_support=False,
                    cache_mode=cache_mode,
                )
        except Exception as exc:
            config.warnings.append(f"Failed to list resources: {exc}")
            logger.error(
                "MCP app resource scan failed",
                data=build_progress_payload(
                    action=ProgressAction.FATAL_ERROR,
                    server_name=server_name,
                    agent_name=self.agent_name,
                    details="Apps",
                    extra={"error_message": str(exc)},
                ),
            )
            if strict:
                raise
            return
        logger.info(
            "MCP app resource scan complete",
            data=build_progress_payload(
                action=ProgressAction.RESOURCE_READ,
                server_name=server_name,
                agent_name=self.agent_name,
                details="Apps",
            ),
        )

        expected_mime_by_uri = {
            str(tool.resource_uri): expected_mime_type(tool.kind)
            for tool in tool_configs
            if tool.resource_uri is not None and tool.kind is not None
        }

        for resource_entry in resources:
            uri_str, app_resource = self._app_resource_candidate(resource_entry, config)
            if app_resource is None:
                continue

            config.resources.append(app_resource)
            if cache_mode == "use":
                await self._read_app_resource(
                    server_name,
                    uri_str,
                    app_resource,
                    config,
                    expected_mime_by_uri,
                    strict=strict,
                )
            else:
                await self._read_app_resource(
                    server_name,
                    uri_str,
                    app_resource,
                    config,
                    expected_mime_by_uri,
                    cache_mode=cache_mode,
                    strict=strict,
                )

    @staticmethod
    def _app_resource_candidate(
        resource_entry: Resource,
        config: AppServerConfig,
    ) -> tuple[str, AppResourceConfig | None]:
        uri = resource_entry.uri
        if not uri:
            return "", None

        uri_str = str(uri)
        if not uri_str.startswith("ui://"):
            return uri_str, None

        try:
            uri_value = AnyUrl(uri_str)
        except Exception as exc:
            warning = f"Ignoring app resource candidate '{uri_str}': invalid URI ({exc})"
            config.warnings.append(warning)
            logger.debug(warning)
            return uri_str, None

        entry_meta = getattr(resource_entry, "meta", None)
        return uri_str, AppResourceConfig(
            uri=uri_value,
            meta=dict(entry_meta) if isinstance(entry_meta, dict) else {},
        )

    async def _read_app_resource(
        self,
        server_name: str,
        uri_str: str,
        app_resource: AppResourceConfig,
        config: AppServerConfig,
        expected_mime_by_uri: dict[str, str],
        *,
        cache_mode: CacheMode = "use",
        strict: bool = False,
    ) -> None:
        try:
            if cache_mode == "use":
                read_result = await self._get_resource_from_server(server_name, uri_str)
            else:
                read_result = await self._get_resource_from_server(
                    server_name,
                    uri_str,
                    cache_mode=cache_mode,
                )
        except Exception as exc:
            warning = f"Failed to read resource '{uri_str}': {exc}"
            app_resource.warning = warning
            config.warnings.append(warning)
            if strict:
                raise
            return

        self._apply_app_resource_contents(app_resource, read_result)
        if not app_resource.is_valid:
            self._warn_invalid_app_resource(
                uri_str,
                app_resource,
                config,
                expected_mime_by_uri,
            )

    @staticmethod
    def _apply_app_resource_contents(
        app_resource: AppResourceConfig,
        read_result: ReadResourceResult,
    ) -> None:
        seen_mime_types: list[str] = []
        for content in read_result.contents:
            mime_type = content.mime_type
            if mime_type:
                seen_mime_types.append(mime_type)
            if kind := integration_kind_for_mime_type(mime_type):
                app_resource.mime_type = mime_type
                app_resource.kind = kind

            content_meta = getattr(content, "meta", None)
            if isinstance(content_meta, dict):
                app_resource.meta.update(content_meta)

        if app_resource.mime_type is None and seen_mime_types:
            app_resource.mime_type = seen_mime_types[0]

    @staticmethod
    def _warn_invalid_app_resource(
        uri_str: str,
        app_resource: AppResourceConfig,
        config: AppServerConfig,
        expected_mime_by_uri: dict[str, str],
    ) -> None:
        observed_type = app_resource.mime_type or "unknown MIME type"
        expected_mime_type = expected_mime_by_uri.get(uri_str)
        openai_mime_type, mcp_apps_mime_type = supported_mime_types()
        expected_label = (
            f"'{expected_mime_type}'"
            if expected_mime_type
            else f"'{openai_mime_type}' or '{mcp_apps_mime_type}'"
        )
        warning = f"served as '{observed_type}' instead of {expected_label}"
        app_resource.warning = warning
        config.warnings.append(f"{uri_str}: {warning}")

    def _link_app_tools_to_resources(
        self,
        config: AppServerConfig,
        tool_configs: list[AppToolConfig],
    ) -> None:
        resource_lookup = {str(resource.uri): resource for resource in config.resources}
        for tool_config in tool_configs:
            if tool_config.resource_uri is None:
                continue

            resource_match = resource_lookup.get(str(tool_config.resource_uri))
            if not resource_match:
                self._warn_missing_app_resource(tool_config, config)
                continue

            self._apply_app_tool_resource_match(tool_config, resource_match, config)

    @staticmethod
    def _warn_missing_app_resource(
        tool_config: AppToolConfig,
        config: AppServerConfig,
    ) -> None:
        resource_label = tool_config.kind.display_name if tool_config.kind else "App integration"
        warning = (
            f"Tool '{tool_config.namespaced_tool_name}' references missing "
            f"{resource_label} resource '{tool_config.resource_uri}'"
        )
        tool_config.warning = warning
        config.warnings.append(warning)
        logger.error(warning)

    @staticmethod
    def _apply_app_tool_resource_match(
        tool_config: AppToolConfig,
        resource_match: AppResourceConfig,
        config: AppServerConfig,
    ) -> None:
        if tool_config.kind is None:
            return
        required_mime_type = expected_mime_type(tool_config.kind)
        if resource_match.kind is tool_config.kind:
            tool_config.linked_resource_uri = resource_match.uri

        if tool_config.is_valid:
            return

        warning = (
            f"Tool '{tool_config.namespaced_tool_name}' references resource "
            f"'{resource_match.uri}' served as '{resource_match.mime_type or 'unknown'}' "
            f"instead of '{required_mime_type}'"
        )
        tool_config.warning = warning
        config.warnings.append(warning)
        logger.warning(warning)

    @staticmethod
    def _warn_if_app_resources_are_unexposed(
        server_name: str,
        config: AppServerConfig,
        tool_configs: list[AppToolConfig],
    ) -> None:
        valid_tool_count = sum(1 for tool in tool_configs if tool.is_valid)
        if config.enabled and valid_tool_count == 0:
            warning = f"App resources detected on server '{server_name}' but no tools expose them"
            config.warnings.append(warning)
            logger.warning(warning)

    def _display_startup_state(self) -> None:
        """Record discovered interactive app integration state."""
        # In interactive contexts the UI helper will render both the agent summary and the
        # app integration status. For non-interactive contexts, discovery warnings are
        # emitted through the logger, so we don't need to duplicate output here.
        if not self._app_integration_configs:
            return

        logger.debug(
            "App integration discovery completed",
            data={
                "agent_name": self.agent_name,
                "server_count": len(self._app_integration_configs),
            },
        )

    async def get_capabilities(self, server_name: str) -> ServerCapabilities | None:
        """Get server capabilities if available."""
        server_name = self._resolve_server_key(server_name)
        if not self.connection_persistence:
            # Check cache under lock (fast path)
            async with self._capabilities_cache_lock:
                cached = self._capabilities_cache.get(server_name)
                if cached is not None:
                    return cached

            # I/O without holding lock — allows concurrent probes for different servers
            try:
                server_registry = self._require_server_registry()
                async with gen_client(
                    server_name=server_name,
                    server_registry=server_registry,
                    callback_runtime=self._create_callback_runtime(server_name),
                    **self._attachment_client_kwargs(server_name),
                ) as connection:
                    capabilities = connection.server_capabilities

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
                callback_runtime=self._create_callback_runtime(server_name),
                **self._attachment_manager_kwargs(server_name),
            )
            return server_conn.server_capabilities
        except Exception as e:
            logger.debug(f"Error getting capabilities for server '{server_name}': {e}")
            return None

    async def _scan_mcp_skill_registry(
        self,
        server_name: str,
        *,
        cache_mode: CacheMode = "use",
    ) -> McpSkillRegistry | None:
        client = _AttachedRegistryScanClient(self, cache_mode=cache_mode)
        return await scan_mcp_skill_registry(
            client,
            server_name,
            server_version=await self._mcp_server_version(server_name),
        )

    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]:
        if not self.initialized:
            await self.load_servers()
        registries: list[McpSkillRegistry] = []
        for server_name in self._attached_server_names:
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
        server_name = self._resolve_server_key(server_name)
        valid = server_name in self.server_names or server_name in self._attachment_configs
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

        return [self.server_display_name(name) for name in self.server_names]

    async def list_tools(self) -> ListToolsResult:
        """
        :return: Tools from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        if not self.initialized:
            await self.load_servers()

        tools: list[Tool] = []

        for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items():
            app_integration_config = self._app_integration_configs.get(namespaced_tool.server_name)
            discovered_tool = None
            matching_tool = None
            if app_integration_config:
                discovered_tool = next(
                    (
                        tool
                        for tool in app_integration_config.tools
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
                mark_tool_metadata(meta, matching_tool)
                tool_copy.meta = meta
            tools.append(tool_copy)

        return ListToolsResult(tools=tools)

    async def _record_server_call(
        self, server_name: str, operation_type: str, success: bool
    ) -> None:
        async with self._stats_lock:
            stats = self._server_stats.setdefault(server_name, ServerStats())
            stats.record(operation_type, success)

            # For stdio servers, also emit synthetic transport events to create activity timeline
            await self._notify_stdio_transport_activity(server_name, operation_type, success)

    async def _record_connection_negotiation(
        self,
        server_name: str,
        server_conn: ServerConnection,
    ) -> None:
        if server_conn.negotiation in {"discover", "initialize"}:
            await self._record_server_call(server_name, server_conn.negotiation, True)

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
                instructions[self.server_display_name(server_name)] = (
                    server_conn.server_instructions,
                    tool_names,
                )
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
            app_integration_config=self._app_integration_configs.get(server_name),
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
        status.protocol_version = server_conn.protocol_version
        status.protocol_era = server_conn.protocol_era
        status.supported_protocol_versions = server_conn.supported_protocol_versions
        status.negotiation = server_conn.negotiation

        status.server_capabilities = server_conn.server_capabilities
        status.mcp_skills_enabled = server_supports_mcp_skills(server_conn.server_capabilities)
        client_info = server_conn._callback_runtime.client_info
        status.client_info_name = client_info.name
        status.client_info_version = client_info.version

        if server_conn._initialized_event.is_set():
            status.is_connected = server_conn.is_healthy()
        else:
            status.is_connected = False
            status.error_message = status.error_message or "initializing..."

        status.error_message = status.error_message or server_conn._error_message
        status.instructions_available = server_conn.server_instructions_available
        status.instructions_enabled = server_conn.server_instructions_enabled
        status.instructions_included = bool(server_conn.server_instructions)
        status.subscription_state = server_conn.subscription_state

        self._apply_ping_status(status, server_conn)
        self._apply_session_status(status, server_conn)
        self._apply_transport_status(status, server_conn)

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
        status.elicitation_mode = server_conn._callback_runtime.effective_elicitation_mode
        status.session_id = server_conn.session_id

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
        status.protocol_mode = server_cfg.protocol_mode
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
        self._apply_config_session_id(status, server_cfg)
        status.sampling_mode = (
            "configured" if server_cfg.sampling is not None else self._auto_sampling_mode()
        )

    def _apply_config_session_id(
        self,
        status: ServerStatus,
        server_cfg: MCPServerSettings,
    ) -> None:
        if status.session_id is not None:
            return
        if server_cfg.transport == "stdio":
            status.session_id = "local"

    def _auto_sampling_mode(self) -> Literal["auto", "off"]:
        auto_sampling = True
        if self.context and self.context.config is not None and self.context.config.mcp is not None:
            auto_sampling = self.context.config.mcp.client.auto_sampling
        return "auto" if auto_sampling else "off"

    async def get_app_integration_configs(self) -> dict[str, AppServerConfig]:
        """Expose discovered app integration configurations keyed by server."""
        if not self.initialized:
            await self.load_servers()
        return dict(self._app_integration_configs)

    async def get_app_integration_config(self, server_name: str) -> AppServerConfig | None:
        """Return app integration configuration for a server, loading if necessary."""
        if not self.initialized:
            await self.load_servers()
        return self._app_integration_configs.get(server_name)

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
            method_name: Name of the high-level client method to call
            method_args: Arguments to pass to the method
            error_factory: Function to create an error return value if the operation fails
            progress_callback: Optional progress callback for operations that support it

        Returns:
            Result from the operation or an error result
        """

        async def try_execute(client: MCPOperationClient) -> R:
            return await self._execute_session_method(
                client,
                method_name=method_name,
                method_args=method_args,
                progress_callback=progress_callback,
            )

        success_flag: bool | None = None
        result: R | None = None

        try:
            result = await self._execute_initial_server_operation(server_name, try_execute)
            success_flag = True
        except ConnectionError as exc:
            if method_name not in _CONNECTION_ERROR_REPLAY_SAFE_METHODS:
                await self._reconnect_for_future_operations(server_name, method_name)
                result = self._handle_session_method_error(
                    exc=exc,
                    server_name=server_name,
                    operation_name=operation_name,
                    method_name=method_name,
                    error_factory=error_factory,
                )
                success_flag = False
            else:
                recovery = await self._handle_connection_error(
                    server_name, try_execute, error_factory
                )
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
                result = self._handle_session_method_error(
                    exc=exc,
                    server_name=server_name,
                    operation_name=operation_name,
                    method_name=method_name,
                    error_factory=error_factory,
                )
                success_flag = False
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
        client: MCPOperationClient,
        *,
        method_name: str,
        method_args: dict[str, Any] | None,
        progress_callback: ProgressFnT | None,
    ) -> R:
        if method_name in {"call_tool", "read_resource", "get_prompt"}:
            kwargs = self._server_method_kwargs(method_name, method_args)
            if method_name == "call_tool":
                result = await client.call_tool(
                    progress_callback=progress_callback,
                    **kwargs,
                )
            elif method_name == "read_resource":
                result = await client.read_resource(**kwargs)
            else:
                result = await client.get_prompt(**kwargs)
            return cast("R", result)

        method = getattr(client, method_name)
        return cast("R", await method(**self._server_method_kwargs(method_name, method_args)))

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
        payload = url_elicitation_required_payload(exc)
        if payload is not None:
            with suppress(Exception):
                set_url_elicitation_required_payload(error_result, payload)
        return error_result

    async def _execute_initial_server_operation(
        self,
        server_name: str,
        try_execute: Callable[[MCPOperationClient], Awaitable[R]],
    ) -> R:
        if self.connection_persistence and not self._should_use_request_scoped_connection(
            server_name
        ):
            return await self._execute_persistent_server_operation(server_name, try_execute)
        return await self._execute_temporary_server_operation(server_name, try_execute)

    async def _execute_persistent_server_operation(
        self,
        server_name: str,
        try_execute: Callable[[MCPOperationClient], Awaitable[R]],
    ) -> R:
        manager = self._require_connection_manager()
        server_connection = await manager.get_server(
            server_name,
            callback_runtime=self._create_callback_runtime(server_name),
            **self._attachment_manager_kwargs(server_name),
        )
        client = server_connection.client
        if client is None:
            raise RuntimeError(f"MCP client runtime not initialized for '{server_name}'")
        return await try_execute(client)

    async def _execute_temporary_server_operation(
        self,
        server_name: str,
        try_execute: Callable[[MCPOperationClient], Awaitable[R]],
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
        async with gen_client(
            server_name,
            server_registry=server_registry,
            callback_runtime=self._create_callback_runtime(server_name),
            **self._attachment_client_kwargs(server_name),
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

        config = self._server_config(server_name)
        if config is None:
            return False
        return resolve_oauth_mode(config, trigger_oauth=None) == "auto" and is_http_auth_challenge(
            exc
        )

    def _log_server_progress(
        self,
        action: ProgressAction,
        server_name: str,
        details: str,
    ) -> None:
        payload = build_progress_payload(
            action=action,
            server_name=server_name,
            agent_name=self.agent_name,
            details=details,
            extra={"error_message": details} if action == ProgressAction.FATAL_ERROR else None,
        )
        log = logger.error if action == ProgressAction.FATAL_ERROR else logger.info
        log("MCP server recovery", data=payload)

    async def _handle_auth_challenge(
        self,
        server_name: str,
        try_execute: Callable,
        error_factory: Callable[[str], R] | None,
        _exc: Exception | None = None,
    ) -> _ServerOperationRecovery[R]:
        self._log_server_progress(
            ProgressAction.CONNECTING,
            server_name,
            "authorization required; reconnecting with OAuth",
        )

        try:
            if self.connection_persistence:
                manager = self._require_connection_manager()
                server_connection = await manager.reconnect_server(
                    server_name,
                    callback_runtime=self._create_callback_runtime(server_name),
                    trigger_oauth=True,
                    **self._attachment_manager_kwargs(server_name),
                )
                await self._record_connection_negotiation(server_name, server_connection)
                server_connection._callback_runtime.mark_subscription_ready()
                client = server_connection.client
                if client is None:
                    raise RuntimeError(f"MCP client runtime not initialized for '{server_name}'")
                result = await try_execute(client)
            else:
                server_registry = self._require_server_registry()
                async with gen_client(
                    server_name,
                    server_registry=server_registry,
                    callback_runtime=self._create_callback_runtime(server_name),
                    trigger_oauth=True,
                    **self._attachment_client_kwargs(server_name),
                ) as client:
                    result = await try_execute(client)
            self._log_server_progress(
                ProgressAction.READY,
                server_name,
                "reconnected with OAuth",
            )
            return _ServerOperationRecovery(result=result, success=True)
        except Exception as retry_exc:
            self._log_server_progress(
                ProgressAction.FATAL_ERROR,
                server_name,
                f"OAuth reconnect failed: {retry_exc}",
            )
            if error_factory:
                return _ServerOperationRecovery(
                    result=error_factory(str(retry_exc)),
                    success=False,
                )
            raise

    async def _handle_connection_error(
        self,
        server_name: str,
        try_execute: Callable[[MCPOperationClient], Awaitable[R]],
        error_factory: Callable[[str], R] | None,
    ) -> _ServerOperationRecovery[R]:
        """Handle ConnectionError by attempting to reconnect to the server."""
        self._log_server_progress(ProgressAction.CONNECTING, server_name, "reconnecting")

        try:
            result = await self._reconnect_and_replay_server_operation(
                server_name,
                try_execute,
            )

            # Success!
            self._log_server_progress(ProgressAction.READY, server_name, "reconnected")
            return _ServerOperationRecovery(result=result, success=True)

        except ServerSessionTerminatedError:
            # After reconnecting for connection error, we got session terminated
            # Don't loop - just report the error
            self._log_server_progress(
                ProgressAction.FATAL_ERROR,
                server_name,
                "session terminated after reconnect; retries exhausted",
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
            self._log_server_progress(
                ProgressAction.FATAL_ERROR,
                server_name,
                f"reconnect failed: {e}",
            )
            error_msg = f"MCP server {server_name} offline - failed to reconnect"
            if error_factory:
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
            raise RuntimeError(error_msg) from e

    async def _reconnect_and_replay_server_operation(
        self,
        server_name: str,
        try_execute: Callable[[MCPOperationClient], Awaitable[R]],
    ) -> R:
        if self.connection_persistence:
            manager = self._require_connection_manager()
            server_connection = await manager.reconnect_server(
                server_name,
                callback_runtime=self._create_callback_runtime(server_name),
                **self._attachment_manager_kwargs(server_name),
            )
            await self._record_connection_negotiation(server_name, server_connection)
            server_connection._callback_runtime.mark_subscription_ready()
            client = server_connection.client
            if client is None:
                raise RuntimeError(f"MCP client runtime not initialized for '{server_name}'")
            return await try_execute(client)

        server_registry = self._require_server_registry()
        async with gen_client(
            server_name,
            server_registry=server_registry,
            callback_runtime=self._create_callback_runtime(server_name),
            **self._attachment_client_kwargs(server_name),
        ) as client:
            return await try_execute(client)

    async def _reconnect_for_future_operations(
        self,
        server_name: str,
        method_name: str,
    ) -> None:
        if not self.connection_persistence:
            return

        self._log_server_progress(
            ProgressAction.CONNECTING,
            server_name,
            f"reconnecting after {method_name}",
        )
        try:
            manager = self._require_connection_manager()
            server_connection = await manager.reconnect_server(
                server_name,
                callback_runtime=self._create_callback_runtime(server_name),
                **self._attachment_manager_kwargs(server_name),
            )
            await self._record_connection_negotiation(server_name, server_connection)
            server_connection._callback_runtime.mark_subscription_ready()
            self._log_server_progress(ProgressAction.READY, server_name, "reconnected")
        except Exception as exc:
            self._log_server_progress(
                ProgressAction.FATAL_ERROR,
                server_name,
                f"reconnect failed: {exc}",
            )
            logger.warning(
                f"MCP server {server_name} failed to reconnect after non-replayable "
                f"{method_name}: {exc}"
            )

    async def _handle_session_terminated(
        self,
        server_name: str,
        try_execute: Callable[[MCPOperationClient], Awaitable[R]],
        error_factory: Callable[[str], R] | None,
        exc: ServerSessionTerminatedError,
    ) -> _ServerOperationRecovery[R]:
        """Handle ServerSessionTerminatedError by attempting to reconnect if configured."""
        server_config = self._server_config(server_name)
        reconnect_enabled = server_config and server_config.reconnect_on_disconnect

        if not reconnect_enabled:
            # Reconnection not enabled - inform user and fail
            self._log_server_progress(
                ProgressAction.FATAL_ERROR,
                server_name,
                "session terminated; reconnect disabled (enable reconnect_on_disconnect)",
            )
            error_msg = f"MCP server {server_name} session terminated - reconnection not enabled"
            if error_factory:
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
            raise exc

        # Attempt reconnection
        self._log_server_progress(
            ProgressAction.CONNECTING,
            server_name,
            "session terminated; reconnecting",
        )

        try:
            result = await self._reconnect_and_replay_server_operation(
                server_name,
                try_execute,
            )

            # Success! Record the reconnection
            await self._record_reconnect(server_name)
            self._log_server_progress(ProgressAction.READY, server_name, "reconnected")
            return _ServerOperationRecovery(result=result, success=True)

        except ServerSessionTerminatedError:
            # Retry after reconnection ALSO failed with session terminated
            # Do NOT attempt another reconnection - this would cause an infinite loop
            self._log_server_progress(
                ProgressAction.FATAL_ERROR,
                server_name,
                "session terminated after reconnect; retries exhausted",
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
            self._log_server_progress(
                ProgressAction.FATAL_ERROR,
                server_name,
                f"reconnect failed: {e}",
            )
            error_msg = f"MCP server {server_name} failed to reconnect: {e}"
            if error_factory:
                return _ServerOperationRecovery(result=error_factory(error_msg), success=False)
            raise RuntimeError(error_msg) from e

    def tool_catalog(self) -> MCPToolCatalog:
        return MCPToolCatalog.snapshot(
            by_namespaced_name=self._namespaced_tool_map,
            by_server=self._server_to_tool_map,
            server_names=self.server_names,
        )

    def resolve_tool_name(self, name: str) -> ToolNameResolution:
        return self.tool_catalog().resolve_tool_name(name)

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
        tool_name_resolution = self.resolve_tool_name(name)
        server_name = tool_name_resolution.server_name
        local_tool_name = tool_name_resolution.local_name

        if server_name is None:
            logger.error(f"Error: Tool '{name}' not found")
            return CallToolResult(
                is_error=True,
                content=[TextContent(type="text", text=f"Tool '{name}' not found")],
            )

        display_server_name = self.server_display_name(server_name)
        namespaced_tool_name = create_namespaced_name(display_server_name, local_tool_name)
        active_tool_handler = request_tool_handler or self._tool_handler

        permission_error = await self._tool_permission_error_result(
            local_tool_name=local_tool_name,
            server_name=display_server_name,
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
            server_name=display_server_name,
            arguments=arguments,
            tool_use_id=tool_use_id,
        )

        logger.info(
            "Requesting tool call",
            data=build_progress_payload(
                action=ProgressAction.CALLING_TOOL,
                tool_name=local_tool_name,
                server_name=display_server_name,
                agent_name=self.agent_name,
                tool_call_id=tool_call_id,
                tool_use_id=tool_use_id,
            ),
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"MCP Tool: {namespaced_tool_name}"):
            trace.get_current_span().set_attribute("tool_name", local_tool_name)
            trace.get_current_span().set_attribute("server_name", display_server_name)
            trace.get_current_span().set_attribute("namespaced_tool_name", namespaced_tool_name)

            # Create progress callback for this tool execution
            progress_callback = self._create_progress_callback(
                display_server_name,
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
                        is_error=True, content=[TextContent(type="text", text=msg)]
                    ),
                    progress_callback=progress_callback,
                )

                await self._complete_tool_execution(
                    active_tool_handler,
                    result=result,
                    local_tool_name=local_tool_name,
                    server_name=display_server_name,
                    tool_call_id=tool_call_id,
                    tool_use_id=tool_use_id,
                )
                return result

            except Exception as e:
                await self._fail_tool_execution(
                    active_tool_handler,
                    exc=e,
                    local_tool_name=local_tool_name,
                    server_name=display_server_name,
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
            is_error=True,
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
        completion_state = "completed" if not result.is_error else "failed"
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
                is_error=result.is_error,
            )

            error_text = None
            if result.is_error and content:
                text_parts = [text for c in content if (text := get_text(c))]
                error_text = "\n".join(text_parts) if text_parts else None
                content = None

            await active_tool_handler.on_tool_complete(
                tool_call_id,
                not result.is_error,
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
            return _PromptNameResolution(
                server_name=self._resolve_server_key(server_name),
                local_name=prompt_name,
            )
        if not is_namespaced_name(prompt_name):
            return _PromptNameResolution(server_name=None, local_name=prompt_name)

        potential_server, local_name = prompt_name.split(SEP, 1)
        resolved_server = self._resolve_server_key(potential_server)
        if resolved_server in self.server_names:
            return _PromptNameResolution(server_name=resolved_server, local_name=local_name)

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

    def _prompt_result_with_metadata(
        self,
        result: GetPromptResult,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, str] | None,
    ) -> GetPromptResult:
        return with_prompt_metadata(
            result,
            namespaced_name=create_namespaced_name(
                self.server_display_name(server_name),
                prompt_name,
            ),
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
                results[self.server_display_name(s_name)] = []

        for s_name in supported_servers:
            results[self.server_display_name(s_name)] = await self._fetch_and_cache_prompts(s_name)

        logger.debug(f"Available prompts across servers: {results}")
        return results

    async def _list_prompts_for_server(self, server_name: str) -> dict[str, list[Prompt]]:
        server_name = self._resolve_server_key(server_name)
        if server_name not in self.server_names:
            logger.error(f"Server '{server_name}' not found")
            return {}

        cached_prompts = await self._cached_prompts_for_server(server_name)
        display_name = self.server_display_name(server_name)
        if cached_prompts is not None:
            return {display_name: cached_prompts}

        if not await self._server_supports_prompts(server_name):
            logger.debug(f"Server '{server_name}' does not support prompts")
            return {display_name: []}

        return {display_name: await self._fetch_and_cache_prompts(server_name)}

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
            return {
                self.server_display_name(server_name): prompts
                for server_name, prompts in self._prompt_cache.items()
            }

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
        async with self._lifecycle_lock:
            if self._closed:
                return
            async with self._attachment_locks.setdefault(server_name, Lock()):
                if server_name not in self._attached_server_names:
                    logger.debug(f"Ignoring tool-list change for unattached server '{server_name}'")
                    return
                logger.info(f"Tool list changed for server '{server_name}', refreshing tools")
                await self._refresh_server_tools(server_name)

    async def _refresh_server_resources(self, server_name: str) -> None:
        _, app_integration_config = await self._evaluate_app_integrations_for_server(server_name)
        self._app_integration_configs[server_name] = app_integration_config

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
                new_namespaced_tools = [
                    NamespacedTool(
                        tool=tool,
                        server_name=server_name,
                        namespaced_tool_name=create_namespaced_name(
                            self.server_display_name(server_name),
                            tool.name,
                        ),
                    )
                    for tool in new_tools
                ]

                self._staged_discovery_tools[server_name] = new_namespaced_tools
                try:
                    _, app_integration_config = await self._evaluate_app_integrations_for_server(
                        server_name
                    )
                finally:
                    self._staged_discovery_tools.pop(server_name, None)

                # Commit tools and their app metadata together so app-only visibility and
                # resource validation cannot lag behind a tools/list_changed notification.
                async with self._tool_map_lock:
                    old_tools = self._server_to_tool_map.get(server_name, [])
                    for old_tool in old_tools:
                        if old_tool.namespaced_tool_name in self._namespaced_tool_map:
                            del self._namespaced_tool_map[old_tool.namespaced_tool_name]

                    self._server_to_tool_map[server_name] = new_namespaced_tools
                    for namespaced_tool in new_namespaced_tools:
                        self._namespaced_tool_map[namespaced_tool.namespaced_tool_name] = (
                            namespaced_tool
                        )
                    self._app_integration_configs[server_name] = app_integration_config

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
        self,
        resource_uri: str,
        server_name: str | None = None,
        *,
        cache_mode: CacheMode = "use",
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
            server_name = self._resolve_server_key(server_name)
            if server_name not in self.server_names:
                raise ValueError(f"Server '{server_name}' not found")

            # Get the resource from the specified server
            return await self._get_resource_from_server(
                server_name,
                resource_uri,
                cache_mode=cache_mode,
            )

        # If no server specified, search all servers
        if not self.server_names:
            raise ValueError("No servers available to get resource from")

        # Try each server in order - simply attempt to get the resource
        for s_name in self.server_names:
            try:
                return await self._get_resource_from_server(
                    s_name,
                    resource_uri,
                    cache_mode=cache_mode,
                )
            except Exception:
                # Continue to next server if not found
                continue

        # If we reach here, we couldn't find the resource on any server
        raise ValueError(f"Resource '{resource_uri}' not found on any server")

    async def _execute_resource_read(
        self,
        server_name: str,
        *,
        uri: str,
        operation_type: str,
        method_name: str,
        noun: str,
        extra_args: dict[str, Any] | None = None,
        cache_mode: CacheMode | None = None,
    ) -> Any:
        """Shared implementation behind ``get_resource`` and ``read_directory``.

        Both ``resources/read`` and ``resources/directory/read`` share the same
        shape: verify the resources capability, emit a READING_RESOURCE progress
        event, validate the URI, dispatch via ``_execute_on_server``, and treat a
        ``None`` result as not-found. ``noun`` ("Resource"/"Directory") is woven
        into the error messages.
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
                details=uri,
                extra={"resource_uri": uri},
            ),
        )

        try:
            uri_value = str(AnyUrl(uri))
        except Exception as e:
            raise ValueError(f"Invalid {noun.lower()} URI: {uri}. Error: {e}") from e

        method_args: dict[str, Any] = {"uri": uri_value}
        if extra_args:
            method_args.update(extra_args)
        if cache_mode is not None and method_name == "read_resource":
            method_args["cache_mode"] = cache_mode

        try:
            result = await self._execute_on_server(
                server_name=server_name,
                operation_type=operation_type,
                operation_name=uri,
                method_name=method_name,
                method_args=method_args,
                # Don't create ValueError, just return None on error so we can catch it.
            )
        except Exception as exc:
            logger.error(
                f"{noun} read failed",
                data=build_progress_payload(
                    action=ProgressAction.FATAL_ERROR,
                    server_name=server_name,
                    agent_name=self.agent_name,
                    details=uri,
                    extra={
                        "resource_uri": uri,
                        "error_message": str(exc),
                    },
                ),
            )
            raise

        # If result is None, the resource was not found
        if result is None:
            error = ValueError(f"{noun} '{uri}' not found on server '{server_name}'")
            logger.error(
                f"{noun} read failed",
                data=build_progress_payload(
                    action=ProgressAction.FATAL_ERROR,
                    server_name=server_name,
                    agent_name=self.agent_name,
                    details=uri,
                    extra={
                        "resource_uri": uri,
                        "error_message": str(error),
                    },
                ),
            )
            raise error

        logger.info(
            f"{noun} read complete",
            data=build_progress_payload(
                action=ProgressAction.RESOURCE_READ,
                server_name=server_name,
                agent_name=self.agent_name,
                details=uri,
                extra={
                    "resource_uri": uri,
                    "success": True,
                },
            ),
        )

        return result

    async def _get_resource_from_server(
        self,
        server_name: str,
        resource_uri: str,
        *,
        cache_mode: CacheMode = "use",
    ) -> ReadResourceResult:
        """Internal helper method to get a resource from a specific server."""
        return await self._execute_resource_read(
            server_name,
            uri=resource_uri,
            operation_type="resources/read",
            method_name="read_resource",
            noun="Resource",
            cache_mode=cache_mode if cache_mode != "use" else None,
        )

    async def read_directory(
        self,
        uri: str,
        *,
        server_name: str | None = None,
        cursor: str | None = None,
    ) -> ListResourcesResult:
        """List the direct children of a directory resource via SEP-2640.

        Routes ``resources/directory/read`` to the named server. Callers should
        only invoke this against servers that declared ``directoryRead``.

        ``server_name`` is required: a walk is scoped to the one server hosting
        the skill. Unlike ``get_resource`` we don't fan out, since a same-named
        directory URI on another server would read off the wrong server.
        """
        if not self.initialized:
            await self.load_servers()

        if server_name is None:
            raise ValueError("read_directory requires an explicit server_name")
        server_name = self._resolve_server_key(server_name)
        if server_name not in self.server_names:
            raise ValueError(f"Server '{server_name}' not found")
        return await self._read_directory_from_server(server_name, uri, cursor=cursor)

    async def list_skills(
        self,
        server_name: str,
        cursor: str | None = None,
    ) -> ListSkillsResult:
        """List skills from one server via the SEP-2640 extension."""
        if not self.initialized:
            await self.load_servers()

        server_name = self._resolve_server_key(server_name)
        if server_name not in self.server_names:
            raise ValueError(f"Server '{server_name}' not found")
        return await self._list_skills_from_server(server_name, cursor=cursor)

    async def _list_skills_from_server(
        self,
        server_name: str,
        *,
        cursor: str | None = None,
    ) -> ListSkillsResult:
        """Internal helper to call ``skills/list`` on a server."""
        method_args = {"cursor": cursor} if cursor is not None else None
        return await self._execute_on_server(
            server_name=server_name,
            operation_type="skills/list",
            operation_name="",
            method_name="list_skills",
            method_args=method_args,
        )

    async def get_skill(self, uri: str, server_name: str) -> GetSkillResult:
        """Get one skill entry from a specific server via SEP-2640."""
        if not self.initialized:
            await self.load_servers()

        server_name = self._resolve_server_key(server_name)
        if server_name not in self.server_names:
            raise ValueError(f"Server '{server_name}' not found")
        return await self._get_skill_from_server(server_name, uri)

    async def _get_skill_from_server(self, server_name: str, uri: str) -> GetSkillResult:
        """Internal helper to call ``skills/get`` on a server."""
        return await self._execute_on_server(
            server_name=server_name,
            operation_type="skills/get",
            operation_name=uri,
            method_name="get_skill",
            method_args={"uri": uri},
        )

    async def _read_directory_from_server(
        self, server_name: str, uri: str, *, cursor: str | None = None
    ) -> ListResourcesResult:
        """Internal helper to call ``resources/directory/read`` on a server."""
        return await self._execute_resource_read(
            server_name,
            uri=uri,
            operation_type="resources/directory/read",
            method_name="read_directory",
            noun="Directory",
            extra_args={"cursor": cursor} if cursor is not None else None,
        )

    async def _list_resources_from_server(
        self,
        server_name: str,
        *,
        check_support: bool = True,
        cache_mode: CacheMode = "use",
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
            method_args={"cache_mode": cache_mode} if cache_mode != "use" else {},
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
            error_factory=lambda _: ListResourceTemplatesResult(resource_templates=[]),
        )

        return result.resource_templates

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
        servers_to_check = (
            [self._resolve_server_key(server_name)] if server_name else self.server_names
        )

        # For each server, try to list its resources
        for s_name in servers_to_check:
            if s_name not in self.server_names:
                logger.error(f"Server '{s_name}' not found")
                continue

            # Initialize empty list for this server
            display_name = self.server_display_name(s_name)
            results[display_name] = []

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
                results[display_name] = formatted_resources
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
        servers_to_check = (
            [self._resolve_server_key(server_name)] if server_name else self.server_names
        )

        for s_name in servers_to_check:
            if s_name not in self.server_names:
                logger.error(f"Server '{s_name}' not found")
                continue

            display_name = self.server_display_name(s_name)
            results[display_name] = []

            if not await self.server_supports_feature(s_name, "resources"):
                logger.debug(f"Server '{s_name}' does not support resources")
                continue

            try:
                templates = await self._list_resource_templates_from_server(
                    s_name, check_support=False
                )
                results[display_name] = list(templates)
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
        server_name = self._resolve_server_key(server_name)
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
        servers_to_check = (
            [self._resolve_server_key(server_name)] if server_name else self.server_names
        )

        # For each server, try to list its tools
        for s_name in servers_to_check:
            if s_name not in self.server_names:
                logger.error(f"Server '{s_name}' not found")
                continue

            # Initialize empty list for this server
            display_name = self.server_display_name(s_name)
            results[display_name] = []

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
                results[display_name] = tools

            except Exception as e:
                logger.error(f"Error fetching tools from {s_name}: {e}")

        return results
