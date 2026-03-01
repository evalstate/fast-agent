"""
AgentACPServer - Exposes FastAgent agents via the Agent Client Protocol (ACP).

This implementation allows fast-agent to act as an ACP agent, enabling editors
and other clients to interact with fast-agent agents over stdio using the ACP protocol.
"""

import asyncio
from dataclasses import dataclass, field
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence, cast

from acp import (
    Agent as ACPAgent,
)
from acp import (
    Client as ACPClient,
)
from acp import (
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    SetSessionModeResponse,
    run_agent,
)
from acp.exceptions import RequestError
from acp.helpers import ContentBlock as ACPContentBlock
from acp.helpers import (
    update_agent_message,
    update_agent_message_text,
    update_agent_thought_text,
    update_user_message,
)
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AuthenticateResponse,
    AuthMethod,
    AvailableCommandsUpdate,
    ClientCapabilities,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    PromptCapabilities,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionInfoUpdate,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SessionResumeCapabilities,
    SseMcpServer,
    StopReason,
    UserMessageChunk,
)
from acp.schema import (
    SessionInfo as AcpSessionInfo,
)

from fast_agent.acp.acp_context import ACPContext, ClientInfo
from fast_agent.acp.acp_context import ClientCapabilities as FAClientCapabilities
from fast_agent.acp.content_conversion import (
    convert_acp_prompt_to_mcp_content_blocks,
    convert_mcp_content_to_acp,
    inline_resources_for_slash_command,
)
from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.permission_store import PermissionStore
from fast_agent.acp.protocols import (
    FilesystemRuntimeCapable,
    InstructionContextCapable,
    PlanTelemetryCapable,
    ShellRuntimeCapable,
    WorkflowTelemetryCapable,
)
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
from fast_agent.acp.tool_progress import ACPToolProgressManager
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.config import MCPServerSettings
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
)
from fast_agent.context import Context
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.instruction_refresh import McpInstructionCapable, build_instruction
from fast_agent.core.instruction_utils import (
    build_agent_instruction_context,
    get_instruction_template,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.interfaces import (
    ACPAwareProtocol,
    AgentProtocol,
    StreamingAgentProtocol,
    ToolRunnerHookCapable,
)
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.terminal_output_limits import calculate_terminal_output_limit_for_model
from fast_agent.llm.usage_tracking import last_turn_usage
from fast_agent.mcp.helpers.content_helpers import is_text_content
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import NoOpToolPermissionHandler
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.session import (
    Session,
    extract_session_title,
    get_session_history_window,
    get_session_manager,
)
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams
from fast_agent.workflow_telemetry import ACPPlanTelemetryProvider, ToolHandlerWorkflowTelemetry

logger = get_logger(__name__)

END_TURN: StopReason = "end_turn"
REFUSAL: StopReason = "refusal"
MAX_TOKENS: StopReason = "max_tokens"
CANCELLED: StopReason = "cancelled"



def map_llm_stop_reason_to_acp(llm_stop_reason: LlmStopReason | None) -> StopReason:
    """
    Map fast-agent LlmStopReason to ACP StopReason.

    Args:
        llm_stop_reason: The stop reason from the LLM response

    Returns:
        The corresponding ACP StopReason value
    """
    if llm_stop_reason is None:
        return END_TURN

    # Use string keys to avoid hashing Enum members with custom equality logic
    key = (
        llm_stop_reason.value
        if isinstance(llm_stop_reason, LlmStopReason)
        else str(llm_stop_reason)
    )
    mapping: dict[str, StopReason] = {
        LlmStopReason.END_TURN.value: END_TURN,
        LlmStopReason.STOP_SEQUENCE.value: END_TURN,  # Normal completion
        LlmStopReason.MAX_TOKENS.value: MAX_TOKENS,
        LlmStopReason.TOOL_USE.value: END_TURN,  # Tool use is normal completion in ACP
        LlmStopReason.PAUSE.value: END_TURN,  # Pause is treated as normal completion
        LlmStopReason.ERROR.value: REFUSAL,  # Errors are mapped to refusal
        LlmStopReason.TIMEOUT.value: REFUSAL,  # Timeouts are mapped to refusal
        LlmStopReason.SAFETY.value: REFUSAL,  # Safety triggers are mapped to refusal
        LlmStopReason.CANCELLED.value: CANCELLED,  # User cancellation
    }

    return mapping.get(key, END_TURN)


def format_agent_name_as_title(agent_name: str) -> str:
    """
    Format agent name as title case for display.

    Examples:
        code_expert -> Code Expert
        general_assistant -> General Assistant

    Args:
        agent_name: The agent name (typically snake_case)

    Returns:
        Title-cased version of the name
    """
    return agent_name.replace("_", " ").title()


@dataclass
class ACPSessionState:
    """Aggregated per-session ACP state for easier lifecycle management."""

    session_id: str
    instance: AgentInstance
    current_agent_name: str | None = None
    progress_manager: ACPToolProgressManager | None = None
    permission_handler: ACPToolPermissionAdapter | None = None
    terminal_runtime: ACPTerminalRuntime | None = None
    filesystem_runtime: ACPFilesystemRuntime | None = None
    slash_handler: SlashCommandHandler | None = None
    acp_context: ACPContext | None = None
    prompt_context: dict[str, str] = field(default_factory=dict)
    resolved_instructions: dict[str, str] = field(default_factory=dict)


def truncate_description(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length, taking the first line only.

    Args:
        text: The text to truncate
        max_length: Maximum length (default 200 chars per spec)

    Returns:
        Truncated text
    """
    # Take first line only
    first_line = text.split("\n")[0]
    # Truncate to max length
    if len(first_line) > max_length:
        return first_line[:max_length]
    return first_line


class AgentACPServer(ACPAgent):
    """
    Exposes FastAgent agents as an ACP agent through stdio.

    This server:
    - Handles ACP connection initialization and capability negotiation
    - Manages sessions (maps sessionId to AgentInstance)
    - Routes prompts to the appropriate fast-agent agent
    - Returns responses in ACP format
    """

    def __init__(
        self,
        primary_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        instance_scope: str,
        server_name: str = "fast-agent-acp",
        server_version: str | None = None,
        skills_directory_override: Sequence[str | Path] | str | Path | None = None,
        permissions_enabled: bool = True,
        get_registry_version: Callable[[], int] | None = None,
        load_card_callback: Callable[[str, str | None], Awaitable[tuple[list[str], list[str]]]]
        | None = None,
        attach_agent_tools_callback: Callable[[str, Sequence[str]], Awaitable[list[str]]]
        | None = None,
        detach_agent_tools_callback: Callable[[str, Sequence[str]], Awaitable[list[str]]]
        | None = None,
        attach_mcp_server_callback: Callable[
            [str, str, MCPServerSettings | None, MCPAttachOptions | None],
            Awaitable[MCPAttachResult],
        ]
        | None = None,
        detach_mcp_server_callback: Callable[[str, str], Awaitable[MCPDetachResult]] | None = None,
        list_attached_mcp_servers_callback: Callable[[str], Awaitable[list[str]]] | None = None,
        list_configured_detached_mcp_servers_callback: Callable[[str], Awaitable[list[str]]]
        | None = None,
        dump_agent_card_callback: Callable[[str], Awaitable[str]] | None = None,
        reload_callback: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """
        Initialize the ACP server.

        Args:
            primary_instance: The primary agent instance (used in shared mode)
            create_instance: Factory function to create new agent instances
            dispose_instance: Function to dispose of agent instances
            instance_scope: How to scope instances ('shared', 'connection', or 'request')
            server_name: Name of the server for capability advertisement
            server_version: Version of the server (defaults to fast-agent version)
            skills_directory_override: Optional skills directory override (relative to session cwd)
            permissions_enabled: Whether to request tool permissions from client (default: True)
            load_card_callback: Optional callback to load AgentCards at runtime
            attach_agent_tools_callback: Optional callback to attach agent tools at runtime
            detach_agent_tools_callback: Optional callback to detach agent tools at runtime
            attach_mcp_server_callback: Optional callback to attach MCP servers at runtime
            detach_mcp_server_callback: Optional callback to detach MCP servers at runtime
            list_attached_mcp_servers_callback: Optional callback to list attached MCP servers
            list_configured_detached_mcp_servers_callback: Optional callback to list configured
                detached MCP servers
            dump_agent_card_callback: Optional callback to dump AgentCards at runtime
            reload_callback: Optional callback to reload AgentCards
        """
        super().__init__()

        self.primary_instance = primary_instance
        self._create_instance_task = create_instance
        self._dispose_instance_task = dispose_instance
        self._instance_scope = instance_scope
        self._get_registry_version = get_registry_version
        self._load_card_callback = load_card_callback
        self._attach_agent_tools_callback = attach_agent_tools_callback
        self._detach_agent_tools_callback = detach_agent_tools_callback
        self._attach_mcp_server_callback = attach_mcp_server_callback
        self._detach_mcp_server_callback = detach_mcp_server_callback
        self._list_attached_mcp_servers_callback = list_attached_mcp_servers_callback
        self._list_configured_detached_mcp_servers_callback = (
            list_configured_detached_mcp_servers_callback
        )
        self._dump_agent_card_callback = dump_agent_card_callback
        self._reload_callback = reload_callback
        self._primary_registry_version = getattr(primary_instance, "registry_version", 0)
        self._shared_reload_lock = asyncio.Lock()
        self._stale_instances: list[AgentInstance] = []
        self.server_name = server_name
        self._skills_directory_override = skills_directory_override
        self._permissions_enabled = permissions_enabled
        # Use provided version or get fast-agent version
        if server_version is None:
            try:
                server_version = get_version("fast-agent-mcp")
            except Exception:
                server_version = "unknown"
        self.server_version = server_version

        # Session management
        self.sessions: dict[str, AgentInstance] = {}
        self._session_lock = asyncio.Lock()

        # Per-session prompt locks to serialize prompt turns.
        # ACP session/update notifications are correlated only by sessionId, so overlapping
        # prompts would interleave updates and become ambiguous.
        self._prompt_locks: dict[str, asyncio.Lock] = {}

        # Track sessions with active prompts to prevent overlapping requests (per ACP protocol)
        self._active_prompts: set[str] = set()

        # Track asyncio tasks per session for proper task-based cancellation
        self._session_tasks: dict[str, asyncio.Task] = {}

        # Aggregated per-session state
        self._session_state: dict[str, ACPSessionState] = {}

        # Connection reference (set during run_async)
        self._connection: ACPClient | None = None

        # Client capabilities and info (set during initialize)
        self._client_supports_terminal: bool = False
        self._client_supports_fs_read: bool = False
        self._client_supports_fs_write: bool = False
        self._client_capabilities: dict | None = None
        self._client_info: dict | None = None
        self._protocol_version: int | None = None

        # Parsed client capabilities and info for ACPContext
        self._parsed_client_capabilities: FAClientCapabilities | None = None
        self._parsed_client_info: ClientInfo | None = None

        # Determine primary agent using FastAgent default flag when available
        self.primary_agent_name = self._select_primary_agent(primary_instance)

        logger.info(
            "AgentACPServer initialized",
            name="acp_server_initialized",
            agent_count=len(primary_instance.agents),
            instance_scope=instance_scope,
            primary_agent=self.primary_agent_name,
        )

    def _calculate_terminal_output_limit(self, agent: Any) -> int:
        """
        Determine a default terminal output byte limit based on the agent's model.

        Args:
            agent: Agent instance that may expose an llm with model metadata.
        """
        # Some workflow agents (e.g., chain/parallel) don't attach an LLM directly.
        llm = getattr(agent, "_llm", None)
        model_name = getattr(llm, "model_name", None)
        return self._calculate_terminal_output_limit_for_model(model_name)

    @staticmethod
    def _calculate_terminal_output_limit_for_model(model_name: str | None) -> int:
        if not model_name:
            return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

        return calculate_terminal_output_limit_for_model(model_name)

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """
        Handle ACP initialization request.

        Negotiates protocol version and advertises capabilities.
        """
        try:
            # Store protocol version
            self._protocol_version = protocol_version

            # Store client info
            if client_info:
                self._client_info = {
                    "name": getattr(client_info, "name", "unknown"),
                    "version": getattr(client_info, "version", "unknown"),
                }
                # Include title if available
                if hasattr(client_info, "title"):
                    self._client_info["title"] = client_info.title

            # Store client capabilities
            if client_capabilities:
                self._client_supports_terminal = bool(
                    getattr(client_capabilities, "terminal", False)
                )

                # Check for filesystem capabilities
                if hasattr(client_capabilities, "fs"):
                    fs_caps = client_capabilities.fs
                    if fs_caps:
                        self._client_supports_fs_read = bool(
                            getattr(fs_caps, "read_text_file", False)
                        )
                        self._client_supports_fs_write = bool(
                            getattr(fs_caps, "write_text_file", False)
                        )

                # Convert capabilities to a dict for status reporting
                self._client_capabilities = {}
                if hasattr(client_capabilities, "fs"):
                    fs_caps = client_capabilities.fs
                    fs_capabilities = self._extract_fs_capabilities(fs_caps)
                    if fs_capabilities:
                        self._client_capabilities["fs"] = fs_capabilities

                if hasattr(client_capabilities, "terminal") and client_capabilities.terminal:
                    self._client_capabilities["terminal"] = True

                # Store _meta if present
                if hasattr(client_capabilities, "_meta"):
                    meta = client_capabilities._meta
                    if meta:
                        self._client_capabilities["_meta"] = (
                            dict(meta) if isinstance(meta, dict) else {}
                        )

            # Parse client capabilities and info for ACPContext
            self._parsed_client_capabilities = FAClientCapabilities(
                terminal=self._client_supports_terminal,
                fs_read=self._client_supports_fs_read,
                fs_write=self._client_supports_fs_write,
                _meta=self._client_capabilities.get("_meta", {})
                if self._client_capabilities
                else {},
            )
            self._parsed_client_info = ClientInfo.from_acp_info(client_info)

            logger.info(
                "ACP initialize request",
                name="acp_initialize",
                client_protocol=protocol_version,
                client_info=client_info,
                client_supports_terminal=self._client_supports_terminal,
                client_supports_fs_read=self._client_supports_fs_read,
                client_supports_fs_write=self._client_supports_fs_write,
            )

            # Build our capabilities
            agent_capabilities = AgentCapabilities(
                prompt_capabilities=PromptCapabilities(
                    image=True,  # Support image content
                    embedded_context=True,  # Support embedded resources
                    audio=False,  # Don't support audio (yet)
                ),
                load_session=True,
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            )

            # Build agent info using Implementation type
            agent_info = Implementation(
                name=self.server_name,
                version=self.server_version,
            )

            # Minimal "agent auth" hint for ACP clients.
            #
            # Per ACP RFD auth-methods, the default type is "agent" when no type is provided.
            # We keep this strictly within the current AuthMethod schema (id/name/description)
            # to avoid requiring client/SDK support for typed auth metadata yet.
            auth_methods = [
                AuthMethod(
                    id="fast-agent-ai-secrets",
                    name="Configure fast-agent",
                    description=(
                        "Set provider keys in fastagent.secrets.yaml or env vars. "
                        "See docs: [Configuration Reference](https://fast-agent.ai/ref/config_file/)"
                    ),
                )
            ]

            response = InitializeResponse(
                protocol_version=protocol_version,  # Echo back the client's version
                agent_capabilities=agent_capabilities,
                agent_info=agent_info,
                auth_methods=auth_methods,
            )

            logger.info(
                "ACP initialize response sent",
                name="acp_initialize_response",
                protocol_version=response.protocolVersion,
            )

            return response
        except Exception as e:
            logger.error(f"Error in initialize: {e}", name="acp_initialize_error", exc_info=True)
            print(f"ERROR in initialize: {e}", file=__import__("sys").stderr)
            raise

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # ACP clients use this hook to trigger a login/setup flow. Our initial implementation
        # is intentionally conservative: we validate the method id and acknowledge the request.
        #
        # The actual credentials (LLM provider keys, MCP server auth, etc.) are configured via
        # fast-agent config/secrets and existing CLI commands; see the advertised method text.
        if method_id != "fast-agent-ai-secrets":
            raise RequestError.invalid_params(
                {
                    "methodId": method_id,
                    "supported": ["fast-agent-ai-secrets"],
                }
            )

        return AuthenticateResponse()

    def _extract_fs_capabilities(self, fs_caps: Any) -> dict[str, bool]:
        """Normalize filesystem capabilities for status reporting."""
        normalized: dict[str, bool] = {}
        if not fs_caps:
            return normalized

        if isinstance(fs_caps, dict):
            for key, value in fs_caps.items():
                if value is not None:
                    normalized[key] = bool(value)
            return normalized

        for attr in ("read_text_file", "write_text_file"):
            if hasattr(fs_caps, attr):
                value = getattr(fs_caps, attr)
                if value is not None:
                    normalized[attr] = bool(value)

        return normalized

    def _build_session_modes(
        self, instance: AgentInstance, session_state: ACPSessionState | None = None
    ) -> SessionModeState:
        """
        Build SessionModeState from an AgentInstance's agents.

        Each agent in the instance becomes an available mode.

        Args:
            instance: The AgentInstance containing agents

        Returns:
            SessionModeState with available modes and current mode ID
        """
        available_modes: list[SessionMode] = []

        resolved_cache = session_state.resolved_instructions if session_state else {}

        # Get tool_only agents to filter from available modes
        tool_only_agents = getattr(instance.app, "_tool_only_agents", set())

        # Create a SessionMode for each agent (excluding tool_only agents)
        for agent_name, agent in instance.agents.items():
            # Skip tool_only agents - they shouldn't appear in mode listings
            if agent_name in tool_only_agents:
                continue
            # Get instruction from resolved cache (if available) or agent's instruction
            instruction = resolved_cache.get(agent_name) or agent.instruction

            # Format description (first line, truncated to 200 chars)
            description = truncate_description(instruction) if instruction else None
            display_name = format_agent_name_as_title(agent_name)

            # Allow ACP-aware agents to supply custom name/description
            if isinstance(agent, ACPAwareProtocol):
                try:
                    mode_info = agent.acp_mode_info()
                except Exception:
                    logger.warning(
                        "Error getting acp_mode_info from agent",
                        name="acp_mode_info_error",
                        agent_name=agent_name,
                        exc_info=True,
                    )
                    mode_info = None

                if mode_info:
                    if mode_info.name:
                        display_name = mode_info.name
                    if mode_info.description:
                        description = mode_info.description

            if description:
                description = truncate_description(description)

            # Create the SessionMode
            mode = SessionMode(
                id=agent_name,
                name=display_name,
                description=description,
            )
            available_modes.append(mode)

        # Current mode is the primary agent name
        current_mode_id = self.primary_agent_name or (
            list(instance.agents.keys())[0] if instance.agents else "default"
        )

        return SessionModeState(
            available_modes=available_modes,
            current_mode_id=current_mode_id,
        )

    async def _build_session_request_params(
        self, agent: Any, session_state: ACPSessionState | None
    ) -> RequestParams | None:
        """
        Apply late-binding template variables to an agent's instruction for this session.
        """
        # Only apply per-session system prompts when the target agent actually has an LLM.
        # Workflow wrappers (chain/parallel) don't attach an LLM and will forward params
        # to their children, which can override their instructions if we keep the prompt.
        if not getattr(agent, "_llm", None):
            return None

        resolved_cache = session_state.resolved_instructions if session_state else {}
        resolved = resolved_cache.get(getattr(agent, "name", ""), None)
        if isinstance(agent, McpInstructionCapable) or resolved is None:
            context = session_state.prompt_context if session_state else None
            if not context:
                return None
            resolved = await self._resolve_instruction_for_session(agent, context)
            if not resolved:
                return None
            if session_state is not None:
                session_state.resolved_instructions[getattr(agent, "name", "")] = resolved
        return RequestParams(systemPrompt=resolved)

    async def _resolve_instruction_for_session(
        self,
        agent: object,
        context: dict[str, str],
    ) -> str | None:
        template = get_instruction_template(agent)
        if not template:
            return None

        aggregator = None
        skill_manifests = None
        has_filesystem_runtime = False
        effective_context = dict(context)
        if isinstance(agent, McpInstructionCapable):
            aggregator = agent.aggregator
            skill_manifests = agent.skill_manifests
            has_filesystem_runtime = agent.has_filesystem_runtime
            if agent.instruction_context:
                effective_context = dict(agent.instruction_context)

        effective_context = build_agent_instruction_context(agent, effective_context)

        return await build_instruction(
            template,
            aggregator=aggregator,
            skill_manifests=skill_manifests,
            has_filesystem_runtime=has_filesystem_runtime,
            context=effective_context,
            source=getattr(agent, "name", None),
        )

    def _resolve_request_cwd(
        self,
        *,
        cwd: str | None,
        request_name: str,
        warn_if_missing: bool = True,
    ) -> str:
        if cwd:
            return cwd
        default_cwd = str(Path.cwd())
        if warn_if_missing:
            logger.warning(
                "Missing cwd for ACP request; defaulting to process cwd",
                name="acp_missing_cwd",
                request=request_name,
                default_cwd=default_cwd,
            )
        return default_cwd

    async def _maybe_refresh_shared_instance(self) -> None:
        if self._instance_scope != "shared" or not self._get_registry_version:
            return
        if self._active_prompts:
            return

        latest_version = self._get_registry_version()
        if latest_version <= self._primary_registry_version:
            return

        async with self._shared_reload_lock:
            if self._active_prompts:
                return
            latest_version = self._get_registry_version()
            if latest_version <= self._primary_registry_version:
                return

            new_instance = await self._create_instance_task()
            old_instance = self.primary_instance
            self.primary_instance = new_instance
            self._primary_registry_version = getattr(
                new_instance, "registry_version", latest_version
            )
            self._stale_instances.append(old_instance)
            self.primary_agent_name = self._select_primary_agent(new_instance)
            await self._refresh_sessions_for_instance(new_instance)

    async def _replace_instance_for_session(
        self,
        session_state: ACPSessionState,
        *,
        dispose_error_name: str,
        await_refresh_session_state: bool,
    ) -> AgentInstance:
        if self._instance_scope == "shared":
            async with self._shared_reload_lock:
                new_instance = await self._create_instance_task()
                old_instance = self.primary_instance
                self.primary_instance = new_instance
                latest_version = (
                    self._get_registry_version() if self._get_registry_version else None
                )
                self._primary_registry_version = getattr(
                    new_instance, "registry_version", latest_version
                )
                self._stale_instances.append(old_instance)
                self.primary_agent_name = self._select_primary_agent(new_instance)
                await self._refresh_sessions_for_instance(new_instance)
            return session_state.instance

        instance = await self._create_instance_task()
        old_instance = session_state.instance
        session_state.instance = instance
        async with self._session_lock:
            self.sessions[session_state.session_id] = instance
        if await_refresh_session_state:
            await self._refresh_session_state(session_state, instance)
        else:
            self._refresh_session_state(session_state, instance)
        if old_instance != self.primary_instance:
            try:
                await self._dispose_instance_task(old_instance)
            except Exception as exc:
                logger.warning(
                    "Failed to dispose old session instance",
                    name=dispose_error_name,
                    session_id=session_state.session_id,
                    error=str(exc),
                )
        return instance

    async def _refresh_sessions_for_instance(self, instance: AgentInstance) -> None:
        async with self._session_lock:
            for session_id, session_state in self._session_state.items():
                self.sessions[session_id] = instance
                session_state.instance = instance
                await self._refresh_session_state(session_state, instance)

    async def _refresh_session_state(
        self, session_state: ACPSessionState, instance: AgentInstance
    ) -> None:
        prompt_context = session_state.prompt_context or {}
        resolved_for_session: dict[str, str] = {}
        for agent_name, agent in instance.agents.items():
            resolved = await self._resolve_instruction_for_session(agent, prompt_context)
            if resolved:
                resolved_for_session[agent_name] = resolved
        session_state.resolved_instructions = resolved_for_session

        for agent_name, agent in instance.agents.items():
            if isinstance(agent, InstructionContextCapable):
                try:
                    context_with_agent = build_agent_instruction_context(agent, prompt_context)
                    agent.set_instruction_context(context_with_agent)
                except Exception as exc:
                    logger.warning(
                        "Failed to set instruction context on agent",
                        name="acp_instruction_context_failed",
                        session_id=session_state.session_id,
                        agent_name=agent_name,
                        error=str(exc),
                    )

        # Register ACP handlers for agents (including newly loaded ones)
        tool_handler = session_state.progress_manager
        permission_handler = session_state.permission_handler

        if tool_handler:
            workflow_telemetry = ToolHandlerWorkflowTelemetry(
                tool_handler, server_name=self.server_name
            )

            for agent_name, agent in instance.agents.items():
                if isinstance(agent, McpAgentProtocol):
                    aggregator = agent.aggregator
                    # Only set if not already set (avoid duplicate registration)
                    if isinstance(aggregator._tool_handler, NoOpToolExecutionHandler):
                        aggregator._tool_handler = tool_handler
                        logger.debug(
                            "ACP tool handler registered (refresh)",
                            name="acp_tool_handler_refresh",
                            session_id=session_state.session_id,
                            agent_name=agent_name,
                        )

                if isinstance(agent, WorkflowTelemetryCapable):
                    if agent.workflow_telemetry is None:
                        agent.workflow_telemetry = workflow_telemetry

                if isinstance(agent, PlanTelemetryCapable):
                    if agent.plan_telemetry is None and self._connection:
                        plan_telemetry = ACPPlanTelemetryProvider(
                            self._connection, session_state.session_id
                        )
                        agent.plan_telemetry = plan_telemetry

                # Register stream listener (set handles duplicates)
                llm = getattr(agent, "_llm", None)
                if llm and hasattr(llm, "add_tool_stream_listener"):
                    try:
                        llm.add_tool_stream_listener(tool_handler.handle_tool_stream_event)
                    except Exception:
                        pass

        if permission_handler:
            for agent_name, agent in instance.agents.items():
                if isinstance(agent, McpAgentProtocol):
                    aggregator = agent.aggregator
                    if isinstance(aggregator._permission_handler, NoOpToolPermissionHandler):
                        aggregator._permission_handler = permission_handler
                        logger.debug(
                            "ACP permission handler registered (refresh)",
                            name="acp_permission_handler_refresh",
                            session_id=session_state.session_id,
                            agent_name=agent_name,
                        )

        if session_state.terminal_runtime:
            for agent_name, agent in instance.agents.items():
                if isinstance(agent, ShellRuntimeCapable) and agent._shell_runtime_enabled:
                    agent.set_external_runtime(session_state.terminal_runtime)

        if session_state.filesystem_runtime:
            for agent_name, agent in instance.agents.items():
                if isinstance(agent, FilesystemRuntimeCapable):
                    agent.set_filesystem_runtime(session_state.filesystem_runtime)
            # Rebuild instructions now that filesystem runtime is available
            # This ensures skill prompts use read_text_file instead of read_skill
            from fast_agent.core.instruction_refresh import rebuild_agent_instruction

            for agent in instance.agents.values():
                await rebuild_agent_instruction(agent)

        slash_handler = self._create_slash_handler(session_state, instance)
        session_state.slash_handler = slash_handler

        current_agent = session_state.current_agent_name
        if not current_agent or current_agent not in instance.agents:
            current_agent = self.primary_agent_name or next(iter(instance.agents.keys()), None)
            session_state.current_agent_name = current_agent
        if current_agent and session_state.slash_handler:
            session_state.slash_handler.set_current_agent(current_agent)

        session_modes = self._build_session_modes(instance, session_state)
        if current_agent and current_agent in instance.agents:
            session_modes = SessionModeState(
                available_modes=session_modes.available_modes,
                current_mode_id=current_agent,
            )

        if session_state.acp_context:
            slash_handler.set_acp_context(session_state.acp_context)
            session_state.acp_context.set_slash_handler(slash_handler)
            session_state.acp_context.set_resolved_instructions(resolved_for_session)
            session_state.acp_context.set_available_modes(session_modes.available_modes)
            if current_agent:
                session_state.acp_context.set_current_mode(current_agent)

    def _create_slash_handler(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
    ) -> SlashCommandHandler:
        async def load_card(
            source: str, parent_name: str | None
        ) -> tuple[AgentInstance, list[str], list[str]]:
            return await self._load_agent_card_for_session(
                session_state, source, attach_to=parent_name
            )

        async def attach_agent_tools(
            parent_name: str, child_names: Sequence[str]
        ) -> tuple[AgentInstance, list[str]]:
            return await self._attach_agent_tools_for_session(
                session_state, parent_name, child_names
            )

        async def detach_agent_tools(
            parent_name: str, child_names: Sequence[str]
        ) -> tuple[AgentInstance, list[str]]:
            return await self._detach_agent_tools_for_session(
                session_state, parent_name, child_names
            )

        async def attach_mcp_server(
            agent_name: str,
            server_name: str,
            server_config: MCPServerSettings | None = None,
            options: MCPAttachOptions | None = None,
        ) -> MCPAttachResult:
            if not self._attach_mcp_server_callback:
                raise RuntimeError("Runtime MCP server attachment is not available.")
            result = await self._attach_mcp_server_callback(
                agent_name,
                server_name,
                server_config,
                options,
            )

            if session_state.acp_context:
                resolved_instruction = None
                agent = instance.agents.get(agent_name)
                if isinstance(agent, InstructionContextCapable):
                    resolved_instruction = agent.instruction
                await session_state.acp_context.invalidate_instruction_cache(
                    agent_name,
                    resolved_instruction,
                )
                await session_state.acp_context.send_available_commands_update()

            return result

        async def detach_mcp_server(agent_name: str, server_name: str) -> MCPDetachResult:
            if not self._detach_mcp_server_callback:
                raise RuntimeError("Runtime MCP server detachment is not available.")
            result = await self._detach_mcp_server_callback(agent_name, server_name)

            if session_state.acp_context:
                resolved_instruction = None
                agent = instance.agents.get(agent_name)
                if isinstance(agent, InstructionContextCapable):
                    resolved_instruction = agent.instruction
                await session_state.acp_context.invalidate_instruction_cache(
                    agent_name,
                    resolved_instruction,
                )
                await session_state.acp_context.send_available_commands_update()

            return result

        async def list_attached_mcp_servers(agent_name: str) -> list[str]:
            if not self._list_attached_mcp_servers_callback:
                raise RuntimeError("Runtime MCP server listing is not available.")
            return await self._list_attached_mcp_servers_callback(agent_name)

        async def list_configured_detached_mcp_servers(agent_name: str) -> list[str]:
            if not self._list_configured_detached_mcp_servers_callback:
                raise RuntimeError("Configured MCP server listing is not available.")
            return await self._list_configured_detached_mcp_servers_callback(agent_name)

        async def dump_agent_card(agent_name: str) -> str:
            if not self._dump_agent_card_callback:
                raise RuntimeError("AgentCard dumping is not available.")
            return await self._dump_agent_card_callback(agent_name)

        async def reload_cards() -> bool:
            return await self._reload_agent_cards_for_session(session_state.session_id)

        async def set_current_mode(agent_name: str) -> None:
            session_state.current_agent_name = agent_name
            if session_state.acp_context:
                await session_state.acp_context.switch_mode(agent_name)

        async def resolve_instruction_for_system(agent_name: str) -> str | None:
            agent = instance.agents.get(agent_name)
            if agent is None:
                return None
            context = session_state.prompt_context or {}
            if not context:
                return None
            resolved = await self._resolve_instruction_for_session(agent, context)
            if resolved:
                session_state.resolved_instructions[agent_name] = resolved
            return resolved

        return SlashCommandHandler(
            session_state.session_id,
            instance,
            self.primary_agent_name or "default",
            noenv=bool(getattr(instance.app, "_noenv_mode", False)),
            client_info=self._client_info,
            client_capabilities=self._client_capabilities,
            protocol_version=self._protocol_version,
            session_instructions=session_state.resolved_instructions,
            instruction_resolver=resolve_instruction_for_system,
            card_loader=load_card if self._load_card_callback else None,
            attach_agent_callback=(
                attach_agent_tools if self._attach_agent_tools_callback else None
            ),
            detach_agent_callback=(
                detach_agent_tools if self._detach_agent_tools_callback else None
            ),
            attach_mcp_server_callback=(
                attach_mcp_server if self._attach_mcp_server_callback else None
            ),
            detach_mcp_server_callback=(
                detach_mcp_server if self._detach_mcp_server_callback else None
            ),
            list_attached_mcp_servers_callback=(
                list_attached_mcp_servers if self._list_attached_mcp_servers_callback else None
            ),
            list_configured_detached_mcp_servers_callback=(
                list_configured_detached_mcp_servers
                if self._list_configured_detached_mcp_servers_callback
                else None
            ),
            dump_agent_callback=(dump_agent_card if self._dump_agent_card_callback else None),
            reload_callback=reload_cards if self._reload_callback else None,
            set_current_mode_callback=set_current_mode,
        )

    async def _load_agent_card_for_session(
        self,
        session_state: ACPSessionState,
        source: str,
        *,
        attach_to: str | None = None,
    ) -> tuple[AgentInstance, list[str], list[str]]:
        if not self._load_card_callback:
            raise RuntimeError("AgentCard loading is not available.")
        loaded_names, attached_names = await self._load_card_callback(source, attach_to)

        instance = await self._replace_instance_for_session(
            session_state,
            dispose_error_name="acp_card_dispose_error",
            await_refresh_session_state=True,
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, loaded_names, attached_names

    async def _attach_agent_tools_for_session(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        if not self._attach_agent_tools_callback:
            raise RuntimeError("Agent tool attachment is not available.")

        attached_names = await self._attach_agent_tools_callback(parent_name, child_names)
        if not attached_names:
            return session_state.instance, []

        instance = await self._replace_instance_for_session(
            session_state,
            dispose_error_name="acp_attach_dispose_error",
            await_refresh_session_state=False,
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, attached_names

    async def _detach_agent_tools_for_session(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        if not self._detach_agent_tools_callback:
            raise RuntimeError("Agent tool detachment is not available.")

        detached_names = await self._detach_agent_tools_callback(parent_name, child_names)
        if not detached_names:
            return session_state.instance, []

        instance = await self._replace_instance_for_session(
            session_state,
            dispose_error_name="acp_detach_dispose_error",
            await_refresh_session_state=False,
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, detached_names

    async def _reload_agent_cards_for_session(self, session_id: str) -> bool:
        if not self._reload_callback:
            return False
        if session_id in self._active_prompts:
            current_task = asyncio.current_task()
            session_task = self._session_tasks.get(session_id)
            if current_task != session_task:
                raise RuntimeError("Cannot reload while a prompt is active for this session.")

        changed = await self._reload_callback()
        if not changed:
            return False

        if self._instance_scope == "shared":
            await self._maybe_refresh_shared_instance()
            return True

        async with self._session_lock:
            session_state = self._session_state.get(session_id)
        if not session_state:
            return True

        instance = await self._create_instance_task()
        old_instance = session_state.instance
        session_state.instance = instance
        async with self._session_lock:
            self.sessions[session_id] = instance
        await self._refresh_session_state(session_state, instance)
        if old_instance != self.primary_instance:
            try:
                await self._dispose_instance_task(old_instance)
            except Exception as exc:
                logger.warning(
                    "Failed to dispose old session instance after reload",
                    name="acp_reload_dispose_error",
                    session_id=session_id,
                    error=str(exc),
                )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return True

    async def _dispose_stale_instances_if_idle(self) -> None:
        if self._active_prompts:
            return
        if not self._stale_instances:
            return
        stale = list(self._stale_instances)
        self._stale_instances.clear()
        for instance in stale:
            await self._dispose_instance_task(instance)

    def _build_status_line_meta(
        self, agent: Any, turn_start_index: int | None
    ) -> dict[str, Any] | None:
        """Build ACP _meta payload for status line usage display."""
        if not agent or not getattr(agent, "usage_accumulator", None):
            return None

        totals = last_turn_usage(agent.usage_accumulator, turn_start_index)
        if not totals:
            return None

        input_tokens = totals["input_tokens"]
        output_tokens = totals["output_tokens"]
        tool_calls = totals["tool_calls"]
        tool_info = f", {tool_calls} tools" if tool_calls > 0 else ""
        context_pct = agent.usage_accumulator.context_usage_percentage
        context_info = f" ({context_pct:.1f}%)" if context_pct is not None else ""
        status_line = f"{input_tokens:,} in, {output_tokens:,} out{tool_info}{context_info}"

        return {"field_meta": {"openhands.dev/metrics": {"status_line": status_line}}}

    @staticmethod
    def _merge_tool_runner_hooks(
        base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None:
        if base is None:
            return extra
        if extra is None:
            return base

        def merge(one, two):
            if one is None:
                return two
            if two is None:
                return one

            async def merged(*args, **kwargs):
                await one(*args, **kwargs)
                await two(*args, **kwargs)

            return merged

        return ToolRunnerHooks(
            before_llm_call=merge(base.before_llm_call, extra.before_llm_call),
            after_llm_call=merge(base.after_llm_call, extra.after_llm_call),
            before_tool_call=merge(base.before_tool_call, extra.before_tool_call),
            after_tool_call=merge(base.after_tool_call, extra.after_tool_call),
            after_turn_complete=merge(
                base.after_turn_complete, extra.after_turn_complete
            ),
        )

    async def _send_status_line_update(
        self, session_id: str, agent: Any, turn_start_index: int | None
    ) -> None:
        if not self._connection:
            return
        status_line_meta = self._build_status_line_meta(agent, turn_start_index)
        if not status_line_meta:
            return
        try:
            message_chunk = update_agent_message_text("")
            await self._connection.session_update(
                session_id=session_id,
                update=message_chunk,
                **status_line_meta,
            )
        except Exception as exc:
            logger.error(
                f"Error sending status line update: {exc}",
                name="acp_status_line_update_error",
                exc_info=True,
            )

    async def _initialize_session_state(
        self,
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
    ) -> tuple[ACPSessionState, SessionModeState]:
        """
        Initialize or refresh ACP session state for a given session id.

        Returns the updated session state and the current mode state.
        """
        _ = mcp_servers
        await self._maybe_refresh_shared_instance()

        async with self._session_lock:
            session_state = self._session_state.get(session_id)
            if session_state:
                instance = session_state.instance
                self.sessions[session_id] = instance
            else:
                # Determine which instance to use based on scope
                if self._instance_scope == "shared":
                    # All sessions share the primary instance
                    instance = self.primary_instance
                elif self._instance_scope in ["connection", "request"]:
                    # Create a new instance for this session
                    instance = await self._create_instance_task()
                else:
                    # Default to shared
                    instance = self.primary_instance

                self.sessions[session_id] = instance
                session_state = ACPSessionState(session_id=session_id, instance=instance)
                self._session_state[session_id] = session_state

            # Serialize prompts per session
            if session_id not in self._prompt_locks:
                self._prompt_locks[session_id] = asyncio.Lock()

            # Create tool progress manager for this session if connection is available
            tool_handler = session_state.progress_manager
            if self._connection and tool_handler is None:
                # Create a progress manager for this session
                tool_handler = ACPToolProgressManager(self._connection, session_id)
                session_state.progress_manager = tool_handler
                workflow_telemetry = ToolHandlerWorkflowTelemetry(
                    tool_handler, server_name=self.server_name
                )

                logger.info(
                    "ACP tool progress manager created for session",
                    name="acp_tool_progress_init",
                    session_id=session_id,
                )

                # Register tool handler with agents' aggregators
                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, McpAgentProtocol):
                        aggregator = agent.aggregator
                        aggregator._tool_handler = tool_handler

                        logger.info(
                            "ACP tool handler registered",
                            name="acp_tool_handler_registered",
                            session_id=session_id,
                            agent_name=agent_name,
                        )

                    if isinstance(agent, WorkflowTelemetryCapable):
                        agent.workflow_telemetry = workflow_telemetry

                    # Set up plan telemetry for agents that support it (e.g., IterativePlanner)
                    if isinstance(agent, PlanTelemetryCapable):
                        plan_telemetry = ACPPlanTelemetryProvider(self._connection, session_id)
                        agent.plan_telemetry = plan_telemetry
                        logger.info(
                            "ACP plan telemetry registered",
                            name="acp_plan_telemetry_registered",
                            session_id=session_id,
                            agent_name=agent_name,
                        )

                    # Register tool handler as stream listener to get early tool start events
                    llm = getattr(agent, "_llm", None)
                    if llm and hasattr(llm, "add_tool_stream_listener"):
                        try:
                            llm.add_tool_stream_listener(tool_handler.handle_tool_stream_event)
                            logger.info(
                                "ACP tool handler registered as stream listener",
                                name="acp_tool_stream_listener_registered",
                                session_id=session_id,
                                agent_name=agent_name,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to register tool stream listener: {e}",
                                name="acp_tool_stream_listener_failed",
                                exc_info=True,
                            )

            # If permissions are enabled, create and register permission handler
            if (
                self._connection
                and self._permissions_enabled
                and session_state.permission_handler is None
            ):
                # Create shared permission store for this session
                session_cwd = cwd or "."
                permission_store = PermissionStore(cwd=session_cwd)

                # Create permission adapter with tool_handler for toolCallId lookup
                permission_handler = ACPToolPermissionAdapter(
                    connection=self._connection,
                    session_id=session_id,
                    store=permission_store,
                    cwd=session_cwd,
                    tool_handler=tool_handler,
                )
                session_state.permission_handler = permission_handler

                # Register permission handler with all agents' aggregators
                permission_agents: list[str] = []
                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, McpAgentProtocol):
                        aggregator = agent.aggregator
                        aggregator._permission_handler = permission_handler
                        permission_agents.append(agent_name)

                        logger.info(
                            "ACP permission handler registered",
                            name="acp_permission_handler_registered",
                            session_id=session_id,
                            agent_name=agent_name,
                        )

                logger.info(
                    "ACP tool permissions enabled for session",
                    name="acp_permissions_init",
                    session_id=session_id,
                    cwd=cwd,
                )

            # If client supports terminals and we have shell runtime enabled,
            # inject ACP terminal runtime to replace local ShellRuntime
            if (
                self._connection
                and self._client_supports_terminal
                and session_state.terminal_runtime is None
            ):
                # Check if any agent has shell runtime enabled
                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, ShellRuntimeCapable) and agent._shell_runtime_enabled:
                        # Create ACPTerminalRuntime for this session
                        default_limit = getattr(
                            agent._shell_runtime,
                            "_output_byte_limit",
                            self._calculate_terminal_output_limit(agent),
                        )
                        # Get permission handler if enabled for this session
                        perm_handler = session_state.permission_handler
                        terminal_runtime = ACPTerminalRuntime(
                            connection=self._connection,
                            session_id=session_id,
                            activation_reason="via ACP terminal support",
                            timeout_seconds=getattr(
                                agent._shell_runtime,
                                "timeout_seconds",
                                90,  # ty: ignore[unresolved-attribute]
                            ),
                            tool_handler=tool_handler,
                            default_output_byte_limit=default_limit,
                            permission_handler=perm_handler,
                        )

                        # Inject into agent
                        agent.set_external_runtime(terminal_runtime)
                        session_state.terminal_runtime = terminal_runtime

                        logger.info(
                            "ACP terminal runtime injected",
                            name="acp_terminal_injected",
                            session_id=session_id,
                            agent_name=agent_name,
                            default_output_limit=default_limit,
                        )

            # If client supports filesystem operations, inject ACP filesystem runtime
            if (
                self._connection
                and (self._client_supports_fs_read or self._client_supports_fs_write)
                and session_state.filesystem_runtime is None
            ):
                # Get permission handler if enabled for this session
                perm_handler = session_state.permission_handler
                # Create ACPFilesystemRuntime for this session with appropriate capabilities
                filesystem_runtime = ACPFilesystemRuntime(
                    connection=self._connection,
                    session_id=session_id,
                    activation_reason="via ACP filesystem support",
                    enable_read=self._client_supports_fs_read,
                    enable_write=self._client_supports_fs_write,
                    tool_handler=tool_handler,
                    permission_handler=perm_handler,
                )
                session_state.filesystem_runtime = filesystem_runtime

                # Inject filesystem runtime into each agent
                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, FilesystemRuntimeCapable):
                        agent.set_filesystem_runtime(filesystem_runtime)
                        logger.info(
                            "ACP filesystem runtime injected",
                            name="acp_filesystem_injected",
                            session_id=session_id,
                            agent_name=agent_name,
                            read_enabled=self._client_supports_fs_read,
                            write_enabled=self._client_supports_fs_write,
                        )

                # Rebuild instructions now that filesystem runtime is available
                # This ensures skill prompts use read_text_file instead of read_skill
                from fast_agent.core.instruction_refresh import rebuild_agent_instruction

                for agent in instance.agents.values():
                    await rebuild_agent_instruction(agent)

        # Track per-session template variables (used for late instruction binding)
        session_context: dict[str, str] = {}
        enrich_with_environment_context(
            session_context, cwd, self._client_info, self._skills_directory_override
        )
        session_state.prompt_context = session_context

        # Cache resolved instructions for this session (without mutating shared instances)
        resolved_for_session: dict[str, str] = {}
        for agent_name, agent in instance.agents.items():
            resolved = await self._resolve_instruction_for_session(agent, session_context)
            if resolved:
                resolved_for_session[agent_name] = resolved
        if resolved_for_session:
            session_state.resolved_instructions = resolved_for_session

        # Set session context on agents that have InstructionBuilder
        # This ensures {{env}}, {{workspaceRoot}}, etc. are available when rebuilding
        for agent_name, agent in instance.agents.items():
            if isinstance(agent, InstructionContextCapable):
                try:
                    context_with_agent = build_agent_instruction_context(agent, session_context)
                    agent.set_instruction_context(context_with_agent)
                except Exception as e:
                    logger.warning(f"Failed to set instruction context on agent {agent_name}: {e}")

        # Create slash command handler for this session
        slash_handler = self._create_slash_handler(session_state, instance)
        session_state.slash_handler = slash_handler

        # Create ACPContext for this session - centralizes all ACP state
        acp_context = session_state.acp_context
        if self._connection:
            if acp_context is None:
                acp_context = ACPContext(
                    connection=self._connection,
                    session_id=session_id,
                    client_capabilities=self._parsed_client_capabilities,
                    client_info=self._parsed_client_info,
                    protocol_version=self._protocol_version,
                )
                session_state.acp_context = acp_context

            # Store references to runtimes and handlers in ACPContext
            if session_state.terminal_runtime:
                acp_context.set_terminal_runtime(session_state.terminal_runtime)
            if session_state.filesystem_runtime:
                acp_context.set_filesystem_runtime(session_state.filesystem_runtime)
            if session_state.permission_handler:
                acp_context.set_permission_handler(session_state.permission_handler)
            if session_state.progress_manager:
                acp_context.set_progress_manager(session_state.progress_manager)

            slash_handler.set_acp_context(acp_context)
            acp_context.set_slash_handler(slash_handler)

            # Share the resolved instructions cache so agents can invalidate it
            acp_context.set_resolved_instructions(session_state.resolved_instructions)

            # Set ACPContext on each agent's Context object (if they have one)
            for agent_name, agent in instance.agents.items():
                context = getattr(agent, "context", None)
                if isinstance(context, Context):
                    context.acp = acp_context
                    logger.debug(
                        "ACPContext set on agent",
                        name="acp_context_set",
                        session_id=session_id,
                        agent_name=agent_name,
                    )

            logger.info(
                "ACPContext created for session",
                name="acp_context_created",
                session_id=session_id,
                has_terminal=acp_context.terminal_runtime is not None,
                has_filesystem=acp_context.filesystem_runtime is not None,
                has_permissions=acp_context.permission_handler is not None,
            )

        # Schedule available_commands_update notification to be sent after response is returned
        # This ensures the client receives session/new response before the session/update notification
        if self._connection:
            asyncio.create_task(self._send_available_commands_update(session_id))

        # Build session modes from the instance's agents
        session_modes = self._build_session_modes(instance, session_state)

        # Initialize the current agent for this session
        session_state.current_agent_name = session_modes.currentModeId

        # Update ACPContext with mode information
        if session_state.acp_context:
            session_state.acp_context.set_available_modes(session_modes.availableModes)
            session_state.acp_context.set_current_mode(session_modes.currentModeId)

        logger.info(
            "Session modes initialized",
            name="acp_session_modes_init",
            session_id=session_id,
            current_mode=session_modes.currentModeId,
            mode_count=len(session_modes.availableModes),
        )

        return session_state, session_modes

    @staticmethod
    def _extract_session_title(metadata: object) -> str | None:
        if not isinstance(metadata, Mapping):
            return None
        return extract_session_title(cast("Mapping[str, object]", metadata))

    def _build_history_updates(
        self,
        history: Sequence[PromptMessageExtended],
    ) -> list[UserMessageChunk | AgentMessageChunk]:
        updates: list[UserMessageChunk | AgentMessageChunk] = []
        for message in history:
            role_value = message.role.value if hasattr(message.role, "value") else str(
                message.role
            )
            if role_value == "user":
                update_builder = update_user_message
            elif role_value == "assistant":
                update_builder = update_agent_message
            else:
                continue

            for content in message.content:
                acp_block = convert_mcp_content_to_acp(content)
                if acp_block is None:
                    continue
                updates.append(update_builder(acp_block))

        return updates

    async def _send_session_history_updates(
        self,
        session_state: ACPSessionState,
        session: Session,
        agent_name: str | None,
    ) -> None:
        if not self._connection:
            return

        try:
            title = self._extract_session_title(session.info.metadata)
            info_update = SessionInfoUpdate(
                session_update="session_info_update",
                title=title,
                updated_at=session.info.last_activity.isoformat(),
            )
            await self._connection.session_update(
                session_id=session_state.session_id,
                update=info_update,
            )

            if not agent_name:
                return
            agent = session_state.instance.agents.get(agent_name)
            if not agent:
                return

            history = list(getattr(agent, "message_history", []))
            if not history:
                return

            updates = self._build_history_updates(history)
            for update in updates:
                await self._connection.session_update(
                    session_id=session_state.session_id,
                    update=update,
                )

            logger.info(
                "Sent session history updates",
                name="acp_session_history_sent",
                session_id=session_state.session_id,
                message_count=len(history),
                update_count=len(updates),
            )
        except Exception as exc:
            logger.error(
                f"Error sending session history updates: {exc}",
                name="acp_session_history_error",
                session_id=session_state.session_id,
                exc_info=True,
            )

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List saved sessions for the current environment."""
        _ = kwargs
        request_cwd = self._resolve_request_cwd(
            cwd=cwd,
            request_name="session/list",
            warn_if_missing=False,
        )
        manager_cwd = Path(request_cwd).expanduser().resolve()
        manager = get_session_manager(cwd=manager_cwd)
        sessions = manager.list_sessions()

        start_index = 0
        if cursor:
            try:
                start_index = int(cursor)
            except ValueError:
                logger.warning(
                    "Invalid session list cursor",
                    name="acp_session_list_cursor_invalid",
                    cursor=cursor,
                )
                start_index = 0

        limit = get_session_history_window()
        if limit > 0:
            page = sessions[start_index : start_index + limit]
            next_cursor = (
                str(start_index + limit) if start_index + limit < len(sessions) else None
            )
        else:
            page = sessions[start_index:]
            next_cursor = None

        session_cwd = manager_cwd
        acp_sessions = []
        for session_info in page:
            title = self._extract_session_title(session_info.metadata)
            acp_sessions.append(
                AcpSessionInfo(
                    session_id=session_info.name,
                    cwd=str(session_cwd),
                    title=title,
                    updated_at=session_info.last_activity.isoformat(),
                )
            )

        return ListSessionsResponse(sessions=acp_sessions, next_cursor=next_cursor)

    async def load_session(
        self,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        session_id: str,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        """Load a saved session and stream history updates."""
        _ = kwargs
        request_cwd = self._resolve_request_cwd(
            cwd=cwd,
            request_name="session/load",
        )
        logger.info(
            "ACP load session request",
            name="acp_load_session",
            session_id=session_id,
            cwd=request_cwd,
            mcp_server_count=len(mcp_servers),
        )
        async with self._session_lock:
            existing_session = session_id in self._session_state

        session_state, session_modes = await self._initialize_session_state(
            session_id,
            cwd=request_cwd,
            mcp_servers=mcp_servers,
        )

        manager = get_session_manager(cwd=Path(request_cwd).expanduser().resolve())
        result = manager.resume_session_agents(
            session_state.instance.agents,
            session_id,
            default_agent_name=self.primary_agent_name,
        )
        if not result:
            logger.error(
                "Session not found for load_session",
                name="acp_load_session_not_found",
                session_id=session_id,
            )
            if not existing_session:
                async with self._session_lock:
                    self.sessions.pop(session_id, None)
                    self._session_state.pop(session_id, None)
                    self._prompt_locks.pop(session_id, None)
                if session_state.instance != self.primary_instance:
                    await self._dispose_instance_task(session_state.instance)
            raise RequestError(
                -32002,
                f"Session not found: {session_id}",
                {
                    "uri": session_id,
                    "reason": "Session not found",
                    "details": (
                        f"Session {session_id} could not be resolved from {request_cwd}"
                    ),
                },
            )

        session = result.session
        loaded = result.loaded
        missing_agents = result.missing_agents
        usage_notices = result.usage_notices
        if missing_agents:
            logger.warning(
                "Missing agents while loading session",
                name="acp_load_session_missing_agents",
                session_id=session_id,
                missing_agents=missing_agents,
            )
        for usage_notice in usage_notices:
            logger.warning(
                usage_notice,
                name="acp_load_session_usage_unavailable",
                session_id=session_id,
            )

        current_agent = session_state.current_agent_name
        if len(loaded) == 1:
            current_agent = next(iter(loaded.keys()))
        if not current_agent or current_agent not in session_state.instance.agents:
            current_agent = self.primary_agent_name or next(
                iter(session_state.instance.agents.keys()),
                None,
            )

        if current_agent:
            session_state.current_agent_name = current_agent
            if session_state.slash_handler:
                session_state.slash_handler.set_current_agent(current_agent)
            if session_state.acp_context:
                session_state.acp_context.set_current_mode(current_agent)

        if current_agent and current_agent != session_modes.currentModeId:
            session_modes = SessionModeState(
                available_modes=session_modes.availableModes,
                current_mode_id=current_agent,
            )
            if session_state.acp_context:
                session_state.acp_context.set_available_modes(
                    session_modes.availableModes
                )
                session_state.acp_context.set_current_mode(current_agent)

        if self._connection:
            asyncio.create_task(
                self._send_session_history_updates(
                    session_state,
                    session,
                    current_agent,
                )
            )

        logger.info(
            "ACP session loaded",
            name="acp_session_loaded",
            session_id=session_id,
            loaded_agents=sorted(loaded.keys()),
        )

        return LoadSessionResponse(modes=session_modes)

    async def resume_session(
        self,
        session_id: str,
        cwd: str | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        """Alias for session/load to support unstable session/resume."""
        _ = kwargs
        request_cwd = self._resolve_request_cwd(
            cwd=cwd,
            request_name="session/resume",
        )
        response = await self.load_session(
            cwd=request_cwd,
            mcp_servers=mcp_servers or [],
            session_id=session_id,
        )
        assert response is not None
        return ResumeSessionResponse(modes=response.modes, models=response.models)

    async def new_session(
        self,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        cwd: str | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """
        Handle new session request.

        Creates a new session and maps it to an AgentInstance based on instance_scope.
        """
        request_cwd = self._resolve_request_cwd(
            cwd=cwd,
            request_name="session/new",
        )
        manager = get_session_manager(cwd=Path(request_cwd).expanduser().resolve())
        session_id = manager.generate_session_id()

        logger.info(
            "ACP new session request",
            name="acp_new_session",
            session_id=session_id,
            instance_scope=self._instance_scope,
            cwd=request_cwd,
            mcp_server_count=len(mcp_servers),
        )

        session_state, session_modes = await self._initialize_session_state(
            session_id,
            cwd=request_cwd,
            mcp_servers=mcp_servers,
        )

        logger.info(
            "ACP new session created",
            name="acp_new_session_created",
            session_id=session_id,
            total_sessions=len(self.sessions),
            terminal_enabled=session_state.terminal_runtime is not None,
            filesystem_enabled=session_state.filesystem_runtime is not None,
        )

        return NewSessionResponse(session_id=session_id, modes=session_modes)

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse | None:
        """
        Handle session mode change request.

        Updates the current agent for the session to route future prompts
        to the selected mode (agent).

        Args:
            mode_id: The ID of the mode (agent) to switch to
            session_id: The session ID

        Returns:
            SetSessionModeResponse (empty response on success)

        Raises:
            ValueError: If session not found or mode ID is invalid
        """
        logger.info(
            "ACP set session mode request",
            name="acp_set_session_mode",
            session_id=session_id,
            mode_id=mode_id,
        )

        # Get the agent instance for this session
        async with self._session_lock:
            instance = self.sessions.get(session_id)
            session_state = self._session_state.get(session_id)

        if not instance:
            logger.error(
                "Session not found for set_session_mode",
                name="acp_set_mode_error",
                session_id=session_id,
            )
            raise ValueError(f"Session not found: {session_id}")

        # Validate that the mode_id exists in the instance's agents
        if mode_id not in instance.agents:
            logger.error(
                "Invalid mode ID for set_session_mode",
                name="acp_set_mode_invalid",
                session_id=session_id,
                mode_id=mode_id,
                available_modes=list(instance.agents.keys()),
            )
            raise ValueError(
                f"Invalid mode ID '{mode_id}'. Available modes: {list(instance.agents.keys())}"
            )

        # Update the session's current agent
        if session_state:
            session_state.current_agent_name = mode_id

        # Update slash handler's current agent so it queries the right agent's commands
        if session_state and session_state.slash_handler:
            session_state.slash_handler.set_current_agent(mode_id)

        # Update ACPContext and send available_commands_update
        # (commands may differ per agent)
        if session_state and session_state.acp_context:
            acp_context = session_state.acp_context
            acp_context.set_current_mode(mode_id)
            await acp_context.send_available_commands_update()

        logger.info(
            "Session mode updated",
            name="acp_set_session_mode_success",
            session_id=session_id,
            new_mode=mode_id,
        )

        return SetSessionModeResponse()

    def _select_primary_agent(self, instance: AgentInstance) -> str | None:
        """
        Pick the default agent to expose as the initial ACP mode.

        Respects AgentConfig.default when set; otherwise falls back to the first agent.
        """
        if not instance.agents:
            return None

        for agent_name, agent in instance.agents.items():
            config = getattr(agent, "config", None)
            if config and getattr(config, "default", False):
                return agent_name

        return next(iter(instance.agents.keys()))

    async def prompt(
        self,
        prompt: list[ACPContentBlock],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Handle prompt request.

        ACP session/update notifications are correlated only by sessionId (no per-turn id).
        To avoid interleaved updates, we serialize prompt turns per session.

        If a client sends overlapping session/prompt requests for the same sessionId, this
        method will await a per-session lock (i.e., queue the prompt) rather than refusing.
        """
        prompt_lock = await self._get_prompt_lock(session_id)
        async with prompt_lock:
            return await self._prompt_locked(prompt=prompt, session_id=session_id, **kwargs)

    async def _get_prompt_lock(self, session_id: str) -> asyncio.Lock:
        """Get/create the lock used to serialize prompts for a session."""
        async with self._session_lock:
            lock = self._prompt_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._prompt_locks[session_id] = lock
            return lock

    async def _prompt_locked(
        self,
        prompt: list[ACPContentBlock],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """
        Handle prompt request.

        Extracts the prompt text, sends it to the fast-agent agent, and sends the response
        back to the client via sessionUpdate notifications.

        Per ACP protocol, only one prompt can be active per session at a time. If a prompt
        is already in progress for this session, this will immediately return a refusal.
        """
        logger.info(
            "ACP prompt request",
            name="acp_prompt",
            session_id=session_id,
        )

        await self._maybe_refresh_shared_instance()

        # Mark this session as having an active prompt

        async with self._session_lock:
            self._active_prompts.add(session_id)

            # Track the current task for proper cancellation via asyncio.Task.cancel()

            current_task = asyncio.current_task()

            if current_task:
                self._session_tasks[session_id] = current_task

        # Use try/finally to ensure session is always removed from active prompts
        try:
            # Get the agent instance for this session
            async with self._session_lock:
                instance = self.sessions.get(session_id)

            if not instance:
                logger.error(
                    "ACP prompt error: session not found",
                    name="acp_prompt_error",
                    session_id=session_id,
                )
                # Return an error response
                return PromptResponse(stop_reason=REFUSAL)

            # Inline resource URIs for slash commands (e.g., /card @file.txt)
            processed_prompt = inline_resources_for_slash_command(prompt)

            # Convert ACP content blocks to MCP format
            mcp_content_blocks = convert_acp_prompt_to_mcp_content_blocks(processed_prompt)

            # Create a PromptMessageExtended with the converted content
            prompt_message = PromptMessageExtended(
                role="user",
                content=mcp_content_blocks,
            )

            # Get current agent for this session (defaults to primary agent if not set).
            # Prefer ACPContext.current_mode so agent-initiated mode switches route correctly.
            session_state = self._session_state.get(session_id)
            acp_context = session_state.acp_context if session_state else None
            current_agent_name = None
            if acp_context is not None:
                current_agent_name = acp_context.current_mode
            if not current_agent_name and session_state:
                current_agent_name = session_state.current_agent_name
            if not current_agent_name:
                current_agent_name = self.primary_agent_name

            # Check if this is a slash command
            # Only process slash commands if the prompt is a single text block
            # This ensures resources, images, and multi-part prompts are never treated as commands
            slash_handler = session_state.slash_handler if session_state else None
            is_single_text_block = len(mcp_content_blocks) == 1 and is_text_content(
                mcp_content_blocks[0]
            )
            prompt_text = prompt_message.all_text() or ""
            if (
                slash_handler
                and is_single_text_block
                and slash_handler.is_slash_command(prompt_text)
            ):
                logger.info(
                    "Processing slash command",
                    name="acp_slash_command",
                    session_id=session_id,
                    prompt_text=prompt_text[:100],  # Log first 100 chars
                )

                # Update slash handler with current agent before executing command
                slash_handler.set_current_agent(current_agent_name or "default")

                # Parse and execute the command
                command_name, arguments = slash_handler.parse_command(prompt_text)
                response_text = await slash_handler.execute_command(command_name, arguments)

                # Send the response via sessionUpdate
                if self._connection and response_text:
                    try:
                        message_chunk = update_agent_message_text(response_text)
                        await self._connection.session_update(
                            session_id=session_id, update=message_chunk
                        )
                        logger.info(
                            "Sent slash command response",
                            name="acp_slash_command_response",
                            session_id=session_id,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error sending slash command response: {e}",
                            name="acp_slash_command_response_error",
                            exc_info=True,
                        )

                # Return success
                return PromptResponse(stop_reason=END_TURN)

            logger.info(
                "Sending prompt to fast-agent",
                name="acp_prompt_send",
                session_id=session_id,
                agent=current_agent_name,
                content_blocks=len(mcp_content_blocks),
            )

            # Send to the fast-agent agent with streaming support
            # Track the stop reason to return in PromptResponse
            acp_stop_reason: StopReason = END_TURN
            status_line_meta: dict[str, Any] | None = None
            try:
                if current_agent_name:
                    agent = instance.agents[current_agent_name]

                    # Set up streaming if connection is available and agent supports it
                    stream_listener = None
                    remove_listener: Callable[[], None] | None = None
                    streaming_tasks: list[asyncio.Task] = []
                    if self._connection and isinstance(agent, StreamingAgentProtocol):
                        connection = self._connection
                        update_lock = asyncio.Lock()

                        async def send_stream_update(chunk: StreamChunk) -> None:
                            """Send sessionUpdate with accumulated text so far."""
                            if not chunk.text:
                                return
                            try:
                                async with update_lock:
                                    if chunk.is_reasoning:
                                        message_chunk = update_agent_thought_text(chunk.text)
                                    else:
                                        message_chunk = update_agent_message_text(chunk.text)
                                    await connection.session_update(
                                        session_id=session_id, update=message_chunk
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error sending stream update: {e}",
                                    name="acp_stream_error",
                                    exc_info=True,
                                )

                        def on_stream_chunk(chunk: StreamChunk):
                            """
                            Sync callback from fast-agent streaming.
                            Sends each chunk as it arrives to the ACP client.
                            """
                            if not chunk or not chunk.text:
                                return
                            task = asyncio.create_task(send_stream_update(chunk))
                            streaming_tasks.append(task)

                        # Register the stream listener and keep the cleanup function
                        stream_listener = on_stream_chunk
                        remove_listener = agent.add_stream_listener(stream_listener)

                        logger.info(
                            "Streaming enabled for prompt",
                            name="acp_streaming_enabled",
                            session_id=session_id,
                        )

                    try:
                        # This will trigger streaming callbacks as chunks arrive
                        session_request_params = await self._build_session_request_params(
                            agent, session_state
                        )
                        turn_start_index = None
                        if isinstance(agent, AgentProtocol) and agent.usage_accumulator is not None:
                            turn_start_index = len(agent.usage_accumulator.turns)
                        previous_hooks = None
                        restore_hooks = False
                        tool_hook_agent: ToolRunnerHookCapable | None = None
                        if (
                            self._connection
                            and isinstance(agent, ToolRunnerHookCapable)
                            and turn_start_index is not None
                        ):
                            tool_hook_agent = agent

                            async def after_llm_call(_runner, message):
                                if message.stop_reason != LlmStopReason.TOOL_USE:
                                    return
                                await self._send_status_line_update(
                                    session_id, agent, turn_start_index
                                )

                            status_hook = ToolRunnerHooks(after_llm_call=after_llm_call)
                            try:
                                previous_hooks = tool_hook_agent.tool_runner_hooks
                                tool_hook_agent.tool_runner_hooks = self._merge_tool_runner_hooks(
                                    previous_hooks, status_hook
                                )
                                restore_hooks = True
                            except AttributeError:
                                previous_hooks = None
                                restore_hooks = False

                        try:
                            result = await agent.generate(
                                prompt_message,
                                request_params=session_request_params,
                            )
                        finally:
                            if restore_hooks and tool_hook_agent is not None:
                                tool_hook_agent.tool_runner_hooks = previous_hooks
                        response_text = result.last_text() or "No content generated"
                        status_line_meta = self._build_status_line_meta(agent, turn_start_index)

                        # Map the LLM stop reason to ACP stop reason
                        try:
                            acp_stop_reason = map_llm_stop_reason_to_acp(result.stop_reason)
                        except Exception as e:
                            logger.error(
                                f"Error mapping stop reason: {e}",
                                name="acp_stop_reason_error",
                                exc_info=True,
                            )
                            # Default to END_TURN on error
                            acp_stop_reason = END_TURN

                        logger.info(
                            "Received complete response from fast-agent",
                            name="acp_prompt_response",
                            session_id=session_id,
                            response_length=len(response_text),
                            llm_stop_reason=str(result.stop_reason) if result.stop_reason else None,
                            acp_stop_reason=acp_stop_reason,
                        )

                        # Wait for all streaming tasks to complete before sending final message
                        # and returning PromptResponse. This ensures all chunks arrive before END_TURN.
                        if streaming_tasks:
                            try:
                                await asyncio.gather(*streaming_tasks)
                                logger.debug(
                                    f"All {len(streaming_tasks)} streaming tasks completed",
                                    name="acp_streaming_complete",
                                    session_id=session_id,
                                    task_count=len(streaming_tasks),
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error waiting for streaming tasks: {e}",
                                    name="acp_streaming_wait_error",
                                    exc_info=True,
                                )

                        # Only send final update if no streaming chunks were sent
                        # When chunks were streamed, the final chunk already contains the complete response
                        # This prevents duplicate messages from being sent to the client
                        if not streaming_tasks and self._connection and response_text:
                            try:
                                message_chunk = update_agent_message_text(response_text)
                                if status_line_meta:
                                    await self._connection.session_update(
                                        session_id=session_id,
                                        update=message_chunk,
                                        **status_line_meta,
                                    )
                                else:
                                    await self._connection.session_update(
                                        session_id=session_id, update=message_chunk
                                    )
                                logger.info(
                                    "Sent final sessionUpdate with complete response (no streaming)",
                                    name="acp_final_update",
                                    session_id=session_id,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error sending final update: {e}",
                                    name="acp_final_update_error",
                                    exc_info=True,
                                )
                        elif streaming_tasks and self._connection and status_line_meta:
                            try:
                                message_chunk = update_agent_message_text("")
                                await self._connection.session_update(
                                    session_id=session_id,
                                    update=message_chunk,
                                    **status_line_meta,
                                )
                                logger.debug(
                                    "Sent status line metadata update after streaming",
                                    name="acp_status_line_update",
                                    session_id=session_id,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error sending status line update: {e}",
                                    name="acp_status_line_update_error",
                                    exc_info=True,
                                )

                    except Exception as send_error:
                        # Make sure listener is cleaned up even on error
                        if stream_listener and remove_listener:
                            try:
                                remove_listener()
                                logger.info(
                                    "Removed stream listener after error",
                                    name="acp_streaming_cleanup_error",
                                    session_id=session_id,
                                )
                            except Exception:
                                logger.warning("Failed to remove ACP stream listener after error")
                        # Re-raise the original error
                        raise send_error

                    finally:
                        # Clean up stream listener (if not already cleaned up in except)
                        if stream_listener and remove_listener:
                            try:
                                remove_listener()
                            except Exception:
                                logger.warning("Failed to remove ACP stream listener")
                            else:
                                logger.info(
                                    "Removed stream listener",
                                    name="acp_streaming_cleanup",
                                    session_id=session_id,
                                )

                else:
                    logger.error("No primary agent available")
            except Exception as e:
                logger.error(
                    f"Error processing prompt: {e}",
                    name="acp_prompt_error",
                    exc_info=True,
                )
                import sys
                import traceback

                print(f"ERROR processing prompt: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise

            # Return response with appropriate stop reason
            return PromptResponse(
                stop_reason=acp_stop_reason,
                field_meta=status_line_meta,
            )
        except asyncio.CancelledError:
            # Task was cancelled - return appropriate response
            logger.info(
                "Prompt cancelled by user",
                name="acp_prompt_cancelled",
                session_id=session_id,
            )
            return PromptResponse(stop_reason="cancelled")
        finally:
            # Always remove session from active prompts and cleanup task
            async with self._session_lock:
                self._active_prompts.discard(session_id)
                self._session_tasks.pop(session_id, None)
            logger.debug(
                "Removed session from active prompts",
                name="acp_prompt_complete",
                session_id=session_id,
            )
            await self._dispose_stale_instances_if_idle()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """
        Handle session/cancel notification from the client.

        This cancels any in-progress prompt for the specified session.
        Per ACP protocol, we should stop all LLM requests and tool invocations
        as soon as possible.

        Uses asyncio.Task.cancel() for proper async cancellation, which raises
        asyncio.CancelledError in the running task.
        """
        logger.info(
            "ACP cancel request received",
            name="acp_cancel",
            session_id=session_id,
        )

        # Get the task for this session and cancel it
        async with self._session_lock:
            task = self._session_tasks.get(session_id)
            if task and not task.done():
                task.cancel()
                logger.info(
                    "Task cancelled for session",
                    name="acp_cancel_task",
                    session_id=session_id,
                )
            else:
                logger.warning(
                    "No active prompt to cancel for session",
                    name="acp_cancel_no_active",
                    session_id=session_id,
                )

    def on_connect(self, conn: ACPClient) -> None:
        """
        Called when connection is established.

        Store connection reference for sending session_update notifications.
        """
        self._connection = conn
        logger.info("ACP connection established via on_connect")

    async def run_async(self) -> None:
        """
        Run the ACP server over stdio.

        Uses the new run_agent helper which handles stdio streams and message routing.
        """
        logger.info("Starting ACP server on stdio")
        # Startup messages are handled by fastagent.py to respect quiet mode and use correct stream

        try:
            # Use the new run_agent helper which handles:
            # - stdio stream setup
            # - AgentSideConnection creation
            # - Message loop
            # The connection is passed to us via on_connect callback
            await run_agent(self, use_unstable_protocol=True)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("ACP server shutting down")
            # Shutdown message is handled by fastagent.py to respect quiet mode

        except Exception as e:
            logger.error(f"ACP server error: {e}", name="acp_server_error", exc_info=True)
            raise

        finally:
            # Clean up sessions
            await self._cleanup_sessions()

    async def _send_available_commands_update(self, session_id: str) -> None:
        """
        Send available_commands_update notification for a session.

        This is called as a background task after NewSessionResponse is returned
        to ensure the client receives the session/new response before the session/update.
        """
        if not self._connection:
            return

        try:
            session_state = self._session_state.get(session_id)
            if not session_state or not session_state.slash_handler:
                return

            available_commands = session_state.slash_handler.get_available_commands()
            commands_update = AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=available_commands,
            )
            await self._connection.session_update(session_id=session_id, update=commands_update)

            logger.info(
                "Sent available_commands_update",
                name="acp_available_commands_sent",
                session_id=session_id,
                command_count=len(available_commands),
            )
        except Exception as e:
            logger.error(
                f"Error sending available_commands_update: {e}",
                name="acp_available_commands_error",
                exc_info=True,
            )

    async def _cleanup_sessions(self) -> None:
        """Clean up all sessions and dispose of agent instances."""
        logger.info(f"Cleaning up {len(self.sessions)} sessions")

        async with self._session_lock:
            # Clean up per-session state
            for session_id, state in list(self._session_state.items()):
                if state.terminal_runtime:
                    try:
                        logger.debug(
                            f"Terminal runtime for session {session_id} will be cleaned up"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error noting terminal cleanup for session {session_id}: {e}",
                            name="acp_terminal_cleanup_error",
                        )

                if state.filesystem_runtime:
                    try:
                        logger.debug(f"Filesystem runtime for session {session_id} cleaned up")
                    except Exception as e:
                        logger.error(
                            f"Error noting filesystem cleanup for session {session_id}: {e}",
                            name="acp_filesystem_cleanup_error",
                        )

                if state.permission_handler:
                    try:
                        await state.permission_handler.clear_session_cache()
                        logger.debug(f"Permission handler for session {session_id} cleaned up")
                    except Exception as e:
                        logger.error(
                            f"Error cleaning up permission handler for session {session_id}: {e}",
                            name="acp_permission_cleanup_error",
                        )

                if state.progress_manager:
                    try:
                        await state.progress_manager.cleanup_session_tools(session_id)
                        logger.debug(f"Progress manager for session {session_id} cleaned up")
                    except Exception as e:
                        logger.error(
                            f"Error cleaning up progress manager for session {session_id}: {e}",
                            name="acp_progress_cleanup_error",
                        )

                if state.acp_context:
                    try:
                        await state.acp_context.cleanup()
                        logger.debug(f"ACPContext for session {session_id} cleaned up")
                    except Exception as e:
                        logger.error(
                            f"Error cleaning up ACPContext for session {session_id}: {e}",
                            name="acp_context_cleanup_error",
                        )

            self._session_state.clear()
            self._session_tasks.clear()
            self._active_prompts.clear()
            self._prompt_locks.clear()

            # Dispose of non-shared instances
            if self._instance_scope in ["connection", "request"]:
                for session_id, instance in self.sessions.items():
                    if instance != self.primary_instance:
                        try:
                            await self._dispose_instance_task(instance)
                        except Exception as e:
                            logger.error(
                                f"Error disposing instance for session {session_id}: {e}",
                                name="acp_cleanup_error",
                            )

            # Dispose of primary instance
            if self.primary_instance:
                try:
                    await self._dispose_instance_task(self.primary_instance)
                except Exception as e:
                    logger.error(
                        f"Error disposing primary instance: {e}",
                        name="acp_cleanup_error",
                    )

            if self._stale_instances:
                for instance in list(self._stale_instances):
                    try:
                        await self._dispose_instance_task(instance)
                    except Exception as e:
                        logger.error(
                            f"Error disposing stale instance: {e}",
                            name="acp_cleanup_error",
                        )
                self._stale_instances.clear()

            self.sessions.clear()

        logger.info("ACP cleanup complete")
