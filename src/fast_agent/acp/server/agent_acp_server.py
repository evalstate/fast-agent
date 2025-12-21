"""
AgentACPServer - Exposes FastAgent agents via the Agent Client Protocol (ACP).

This implementation allows fast-agent to act as an ACP agent, enabling editors
and other clients to interact with fast-agent agents over stdio using the ACP protocol.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from acp import Agent as ACPAgent
from acp import (
    Client,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    SetSessionModeResponse,
    run_agent,
)
from acp import (
    Client as ACPClient,
)
from acp.helpers import (
    ContentBlock as ACPContentBlock,
)
from acp.helpers import (
    update_agent_message_text,
    update_agent_thought_text,
)
from acp.schema import (
    AgentCapabilities,
    AvailableCommandsUpdate,
    ClientCapabilities,
    HttpMcpServer,
    Implementation,
    McpServerStdio,
    PromptCapabilities,
    SessionMode,
    SessionModeState,
    SseMcpServer,
    StopReason,
)

from fast_agent.acp.acp_context import ACPContext, ClientInfo
from fast_agent.acp.acp_context import ClientCapabilities as FAClientCapabilities
from fast_agent.acp.content_conversion import convert_acp_prompt_to_mcp_content_blocks
from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.permission_store import PermissionStore
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
from fast_agent.acp.tool_progress import ACPToolProgressManager
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
    TERMINAL_AVG_BYTES_PER_TOKEN,
    TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO,
    TERMINAL_OUTPUT_TOKEN_RATIO,
)
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import (
    apply_template_variables,
    enrich_with_environment_context,
)
from fast_agent.interfaces import ACPAwareProtocol, StreamingAgentProtocol
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.usage_tracking import last_turn_usage
from fast_agent.mcp.helpers.content_helpers import is_text_content
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
        """
        super().__init__()

        self.primary_instance = primary_instance
        self._create_instance_task = create_instance
        self._dispose_instance_task = dispose_instance
        self._instance_scope = instance_scope
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

        # Track sessions with active prompts to prevent overlapping requests (per ACP protocol)
        self._active_prompts: set[str] = set()

        # Track asyncio tasks per session for proper task-based cancellation
        self._session_tasks: dict[str, asyncio.Task] = {}

        # Aggregated per-session state
        self._session_state: dict[str, ACPSessionState] = {}

        # Connection reference (set during run_async)
        self._connection: Client | None = None

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

        max_tokens = ModelDatabase.get_max_output_tokens(model_name)
        if not max_tokens:
            return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

        terminal_token_budget = max(int(max_tokens * TERMINAL_OUTPUT_TOKEN_RATIO), 1)
        terminal_token_budget = max(
            int(terminal_token_budget * (1 - TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO)), 1
        )
        terminal_byte_budget = int(terminal_token_budget * TERMINAL_AVG_BYTES_PER_TOKEN)

        terminal_byte_budget = min(terminal_byte_budget, MAX_TERMINAL_OUTPUT_BYTE_LIMIT)
        return max(DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT, terminal_byte_budget)

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
                            getattr(fs_caps, "readTextFile", False)
                        )
                        self._client_supports_fs_write = bool(
                            getattr(fs_caps, "writeTextFile", False)
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
                _meta=self._client_capabilities.get("_meta", {}) if self._client_capabilities else {},
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
                # We don't support loadSession yet
                load_session=False,
            )

            # Build agent info using Implementation type
            agent_info = Implementation(
                name=self.server_name,
                version=self.server_version,
            )

            response = InitializeResponse(
                protocol_version=protocol_version,  # Echo back the client's version
                agent_capabilities=agent_capabilities,
                agent_info=agent_info,
                auth_methods=[],  # No authentication for now
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

        for attr in ("readTextFile", "writeTextFile", "readFile", "writeFile"):
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

        # Create a SessionMode for each agent
        for agent_name, agent in instance.agents.items():
            # Get instruction from agent's config
            instruction = ""
            resolved_instruction = resolved_cache.get(agent_name)
            if resolved_instruction:
                instruction = resolved_instruction
            elif hasattr(agent, "_config") and hasattr(agent._config, "instruction"):
                instruction = agent._config.instruction
            elif hasattr(agent, "instruction"):
                instruction = agent.instruction

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

    def _build_session_request_params(
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

        # Prefer cached resolved instructions to avoid reprocessing templates
        resolved_cache = session_state.resolved_instructions if session_state else {}
        resolved = resolved_cache.get(getattr(agent, "name", ""), None)
        if not resolved:
            context = session_state.prompt_context if session_state else None
            if not context:
                return None
            template = getattr(agent, "instruction", None)
            if not template:
                return None
            resolved = apply_template_variables(template, context)
            if resolved == template:
                return None
        return RequestParams(systemPrompt=resolved)

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

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: Any,
    ) -> NewSessionResponse:
        """
        Handle new session request.

        Creates a new session and maps it to an AgentInstance based on instance_scope.
        """
        session_id = str(uuid.uuid4())

        logger.info(
            "ACP new session request",
            name="acp_new_session",
            session_id=session_id,
            instance_scope=self._instance_scope,
            cwd=cwd,
            mcp_server_count=len(mcp_servers),
        )

        async with self._session_lock:
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

            # Create tool progress manager for this session if connection is available
            tool_handler = None
            if self._connection:
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
                    if hasattr(agent, "_aggregator"):
                        aggregator = agent._aggregator
                        aggregator._tool_handler = tool_handler

                        logger.info(
                            "ACP tool handler registered",
                            name="acp_tool_handler_registered",
                            session_id=session_id,
                            agent_name=agent_name,
                        )

                    if hasattr(agent, "workflow_telemetry"):
                        agent.workflow_telemetry = workflow_telemetry

                    # Set up plan telemetry for agents that support it (e.g., IterativePlanner)
                    if hasattr(agent, "plan_telemetry"):
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
                if self._permissions_enabled:
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
                    for agent_name, agent in instance.agents.items():
                        if hasattr(agent, "_aggregator"):
                            aggregator = agent._aggregator
                            aggregator._permission_handler = permission_handler

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
                if self._client_supports_terminal:
                    # Check if any agent has shell runtime enabled
                    for agent_name, agent in instance.agents.items():
                        if (
                            hasattr(agent, "_shell_runtime_enabled")
                            and agent._shell_runtime_enabled
                        ):
                            # Create ACPTerminalRuntime for this session
                            default_limit = self._calculate_terminal_output_limit(agent)
                            # Get permission handler if enabled for this session
                            perm_handler = session_state.permission_handler
                            terminal_runtime = ACPTerminalRuntime(
                                connection=self._connection,
                                session_id=session_id,
                                activation_reason="via ACP terminal support",
                                timeout_seconds=getattr(
                                    agent._shell_runtime, "timeout_seconds", 90
                                ),
                                tool_handler=tool_handler,
                                default_output_byte_limit=default_limit,
                                permission_handler=perm_handler,
                            )

                            # Inject into agent
                            if hasattr(agent, "set_external_runtime"):
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
                if self._client_supports_fs_read or self._client_supports_fs_write:
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
                        if hasattr(agent, "set_filesystem_runtime"):
                            agent.set_filesystem_runtime(filesystem_runtime)
                            logger.info(
                                "ACP filesystem runtime injected",
                                name="acp_filesystem_injected",
                                session_id=session_id,
                                agent_name=agent_name,
                                read_enabled=self._client_supports_fs_read,
                                write_enabled=self._client_supports_fs_write,
                            )

        # Track per-session template variables (used for late instruction binding)
        session_context: dict[str, str] = {}
        enrich_with_environment_context(
            session_context, cwd, self._client_info, self._skills_directory_override
        )
        session_state.prompt_context = session_context

        # Cache resolved instructions for this session (without mutating shared instances)
        resolved_for_session: dict[str, str] = {}
        for agent_name, agent in instance.agents.items():
            template = getattr(agent, "instruction", None)
            if not template:
                continue
            resolved = apply_template_variables(template, session_context)
            if resolved:
                resolved_for_session[agent_name] = resolved
        if resolved_for_session:
            session_state.resolved_instructions = resolved_for_session

        # Set session context on agents that have InstructionBuilder
        # This ensures {{env}}, {{workspaceRoot}}, etc. are available when rebuilding
        for agent_name, agent in instance.agents.items():
            if hasattr(agent, "set_instruction_context"):
                try:
                    agent.set_instruction_context(session_context)
                except Exception as e:
                    logger.warning(
                        f"Failed to set instruction context on agent {agent_name}: {e}"
                    )

        # Create slash command handler for this session
        resolved_prompts = session_state.resolved_instructions

        slash_handler = SlashCommandHandler(
            session_id,
            instance,
            self.primary_agent_name or "default",
            client_info=self._client_info,
            client_capabilities=self._client_capabilities,
            protocol_version=self._protocol_version,
            session_instructions=resolved_prompts,
        )
        session_state.slash_handler = slash_handler

        # Create ACPContext for this session - centralizes all ACP state
        if self._connection:
            acp_context = ACPContext(
                connection=self._connection,
                session_id=session_id,
                client_capabilities=self._parsed_client_capabilities,
                client_info=self._parsed_client_info,
                protocol_version=self._protocol_version,
            )

            # Store references to runtimes and handlers in ACPContext
            if session_state.terminal_runtime:
                acp_context.set_terminal_runtime(session_state.terminal_runtime)
            if session_state.filesystem_runtime:
                acp_context.set_filesystem_runtime(session_state.filesystem_runtime)
            if session_state.permission_handler:
                acp_context.set_permission_handler(session_state.permission_handler)
            if session_state.progress_manager:
                acp_context.set_progress_manager(session_state.progress_manager)

            acp_context.set_slash_handler(slash_handler)

            # Share the resolved instructions cache so agents can invalidate it
            acp_context.set_resolved_instructions(session_state.resolved_instructions)

            # Store ACPContext
            session_state.acp_context = acp_context

            # Set ACPContext on each agent's Context object (if they have one)
            for agent_name, agent in instance.agents.items():
                if hasattr(agent, "_context") and agent._context is not None:
                    agent._context.acp = acp_context
                    logger.debug(
                        "ACPContext set on agent",
                        name="acp_context_set",
                        session_id=session_id,
                        agent_name=agent_name,
                    )
                elif hasattr(agent, "context"):
                    # Try via context property
                    try:
                        agent.context.acp = acp_context
                        logger.debug(
                            "ACPContext set on agent via context property",
                            name="acp_context_set",
                            session_id=session_id,
                            agent_name=agent_name,
                        )
                    except Exception:
                        # Agent may not have a context available
                        pass

            logger.info(
                "ACPContext created for session",
                name="acp_context_created",
                session_id=session_id,
                has_terminal=acp_context.terminal_runtime is not None,
                has_filesystem=acp_context.filesystem_runtime is not None,
                has_permissions=acp_context.permission_handler is not None,
            )
            session_state.acp_context = acp_context

        logger.info(
            "ACP new session created",
            name="acp_new_session_created",
            session_id=session_id,
            total_sessions=len(self.sessions),
            terminal_enabled=session_state.terminal_runtime is not None,
            filesystem_enabled=session_state.filesystem_runtime is not None,
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

        return NewSessionResponse(
            session_id=session_id,
            modes=session_modes,
        )

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

        # Check for overlapping prompt requests (per ACP protocol requirement)
        async with self._session_lock:
            if session_id in self._active_prompts:
                logger.warning(
                    "Overlapping prompt request detected - refusing",
                    name="acp_prompt_overlap",
                    session_id=session_id,
                )
                # Return immediate refusal - ACP protocol requires sequential prompts per session
                return PromptResponse(stop_reason=REFUSAL)

            # Mark this session as having an active prompt
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

            # Convert ACP content blocks to MCP format
            mcp_content_blocks = convert_acp_prompt_to_mcp_content_blocks(prompt)

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
                        session_request_params = self._build_session_request_params(
                            agent, session_state
                        )
                        turn_start_index = None
                        if getattr(agent, "usage_accumulator", None) is not None:
                            turn_start_index = len(agent.usage_accumulator.turns)
                        result = await agent.generate(
                            prompt_message,
                            request_params=session_request_params,
                        )
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
                _meta=status_line_meta,
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
            await run_agent(self)
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
                        logger.debug(f"Terminal runtime for session {session_id} will be cleaned up")
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

            self.sessions.clear()

        logger.info("ACP cleanup complete")
