"""
AgentACPServer - Exposes FastAgent agents via the Agent Client Protocol (ACP).

This implementation allows fast-agent to act as an ACP agent, enabling editors
and other clients to interact with fast-agent agents over stdio using the ACP protocol.
"""

import asyncio
import uuid
from importlib.metadata import version as get_version
from typing import Any, Awaitable, Callable

from acp import Agent as ACPAgent
from acp import (
    AgentSideConnection,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
)
from acp.helpers import session_notification, update_agent_message_text
from acp.schema import (
    AgentCapabilities,
    Implementation,
    PromptCapabilities,
    SessionMode,
    SessionModeState,
    StopReason,
)
from acp.stdio import stdio_streams

from fast_agent.acp.content_conversion import convert_acp_prompt_to_mcp_content_blocks
from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.acp.tool_progress import ACPToolProgressManager
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    TERMINAL_AVG_BYTES_PER_TOKEN,
    TERMINAL_OUTPUT_TOKEN_RATIO,
)
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import (
    apply_template_variables,
    enrich_with_environment_context,
)
from fast_agent.interfaces import StreamingAgentProtocol
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.mcp.helpers.content_helpers import is_text_content
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams
from fast_agent.workflow_telemetry import ACPPlanTelemetryProvider, ToolHandlerWorkflowTelemetry

logger = get_logger(__name__)

END_TURN: StopReason = "end_turn"
REFUSAL: StopReason = "refusal"


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

    mapping = {
        LlmStopReason.END_TURN.value: END_TURN,
        LlmStopReason.STOP_SEQUENCE.value: END_TURN,  # Normal completion
        LlmStopReason.MAX_TOKENS.value: "max_tokens",
        LlmStopReason.TOOL_USE.value: END_TURN,  # Tool use is normal completion in ACP
        LlmStopReason.PAUSE.value: END_TURN,  # Pause is treated as normal completion
        LlmStopReason.ERROR.value: REFUSAL,  # Errors are mapped to refusal
        LlmStopReason.TIMEOUT.value: REFUSAL,  # Timeouts are mapped to refusal
        LlmStopReason.SAFETY.value: REFUSAL,  # Safety triggers are mapped to refusal
        LlmStopReason.CANCELLED.value: "cancelled",  # User cancellation
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
        skills_directory_override: str | None = None,
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
        """
        super().__init__()

        self.primary_instance = primary_instance
        self._create_instance_task = create_instance
        self._dispose_instance_task = dispose_instance
        self._instance_scope = instance_scope
        self.server_name = server_name
        self._skills_directory_override = skills_directory_override
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

        # Track current agent per session for ACP mode support
        self._session_current_agent: dict[str, str] = {}

        # Terminal runtime tracking (for cleanup)
        self._session_terminal_runtimes: dict[str, ACPTerminalRuntime] = {}

        # Filesystem runtime tracking
        self._session_filesystem_runtimes: dict[str, ACPFilesystemRuntime] = {}

        # Slash command handlers for each session
        self._session_slash_handlers: dict[str, SlashCommandHandler] = {}

        # Late-binding prompt context by session (e.g., client-provided cwd)
        self._session_prompt_context: dict[str, dict[str, str]] = {}

        # Per-session resolved instructions (do not mutate shared Agent instances)
        self._session_resolved_instructions: dict[str, dict[str, str]] = {}

        # Connection reference (set during run_async)
        self._connection: AgentSideConnection | None = None

        # Client capabilities and info (set during initialize)
        self._client_supports_terminal: bool = False
        self._client_supports_fs_read: bool = False
        self._client_supports_fs_write: bool = False
        self._client_capabilities: dict | None = None
        self._client_info: dict | None = None
        self._protocol_version: str | None = None

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
        model_name = getattr(getattr(agent, "llm", None), "model_name", None)
        if not model_name:
            return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

        max_tokens = ModelDatabase.get_max_output_tokens(model_name)
        if not max_tokens:
            return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

        estimated_tokens = max(int(max_tokens * TERMINAL_OUTPUT_TOKEN_RATIO), 1)
        estimated_bytes = int(estimated_tokens * TERMINAL_AVG_BYTES_PER_TOKEN)
        return max(DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT, estimated_bytes)

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """
        Handle ACP initialization request.

        Negotiates protocol version and advertises capabilities.
        """
        try:
            # Store protocol version
            self._protocol_version = params.protocolVersion

            # Store client info
            if params.clientInfo:
                self._client_info = {
                    "name": getattr(params.clientInfo, "name", "unknown"),
                    "version": getattr(params.clientInfo, "version", "unknown"),
                }
                # Include title if available
                if hasattr(params.clientInfo, "title"):
                    self._client_info["title"] = params.clientInfo.title

            # Store client capabilities
            if params.clientCapabilities:
                self._client_supports_terminal = bool(
                    getattr(params.clientCapabilities, "terminal", False)
                )

                # Check for filesystem capabilities
                if hasattr(params.clientCapabilities, "fs"):
                    fs_caps = params.clientCapabilities.fs
                    if fs_caps:
                        self._client_supports_fs_read = bool(
                            getattr(fs_caps, "readTextFile", False)
                        )
                        self._client_supports_fs_write = bool(
                            getattr(fs_caps, "writeTextFile", False)
                        )

                # Convert capabilities to a dict for status reporting
                self._client_capabilities = {}
                if hasattr(params.clientCapabilities, "fs"):
                    fs_caps = params.clientCapabilities.fs
                    fs_capabilities = self._extract_fs_capabilities(fs_caps)
                    if fs_capabilities:
                        self._client_capabilities["fs"] = fs_capabilities

                if (
                    hasattr(params.clientCapabilities, "terminal")
                    and params.clientCapabilities.terminal
                ):
                    self._client_capabilities["terminal"] = True

                # Store _meta if present
                if hasattr(params.clientCapabilities, "_meta"):
                    meta = params.clientCapabilities._meta
                    if meta:
                        self._client_capabilities["_meta"] = (
                            dict(meta) if isinstance(meta, dict) else {}
                        )

            logger.info(
                "ACP initialize request",
                name="acp_initialize",
                client_protocol=params.protocolVersion,
                client_info=params.clientInfo,
                client_supports_terminal=self._client_supports_terminal,
                client_supports_fs_read=self._client_supports_fs_read,
                client_supports_fs_write=self._client_supports_fs_write,
            )

            # Build our capabilities
            agent_capabilities = AgentCapabilities(
                promptCapabilities=PromptCapabilities(
                    image=True,  # Support image content
                    embeddedContext=True,  # Support embedded resources
                    audio=False,  # Don't support audio (yet)
                ),
                # We don't support loadSession yet
                loadSession=False,
            )

            # Build agent info using Implementation type
            agent_info = Implementation(
                name=self.server_name,
                version=self.server_version,
            )

            response = InitializeResponse(
                protocolVersion=params.protocolVersion,  # Echo back the client's version
                agentCapabilities=agent_capabilities,
                agentInfo=agent_info,
                authMethods=[],  # No authentication for now
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
        self, instance: AgentInstance, session_id: str | None = None
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

        resolved_cache = self._session_resolved_instructions.get(session_id or "", {})

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

            # Create the SessionMode
            mode = SessionMode(
                id=agent_name,
                name=format_agent_name_as_title(agent_name),
                description=description,
            )
            available_modes.append(mode)

        # Current mode is the primary agent name
        current_mode_id = self.primary_agent_name or (
            list(instance.agents.keys())[0] if instance.agents else "default"
        )

        return SessionModeState(
            availableModes=available_modes,
            currentModeId=current_mode_id,
        )

    def _build_session_request_params(self, agent: Any, session_id: str) -> RequestParams | None:
        """
        Apply late-binding template variables to an agent's instruction for this session.
        """
        # Prefer cached resolved instructions to avoid reprocessing templates
        resolved_cache = self._session_resolved_instructions.get(session_id, {})
        resolved = resolved_cache.get(getattr(agent, "name", ""), None)
        if not resolved:
            context = self._session_prompt_context.get(session_id)
            if not context:
                return None
            template = getattr(agent, "instruction", None)
            if not template:
                return None
            resolved = apply_template_variables(template, context)
            if resolved == template:
                return None
        return RequestParams(systemPrompt=resolved)

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
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
            cwd=params.cwd,
            mcp_server_count=len(params.mcpServers),
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

            # Create tool progress manager for this session if connection is available
            tool_handler = None
            if self._connection:
                # Create a progress manager for this session
                tool_handler = ACPToolProgressManager(self._connection, session_id)
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
                    if hasattr(agent, "llm") and hasattr(agent.llm, "add_tool_stream_listener"):
                        try:
                            agent.llm.add_tool_stream_listener(tool_handler.handle_tool_stream_event)
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

                # If client supports terminals and we have shell runtime enabled,
                # inject ACP terminal runtime to replace local ShellRuntime
                if self._client_supports_terminal:
                    # Check if any agent has shell runtime enabled
                    for agent_name, agent in instance.agents.items():
                        if hasattr(agent, "_shell_runtime_enabled") and agent._shell_runtime_enabled:
                            # Create ACPTerminalRuntime for this session
                            default_limit = self._calculate_terminal_output_limit(agent)
                            terminal_runtime = ACPTerminalRuntime(
                                connection=self._connection,
                                session_id=session_id,
                                activation_reason="via ACP terminal support",
                                timeout_seconds=getattr(agent._shell_runtime, "timeout_seconds", 90),
                                tool_handler=tool_handler,
                                default_output_byte_limit=default_limit,
                            )

                            # Inject into agent
                            if hasattr(agent, "set_external_runtime"):
                                agent.set_external_runtime(terminal_runtime)
                                self._session_terminal_runtimes[session_id] = terminal_runtime

                                logger.info(
                                    "ACP terminal runtime injected",
                                    name="acp_terminal_injected",
                                    session_id=session_id,
                                    agent_name=agent_name,
                                    default_output_limit=default_limit,
                                )

                # If client supports filesystem operations, inject ACP filesystem runtime
                if self._client_supports_fs_read or self._client_supports_fs_write:
                    # Create ACPFilesystemRuntime for this session with appropriate capabilities
                    filesystem_runtime = ACPFilesystemRuntime(
                        connection=self._connection,
                        session_id=session_id,
                        activation_reason="via ACP filesystem support",
                        enable_read=self._client_supports_fs_read,
                        enable_write=self._client_supports_fs_write,
                        tool_handler=tool_handler,
                    )
                    self._session_filesystem_runtimes[session_id] = filesystem_runtime

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
            session_context, params.cwd, self._client_info, self._skills_directory_override
        )
        self._session_prompt_context[session_id] = session_context

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
            self._session_resolved_instructions[session_id] = resolved_for_session

        # Create slash command handler for this session
        resolved_prompts = self._session_resolved_instructions.get(session_id, {})

        slash_handler = SlashCommandHandler(
            session_id,
            instance,
            self.primary_agent_name,
            client_info=self._client_info,
            client_capabilities=self._client_capabilities,
            protocol_version=self._protocol_version,
            session_instructions=resolved_prompts,
        )
        self._session_slash_handlers[session_id] = slash_handler

        logger.info(
            "ACP new session created",
            name="acp_new_session_created",
            session_id=session_id,
            total_sessions=len(self.sessions),
            terminal_enabled=session_id in self._session_terminal_runtimes,
            filesystem_enabled=session_id in self._session_filesystem_runtimes,
        )

        # Schedule available_commands_update notification to be sent after response is returned
        # This ensures the client receives session/new response before the session/update notification
        if self._connection:
            asyncio.create_task(self._send_available_commands_update(session_id, slash_handler))

        # Build session modes from the instance's agents
        session_modes = self._build_session_modes(instance, session_id)

        # Initialize the current agent for this session
        self._session_current_agent[session_id] = session_modes.currentModeId

        logger.info(
            "Session modes initialized",
            name="acp_session_modes_init",
            session_id=session_id,
            current_mode=session_modes.currentModeId,
            mode_count=len(session_modes.availableModes),
        )

        return NewSessionResponse(
            sessionId=session_id,
            modes=session_modes,
        )

    async def setSessionMode(self, params: SetSessionModeRequest) -> SetSessionModeResponse:
        """
        Handle session mode change request.

        Updates the current agent for the session to route future prompts
        to the selected mode (agent).

        Args:
            params: SetSessionModeRequest with sessionId and modeId

        Returns:
            SetSessionModeResponse (empty response on success)

        Raises:
            ValueError: If session not found or mode ID is invalid
        """
        session_id = params.sessionId
        mode_id = params.modeId

        logger.info(
            "ACP set session mode request",
            name="acp_set_session_mode",
            session_id=session_id,
            mode_id=mode_id,
        )

        # Get the agent instance for this session
        async with self._session_lock:
            instance = self.sessions.get(session_id)

        if not instance:
            logger.error(
                "Session not found for setSessionMode",
                name="acp_set_mode_error",
                session_id=session_id,
            )
            raise ValueError(f"Session not found: {session_id}")

        # Validate that the mode_id exists in the instance's agents
        if mode_id not in instance.agents:
            logger.error(
                "Invalid mode ID for setSessionMode",
                name="acp_set_mode_invalid",
                session_id=session_id,
                mode_id=mode_id,
                available_modes=list(instance.agents.keys()),
            )
            raise ValueError(
                f"Invalid mode ID '{mode_id}'. Available modes: {list(instance.agents.keys())}"
            )

        # Update the session's current agent
        self._session_current_agent[session_id] = mode_id

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

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """
        Handle prompt request.

        Extracts the prompt text, sends it to the fast-agent agent, and sends the response
        back to the client via sessionUpdate notifications.

        Per ACP protocol, only one prompt can be active per session at a time. If a prompt
        is already in progress for this session, this will immediately return a refusal.
        """
        session_id = params.sessionId

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
                return PromptResponse(stopReason=REFUSAL)

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
                return PromptResponse(stopReason=REFUSAL)

            # Convert ACP content blocks to MCP format
            mcp_content_blocks = convert_acp_prompt_to_mcp_content_blocks(params.prompt)

            # Create a PromptMessageExtended with the converted content
            prompt_message = PromptMessageExtended(
                role="user",
                content=mcp_content_blocks,
            )

            # Get current agent for this session (defaults to primary agent if not set)
            current_agent_name = self._session_current_agent.get(
                session_id, self.primary_agent_name
            )

            # Check if this is a slash command
            # Only process slash commands if the prompt is a single text block
            # This ensures resources, images, and multi-part prompts are never treated as commands
            slash_handler = self._session_slash_handlers.get(session_id)
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
                slash_handler.set_current_agent(current_agent_name)

                # Parse and execute the command
                command_name, arguments = slash_handler.parse_command(prompt_text)
                response_text = await slash_handler.execute_command(command_name, arguments)

                # Send the response via sessionUpdate
                if self._connection and response_text:
                    try:
                        message_chunk = update_agent_message_text(response_text)
                        notification = session_notification(session_id, message_chunk)
                        await self._connection.sessionUpdate(notification)
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
                return PromptResponse(stopReason=END_TURN)

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
            try:
                if current_agent_name:
                    agent = instance.agents[current_agent_name]

                    # Set up streaming if connection is available and agent supports it
                    stream_listener = None
                    remove_listener: Callable[[], None] | None = None
                    streaming_tasks: list[asyncio.Task] = []
                    if self._connection and isinstance(agent, StreamingAgentProtocol):
                        update_lock = asyncio.Lock()

                        async def send_stream_update(chunk: str):
                            """Send sessionUpdate with accumulated text so far."""
                            if not chunk:
                                return
                            try:
                                async with update_lock:
                                    message_chunk = update_agent_message_text(chunk)
                                    notification = session_notification(session_id, message_chunk)
                                    await self._connection.sessionUpdate(notification)
                            except Exception as e:
                                logger.error(
                                    f"Error sending stream update: {e}",
                                    name="acp_stream_error",
                                    exc_info=True,
                                )

                        def on_stream_chunk(chunk: str):
                            """
                            Sync callback from fast-agent streaming.
                            Sends each chunk as it arrives to the ACP client.
                            """
                            logger.debug(
                                f"Stream chunk received: {len(chunk)} chars",
                                name="acp_stream_chunk",
                                session_id=session_id,
                                chunk_length=len(chunk),
                            )

                            # Send update asynchronously (don't await in sync callback)
                            # Track task to ensure all chunks complete before returning PromptResponse
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
                            agent, session_id
                        )
                        result = await agent.generate(
                            prompt_message,
                            request_params=session_request_params,
                        )
                        response_text = result.last_text() or "No content generated"

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
                                notification = session_notification(session_id, message_chunk)
                                await self._connection.sessionUpdate(notification)
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
                                logger.exception("Failed to remove ACP stream listener after error")
                        # Re-raise the original error
                        raise send_error

                    finally:
                        # Clean up stream listener (if not already cleaned up in except)
                        if stream_listener and remove_listener:
                            try:
                                remove_listener()
                            except Exception:
                                logger.exception("Failed to remove ACP stream listener")
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
                stopReason=acp_stop_reason,
            )
        except asyncio.CancelledError:
            # Task was cancelled - return appropriate response
            logger.info(
                "Prompt cancelled by user",
                name="acp_prompt_cancelled",
                session_id=session_id,
            )
            return PromptResponse(stopReason="cancelled")
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

    async def cancel(self, params: CancelNotification) -> None:
        """
        Handle session/cancel notification from the client.

        This cancels any in-progress prompt for the specified session.
        Per ACP protocol, we should stop all LLM requests and tool invocations
        as soon as possible.

        Uses asyncio.Task.cancel() for proper async cancellation, which raises
        asyncio.CancelledError in the running task.
        """
        session_id = params.sessionId

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

    async def run_async(self) -> None:
        """
        Run the ACP server over stdio.

        This creates the stdio streams and sets up the ACP connection.
        """
        logger.info("Starting ACP server on stdio")
        # Startup messages are handled by fastagent.py to respect quiet mode and use correct stream

        try:
            # Get stdio streams
            reader, writer = await stdio_streams()

            # Create the ACP connection
            # Note: AgentSideConnection expects (writer, reader) order
            # - input_stream (writer) = where agent writes TO client
            # - output_stream (reader) = where agent reads FROM client
            connection = AgentSideConnection(
                lambda conn: self,
                writer,  # input_stream = StreamWriter for agent output
                reader,  # output_stream = StreamReader for agent input
            )

            # Store the connection reference so we can send sessionUpdate notifications
            self._connection = connection

            logger.info("ACP connection established, waiting for messages")

            # Keep the connection alive
            # The connection will handle incoming messages automatically
            # We just need to wait until it's closed or interrupted
            try:
                # Wait indefinitely - the connection will process messages in the background
                # The Connection class automatically starts a receive loop on creation
                shutdown_event = asyncio.Event()
                await shutdown_event.wait()
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("ACP server shutting down")
                # Shutdown message is handled by fastagent.py to respect quiet mode
            finally:
                # Close the connection properly
                await connection._conn.close()

        except Exception as e:
            logger.error(f"ACP server error: {e}", name="acp_server_error", exc_info=True)
            raise

        finally:
            # Clean up sessions
            await self._cleanup_sessions()

    async def _send_available_commands_update(
        self, session_id: str, slash_handler: SlashCommandHandler
    ) -> None:
        """
        Send available_commands_update notification for a session.

        This is called as a background task after NewSessionResponse is returned
        to ensure the client receives the session/new response before the session/update.
        """
        if not self._connection:
            return

        try:
            available_commands = slash_handler.get_available_commands()
            commands_update = {
                "sessionUpdate": "available_commands_update",
                "availableCommands": available_commands,
            }
            notification = session_notification(session_id, commands_update)
            await self._connection.sessionUpdate(notification)

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
            # Clean up terminal runtimes (must release as per ACP spec)
            for session_id, terminal_runtime in list(self._session_terminal_runtimes.items()):
                try:
                    # Terminal runtime cleanup happens automatically via _release_terminal
                    # in each execute() call, but we log here for completeness
                    logger.debug(f"Terminal runtime for session {session_id} will be cleaned up")
                except Exception as e:
                    logger.error(
                        f"Error noting terminal cleanup for session {session_id}: {e}",
                        name="acp_terminal_cleanup_error",
                    )

            self._session_terminal_runtimes.clear()

            # Clean up filesystem runtimes
            for session_id, filesystem_runtime in list(self._session_filesystem_runtimes.items()):
                try:
                    logger.debug(f"Filesystem runtime for session {session_id} cleaned up")
                except Exception as e:
                    logger.error(
                        f"Error noting filesystem cleanup for session {session_id}: {e}",
                        name="acp_filesystem_cleanup_error",
                    )

            self._session_filesystem_runtimes.clear()

            # Clean up slash command handlers
            self._session_slash_handlers.clear()

            # Clean up session current agent mapping
            self._session_current_agent.clear()

            # Clear tasks
            self._session_tasks.clear()

            # Clear stored prompt contexts
            self._session_prompt_context.clear()
            self._session_resolved_instructions.clear()

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
