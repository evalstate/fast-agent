"""
AgentACPServer - Exposes FastAgent agents via the Agent Client Protocol (ACP).

This implementation allows fast-agent to act as an ACP agent, enabling editors
and other clients to interact with fast-agent agents over stdio using the ACP protocol.
"""

import asyncio
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence, cast

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
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    AuthMethodAgent,
    AvailableCommandsUpdate,
    EnvVarAuthMethod,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    PromptCapabilities,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionListCapabilities,
    SessionModeState,
    SessionResumeCapabilities,
    SseMcpServer,
    TerminalAuthMethod,
)
from acp.schema import (
    ClientCapabilities as ACPClientCapabilities,
)
from acp.schema import (
    FileSystemCapabilities as ACPFileSystemCapabilities,
)

from fast_agent.acp.acp_context import ClientCapabilities as FAClientCapabilities
from fast_agent.acp.acp_context import ClientInfo
from fast_agent.acp.server.models import ACPSessionState
from fast_agent.acp.server.prompt_flow import ACPPromptFlow, PromptFlowHost
from fast_agent.acp.server.session_runtime import ACPServerSessionRuntime, SessionRuntimeHost
from fast_agent.acp.server.session_store import ACPServerSessionStore, SessionStoreHost
from fast_agent.acp.server.slash_runtime import ACPServerSlashRuntime, SlashRuntimeHost
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.config import MCPServerSettings
from fast_agent.constants import DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
from fast_agent.core.default_agent import agent_is_default, resolve_default_agent_name
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import AgentProtocol, LlmCapableProtocol
from fast_agent.llm.terminal_output_limits import (
    calculate_terminal_output_limit_for_model,
    calculate_terminal_output_limit_for_resolved_model,
)
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
from fast_agent.session import get_session_manager
from fast_agent.types import RequestParams
from fast_agent.ui.interactive_diagnostics import write_interactive_trace

logger = get_logger(__name__)

ACP_AUTH_METHOD_ID = "fast-agent-ai-secrets"
ACP_AUTH_DOCS_URL = "https://fast-agent.ai/ref/config_file/"
ACP_AUTH_CONFIG_FILE = "fastagent.secrets.yaml"
ACP_AUTH_RECOMMENDED_COMMANDS: tuple[str, ...] = (
    "fast-agent check",
    "fast-agent model doctor",
    "fast-agent model setup",
)


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
        bootstrap_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        server_name: str = "fast-agent-acp",
        server_version: str | None = None,
        skills_directory_override: Sequence[str | Path] | str | Path | None = None,
        permissions_enabled: bool = True,
        load_card_callback: Callable[[str, str | None], Awaitable[tuple[list[str], list[str]]]]
        | None = None,
        attach_agent_tools_callback: Callable[[str, Sequence[str]], Awaitable[list[str]]]
        | None = None,
        detach_agent_tools_callback: Callable[[str, Sequence[str]], Awaitable[list[str]]]
        | None = None,
        dump_agent_card_callback: Callable[[str], Awaitable[str]] | None = None,
        reload_callback: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """
        Initialize the ACP server.

        Args:
            bootstrap_instance: Initial agent instance used for capabilities and cleanup
            create_instance: Factory function to create new agent instances
            dispose_instance: Function to dispose of agent instances
            server_name: Name of the server for capability advertisement
            server_version: Version of the server (defaults to fast-agent version)
            skills_directory_override: Optional skills directory override (relative to session cwd)
            permissions_enabled: Whether to request tool permissions from client (default: True)
            load_card_callback: Optional callback to load AgentCards at runtime
            attach_agent_tools_callback: Optional callback to attach agent tools at runtime
            detach_agent_tools_callback: Optional callback to detach agent tools at runtime
            dump_agent_card_callback: Optional callback to dump AgentCards at runtime
            reload_callback: Optional callback to reload AgentCards
        """
        super().__init__()

        self._bootstrap_instance = bootstrap_instance
        self._create_instance_task = create_instance
        self._dispose_instance_task = dispose_instance
        self._load_card_callback = load_card_callback
        self._attach_agent_tools_callback = attach_agent_tools_callback
        self._detach_agent_tools_callback = detach_agent_tools_callback
        self._dump_agent_card_callback = dump_agent_card_callback
        self._reload_callback = reload_callback
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
        self.primary_agent_name = self._select_primary_agent(bootstrap_instance)
        self._session_runtime = ACPServerSessionRuntime(cast("SessionRuntimeHost", self))
        self._session_store = ACPServerSessionStore(cast("SessionStoreHost", self))
        self._slash_runtime = ACPServerSlashRuntime(cast("SlashRuntimeHost", self))
        self._prompt_flow = ACPPromptFlow(cast("PromptFlowHost", self))

        logger.info(
            "AgentACPServer initialized",
            name="acp_server_initialized",
            agent_count=len(bootstrap_instance.agents),
            primary_agent=self.primary_agent_name,
        )

    def _calculate_terminal_output_limit(self, agent: Any) -> int:
        """
        Determine a default terminal output byte limit based on the agent's model.

        Args:
            agent: Agent instance that may expose an llm with model metadata.
        """
        # Some workflow agents (e.g., chain/parallel) don't attach an LLM directly.
        llm = agent.llm if isinstance(agent, LlmCapableProtocol) else None
        resolved_model = llm.resolved_model if llm else None
        if resolved_model is not None:
            return calculate_terminal_output_limit_for_resolved_model(resolved_model)
        model_name = llm.model_name if llm else None
        return self._calculate_terminal_output_limit_for_model(model_name)

    @staticmethod
    def _calculate_terminal_output_limit_for_model(model_name: str | None) -> int:
        if not model_name:
            return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

        return calculate_terminal_output_limit_for_model(model_name)

    def _build_auth_meta(self) -> dict[str, Any]:
        """Return static setup guidance shared by initialize/authenticate/auth errors."""
        return {
            "configFile": ACP_AUTH_CONFIG_FILE,
            "docsUrl": ACP_AUTH_DOCS_URL,
            "recommendedCommands": list(ACP_AUTH_RECOMMENDED_COMMANDS),
            "description": (
                f"Configure provider keys in {ACP_AUTH_CONFIG_FILE} or environment variables. "
                "For interactive setup, run `fast-agent model setup` in a terminal."
            ),
        }

    def _build_auth_required_data(
        self,
        error: ProviderKeyError,
        *,
        agent: AgentProtocol | object | None = None,
    ) -> dict[str, Any]:
        """Translate provider auth failures into ACP AUTH_REQUIRED payload data."""
        data: dict[str, Any] = {
            "methodId": ACP_AUTH_METHOD_ID,
            "message": error.message,
            "details": error.details,
            **self._build_auth_meta(),
        }

        llm = agent.llm if isinstance(agent, LlmCapableProtocol) else None
        provider = llm.provider if llm else None
        provider_name = provider.value if provider else None
        provider_display_name = provider.display_name if provider else None
        if isinstance(provider_name, str) and provider_name:
            from fast_agent.llm.provider_key_manager import ProviderKeyManager

            env_var = ProviderKeyManager.get_env_key_name(provider_name)
            if env_var:
                data["envVars"] = [env_var]
            if isinstance(provider_display_name, str) and provider_display_name:
                data["provider"] = provider_display_name
                if not data["details"] and env_var:
                    data["details"] = (
                        f"Add the {provider_display_name} credentials to "
                        f"{ACP_AUTH_CONFIG_FILE} or set {env_var}."
                    )

        return data

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ACPClientCapabilities | None = None,
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
                    "name": client_info.name,
                    "version": client_info.version,
                }
                if client_info.title:
                    self._client_info["title"] = client_info.title

            # Store client capabilities
            if client_capabilities:
                fs_caps = client_capabilities.fs
                self._client_supports_terminal = bool(client_capabilities.terminal)
                self._client_supports_fs_read = bool(fs_caps.read_text_file) if fs_caps else False
                self._client_supports_fs_write = bool(fs_caps.write_text_file) if fs_caps else False

                # Convert capabilities to a dict for status reporting
                self._client_capabilities = {}
                fs_capabilities = self._extract_fs_capabilities(fs_caps)
                if fs_capabilities:
                    self._client_capabilities["fs"] = fs_capabilities

                if client_capabilities.terminal:
                    self._client_capabilities["terminal"] = True

                # Store _meta if present
                meta = client_capabilities.field_meta
                if meta:
                    self._client_capabilities["_meta"] = (
                        dict(meta) if isinstance(meta, dict) else {}
                    )

            # Parse client capabilities and info for ACPContext
            self._parsed_client_capabilities = FAClientCapabilities.from_acp_capabilities(
                client_capabilities
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
            # In ACP 0.9.x this uses the explicit agent auth schema, but we still
            # keep it to the minimal id/name/description shape.
            auth_methods: list[EnvVarAuthMethod | TerminalAuthMethod | AuthMethodAgent] = [
                AuthMethodAgent(
                    id=ACP_AUTH_METHOD_ID,
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
        if method_id != ACP_AUTH_METHOD_ID:
            raise RequestError.invalid_params(
                {
                    "methodId": method_id,
                    "supported": [ACP_AUTH_METHOD_ID],
                }
            )

        return AuthenticateResponse(field_meta=self._build_auth_meta())

    def _extract_fs_capabilities(
        self,
        fs_caps: ACPFileSystemCapabilities | dict[str, bool | None] | None,
    ) -> dict[str, bool]:
        """Normalize filesystem capabilities for status reporting."""
        normalized: dict[str, bool] = {}
        if not fs_caps:
            return normalized

        if isinstance(fs_caps, dict):
            for key, value in fs_caps.items():
                if value is not None:
                    normalized[key] = bool(value)
            return normalized

        if fs_caps.read_text_file is not None:
            normalized["read_text_file"] = bool(fs_caps.read_text_file)
        if fs_caps.write_text_file is not None:
            normalized["write_text_file"] = bool(fs_caps.write_text_file)

        return normalized

    def _build_session_modes(
        self, instance: AgentInstance, session_state: ACPSessionState | None = None
    ) -> SessionModeState:
        return self._session_runtime.build_session_modes(instance, session_state)

    async def _build_session_request_params(
        self, agent: AgentProtocol, session_state: ACPSessionState | None
    ) -> RequestParams | None:
        return await self._session_runtime.build_session_request_params(agent, session_state)

    async def _resolve_instruction_for_session(
        self,
        agent: AgentProtocol,
        context: dict[str, str],
    ) -> str | None:
        return await self._session_runtime.resolve_instruction_for_session(agent, context)

    def _resolve_request_cwd(
        self,
        *,
        cwd: str | None,
        request_name: str,
        required: bool,
    ) -> str | None:
        if cwd is None:
            if not required:
                return None
            raise RequestError.invalid_params(
                {
                    "cwd": cwd,
                    "request": request_name,
                    "reason": "cwd is required and must be an absolute path",
                }
            )

        path = Path(cwd).expanduser()
        if not path.is_absolute():
            raise RequestError.invalid_params(
                {
                    "cwd": cwd,
                    "request": request_name,
                    "reason": "cwd must be an absolute path",
                }
            )
        return str(path.resolve())

    def _get_session_manager(self, *, cwd: Path | None = None) -> Any:
        return get_session_manager(cwd=cwd)

    @staticmethod
    def _encode_session_list_cursor(offset: int) -> str:
        return ACPServerSessionStore._encode_session_list_cursor(offset)

    @staticmethod
    def _decode_session_list_cursor(cursor: str) -> int:
        return ACPServerSessionStore._decode_session_list_cursor(cursor)

    async def _replace_instance_for_session(
        self,
        session_state: ACPSessionState,
        *,
        dispose_error_name: str,
        await_refresh_session_state: bool,
    ) -> AgentInstance:
        return await self._session_runtime.replace_instance_for_session(
            session_state,
            dispose_error_name=dispose_error_name,
            await_refresh_session_state=await_refresh_session_state,
        )

    async def _refresh_session_state(
        self, session_state: ACPSessionState, instance: AgentInstance
    ) -> None:
        await self._session_runtime.refresh_session_state(session_state, instance)

    async def _attach_mcp_server_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        return await self._session_runtime.attach_session_mcp_server(
            session_state,
            agent_name=agent_name,
            server_name=server_name,
            server_config=server_config,
            options=options,
        )

    async def _detach_mcp_server_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        return await self._session_runtime.detach_session_mcp_server(
            session_state,
            agent_name=agent_name,
            server_name=server_name,
        )

    async def _list_attached_mcp_servers_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
    ) -> list[str]:
        return await self._session_runtime.list_attached_mcp_servers(
            session_state,
            agent_name=agent_name,
        )

    async def _list_configured_detached_mcp_servers_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
    ) -> list[str]:
        return await self._session_runtime.list_configured_detached_mcp_servers(
            session_state,
            agent_name=agent_name,
        )

    def _create_slash_handler(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
    ) -> Any:
        return self._slash_runtime.create_slash_handler(session_state, instance)

    async def _load_agent_card_for_session(
        self,
        session_state: ACPSessionState,
        source: str,
        *,
        attach_to: str | None = None,
    ) -> tuple[AgentInstance, list[str], list[str]]:
        return await self._slash_runtime.load_agent_card_for_session(
            session_state,
            source,
            attach_to=attach_to,
        )

    async def _attach_agent_tools_for_session(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        return await self._slash_runtime.attach_agent_tools_for_session(
            session_state,
            parent_name,
            child_names,
        )

    async def _detach_agent_tools_for_session(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        return await self._slash_runtime.detach_agent_tools_for_session(
            session_state,
            parent_name,
            child_names,
        )

    async def _reload_agent_cards_for_session(self, session_id: str) -> bool:
        return await self._slash_runtime.reload_agent_cards_for_session(session_id)

    def _build_status_line_meta(
        self, agent: Any, turn_start_index: int | None
    ) -> dict[str, Any] | None:
        return self._session_runtime.build_status_line_meta(agent, turn_start_index)

    @staticmethod
    def _merge_tool_runner_hooks(
        base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None:
        return ACPServerSessionRuntime.merge_tool_runner_hooks(base, extra)

    async def _send_status_line_update(
        self, session_id: str, agent: Any, turn_start_index: int | None
    ) -> None:
        await self._session_runtime.send_status_line_update(session_id, agent, turn_start_index)

    async def _initialize_session_state(
        self,
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
    ) -> tuple[ACPSessionState, SessionModeState]:
        return await self._session_runtime.initialize_session_state(
            session_id,
            cwd=cwd,
            mcp_servers=mcp_servers,
        )

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        return await self._session_store.list_sessions(cursor=cursor, cwd=cwd, **kwargs)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        return await self._session_store.load_session(
            cwd=cwd,
            session_id=session_id,
            mcp_servers=mcp_servers,
            **kwargs,
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        return await self._session_store.resume_session(
            cwd=cwd,
            session_id=session_id,
            mcp_servers=mcp_servers,
            **kwargs,
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """
        Handle new session request.

        Creates a new ACP session with its own dedicated agent instance.
        """
        request_cwd = self._resolve_request_cwd(
            cwd=cwd,
            request_name="session/new",
            required=True,
        )
        assert request_cwd is not None
        manager = self._get_session_manager(cwd=Path(request_cwd))
        session_id = manager.generate_session_id()

        logger.info(
            "ACP new session request",
            name="acp_new_session",
            session_id=session_id,
            cwd=request_cwd,
            mcp_server_count=len(mcp_servers or []),
        )

        session_state, session_modes = await self._initialize_session_state(
            session_id,
            cwd=request_cwd,
            mcp_servers=mcp_servers or [],
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
        try:
            return instance.app.resolve_target_agent_name()
        except Exception:
            pass

        return resolve_default_agent_name(
            instance.agents,
            is_default=lambda _name, agent: agent_is_default(agent),
        )

    def _resolve_primary_agent_name(self, instance: AgentInstance) -> str | None:
        """Recompute and cache the ACP primary agent for a specific instance."""
        primary_agent_name = self._select_primary_agent(instance)
        self.primary_agent_name = primary_agent_name
        return primary_agent_name

    def _resolve_session_fallback_agent_name(self, instance: AgentInstance) -> str | None:
        return self._resolve_primary_agent_name(instance)

    async def prompt(
        self,
        prompt: list[ACPContentBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        return await self._prompt_flow.prompt(
            prompt=prompt,
            session_id=session_id,
            message_id=message_id,
            **kwargs,
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
        write_interactive_trace("acp.cancel.request", session_id=session_id)

        # Get the task for this session and cancel it
        async with self._session_lock:
            task = self._session_tasks.get(session_id)
            if task and not task.done():
                task.cancel()
                write_interactive_trace("acp.cancel.task_cancelled", session_id=session_id)
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

            disposed_instances: set[int] = set()
            for session_id, instance in self.sessions.items():
                instance_id = id(instance)
                if instance_id in disposed_instances:
                    continue
                disposed_instances.add(instance_id)
                try:
                    await self._dispose_instance_task(instance)
                except Exception as e:
                    logger.error(
                        f"Error disposing instance for session {session_id}: {e}",
                        name="acp_cleanup_error",
                    )

            bootstrap_instance = self._bootstrap_instance
            if bootstrap_instance and id(bootstrap_instance) not in disposed_instances:
                try:
                    await self._dispose_instance_task(bootstrap_instance)
                except Exception as e:
                    logger.error(
                        f"Error disposing ACP bootstrap instance: {e}",
                        name="acp_cleanup_error",
                    )

            self.sessions.clear()

        logger.info("ACP cleanup complete")
