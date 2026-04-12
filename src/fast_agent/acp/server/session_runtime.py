from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, Sequence, cast

from acp.helpers import update_agent_message_text
from acp.schema import HttpMcpServer, McpServerStdio, SessionMode, SessionModeState, SseMcpServer

from fast_agent.acp.acp_context import ACPContext, ClientInfo
from fast_agent.acp.acp_context import ClientCapabilities as FAClientCapabilities
from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.permission_store import PermissionStore
from fast_agent.acp.protocols import (
    FilesystemRuntimeCapable,
    InstructionContextCapable,
    PlanTelemetryCapable,
    ShellRuntimeCapable,
    WorkflowTelemetryCapable,
)
from fast_agent.acp.server.common import (
    format_agent_name_as_title,
    truncate_description,
)
from fast_agent.acp.server.mcp_server_conversion import (
    ACPConfiguredMCPServer,
    convert_acp_mcp_server,
)
from fast_agent.acp.server.models import ACPSessionState, SessionMCPServerState
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
from fast_agent.acp.tool_progress import ACPToolProgressManager
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.context import Context
from fast_agent.core.instruction_refresh import (
    McpInstructionCapable,
    build_instruction,
    resolve_instruction_skill_manifests,
)
from fast_agent.core.instruction_utils import (
    build_agent_instruction_context,
    get_instruction_template,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.interfaces import ACPAwareProtocol, AgentProtocol, LlmCapableProtocol
from fast_agent.llm.usage_tracking import last_turn_usage
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.types import RequestParams
from fast_agent.workflow_telemetry import ACPPlanTelemetryProvider, ToolHandlerWorkflowTelemetry

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.core.instruction_refresh import ConfiguredMcpInstructionCapable
    from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult

logger = get_logger(__name__)


class SessionRuntimeHost(Protocol):
    _create_instance_task: Any
    _dispose_instance_task: Any
    server_name: str
    sessions: dict[str, AgentInstance]
    _session_lock: asyncio.Lock
    _prompt_locks: dict[str, asyncio.Lock]
    _session_state: dict[str, ACPSessionState]
    _connection: Any
    _client_supports_terminal: bool
    _client_supports_fs_read: bool
    _client_supports_fs_write: bool
    _client_info: dict[str, Any] | None
    _skills_directory_override: Sequence[str] | str | None
    _parsed_client_capabilities: FAClientCapabilities | None
    _parsed_client_info: ClientInfo | None
    _protocol_version: int | None
    primary_agent_name: str | None
    _permissions_enabled: bool

    def _resolve_primary_agent_name(self, instance: AgentInstance) -> str | None: ...

    def _calculate_terminal_output_limit(self, agent: Any) -> int: ...

    def _create_slash_handler(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
    ) -> Any: ...

    async def _send_available_commands_update(self, session_id: str) -> None: ...


class ACPServerSessionRuntime:
    def __init__(self, host: SessionRuntimeHost) -> None:
        self._host = host

    async def _rebuild_agent_instructions_for_filesystem(
        self,
        instance: AgentInstance,
    ) -> None:
        from fast_agent.core.instruction_refresh import rebuild_agent_instruction

        for agent in instance.agents.values():
            await rebuild_agent_instruction(agent)

    @staticmethod
    def _copy_requested_mcp_servers(
        mcp_servers: Sequence[ACPConfiguredMCPServer],
    ) -> dict[str, SessionMCPServerState]:
        return {
            server.name: SessionMCPServerState(
                server_name=server.name,
                server_config=deepcopy(convert_acp_mcp_server(server)),
                attached=True,
            )
            for server in mcp_servers
        }

    @staticmethod
    def _resolve_server_config(
        server_config: MCPServerSettings | None,
        inherited_state: SessionMCPServerState | None = None,
    ) -> MCPServerSettings | None:
        if server_config is not None:
            return deepcopy(server_config)
        if inherited_state and inherited_state.server_config is not None:
            return deepcopy(inherited_state.server_config)
        return None

    @staticmethod
    def _effective_session_mcp_servers_for_agent(
        session_state: ACPSessionState,
        agent_name: str,
    ) -> dict[str, SessionMCPServerState]:
        effective = {
            server_name: replace(server_state)
            for server_name, server_state in session_state.session_mcp_servers.items()
        }
        for server_name, server_state in session_state.agent_mcp_servers.get(agent_name, {}).items():
            effective[server_name] = SessionMCPServerState(
                server_name=server_name,
                server_config=ACPServerSessionRuntime._resolve_server_config(
                    server_state.server_config,
                    effective.get(server_name),
                ),
                attached=server_state.attached,
            )
        return effective

    def _diff_requested_session_mcp_servers(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
        requested_mcp_servers: dict[str, SessionMCPServerState],
    ) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
        removed_servers: list[tuple[str, str]] = []
        reconnect_servers: set[tuple[str, str]] = set()

        for agent_name, _agent in self._mcp_capable_agents(instance):
            agent_overrides = session_state.agent_mcp_servers.get(agent_name, {})

            # Only session-level MCP requests should be diffed here.
            # Runtime overlays created by /mcp connect or /mcp detach survive reloads.
            for server_name, current_state in session_state.session_mcp_servers.items():
                desired_state = requested_mcp_servers.get(server_name)
                if (
                    current_state.attached
                    and (desired_state is None or not desired_state.attached)
                    and server_name not in agent_overrides
                ):
                    removed_servers.append((agent_name, server_name))

                if (
                    desired_state is not None
                    and desired_state.attached
                    and current_state.attached
                    and current_state.server_config != desired_state.server_config
                    and server_name not in agent_overrides
                ):
                    reconnect_servers.add((agent_name, server_name))

        return removed_servers, reconnect_servers

    @staticmethod
    def _set_agent_overlay_state(
        session_state: ACPSessionState,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None,
        attached: bool,
    ) -> None:
        agent_overlay = session_state.agent_mcp_servers.setdefault(agent_name, {})
        agent_overlay[server_name] = SessionMCPServerState(
            server_name=server_name,
            server_config=deepcopy(server_config),
            attached=attached,
        )

    @staticmethod
    def _mcp_capable_agents(
        instance: AgentInstance,
    ) -> list[tuple[str, McpAgentProtocol]]:
        return [
            (agent_name, agent)
            for agent_name, agent in instance.agents.items()
            if isinstance(agent, McpAgentProtocol)
        ]

    @staticmethod
    def _require_runtime_mcp_manager(
        instance: AgentInstance,
    ) -> AgentApp:
        return instance.app

    async def _attach_server_to_agent(
        self,
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        manager = self._require_runtime_mcp_manager(instance)
        return await manager.attach_mcp_server(
            agent_name,
            server_name,
            server_config=server_config,
            options=options,
        )

    async def _detach_server_from_agent(
        self,
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        manager = self._require_runtime_mcp_manager(instance)
        return await manager.detach_mcp_server(agent_name, server_name)

    async def _apply_session_mcp_overlay(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
        *,
        force_reconnect_targets: set[tuple[str, str]] | None = None,
    ) -> None:
        mcp_agents = self._mcp_capable_agents(instance)
        if not mcp_agents:
            if session_state.session_mcp_servers or session_state.agent_mcp_servers:
                raise RuntimeError("ACP session requested MCP servers but no MCP-capable agents exist.")
            return

        for agent_name, _agent in mcp_agents:
            effective_states = self._effective_session_mcp_servers_for_agent(session_state, agent_name)
            for server_state in effective_states.values():
                if not server_state.attached:
                    await self._detach_server_from_agent(
                        instance,
                        agent_name=agent_name,
                        server_name=server_state.server_name,
                    )
                    continue

                attach_options = None
                if (
                    force_reconnect_targets
                    and (agent_name, server_state.server_name) in force_reconnect_targets
                ):
                    attach_options = replace(MCPAttachOptions(), force_reconnect=True)
                result = await self._attach_server_to_agent(
                    instance,
                    agent_name=agent_name,
                    server_name=server_state.server_name,
                    server_config=deepcopy(server_state.server_config),
                    options=attach_options,
                )
                logger.info(
                    "ACP session MCP server attached",
                    name="acp_session_mcp_server_attached",
                    agent_name=agent_name,
                    server_name=server_state.server_name,
                    transport=result.transport,
                    already_attached=result.already_attached,
                    tools_total=result.tools_total,
                    prompts_total=result.prompts_total,
                )

    async def attach_session_mcp_server(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        instance = session_state.instance
        existing_state = self._effective_session_mcp_servers_for_agent(session_state, agent_name).get(
            server_name
        )
        effective_config = self._resolve_server_config(
            server_config,
            existing_state,
        )
        attach_options = options
        if (
            existing_state is not None
            and existing_state.attached
            and existing_state.server_config != effective_config
        ):
            attach_options = replace(options or MCPAttachOptions(), force_reconnect=True)
        result = await self._attach_server_to_agent(
            instance,
            agent_name=agent_name,
            server_name=server_name,
            server_config=effective_config,
            options=attach_options,
        )
        logger.info(
            "ACP session MCP server attached",
            name="acp_session_mcp_server_attached",
            agent_name=agent_name,
            server_name=server_name,
            transport=result.transport,
            already_attached=result.already_attached,
            tools_total=result.tools_total,
            prompts_total=result.prompts_total,
        )
        self._set_agent_overlay_state(
            session_state,
            agent_name=agent_name,
            server_name=server_name,
            server_config=effective_config,
            attached=True,
        )
        return result

    async def detach_session_mcp_server(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        instance = session_state.instance
        existing_state = self._effective_session_mcp_servers_for_agent(session_state, agent_name).get(
            server_name
        )
        result = await self._detach_server_from_agent(
            instance,
            agent_name=agent_name,
            server_name=server_name,
        )
        self._set_agent_overlay_state(
            session_state,
            agent_name=agent_name,
            server_name=server_name,
            server_config=self._resolve_server_config(None, existing_state),
            attached=False,
        )
        return result

    async def list_attached_mcp_servers(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
    ) -> list[str]:
        manager = self._require_runtime_mcp_manager(session_state.instance)
        return await manager.list_attached_mcp_servers(agent_name)

    async def list_configured_detached_mcp_servers(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
    ) -> list[str]:
        manager = self._require_runtime_mcp_manager(session_state.instance)
        configured = set(await manager.list_configured_detached_mcp_servers(agent_name))
        configured.update(
            server_name
            for server_name, server_state in self._effective_session_mcp_servers_for_agent(
                session_state,
                agent_name,
            ).items()
            if not server_state.attached and server_state.server_config is not None
        )
        return sorted(configured)

    def build_session_modes(
        self, instance: AgentInstance, session_state: ACPSessionState | None = None
    ) -> SessionModeState:
        available_modes: list[SessionMode] = []
        resolved_cache = session_state.resolved_instructions if session_state else {}
        force_include = session_state.current_agent_name if session_state else None
        visible_agent_names = instance.app.visible_agent_names(force_include=force_include)

        for agent_name in visible_agent_names:
            agent = instance.agents[agent_name]

            instruction = resolved_cache.get(agent_name) or agent.instruction
            description = truncate_description(instruction) if instruction else None
            display_name = format_agent_name_as_title(agent_name)

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

            available_modes.append(
                SessionMode(
                    id=agent_name,
                    name=display_name,
                    description=description,
                )
            )

        current_mode_id = (
            self._host._resolve_primary_agent_name(instance)
            or next(iter(visible_agent_names), None)
            or (list(instance.agents.keys())[0] if instance.agents else "default")
        )
        return SessionModeState(
            available_modes=available_modes,
            current_mode_id=current_mode_id,
        )

    async def build_session_request_params(
        self, agent: object, session_state: ACPSessionState | None
    ) -> RequestParams | None:
        if not isinstance(agent, LlmCapableProtocol) or agent.llm is None:
            return None

        resolved_cache = session_state.resolved_instructions if session_state else {}
        agent_name = agent.name if isinstance(agent, AgentProtocol) else ""
        resolved = resolved_cache.get(agent_name, None)
        if isinstance(agent, McpInstructionCapable) or resolved is None:
            context = session_state.prompt_context if session_state else None
            if not context:
                return None
            resolved = await self.resolve_instruction_for_session(agent, context)
            if not resolved:
                return None
            if session_state is not None:
                session_state.resolved_instructions[agent_name] = resolved
        return RequestParams(systemPrompt=resolved)

    async def resolve_instruction_for_session(
        self,
        agent: object,
        context: dict[str, str],
    ) -> str | None:
        try:
            template = get_instruction_template(cast("Any", agent))
        except AttributeError:
            return None
        if not template:
            return None

        aggregator = None
        skill_manifests = None
        skill_read_tool_name = "read_skill"
        effective_context = dict(context)
        if isinstance(agent, McpInstructionCapable):
            configured_agent = cast("ConfiguredMcpInstructionCapable", agent)
            aggregator = agent.aggregator
            skill_manifests = resolve_instruction_skill_manifests(
                configured_agent,
                configured_agent.skill_manifests,
            )
            skill_read_tool_name = configured_agent.skill_read_tool_name
            if agent.instruction_context:
                effective_context = dict(agent.instruction_context)

        try:
            effective_context = build_agent_instruction_context(
                cast("Any", agent),
                effective_context,
            )
        except AttributeError:
            return None
        return await build_instruction(
            template,
            aggregator=aggregator,
            skill_manifests=skill_manifests,
            skill_read_tool_name=skill_read_tool_name,
            context=effective_context,
            source=agent.name if isinstance(agent, AgentProtocol) else None,
        )

    async def replace_instance_for_session(
        self,
        session_state: ACPSessionState,
        *,
        dispose_error_name: str,
        await_refresh_session_state: bool,
    ) -> AgentInstance:
        instance = await self._host._create_instance_task()
        old_instance = session_state.instance
        session_state.instance = instance
        async with self._host._session_lock:
            self._host.sessions[session_state.session_id] = instance
        if await_refresh_session_state:
            await self.refresh_session_state(session_state, instance)
        else:
            asyncio.create_task(self.refresh_session_state(session_state, instance))
        try:
            await self._host._dispose_instance_task(old_instance)
        except Exception as exc:
            logger.warning(
                "Failed to dispose old session instance",
                name=dispose_error_name,
                session_id=session_state.session_id,
                error=str(exc),
            )
        return instance

    async def refresh_session_state(
        self, session_state: ACPSessionState, instance: AgentInstance
    ) -> None:
        await self._apply_session_mcp_overlay(session_state, instance)
        self._apply_session_agent_bindings(
            session_state,
            instance,
            bind_tool_handler=session_state.progress_manager is not None,
            bind_permission_handler=session_state.permission_handler is not None,
            bind_runtimes=True,
            register_stream_listeners=session_state.progress_manager is not None,
        )
        if session_state.filesystem_runtime:
            await self._rebuild_agent_instructions_for_filesystem(instance)
        await self._finalize_session_instance_state(
            session_state,
            instance,
            session_cwd=session_state.session_cwd,
            prompt_context=session_state.prompt_context or {},
        )

    def _apply_session_agent_bindings(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
        *,
        bind_tool_handler: bool,
        bind_permission_handler: bool,
        bind_runtimes: bool,
        register_stream_listeners: bool,
    ) -> None:
        tool_handler = session_state.progress_manager
        permission_handler = session_state.permission_handler
        workflow_telemetry = (
            ToolHandlerWorkflowTelemetry(tool_handler, server_name=self._host.server_name)
            if tool_handler and bind_tool_handler
            else None
        )

        for agent_name, agent in instance.agents.items():
            if isinstance(agent, McpAgentProtocol):
                if (
                    bind_tool_handler
                    and tool_handler
                    and agent.aggregator.tool_execution_handler is not tool_handler
                ):
                    agent.aggregator.set_tool_execution_handler(tool_handler)

                if (
                    bind_permission_handler
                    and permission_handler
                    and agent.aggregator.permission_handler is not permission_handler
                ):
                    agent.aggregator.set_permission_handler(permission_handler)

            if workflow_telemetry and isinstance(agent, WorkflowTelemetryCapable):
                agent.workflow_telemetry = workflow_telemetry

            if bind_tool_handler and isinstance(agent, PlanTelemetryCapable) and self._host._connection:
                agent.plan_telemetry = ACPPlanTelemetryProvider(
                    self._host._connection,
                    session_state.session_id,
                )

            llm = agent.llm if isinstance(agent, LlmCapableProtocol) else None
            if register_stream_listeners and tool_handler and llm is not None:
                try:
                    llm.add_tool_stream_listener(tool_handler.handle_tool_stream_event)
                except Exception:
                    pass

            if (
                bind_runtimes
                and session_state.terminal_runtime
                and isinstance(agent, ShellRuntimeCapable)
                and agent.shell_runtime_enabled
            ):
                agent.set_external_runtime(session_state.terminal_runtime)

            if (
                bind_runtimes
                and session_state.filesystem_runtime
                and isinstance(agent, FilesystemRuntimeCapable)
            ):
                agent.set_filesystem_runtime(session_state.filesystem_runtime)

    async def _resolve_session_instructions(
        self,
        instance: AgentInstance,
        prompt_context: dict[str, str],
    ) -> dict[str, str]:
        resolved_for_session: dict[str, str] = {}
        for agent_name, agent in instance.agents.items():
            resolved = await self.resolve_instruction_for_session(agent, prompt_context)
            if resolved:
                resolved_for_session[agent_name] = resolved
        return resolved_for_session

    def _apply_instruction_context(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
        prompt_context: dict[str, str],
    ) -> None:
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

    def _ensure_session_acp_context(
        self,
        session_state: ACPSessionState,
        *,
        session_cwd: str | None,
    ) -> ACPContext | None:
        if not self._host._connection:
            return None

        acp_context = session_state.acp_context
        if acp_context is None:
            acp_context = ACPContext(
                connection=self._host._connection,
                session_id=session_state.session_id,
                session_cwd=session_cwd,
                session_store_scope=session_state.session_store_scope,
                session_store_cwd=session_state.session_store_cwd,
                client_capabilities=self._host._parsed_client_capabilities,
                client_info=self._host._parsed_client_info,
                protocol_version=self._host._protocol_version,
            )
            session_state.acp_context = acp_context
        else:
            acp_context.set_session_cwd(session_cwd)
            acp_context.set_session_store(
                session_state.session_store_scope,
                session_state.session_store_cwd,
            )

        if session_state.terminal_runtime:
            acp_context.set_terminal_runtime(session_state.terminal_runtime)
        if session_state.filesystem_runtime:
            acp_context.set_filesystem_runtime(session_state.filesystem_runtime)
        if session_state.permission_handler:
            acp_context.set_permission_handler(session_state.permission_handler)
        if session_state.progress_manager:
            acp_context.set_progress_manager(session_state.progress_manager)
        return acp_context

    async def _finalize_session_instance_state(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
        *,
        session_cwd: str | None,
        prompt_context: dict[str, str],
    ) -> SessionModeState:
        primary_agent_name = self._host._resolve_primary_agent_name(instance)
        session_state.prompt_context = prompt_context
        session_state.resolved_instructions = await self._resolve_session_instructions(
            instance,
            prompt_context,
        )
        self._apply_instruction_context(session_state, instance, prompt_context)

        slash_handler = self._host._create_slash_handler(session_state, instance)
        session_state.slash_handler = slash_handler

        acp_context = self._ensure_session_acp_context(
            session_state,
            session_cwd=session_cwd,
        )
        if acp_context is not None:
            slash_handler.set_acp_context(acp_context)
            acp_context.set_slash_handler(slash_handler)
            acp_context.set_resolved_instructions(session_state.resolved_instructions)

            for agent_name, agent in instance.agents.items():
                try:
                    context = agent.context
                except AttributeError:
                    continue
                if isinstance(context, Context):
                    context.acp = acp_context
                    logger.debug(
                        "ACPContext set on agent",
                        name="acp_context_set",
                        session_id=session_state.session_id,
                        agent_name=agent_name,
                    )

            logger.info(
                "ACPContext created for session",
                name="acp_context_created",
                session_id=session_state.session_id,
                has_terminal=acp_context.terminal_runtime is not None,
                has_filesystem=acp_context.filesystem_runtime is not None,
                has_permissions=acp_context.permission_handler is not None,
            )

        if self._host._connection:
            asyncio.create_task(self._host._send_available_commands_update(session_state.session_id))

        current_agent = session_state.current_agent_name
        if not current_agent or current_agent not in instance.agents:
            current_agent = primary_agent_name or next(iter(instance.agents.keys()), None)
            session_state.current_agent_name = current_agent
        if current_agent:
            slash_handler.set_current_agent(current_agent)

        session_modes = self.build_session_modes(instance, session_state)
        if current_agent and current_agent in instance.agents:
            session_modes = SessionModeState(
                available_modes=session_modes.available_modes,
                current_mode_id=current_agent,
            )

        if acp_context is not None:
            acp_context.set_available_modes(session_modes.available_modes)
            if current_agent:
                acp_context.set_current_mode(current_agent)

        return session_modes

    async def initialize_session_state(
        self,
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
    ) -> tuple[ACPSessionState, SessionModeState]:
        requested_mcp_servers = self._copy_requested_mcp_servers(mcp_servers)
        removed_session_mcp_servers: list[tuple[str, str]] = []
        force_reconnect_targets: set[tuple[str, str]] = set()

        async with self._host._session_lock:
            session_state = self._host._session_state.get(session_id)
            if session_state:
                requested_mcp_servers_changed = (
                    session_state.session_mcp_servers != requested_mcp_servers
                )
                if requested_mcp_servers_changed:
                    removed_session_mcp_servers, force_reconnect_targets = (
                        self._diff_requested_session_mcp_servers(
                            session_state,
                            session_state.instance,
                            requested_mcp_servers,
                        )
                    )
                    session_state.session_mcp_servers = requested_mcp_servers
                instance = session_state.instance
            else:
                instance = await self._host._create_instance_task()
                self._host.sessions[session_id] = instance
                session_state = ACPSessionState(session_id=session_id, instance=instance)
                session_state.session_mcp_servers = requested_mcp_servers
                self._host._session_state[session_id] = session_state

            session_state.session_cwd = cwd
            if session_state.session_store_scope == "workspace":
                session_state.session_store_cwd = cwd

            tool_handler = session_state.progress_manager
            tool_handler_created = False
            permission_handler_created = False
            terminal_runtime_created = False
            filesystem_runtime_created = False
            if self._host._connection and tool_handler is None:
                tool_handler = ACPToolProgressManager(self._host._connection, session_id)
                session_state.progress_manager = tool_handler
                tool_handler_created = True
                logger.info(
                    "ACP tool progress manager created for session",
                    name="acp_tool_progress_init",
                    session_id=session_id,
                )

            if (
                self._host._connection
                and self._host._permissions_enabled
                and session_state.permission_handler is None
            ):
                session_cwd = cwd or "."
                permission_store = PermissionStore(cwd=session_cwd)
                permission_handler = ACPToolPermissionAdapter(
                    connection=self._host._connection,
                    session_id=session_id,
                    store=permission_store,
                    cwd=session_cwd,
                    tool_handler=tool_handler,
                )
                session_state.permission_handler = permission_handler
                permission_handler_created = True

                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, McpAgentProtocol):
                        agent.aggregator.set_permission_handler(permission_handler)
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

            if (
                self._host._connection
                and self._host._client_supports_terminal
                and session_state.terminal_runtime is None
            ):
                for agent_name, agent in instance.agents.items():
                    if not isinstance(agent, ShellRuntimeCapable) or not agent.shell_runtime_enabled:
                        continue
                    shell_runtime = agent.shell_runtime
                    if shell_runtime is None:
                        continue
                    default_limit = shell_runtime.output_byte_limit
                    perm_handler = session_state.permission_handler
                    terminal_runtime = ACPTerminalRuntime(
                        connection=self._host._connection,
                        session_id=session_id,
                        activation_reason="via ACP terminal support",
                        timeout_seconds=shell_runtime.timeout_seconds,
                        tool_handler=tool_handler,
                        default_output_byte_limit=default_limit,
                        permission_handler=perm_handler,
                    )
                    agent.set_external_runtime(terminal_runtime)
                    session_state.terminal_runtime = terminal_runtime
                    terminal_runtime_created = True

                    logger.info(
                        "ACP terminal runtime injected",
                        name="acp_terminal_injected",
                        session_id=session_id,
                        agent_name=agent_name,
                        default_output_limit=default_limit,
                    )

            if (
                self._host._connection
                and (self._host._client_supports_fs_read or self._host._client_supports_fs_write)
                and session_state.filesystem_runtime is None
            ):
                perm_handler = session_state.permission_handler
                filesystem_runtime = ACPFilesystemRuntime(
                    connection=self._host._connection,
                    session_id=session_id,
                    activation_reason="via ACP filesystem support",
                    enable_read=self._host._client_supports_fs_read,
                    enable_write=self._host._client_supports_fs_write,
                    tool_handler=tool_handler,
                    permission_handler=perm_handler,
                )
                session_state.filesystem_runtime = filesystem_runtime
                filesystem_runtime_created = True

                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, FilesystemRuntimeCapable):
                        agent.set_filesystem_runtime(filesystem_runtime)
                        logger.info(
                            "ACP filesystem runtime injected",
                            name="acp_filesystem_injected",
                            session_id=session_id,
                            agent_name=agent_name,
                            read_enabled=self._host._client_supports_fs_read,
                            write_enabled=self._host._client_supports_fs_write,
                        )

        for agent_name, server_name in removed_session_mcp_servers:
            await self._detach_server_from_agent(
                instance,
                agent_name=agent_name,
                server_name=server_name,
            )
        await self._apply_session_mcp_overlay(
            session_state,
            instance,
            force_reconnect_targets=force_reconnect_targets,
        )

        session_context: dict[str, str] = {}
        enrich_with_environment_context(
            session_context,
            cwd,
            self._host._client_info,
            self._host._skills_directory_override,
        )
        self._apply_session_agent_bindings(
            session_state,
            instance,
            bind_tool_handler=tool_handler_created,
            bind_permission_handler=permission_handler_created,
            bind_runtimes=terminal_runtime_created or filesystem_runtime_created,
            register_stream_listeners=tool_handler_created,
        )
        if filesystem_runtime_created:
            await self._rebuild_agent_instructions_for_filesystem(instance)
        session_modes = await self._finalize_session_instance_state(
            session_state,
            instance,
            session_cwd=cwd,
            prompt_context=session_context,
        )

        logger.info(
            "Session modes initialized",
            name="acp_session_modes_init",
            session_id=session_id,
            current_mode=session_modes.current_mode_id,
            mode_count=len(session_modes.available_modes),
        )

        return session_state, session_modes

    def build_status_line_meta(
        self, agent: AgentProtocol | None, turn_start_index: int | None
    ) -> dict[str, Any] | None:
        if agent is None or agent.usage_accumulator is None:
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
    def merge_tool_runner_hooks(
        base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None:
        if base is None:
            return extra
        if extra is None:
            return base

        def merge(one: Any, two: Any) -> Any:
            if one is None:
                return two
            if two is None:
                return one

            async def merged(*args: Any, **kwargs: Any) -> None:
                await one(*args, **kwargs)
                await two(*args, **kwargs)

            return merged

        return ToolRunnerHooks(
            before_llm_call=merge(base.before_llm_call, extra.before_llm_call),
            after_llm_call=merge(base.after_llm_call, extra.after_llm_call),
            before_tool_call=merge(base.before_tool_call, extra.before_tool_call),
            after_tool_call=merge(base.after_tool_call, extra.after_tool_call),
            after_turn_complete=merge(base.after_turn_complete, extra.after_turn_complete),
        )

    async def send_status_line_update(
        self, session_id: str, agent: AgentProtocol | None, turn_start_index: int | None
    ) -> None:
        if not self._host._connection:
            return
        status_line_meta = self.build_status_line_meta(agent, turn_start_index)
        if not status_line_meta:
            return
        try:
            message_chunk = update_agent_message_text("")
            await self._host._connection.session_update(
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
