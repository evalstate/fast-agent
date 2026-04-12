from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from fast_agent.acp.protocols import InstructionContextCapable
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.acp.server.models import ACPSessionState
    from fast_agent.config import MCPServerSettings
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult

logger = get_logger(__name__)


class SlashRuntimeHost(Protocol):
    primary_agent_name: str | None
    _load_card_callback: Any
    _attach_agent_tools_callback: Any
    _detach_agent_tools_callback: Any
    _dump_agent_card_callback: Any
    _reload_callback: Any
    _active_prompts: set[str]
    _session_tasks: dict[str, asyncio.Task]
    _session_lock: asyncio.Lock
    _session_state: dict[str, ACPSessionState]
    sessions: dict[str, AgentInstance]
    _create_instance_task: Any
    _dispose_instance_task: Any
    _client_info: dict[str, Any] | None
    _client_capabilities: dict[str, Any] | None
    _protocol_version: int | None

    def _resolve_primary_agent_name(self, instance: AgentInstance) -> str | None: ...

    async def _replace_instance_for_session(
        self,
        session_state: ACPSessionState,
        *,
        dispose_error_name: str,
        await_refresh_session_state: bool,
    ) -> AgentInstance: ...

    async def _refresh_session_state(
        self, session_state: ACPSessionState, instance: AgentInstance
    ) -> None: ...

    async def _attach_mcp_server_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult: ...

    async def _detach_mcp_server_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult: ...

    async def _list_attached_mcp_servers_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
    ) -> list[str]: ...

    async def _list_configured_detached_mcp_servers_for_session(
        self,
        session_state: ACPSessionState,
        *,
        agent_name: str,
    ) -> list[str]: ...

    async def _resolve_instruction_for_session(
        self,
        agent: AgentProtocol,
        context: dict[str, str],
    ) -> str | None: ...


class ACPServerSlashRuntime:
    def __init__(self, host: SlashRuntimeHost) -> None:
        self._host = host

    def create_slash_handler(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
    ) -> SlashCommandHandler:
        async def load_card(
            source: str, parent_name: str | None
        ) -> tuple[AgentInstance, list[str], list[str]]:
            return await self.load_agent_card_for_session(
                session_state, source, attach_to=parent_name
            )

        async def attach_agent_tools(
            parent_name: str, child_names: Sequence[str]
        ) -> tuple[AgentInstance, list[str]]:
            return await self.attach_agent_tools_for_session(
                session_state, parent_name, child_names
            )

        async def detach_agent_tools(
            parent_name: str, child_names: Sequence[str]
        ) -> tuple[AgentInstance, list[str]]:
            return await self.detach_agent_tools_for_session(
                session_state, parent_name, child_names
            )

        async def attach_mcp_server(
            agent_name: str,
            server_name: str,
            server_config: MCPServerSettings | None = None,
            options: MCPAttachOptions | None = None,
        ) -> MCPAttachResult:
            result = await self._host._attach_mcp_server_for_session(
                session_state,
                agent_name=agent_name,
                server_name=server_name,
                server_config=server_config,
                options=options,
            )
            current_instance = session_state.instance
            if session_state.slash_handler:
                session_state.slash_handler.instance = current_instance

            if session_state.acp_context:
                resolved_instruction = None
                agent = current_instance.agents.get(agent_name)
                if isinstance(agent, InstructionContextCapable):
                    resolved_instruction = agent.instruction
                await session_state.acp_context.invalidate_instruction_cache(
                    agent_name,
                    resolved_instruction,
                )
                await session_state.acp_context.send_available_commands_update()

            return result

        async def detach_mcp_server(agent_name: str, server_name: str) -> MCPDetachResult:
            result = await self._host._detach_mcp_server_for_session(
                session_state,
                agent_name=agent_name,
                server_name=server_name,
            )
            current_instance = session_state.instance
            if session_state.slash_handler:
                session_state.slash_handler.instance = current_instance

            if session_state.acp_context:
                resolved_instruction = None
                agent = current_instance.agents.get(agent_name)
                if isinstance(agent, InstructionContextCapable):
                    resolved_instruction = agent.instruction
                await session_state.acp_context.invalidate_instruction_cache(
                    agent_name,
                    resolved_instruction,
                )
                await session_state.acp_context.send_available_commands_update()

            return result

        async def list_attached_mcp_servers(agent_name: str) -> list[str]:
            return await self._host._list_attached_mcp_servers_for_session(
                session_state,
                agent_name=agent_name,
            )

        async def list_configured_detached_mcp_servers(agent_name: str) -> list[str]:
            return await self._host._list_configured_detached_mcp_servers_for_session(
                session_state,
                agent_name=agent_name,
            )

        async def dump_agent_card(agent_name: str) -> str:
            if not self._host._dump_agent_card_callback:
                raise RuntimeError("AgentCard dumping is not available.")
            return await self._host._dump_agent_card_callback(agent_name)

        async def reload_cards() -> bool:
            return await self.reload_agent_cards_for_session(session_state.session_id)

        async def set_current_mode(agent_name: str) -> None:
            session_state.current_agent_name = agent_name
            if session_state.acp_context:
                await session_state.acp_context.switch_mode(agent_name)

        async def resolve_instruction_for_system(agent_name: str) -> str | None:
            current_instance = session_state.instance
            agent = current_instance.agents.get(agent_name)
            if agent is None:
                return None
            context = session_state.prompt_context or {}
            if not context:
                return None
            resolved = await self._host._resolve_instruction_for_session(agent, context)
            if resolved:
                session_state.resolved_instructions[agent_name] = resolved
            return resolved

        return SlashCommandHandler(
            session_state.session_id,
            instance,
            self._host._resolve_primary_agent_name(instance) or "default",
            noenv=instance.app.noenv_mode,
            client_info=self._host._client_info,
            client_capabilities=self._host._client_capabilities,
            protocol_version=self._host._protocol_version,
            session_instructions=session_state.resolved_instructions,
            instruction_resolver=resolve_instruction_for_system,
            card_loader=load_card if self._host._load_card_callback else None,
            attach_agent_callback=(
                attach_agent_tools if self._host._attach_agent_tools_callback else None
            ),
            detach_agent_callback=(
                detach_agent_tools if self._host._detach_agent_tools_callback else None
            ),
            attach_mcp_server_callback=(
                attach_mcp_server
            ),
            detach_mcp_server_callback=(
                detach_mcp_server
            ),
            list_attached_mcp_servers_callback=(
                list_attached_mcp_servers
            ),
            list_configured_detached_mcp_servers_callback=(
                list_configured_detached_mcp_servers
            ),
            dump_agent_callback=(
                dump_agent_card if self._host._dump_agent_card_callback else None
            ),
            reload_callback=reload_cards if self._host._reload_callback else None,
            set_current_mode_callback=set_current_mode,
        )

    async def load_agent_card_for_session(
        self,
        session_state: ACPSessionState,
        source: str,
        *,
        attach_to: str | None = None,
    ) -> tuple[AgentInstance, list[str], list[str]]:
        if not self._host._load_card_callback:
            raise RuntimeError("AgentCard loading is not available.")
        loaded_names, attached_names = await self._host._load_card_callback(source, attach_to)

        instance = await self._host._replace_instance_for_session(
            session_state,
            dispose_error_name="acp_card_dispose_error",
            await_refresh_session_state=True,
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, loaded_names, attached_names

    async def attach_agent_tools_for_session(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        if not self._host._attach_agent_tools_callback:
            raise RuntimeError("Agent tool attachment is not available.")

        attached_names = await self._host._attach_agent_tools_callback(parent_name, child_names)
        if not attached_names:
            return session_state.instance, []

        instance = await self._host._replace_instance_for_session(
            session_state,
            dispose_error_name="acp_attach_dispose_error",
            await_refresh_session_state=False,
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, attached_names

    async def detach_agent_tools_for_session(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        if not self._host._detach_agent_tools_callback:
            raise RuntimeError("Agent tool detachment is not available.")

        detached_names = await self._host._detach_agent_tools_callback(parent_name, child_names)
        if not detached_names:
            return session_state.instance, []

        instance = await self._host._replace_instance_for_session(
            session_state,
            dispose_error_name="acp_detach_dispose_error",
            await_refresh_session_state=False,
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, detached_names

    async def reload_agent_cards_for_session(self, session_id: str) -> bool:
        if not self._host._reload_callback:
            return False
        if session_id in self._host._active_prompts:
            current_task = asyncio.current_task()
            session_task = self._host._session_tasks.get(session_id)
            if current_task != session_task:
                raise RuntimeError("Cannot reload while a prompt is active for this session.")

        changed = await self._host._reload_callback()
        if not changed:
            return False

        async with self._host._session_lock:
            session_state = self._host._session_state.get(session_id)
        if not session_state:
            return True

        instance = await self._host._create_instance_task()
        old_instance = session_state.instance
        session_state.instance = instance
        async with self._host._session_lock:
            self._host.sessions[session_id] = instance
        await self._host._refresh_session_state(session_state, instance)
        try:
            await self._host._dispose_instance_task(old_instance)
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
