from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

from fast_agent.acp.protocols import InstructionContextCapable
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.acp.server.live_session_registry import ACPLiveSessionRegistry
    from fast_agent.acp.server.models import ACPSessionState
    from fast_agent.config import MCPServerSettings
    from fast_agent.core.agent_app import AgentCardLoadResult
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
    _live_sessions: ACPLiveSessionRegistry
    _session_lock: asyncio.Lock
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

    async def _hydrate_session_state_from_persisted_session(
        self,
        session_state: ACPSessionState,
    ) -> bool: ...

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
        return SlashCommandHandler(
            session_state.session_id,
            instance,
            self._host._resolve_primary_agent_name(instance) or "default",
            noenv=instance.app.noenv_mode,
            client_info=self._host._client_info,
            client_capabilities=self._host._client_capabilities,
            protocol_version=self._host._protocol_version,
            session_instructions=session_state.resolved_instructions,
            instruction_resolver=partial(self._resolve_instruction_for_system, session_state),
            card_loader=(
                partial(self._load_card, session_state) if self._host._load_card_callback else None
            ),
            attach_agent_callback=(
                partial(self._attach_agent_tools, session_state)
                if self._host._attach_agent_tools_callback
                else None
            ),
            detach_agent_callback=(
                partial(self._detach_agent_tools, session_state)
                if self._host._detach_agent_tools_callback
                else None
            ),
            attach_mcp_server_callback=partial(self._attach_mcp_server, session_state),
            detach_mcp_server_callback=partial(self._detach_mcp_server, session_state),
            list_attached_mcp_servers_callback=partial(
                self._list_attached_mcp_servers, session_state
            ),
            list_configured_detached_mcp_servers_callback=partial(
                self._list_configured_detached_mcp_servers, session_state
            ),
            dump_agent_callback=(
                partial(self._dump_agent_card) if self._host._dump_agent_card_callback else None
            ),
            reload_callback=(
                partial(self.reload_agent_cards_for_session, session_state.session_id)
                if self._host._reload_callback
                else None
            ),
            set_current_mode_callback=partial(self._set_current_mode, session_state),
        )

    async def _load_card(
        self,
        session_state: ACPSessionState,
        source: str,
        parent_name: str | None,
    ) -> tuple[AgentInstance, AgentCardLoadResult]:
        return await self.load_agent_card_for_session(session_state, source, attach_to=parent_name)

    async def _attach_agent_tools(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        return await self.attach_agent_tools_for_session(session_state, parent_name, child_names)

    async def _detach_agent_tools(
        self,
        session_state: ACPSessionState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> tuple[AgentInstance, list[str]]:
        return await self.detach_agent_tools_for_session(session_state, parent_name, child_names)

    async def _attach_mcp_server(
        self,
        session_state: ACPSessionState,
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
        await self._refresh_session_after_mcp_change(session_state, agent_name)
        return result

    async def _detach_mcp_server(
        self,
        session_state: ACPSessionState,
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        result = await self._host._detach_mcp_server_for_session(
            session_state,
            agent_name=agent_name,
            server_name=server_name,
        )
        await self._refresh_session_after_mcp_change(session_state, agent_name)
        return result

    async def _refresh_session_after_mcp_change(
        self, session_state: ACPSessionState, agent_name: str
    ) -> None:
        current_instance = session_state.instance
        if session_state.slash_handler:
            session_state.slash_handler.instance = current_instance

        if not session_state.acp_context:
            return

        resolved_instruction = None
        agent = current_instance.agents.get(agent_name)
        if isinstance(agent, InstructionContextCapable):
            resolved_instruction = agent.instruction
        await session_state.acp_context.invalidate_instruction_cache(
            agent_name,
            resolved_instruction,
        )
        await session_state.acp_context.send_available_commands_update()

    async def _list_attached_mcp_servers(
        self, session_state: ACPSessionState, agent_name: str
    ) -> list[str]:
        return await self._host._list_attached_mcp_servers_for_session(
            session_state,
            agent_name=agent_name,
        )

    async def _list_configured_detached_mcp_servers(
        self, session_state: ACPSessionState, agent_name: str
    ) -> list[str]:
        return await self._host._list_configured_detached_mcp_servers_for_session(
            session_state,
            agent_name=agent_name,
        )

    async def _dump_agent_card(self, agent_name: str) -> str:
        if not self._host._dump_agent_card_callback:
            raise RuntimeError("AgentCard dumping is not available.")
        return await self._host._dump_agent_card_callback(agent_name)

    @staticmethod
    async def _set_current_mode(session_state: ACPSessionState, agent_name: str) -> None:
        session_state.set_current_agent(agent_name, sync_context=False)
        if session_state.acp_context:
            await session_state.acp_context.switch_mode(agent_name)

    async def _resolve_instruction_for_system(
        self, session_state: ACPSessionState, agent_name: str
    ) -> str | None:
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

    async def _replace_instance_and_hydrate_session(
        self,
        session_state: ACPSessionState,
        *,
        dispose_error_name: str,
    ) -> AgentInstance:
        instance = await self._host._replace_instance_for_session(
            session_state,
            dispose_error_name=dispose_error_name,
            await_refresh_session_state=True,
        )
        await self._host._hydrate_session_state_from_persisted_session(session_state)
        return instance

    async def load_agent_card_for_session(
        self,
        session_state: ACPSessionState,
        source: str,
        *,
        attach_to: str | None = None,
    ) -> tuple[AgentInstance, AgentCardLoadResult]:
        if not self._host._load_card_callback:
            raise RuntimeError("AgentCard loading is not available.")
        loaded_card = await self._host._load_card_callback(source, attach_to)

        instance = await self._replace_instance_and_hydrate_session(
            session_state,
            dispose_error_name="acp_card_dispose_error",
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, loaded_card

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

        instance = await self._replace_instance_and_hydrate_session(
            session_state,
            dispose_error_name="acp_attach_dispose_error",
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

        instance = await self._replace_instance_and_hydrate_session(
            session_state,
            dispose_error_name="acp_detach_dispose_error",
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return instance, detached_names

    async def reload_agent_cards_for_session(self, session_id: str) -> bool:
        if not self._host._reload_callback:
            return False
        live_sessions = self._host._live_sessions
        if session_id in live_sessions.active_prompts:
            current_task = asyncio.current_task()
            session_task = live_sessions.session_tasks.get(session_id)
            if current_task != session_task:
                raise RuntimeError("Cannot reload while a prompt is active for this session.")

        changed = await self._host._reload_callback()
        if not changed:
            return False

        async with self._host._session_lock:
            session_state = live_sessions.session_state.get(session_id)
        if not session_state:
            return True

        await self._replace_instance_and_hydrate_session(
            session_state,
            dispose_error_name="acp_reload_dispose_error",
        )

        if session_state.acp_context:
            await session_state.acp_context.send_available_commands_update()

        return True
