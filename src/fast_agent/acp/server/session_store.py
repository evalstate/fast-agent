from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, Sequence, cast

from acp.exceptions import RequestError
from acp.helpers import update_agent_message, update_user_message
from acp.schema import (
    AgentMessageChunk,
    HttpMcpServer,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    ResumeSessionResponse,
    SessionInfoUpdate,
    SessionModeState,
    SseMcpServer,
    UserMessageChunk,
)
from acp.schema import (
    SessionInfo as AcpSessionInfo,
)

if TYPE_CHECKING:
    from fast_agent.acp.server.models import ACPSessionState
    from fast_agent.types import PromptMessageExtended

from fast_agent.acp.content_conversion import convert_mcp_content_to_acp
from fast_agent.core.logging.logger import get_logger
from fast_agent.session import Session, extract_session_title, get_session_history_window

logger = get_logger(__name__)


class SessionStoreHost(Protocol):
    _connection: Any
    _session_lock: Any
    _session_state: dict[str, ACPSessionState]
    sessions: dict[str, Any]
    _prompt_locks: dict[str, Any]
    _dispose_instance_task: Any

    def _resolve_request_cwd(
        self,
        *,
        cwd: str | None,
        request_name: str,
        required: bool,
    ) -> str | None: ...

    async def _initialize_session_state(
        self,
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
    ) -> tuple[ACPSessionState, SessionModeState]: ...

    def _resolve_session_fallback_agent_name(self, instance: Any) -> str | None: ...

    def _get_session_manager(self, *, cwd: Path | None = None) -> Any: ...


class ACPServerSessionStore:
    def __init__(self, host: SessionStoreHost) -> None:
        self._host = host

    @staticmethod
    def extract_session_title(metadata: object) -> str | None:
        if not isinstance(metadata, Mapping):
            return None
        return extract_session_title(cast("Mapping[str, object]", metadata))

    @staticmethod
    def extract_session_cwd(metadata: object) -> str | None:
        if not isinstance(metadata, Mapping):
            return None
        cwd = cast("Mapping[str, object]", metadata).get("cwd")
        if isinstance(cwd, str) and cwd.strip():
            return cwd
        return None

    @staticmethod
    def legacy_session_cwd(manager: Any) -> str:
        workspace_dir = getattr(manager, "workspace_dir", None)
        if isinstance(workspace_dir, Path):
            return str(workspace_dir.resolve())
        if isinstance(workspace_dir, str) and workspace_dir.strip():
            return str(Path(workspace_dir).expanduser().resolve())
        return str(Path(manager.base_dir).resolve().parent.parent)

    def session_manager_entries(self, cwd: str | None) -> list[tuple[Any, str]]:
        if cwd is None:
            manager = self._host._get_session_manager()
            return [(manager, self.legacy_session_cwd(manager))]

        request_manager = self._host._get_session_manager(cwd=Path(cwd))
        entries = [(request_manager, self.legacy_session_cwd(request_manager))]
        app_manager = self._host._get_session_manager()
        if Path(app_manager.base_dir).resolve() != Path(request_manager.base_dir).resolve():
            entries.append((app_manager, self.legacy_session_cwd(app_manager)))
        return entries

    def build_history_updates(
        self,
        history: Sequence[PromptMessageExtended],
    ) -> list[UserMessageChunk | AgentMessageChunk]:
        updates: list[UserMessageChunk | AgentMessageChunk] = []
        for message in history:
            role_value = str(message.role)
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

    async def send_session_history_updates(
        self,
        session_state: ACPSessionState,
        session: Session,
        agent_name: str | None,
    ) -> None:
        if not self._host._connection:
            return

        try:
            title = self.extract_session_title(session.info.metadata)
            info_payload: dict[str, Any] = {
                "session_update": "session_info_update",
                "updated_at": session.info.last_activity.isoformat(),
            }
            if title is not None:
                info_payload["title"] = title
            info_update = SessionInfoUpdate(**info_payload)
            await self._host._connection.session_update(
                session_id=session_state.session_id,
                update=info_update,
            )

            if not agent_name:
                return
            agent = session_state.instance.agents.get(agent_name)
            if not agent:
                return

            history = list(agent.message_history)
            if not history:
                return

            updates = self.build_history_updates(history)
            for update in updates:
                await self._host._connection.session_update(
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
        _ = kwargs
        filter_cwd = self._host._resolve_request_cwd(
            cwd=cwd,
            request_name="session/list",
            required=False,
        )
        session_entries = self.session_manager_entries(filter_cwd)

        sessions_by_id: dict[str, tuple[Any, str]] = {}
        for manager, legacy_cwd in session_entries:
            for session_info in manager.list_sessions():
                session_cwd = self.extract_session_cwd(session_info.metadata) or legacy_cwd
                if filter_cwd is not None and session_cwd != filter_cwd:
                    continue

                existing_entry = sessions_by_id.get(session_info.name)
                if existing_entry is None:
                    sessions_by_id[session_info.name] = (session_info, session_cwd)

        sessions = sorted(
            sessions_by_id.values(),
            key=lambda item: item[0].last_activity,
            reverse=True,
        )

        start_index = 0
        if cursor:
            start_index = self._decode_session_list_cursor(cursor)

        limit = get_session_history_window()
        if limit > 0:
            page = sessions[start_index : start_index + limit]
            next_cursor = (
                self._encode_session_list_cursor(start_index + limit)
                if start_index + limit < len(sessions)
                else None
            )
        else:
            page = sessions[start_index:]
            next_cursor = None

        acp_sessions = []
        for session_info, session_cwd in page:
            title = self.extract_session_title(session_info.metadata)
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
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        _ = kwargs
        request_cwd = self._host._resolve_request_cwd(
            cwd=cwd,
            request_name="session/load",
            required=True,
        )
        assert request_cwd is not None
        logger.info(
            "ACP load session request",
            name="acp_load_session",
            session_id=session_id,
            cwd=request_cwd,
            mcp_server_count=len(mcp_servers or []),
        )
        async with self._host._session_lock:
            existing_session = session_id in self._host._session_state

        persisted_session = None
        manager = None
        manager_store_scope: Literal["workspace", "app"] = "workspace"
        manager_store_cwd: str | None = request_cwd
        for index, (candidate_manager, _legacy_cwd) in enumerate(
            self.session_manager_entries(request_cwd)
        ):
            candidate_session = candidate_manager.get_session(session_id)
            if candidate_session is None:
                continue
            persisted_cwd = self.extract_session_cwd(candidate_session.info.metadata)
            if persisted_cwd and str(Path(persisted_cwd).expanduser().resolve()) != request_cwd:
                logger.warning(
                    "ACP load session cwd mismatch",
                    name="acp_load_session_cwd_mismatch",
                    session_id=session_id,
                    requested_cwd=request_cwd,
                    persisted_cwd=persisted_cwd,
                )
                continue
            manager = candidate_manager
            persisted_session = candidate_session
            manager_store_scope = "workspace" if index == 0 else "app"
            manager_store_cwd = request_cwd if manager_store_scope == "workspace" else None
            break
        if not persisted_session:
            self._raise_session_not_found(session_id=session_id, request_cwd=request_cwd)
        assert manager is not None
        assert persisted_session is not None

        session_state, session_modes = await self._host._initialize_session_state(
            session_id,
            cwd=request_cwd,
            mcp_servers=mcp_servers or [],
        )
        session_state.session_store_scope = manager_store_scope
        session_state.session_store_cwd = manager_store_cwd
        if session_state.acp_context:
            session_state.acp_context.set_session_store(
                manager_store_scope,
                manager_store_cwd,
            )

        fallback_agent_name = self._host._resolve_session_fallback_agent_name(
            session_state.instance
        )
        result = manager.resume_session_agents(
            session_state.instance.agents,
            session_id,
            fallback_agent_name=fallback_agent_name,
        )
        if not result:
            if not existing_session:
                async with self._host._session_lock:
                    self._host.sessions.pop(session_id, None)
                    self._host._session_state.pop(session_id, None)
                    self._host._prompt_locks.pop(session_id, None)
                await self._host._dispose_instance_task(session_state.instance)
            self._raise_session_not_found(session_id=session_id, request_cwd=request_cwd)

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
            current_agent = fallback_agent_name or next(
                iter(session_state.instance.agents.keys()),
                None,
            )

        if current_agent:
            session_state.current_agent_name = current_agent
            if session_state.slash_handler:
                session_state.slash_handler.set_current_agent(current_agent)
            if session_state.acp_context:
                session_state.acp_context.set_current_mode(current_agent)

        if current_agent and current_agent != session_modes.current_mode_id:
            session_modes = SessionModeState(
                available_modes=session_modes.available_modes,
                current_mode_id=current_agent,
            )
            if session_state.acp_context:
                session_state.acp_context.set_available_modes(session_modes.available_modes)
                session_state.acp_context.set_current_mode(current_agent)

        if self._host._connection:
            # ACP session/load must return only after the replayed
            # session/update history has been streamed to the client.
            await self.send_session_history_updates(
                session_state,
                session,
                current_agent,
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
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        """Alias for session/load to support unstable session/resume."""
        _ = kwargs
        request_cwd = self._host._resolve_request_cwd(
            cwd=cwd,
            request_name="session/resume",
            required=True,
        )
        assert request_cwd is not None
        response = await self.load_session(
            cwd=request_cwd,
            mcp_servers=mcp_servers or [],
            session_id=session_id,
        )
        assert response is not None
        return ResumeSessionResponse(modes=response.modes, models=response.models)

    @staticmethod
    def _encode_session_list_cursor(offset: int) -> str:
        import base64
        import json

        payload = json.dumps(
            {"version": 1, "offset": offset},
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_session_list_cursor(cursor: str) -> int:
        import base64
        import json

        padding = "=" * (-len(cursor) % 4)
        try:
            payload = json.loads(
                base64.urlsafe_b64decode(f"{cursor}{padding}".encode("ascii")).decode("utf-8")
            )
        except Exception as exc:
            raise RequestError.invalid_params(
                {
                    "cursor": cursor,
                    "reason": "Invalid session list cursor",
                }
            ) from exc

        offset = payload.get("offset") if isinstance(payload, dict) else None
        version = payload.get("version") if isinstance(payload, dict) else None
        if not isinstance(offset, int) or offset < 0 or version != 1:
            raise RequestError.invalid_params(
                {
                    "cursor": cursor,
                    "reason": "Invalid session list cursor",
                }
            )
        return offset

    @staticmethod
    def _raise_session_not_found(*, session_id: str, request_cwd: str) -> None:
        logger.error(
            "Session not found for load_session",
            name="acp_load_session_not_found",
            session_id=session_id,
        )
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
