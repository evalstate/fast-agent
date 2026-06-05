"""Headless, session-oriented harness API for fast-agent."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union

from mcp.types import PromptMessage
from pydantic import BaseModel

from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from types import TracebackType

    from fast_agent.core.fastagent import AgentInstance, FastAgent, RunRuntime, RunSettings
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session.session_manager import Session, SessionManager

ModelT = TypeVar("ModelT", bound=BaseModel)
HARNESS_SESSION_ID_MAX_LENGTH = 128
HARNESS_SESSION_ID_PATTERN = re.compile(
    rf"^[A-Za-z0-9](?:[A-Za-z0-9_-]{{0,{HARNESS_SESSION_ID_MAX_LENGTH - 2}}}[A-Za-z0-9])?$"
)
MessageParam: TypeAlias = Union[
    str,
    PromptMessage,
    PromptMessageExtended,
    Sequence[Union[str, PromptMessage, PromptMessageExtended]],
]


@dataclass(slots=True)
class _HarnessSessionRecord:
    session_id: str
    default_agent_name: str | None
    instance: AgentInstance
    persisted_session: Session | None = None
    session_manager: SessionManager | None = None
    active_operation: str | None = None
    closed: bool = False


class HarnessSession:
    """A stable conversation session backed by one owned ``AgentInstance``."""

    def __init__(self, manager: HarnessSessions, record: _HarnessSessionRecord) -> None:
        self._manager = manager
        self._record = record

    @property
    def id(self) -> str:
        """Normalized session ID."""
        return self._record.session_id

    @property
    def default_agent_name(self) -> str | None:
        """Default agent used when calls omit ``agent_name``."""
        return self._record.default_agent_name

    async def send(
        self,
        message: MessageParam,
        *,
        agent_name: str | None = None,
        request_params: RequestParams | None = None,
    ) -> str:
        """Send a message and return assistant text."""
        agent = await self._begin_operation("send", agent_name)
        try:
            result = await agent.send(message, request_params)
            await self._save_persisted_history(agent)
            return result
        finally:
            await self._end_operation("send")

    async def generate(
        self,
        messages: MessageParam,
        *,
        agent_name: str | None = None,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        """Generate a message and return the full assistant message."""
        agent = await self._begin_operation("generate", agent_name)
        try:
            result = await agent.generate(messages, request_params)
            await self._save_persisted_history(agent)
            return result
        finally:
            await self._end_operation("generate")

    async def structured(
        self,
        messages: MessageParam,
        model: type[ModelT],
        *,
        agent_name: str | None = None,
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """Generate structured output parsed as a Pydantic model."""
        agent = await self._begin_operation("structured", agent_name)
        try:
            result = await agent.structured(messages, model, request_params)
            await self._save_persisted_history(agent)
            return result
        finally:
            await self._end_operation("structured")

    async def structured_schema(
        self,
        messages: MessageParam,
        schema: dict[str, Any],
        *,
        agent_name: str | None = None,
        request_params: RequestParams | None = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        """Generate structured JSON validated against a raw schema."""
        agent = await self._begin_operation("structured_schema", agent_name)
        try:
            result = await agent.structured_schema(messages, schema, request_params)
            await self._save_persisted_history(agent)
            return result
        finally:
            await self._end_operation("structured_schema")

    async def clear(
        self,
        *,
        agent_name: str | None = None,
        clear_prompts: bool = False,
    ) -> None:
        """Clear conversation state for the resolved target agent."""
        agent = await self._begin_operation("clear", agent_name)
        try:
            agent.clear(clear_prompts=clear_prompts)
            await self._save_persisted_history(agent)
        finally:
            await self._end_operation("clear")

    async def delete(self) -> None:
        """Delete this session and dispose its owned instance."""
        await self._manager.delete(self.id)

    async def _begin_operation(
        self,
        operation: str,
        agent_name: str | None,
    ) -> AgentProtocol:
        async with self._manager._lock:
            self._raise_if_closed()
            if self._record.active_operation is not None:
                raise RuntimeError(
                    f"Session '{self.id}' is already running {self._record.active_operation}; "
                    "start another session for parallel conversation branches."
                )
            agent = self._resolve_agent(agent_name)
            self._record.active_operation = operation
            return agent

    async def _end_operation(self, operation: str) -> None:
        async with self._manager._lock:
            if self._record.active_operation == operation:
                self._record.active_operation = None

    def _resolve_agent(self, agent_name: str | None) -> AgentProtocol:
        target_name = agent_name or self.default_agent_name
        resolved_name = self._record.instance.app.resolve_target_agent_name(target_name)
        if resolved_name is None:
            raise ValueError("No agents provided!")
        agent = self._record.instance.app.get_agent(resolved_name)
        if agent is None:
            raise ValueError(f"Agent '{resolved_name}' not found")
        return agent

    def _raise_if_closed(self) -> None:
        if self._record.closed:
            raise RuntimeError(f"Session '{self.id}' is closed.")

    async def _save_persisted_history(self, agent: AgentProtocol) -> None:
        persisted_session = self._record.persisted_session
        if persisted_session is None:
            return
        await persisted_session.save_history(
            agent,
            agent_registry=self._record.instance.agents,
        )


class HarnessSessions:
    """Manager for in-memory harness sessions."""

    def __init__(
        self,
        *,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        create_persisted_session: Callable[
            [str, AgentInstance, str | None],
            Awaitable[tuple[SessionManager, Session] | None],
        ]
        | None = None,
        delete_persisted_session: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._create_instance = create_instance
        self._dispose_instance = dispose_instance
        self._create_persisted_session = create_persisted_session
        self._delete_persisted_session = delete_persisted_session
        self._lock = asyncio.Lock()
        self._sessions: dict[str, HarnessSession] = {}

    async def get(self, session_id: str | None = None) -> HarnessSession:
        """Return an existing session."""
        normalized_id = _normalize_session_id(session_id)
        async with self._lock:
            try:
                return self._sessions[normalized_id]
            except KeyError as exc:
                raise KeyError(f"Session '{normalized_id}' not found") from exc

    async def create(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> HarnessSession:
        """Create a new session and its owned ``AgentInstance``."""
        normalized_id = _normalize_session_id(session_id)
        async with self._lock:
            if normalized_id in self._sessions:
                raise ValueError(f"Session '{normalized_id}' already exists")

            instance = await self._create_instance()
            record: _HarnessSessionRecord | None = None
            try:
                record = _HarnessSessionRecord(
                    session_id=normalized_id,
                    default_agent_name=agent_name,
                    instance=instance,
                )
                session = HarnessSession(self, record)
                if agent_name is not None:
                    session._resolve_agent(agent_name)

                persisted = None
                if self._create_persisted_session is not None:
                    persisted = await self._create_persisted_session(
                        normalized_id,
                        instance,
                        agent_name,
                    )
                manager, persisted_session = persisted if persisted is not None else (None, None)
                record.session_manager = manager
                record.persisted_session = persisted_session
            except Exception:
                if record is not None:
                    record.closed = True
                await self._dispose_instance(instance)
                raise

            self._sessions[normalized_id] = session
            return session

    async def get_or_create(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> HarnessSession:
        """Return an existing session or create one."""
        normalized_id = _normalize_session_id(session_id)
        async with self._lock:
            existing = self._sessions.get(normalized_id)
        if existing is not None:
            return existing
        try:
            return await self.create(normalized_id, agent_name=agent_name)
        except ValueError as exc:
            try:
                return await self.get(normalized_id)
            except KeyError:
                raise exc

    async def delete(self, session_id: str | None = None) -> None:
        """Delete a session if present; deleting a missing session is a no-op."""
        normalized_id = _normalize_session_id(session_id)
        async with self._lock:
            session = self._sessions.get(normalized_id)
            if session is None:
                return
            operation = session._record.active_operation
            if operation is not None:
                raise RuntimeError(
                    f"Session '{normalized_id}' is running {operation}; wait before deleting it."
                )
            del self._sessions[normalized_id]
            session._record.closed = True
            instance = session._record.instance

        await self._dispose_instance(instance)
        if self._delete_persisted_session is not None:
            await self._delete_persisted_session(normalized_id)

    async def _close_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                session._record.closed = True

        for session in sessions:
            with suppress(Exception):
                await self._dispose_instance(session._record.instance)


class AgentHarness:
    """Async context manager for headless fast-agent sessions."""

    def __init__(self, fast_agent: FastAgent, *, model: str | None = None) -> None:
        self._fast_agent = fast_agent
        self._model = model
        self._sessions: HarnessSessions | None = None
        self._runtime: RunRuntime | None = None
        self._settings: RunSettings | None = None
        self._app_context: Any | None = None
        self._span_context: Any | None = None

    @property
    def sessions(self) -> HarnessSessions:
        """Session manager for this harness."""
        if self._sessions is None:
            raise RuntimeError("Harness is not running.")
        return self._sessions

    async def __aenter__(self) -> AgentHarness:
        from opentelemetry import trace

        await self._fast_agent.app.initialize()
        self._settings = self._fast_agent._prepare_run_settings(
            model_override=self._model,
            force_headless=True,
        )
        self._span_context = trace.get_tracer(__name__).start_as_current_span(
            self._fast_agent.name
        )
        self._span_context.__enter__()
        self._app_context = self._fast_agent.app.run()

        try:
            await self._app_context.__aenter__()
            default_skills = self._fast_agent._load_default_skills_for_run()
            self._load_environment_agent_cards()
            self._fast_agent._apply_skills_to_agent_configs(default_skills)

            if self._settings.quiet_mode:
                self._fast_agent._configure_quiet_mode_for_run()

            self._fast_agent._validate_run_preconditions()
            self._runtime = self._fast_agent._create_run_runtime(self._settings)
            self._sessions = HarnessSessions(
                create_instance=self._create_instance,
                dispose_instance=self._dispose_instance,
                create_persisted_session=self._create_persisted_session,
                delete_persisted_session=self._delete_persisted_session,
            )
            return self
        except Exception:
            await self.__aexit__(*sys.exc_info())
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._sessions is not None:
            await self._sessions._close_all()
            self._sessions = None

        if self._runtime is not None:
            for instance in list(self._runtime.managed_instances):
                with suppress(Exception):
                    await self._dispose_instance(instance)
            self._runtime = None

        if self._settings is not None:
            await self._fast_agent._finalize_run(
                None,
                {},
                had_error=exc_type is not None,
                settings=self._settings,
            )
            self._settings = None

        if self._app_context is not None:
            await self._app_context.__aexit__(exc_type, exc, traceback)
            self._app_context = None

        if self._span_context is not None:
            self._span_context.__exit__(exc_type, exc, traceback)
            self._span_context = None

    async def session(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> HarnessSession:
        """Return an existing session or create it."""
        return await self.sessions.get_or_create(session_id, agent_name=agent_name)

    def _load_environment_agent_cards(self) -> None:
        from fast_agent.core.agent_card_paths import is_agent_card_path
        from fast_agent.paths import resolve_environment_paths

        settings = self._fast_agent.context.config
        if settings is None:
            return

        agent_cards_dir = resolve_environment_paths(settings).agent_cards
        if not agent_cards_dir.is_dir():
            return
        if not any(entry.is_file() and is_agent_card_path(entry) for entry in agent_cards_dir.iterdir()):
            return
        self._fast_agent.load_agents(agent_cards_dir)

    async def _create_persisted_session(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> tuple[SessionManager, Session] | None:
        settings = self._fast_agent.context.config
        if settings is None or settings._fast_agent_noenv or not settings.session_history:
            return None

        from fast_agent.session import SessionHydrator
        from fast_agent.session.session_manager import SessionManager

        manager = SessionManager(
            environment_override=settings.environment_dir,
        )
        persisted_session = manager.create_session_with_id(
            session_id,
            metadata={"harness_session_id": session_id},
            metadata_id_key="harness_session_id",
        )
        fallback_agent_name = instance.app.resolve_target_agent_name(default_agent_name)
        hydration = await SessionHydrator().hydrate_session(
            session=persisted_session,
            agents=instance.agents,
            fallback_agent_name=fallback_agent_name,
        )
        return manager, hydration.session

    async def _delete_persisted_session(self, session_id: str) -> None:
        settings = self._fast_agent.context.config
        if settings is None or settings._fast_agent_noenv or not settings.session_history:
            return

        from fast_agent.session.session_manager import SessionManager

        manager = SessionManager(
            environment_override=settings.environment_dir,
        )
        manager.delete_session(session_id)

    async def _create_instance(self) -> AgentInstance:
        if self._runtime is None:
            raise RuntimeError("Harness is not running.")
        settings = self._fast_agent.context.config
        original_session_history = settings.session_history if settings is not None else None
        if settings is not None:
            settings.session_history = False
        try:
            instance = await self._fast_agent._instantiate_agent_instance(self._runtime)
        finally:
            if settings is not None and original_session_history is not None:
                settings.session_history = original_session_history
        self._fast_agent._configure_runtime_mcp_callbacks(instance.app)
        self._fast_agent._configure_streaming_for_run(instance.agents)
        return instance

    async def _dispose_instance(self, instance: AgentInstance) -> None:
        if self._runtime is None:
            await instance.shutdown()
            return
        await self._fast_agent._dispose_agent_instance(self._runtime, instance)


def _normalize_session_id(session_id: str | None) -> str:
    if session_id is None:
        return "default"
    normalized_id = session_id.strip()
    if not normalized_id:
        raise ValueError("Session ID must not be empty")
    if HARNESS_SESSION_ID_PATTERN.fullmatch(normalized_id) is None:
        raise ValueError(
            f"Session ID must be 1-{HARNESS_SESSION_ID_MAX_LENGTH} characters, start and end "
            "with a letter or digit, and contain only letters, digits, dashes, or underscores"
        )
    return normalized_id


__all__ = [
    "AgentHarness",
    "HarnessSessions",
    "HarnessSession",
    "HARNESS_SESSION_ID_MAX_LENGTH",
    "HARNESS_SESSION_ID_PATTERN",
]
