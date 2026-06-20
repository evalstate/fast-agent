"""Headless, session-oriented harness API for fast-agent."""

from __future__ import annotations

import re
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union

from mcp.types import PromptMessage
from pydantic import BaseModel

from fast_agent.core.agent_instance_factory import (
    AgentInstanceFactory,
    CallableAgentInstanceFactory,
)
from fast_agent.core.harness_persistence import (
    CallbackHarnessSessionPersistence,
    FileHarnessSessionPersistence,
    HarnessSessionPersistence,
)
from fast_agent.core.live_session_registry import InMemoryLiveSessionRegistry
from fast_agent.core.run_lifecycle import FastAgentRunLifecycle, FastAgentRunLifecycleState
from fast_agent.types import (
    AgentAuth,
    AgentRequest,
    AgentResponse,
    PromptMessageExtended,
    RequestParams,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from fast_agent.a2a.task_api import A2ATaskHandle
    from fast_agent.config import CompactionSettings
    from fast_agent.core.fastagent import AgentInstance, FastAgent, RunRuntime, RunSettings
    from fast_agent.history.compaction import CompactionResult
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session.session_manager import Session, SessionManager
    from fast_agent.tools.session_environment import ShellExecutionResult, ShellExecutor

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
    session: HarnessSession | None = None
    persistence_handle: object | None = None
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

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        """Invoke the session's target agent using an AgentRequest envelope."""
        with _agent_auth_context(request.auth):
            message = await self.generate(
                request.message,
                agent_name=request.agent,
                request_params=request.params,
            )
        return AgentResponse(message=message, metadata=dict(request.metadata))

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

    async def compact(
        self,
        *,
        agent_name: str | None = None,
        instructions: str | None = None,
    ) -> "CompactionResult":
        """Compact the resolved agent's history into a checkpoint summary.

        Mirrors the ``/compact`` command: summarizes older turns via the agent's
        own model, keeps recent turns verbatim, and persists the compacted
        history. Honors ``compaction.keep_turns`` and ``compaction.prompt`` from
        config. ``instructions`` adds one-off focus for this summary.

        Raises ``CompactionSkipped`` when there is nothing worth compacting and
        ``CompactionError`` when the summarization call fails (history is left
        untouched in both cases).
        """
        from typing import cast

        from fast_agent.history.compaction import (
            CompactableAgent,
            compact_conversation,
        )

        agent = await self._begin_operation("compact", agent_name)
        try:
            result = await compact_conversation(
                cast("CompactableAgent", agent),
                settings=self._compaction_settings(agent),
                instructions=instructions,
            )
            await self._save_persisted_history(agent)
            return result
        finally:
            await self._end_operation("compact")

    async def shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        """Run a shell command for this session without adding it to chat history."""
        await self._begin_session_operation("shell")
        try:
            return await self._manager._execute_shell(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
        finally:
            await self._end_operation("shell")

    async def delete(self) -> None:
        """Delete this session and dispose its owned instance."""
        await self._manager.delete(self.id)

    async def _begin_operation(
        self,
        operation: str,
        agent_name: str | None,
    ) -> AgentProtocol:
        await self._begin_session_operation(operation)
        try:
            return self._resolve_agent(agent_name)
        except Exception:
            await self._end_operation(operation)
            raise

    async def _begin_session_operation(self, operation: str) -> None:
        async with self._manager._lock:
            self._raise_if_closed()
            if self._record.active_operation is not None:
                raise RuntimeError(
                    f"Session '{self.id}' is already running {self._record.active_operation}; "
                    "start another session for parallel conversation branches."
                )
            self._record.active_operation = operation

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

    def _compaction_settings(self, agent: AgentProtocol) -> "CompactionSettings":
        from fast_agent.config import CompactionSettings

        context = getattr(agent, "context", None)
        config = context.config if context is not None else None
        if config is not None and config.compaction is not None:
            return config.compaction
        return CompactionSettings()

    async def _save_persisted_history(self, agent: AgentProtocol) -> None:
        persistence_handle = self._record.persistence_handle
        persistence = self._manager._persistence
        if persistence_handle is None or persistence is None:
            return
        await persistence.save(
            persistence_handle,
            agent,
            agent_registry=self._record.instance.agents,
        )


class HarnessSessions:
    """Manager for in-memory harness sessions."""

    def __init__(
        self,
        *,
        instance_factory: AgentInstanceFactory | None = None,
        create_instance: Callable[[], Awaitable[AgentInstance]] | None = None,
        dispose_instance: Callable[[AgentInstance], Awaitable[None]] | None = None,
        persistence: HarnessSessionPersistence | None = None,
        create_persisted_session: Callable[
            [str, AgentInstance, str | None],
            Awaitable[tuple[SessionManager, Session] | None],
        ]
        | None = None,
        delete_persisted_session: Callable[[str], Awaitable[None]] | None = None,
        shell_executor: ShellExecutor | None = None,
    ) -> None:
        instance_factory = _resolve_instance_factory(
            instance_factory=instance_factory,
            create_instance=create_instance,
            dispose_instance=dispose_instance,
        )
        self._persistence = _resolve_persistence(
            persistence=persistence,
            create_persisted_session=create_persisted_session,
            delete_persisted_session=delete_persisted_session,
        )
        self._shell_executor = shell_executor
        self._registry: InMemoryLiveSessionRegistry[_HarnessSessionRecord, str | None] = (
            InMemoryLiveSessionRegistry(
                instance_factory=instance_factory,
                create_record=self._create_record,
                record_instance=lambda record: record.instance,
                close_record=self._close_record,
            )
        )
        self._lock = self._registry.lock

    async def get(self, session_id: str | None = None) -> HarnessSession:
        """Return an existing session."""
        normalized_id = _normalize_session_id(session_id)
        record = await self._registry.get(normalized_id)
        return _session_from_record(record)

    async def create(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> HarnessSession:
        """Create a new session and its owned ``AgentInstance``."""
        normalized_id = _normalize_session_id(session_id)
        record = await self._registry.create(
            normalized_id,
            context=agent_name,
        )
        return _session_from_record(record)

    async def get_or_create(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> HarnessSession:
        """Return an existing session or create one."""
        normalized_id = _normalize_session_id(session_id)
        record = await self._registry.get_or_create(
            normalized_id,
            context=agent_name,
        )
        return _session_from_record(record)

    async def delete(self, session_id: str | None = None) -> None:
        """Delete a session if present; deleting a missing session is a no-op."""
        normalized_id = _normalize_session_id(session_id)
        record = await self._registry.delete(
            normalized_id,
            before_delete=self._raise_if_active,
        )
        if record is None:
            return
        if self._persistence is not None:
            await self._persistence.delete(normalized_id)

    async def _execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        timeout: float | None,
    ) -> ShellExecutionResult:
        if self._shell_executor is None:
            raise RuntimeError("Harness shell executor is not configured.")
        return await self._shell_executor.execute_shell(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    async def _close_all(self) -> None:
        await self._registry.close_all()

    async def _create_record(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> _HarnessSessionRecord:
        record = _HarnessSessionRecord(
            session_id=session_id,
            default_agent_name=default_agent_name,
            instance=instance,
        )
        session = HarnessSession(self, record)
        record.session = session
        try:
            if default_agent_name is not None:
                session._resolve_agent(default_agent_name)

            if self._persistence is not None:
                record.persistence_handle = await self._persistence.create_or_load(
                    session_id,
                    instance,
                    default_agent_name,
                )
            return record
        except Exception:
            record.closed = True
            raise

    @staticmethod
    def _close_record(record: _HarnessSessionRecord) -> None:
        record.closed = True

    @staticmethod
    def _raise_if_active(record: _HarnessSessionRecord) -> None:
        operation = record.active_operation
        if operation is not None:
            raise RuntimeError(
                f"Session '{record.session_id}' is running {operation}; wait before deleting it."
            )


class AgentHarness:
    """Async context manager for headless fast-agent sessions."""

    def __init__(self, fast_agent: FastAgent, *, model: str | None = None) -> None:
        self._fast_agent = fast_agent
        self._model = model
        self._sessions: HarnessSessions | None = None
        self._runtime: RunRuntime | None = None
        self._settings: RunSettings | None = None
        self._lifecycle: FastAgentRunLifecycle | None = None
        self._lifecycle_state: FastAgentRunLifecycleState | None = None
        self._shell_executor: ShellExecutor | None = None

    @property
    def sessions(self) -> HarnessSessions:
        """Session manager for this harness."""
        if self._sessions is None:
            raise RuntimeError("Harness is not running.")
        return self._sessions

    async def __aenter__(self) -> AgentHarness:
        self._lifecycle = FastAgentRunLifecycle(self._fast_agent)
        try:
            self._lifecycle_state = await self._lifecycle.enter(
                model_override=self._model,
                force_headless=True,
                before_apply_skills=self._load_environment_agent_cards,
            )
            self._settings = self._lifecycle_state.settings
            self._runtime = self._lifecycle_state.runtime
            self._shell_executor = self._runtime.shell_executor
            self._sessions = HarnessSessions(
                instance_factory=CallableAgentInstanceFactory(
                    create=self._create_instance,
                    dispose=self._dispose_instance,
                ),
                persistence=self._harness_persistence(),
                shell_executor=self._shell_executor,
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
            self._shell_executor = None

        if self._lifecycle is not None and self._lifecycle_state is not None:
            await self._lifecycle.exit(
                self._lifecycle_state,
                None,
                {},
                had_error=exc_type is not None,
                exc_type=exc_type,
                exc=exc,
                traceback=traceback,
            )
            self._lifecycle_state = None
            self._lifecycle = None
            self._settings = None

    async def session(
        self,
        session_id: str | None = None,
        *,
        agent_name: str | None = None,
    ) -> HarnessSession:
        """Return an existing session or create it."""
        return await self.sessions.get_or_create(session_id, agent_name=agent_name)

    @contextmanager
    def request_context(
        self,
        request: AgentRequest | None = None,
        *,
        auth: AgentAuth | None = None,
        bearer_token: str | None = None,
    ):
        """Set request-scoped auth for provider and MCP pass-through calls."""
        resolved_auth = auth if auth is not None else request.auth if request is not None else None
        if bearer_token is not None:
            resolved_auth = AgentAuth.bearer(bearer_token)
        with _agent_auth_context(resolved_auth):
            yield

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        """Invoke an agent through the harness session manager."""
        session = await self.session(request.session_id, agent_name=request.agent)
        return await session.invoke(request)

    async def shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        """Run a shell command and return structured stdout/stderr/exit code."""
        if self._shell_executor is None:
            raise RuntimeError("Harness is not running.")
        return await self._shell_executor.execute_shell(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    async def start_task(self, message: str = "fast-agent is working") -> A2ATaskHandle:
        """Publish an A2A task working update when the harness is used inside A2A serving."""
        from fast_agent.a2a.task_api import start_task

        return await start_task(message)

    async def return_artifact(
        self,
        text: str,
        *,
        name: str = "response",
        artifact_id: str | None = None,
        append: bool = False,
        last_chunk: bool = True,
    ) -> A2ATaskHandle:
        """Publish an A2A task artifact when the harness is used inside A2A serving."""
        from fast_agent.a2a.task_api import return_artifact

        return await return_artifact(
            text,
            name=name,
            artifact_id=artifact_id,
            append=append,
            last_chunk=last_chunk,
        )

    def _load_environment_agent_cards(self) -> None:
        from fast_agent.core.agent_card_paths import is_agent_card_path
        from fast_agent.paths import resolve_environment_paths

        settings = self._fast_agent.context.config
        if settings is None:
            return

        agent_cards_dir = resolve_environment_paths(settings).agent_cards
        if not agent_cards_dir.is_dir():
            return
        if not any(
            entry.is_file() and is_agent_card_path(entry) for entry in agent_cards_dir.iterdir()
        ):
            return
        self._fast_agent.load_agents(agent_cards_dir)

    def _harness_persistence(self) -> HarnessSessionPersistence | None:
        settings = self._fast_agent.context.config
        if settings is None or settings._fast_agent_noenv or not settings.session_history:
            return None
        return FileHarnessSessionPersistence(settings.environment_dir)

    async def _create_persisted_session(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> tuple[SessionManager, Session] | None:
        settings = self._fast_agent.context.config
        if settings is None or settings._fast_agent_noenv or not settings.session_history:
            return None

        from fast_agent.session.session_manager import SessionManager

        manager = SessionManager(
            environment_override=settings.environment_dir,
        )
        persisted_session = manager.create_session_with_id(
            session_id,
            metadata={"harness_session_id": session_id},
            metadata_id_key="harness_session_id",
        )

        from fast_agent.session import SessionHydrator

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


def _resolve_instance_factory(
    *,
    instance_factory: AgentInstanceFactory | None,
    create_instance: Callable[[], Awaitable[AgentInstance]] | None,
    dispose_instance: Callable[[AgentInstance], Awaitable[None]] | None,
) -> AgentInstanceFactory:
    if instance_factory is not None:
        if create_instance is not None or dispose_instance is not None:
            raise ValueError(
                "Pass either instance_factory or create_instance/dispose_instance, not both"
            )
        return instance_factory

    if create_instance is None or dispose_instance is None:
        raise ValueError("create_instance and dispose_instance are required")

    return CallableAgentInstanceFactory(
        create=create_instance,
        dispose=dispose_instance,
    )


def _session_from_record(record: _HarnessSessionRecord) -> HarnessSession:
    session = record.session
    if session is None:
        raise RuntimeError(f"Session '{record.session_id}' is missing its facade")
    return session


def _resolve_persistence(
    *,
    persistence: HarnessSessionPersistence | None,
    create_persisted_session: Callable[
        [str, AgentInstance, str | None],
        Awaitable[tuple[SessionManager, Session] | None],
    ]
    | None,
    delete_persisted_session: Callable[[str], Awaitable[None]] | None,
) -> HarnessSessionPersistence | None:
    if persistence is not None:
        if create_persisted_session is not None or delete_persisted_session is not None:
            raise ValueError(
                "Pass either persistence or create_persisted_session/delete_persisted_session, not both"
            )
        return persistence

    if create_persisted_session is None:
        if delete_persisted_session is not None:
            raise ValueError("delete_persisted_session requires create_persisted_session")
        return None

    return CallbackHarnessSessionPersistence(
        create_persisted_session=create_persisted_session,
        delete_persisted_session=delete_persisted_session,
    )


@contextmanager
def _agent_auth_context(auth: AgentAuth | None):
    from fast_agent.mcp.auth.context import request_bearer_token

    if auth is None:
        yield
        return

    saved_token = request_bearer_token.set(auth.token)
    try:
        yield
    finally:
        request_bearer_token.reset(saved_token)


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
