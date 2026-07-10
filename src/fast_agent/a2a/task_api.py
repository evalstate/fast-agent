"""Request-local helpers for fast-agent A2A server tasks."""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING

from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus

if TYPE_CHECKING:
    from a2a.server.events.event_queue import EventQueue
    from a2a.server.tasks.task_updater import TaskUpdater


@dataclass(frozen=True, slots=True)
class A2ATaskHandle:
    """A2A task identity available while handling an A2A request."""

    task_id: str
    context_id: str


@dataclass(slots=True)
class _A2ATaskRuntime:
    updater: TaskUpdater
    event_queue: EventQueue
    request_message: Message
    task_started: bool = False
    returned_message: bool = False

    @property
    def handle(self) -> A2ATaskHandle:
        return A2ATaskHandle(
            task_id=self.updater.task_id,
            context_id=self.updater.context_id,
        )

    async def ensure_task_started(self) -> None:
        if self.task_started:
            return
        await self.event_queue.enqueue_event(
            Task(
                id=self.updater.task_id,
                context_id=self.updater.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[self.request_message],
            )
        )
        self.task_started = True


_current_a2a_task: ContextVar[_A2ATaskRuntime | None] = ContextVar(
    "fast_agent_current_a2a_task",
    default=None,
)


def current_task() -> A2ATaskHandle | None:
    """Return the current A2A task handle, if code is running inside an A2A request."""
    runtime = _current_a2a_task.get()
    return runtime.handle if runtime is not None else None


async def start_task(message: str = "fast-agent is working") -> A2ATaskHandle:
    """Publish an A2A working status update for the current request."""
    runtime = _require_a2a_task()
    await runtime.ensure_task_started()
    await runtime.updater.start_work(
        message=runtime.updater.new_agent_message(parts=[Part(text=message)])
    )
    return runtime.handle


async def return_artifact(
    text: str,
    *,
    name: str = "response",
    artifact_id: str | None = None,
    append: bool = False,
    last_chunk: bool = True,
) -> A2ATaskHandle:
    """Publish a text artifact update for the current A2A task."""
    runtime = _require_a2a_task()
    await runtime.ensure_task_started()
    await runtime.updater.add_artifact(
        parts=[Part(text=text)],
        artifact_id=artifact_id,
        name=name,
        append=append,
        last_chunk=last_chunk,
    )
    return runtime.handle


async def return_message(
    text: str,
    *,
    metadata: dict[str, object] | None = None,
) -> A2ATaskHandle:
    """Publish a standalone A2A agent message for the current request."""
    return await _return_message_parts([Part(text=text)], metadata=metadata)


async def _return_message_parts(
    parts: list[Part],
    *,
    metadata: dict[str, object] | None = None,
) -> A2ATaskHandle:
    """Publish standalone A2A agent message parts for the current request."""
    runtime = _require_a2a_task()
    if runtime.task_started:
        raise RuntimeError("Cannot return a standalone A2A message after starting a task.")
    runtime.returned_message = True
    await runtime.event_queue.enqueue_event(
        Message(
            role=Role.ROLE_AGENT,
            context_id=runtime.updater.context_id,
            message_id=str(uuid.uuid4()),
            metadata=metadata,
            parts=parts,
        )
    )
    return runtime.handle


async def _ensure_current_task_started() -> None:
    runtime = _require_a2a_task()
    await runtime.ensure_task_started()


def _current_task_returned_message() -> bool:
    runtime = _current_a2a_task.get()
    return runtime.returned_message if runtime is not None else False


def _current_task_started() -> bool:
    runtime = _current_a2a_task.get()
    return runtime.task_started if runtime is not None else False


def _set_current_task(
    updater: "TaskUpdater",
    *,
    event_queue: "EventQueue",
    request_message: Message,
) -> Token[_A2ATaskRuntime | None]:
    return _current_a2a_task.set(
        _A2ATaskRuntime(
            updater=updater,
            event_queue=event_queue,
            request_message=request_message,
        )
    )


def _reset_current_task(token: Token[_A2ATaskRuntime | None]) -> None:
    _current_a2a_task.reset(token)


def _require_a2a_task() -> _A2ATaskRuntime:
    runtime = _current_a2a_task.get()
    if runtime is None:
        raise RuntimeError("A2A task APIs are only available while handling an A2A request.")
    return runtime


__all__ = [
    "A2ATaskHandle",
    "current_task",
    "return_message",
    "return_artifact",
    "start_task",
]
