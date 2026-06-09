"""Request-local helpers for fast-agent A2A server tasks."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING

from a2a.types import Part

if TYPE_CHECKING:
    from a2a.server.tasks.task_updater import TaskUpdater


@dataclass(frozen=True, slots=True)
class A2ATaskHandle:
    """A2A task identity available while handling an A2A request."""

    task_id: str
    context_id: str


@dataclass(frozen=True, slots=True)
class _A2ATaskRuntime:
    updater: TaskUpdater

    @property
    def handle(self) -> A2ATaskHandle:
        return A2ATaskHandle(
            task_id=self.updater.task_id,
            context_id=self.updater.context_id,
        )


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
    await runtime.updater.add_artifact(
        parts=[Part(text=text)],
        artifact_id=artifact_id,
        name=name,
        append=append,
        last_chunk=last_chunk,
    )
    return runtime.handle


def _set_current_task(updater: "TaskUpdater") -> Token[_A2ATaskRuntime | None]:
    return _current_a2a_task.set(_A2ATaskRuntime(updater=updater))


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
    "return_artifact",
    "start_task",
]
