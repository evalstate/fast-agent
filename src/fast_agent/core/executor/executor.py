import asyncio
import contextvars
import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    TypeGuard,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, field_validator

from fast_agent.context_dependent import ContextDependent
from fast_agent.core.executor.workflow_signal import (
    AsyncioSignalHandler,
    Signal,
    SignalHandler,
    SignalValueT,
)
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.context import Context

logger = get_logger(__name__)

# Type variable for the return type of tasks
R = TypeVar("R")
ExecutorTask = Awaitable[R] | Callable[..., R]


def _is_awaitable_task(task: ExecutorTask[R]) -> TypeGuard[Awaitable[R]]:
    return inspect.isawaitable(task)


def _is_callable_task(task: ExecutorTask[R]) -> TypeGuard[Callable[..., R]]:
    return callable(task)


class ExecutorConfig(BaseModel):
    """Configuration for executors."""

    max_concurrent_activities: int | None = None  # Unbounded by default
    timeout_seconds: timedelta | None = None  # No timeout by default

    model_config = ConfigDict(extra="forbid")

    @field_validator("max_concurrent_activities")
    @classmethod
    def _validate_max_concurrent_activities(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_concurrent_activities must be greater than zero")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout_seconds(cls, value: timedelta | None) -> timedelta | None:
        if value is not None and value.total_seconds() <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        return value


class Executor(ABC, ContextDependent):
    """Abstract base class for different execution backends"""

    def __init__(
        self,
        engine: str,
        config: ExecutorConfig | None = None,
        signal_bus: SignalHandler | None = None,
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(context=context, **kwargs)
        del engine

        if config:
            self.config = config
        else:
            # TODO: saqadri - executor config should be loaded from settings
            # ctx = get_current_context()
            self.config = ExecutorConfig()

        self.signal_bus = signal_bus

    @abstractmethod
    async def execute(
        self,
        *tasks: ExecutorTask[R],
        **kwargs: Any,
    ) -> list[R | BaseException]:
        """Execute a list of tasks and return their results"""

    @abstractmethod
    def execute_streaming(
        self,
        *tasks: ExecutorTask[R],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        """Execute tasks and yield results as they complete"""

    async def map(
        self,
        func: Callable[..., R],
        inputs: list[Any],
        **kwargs: Any,
    ) -> list[R | BaseException]:
        """
        Run `func(item)` for each item in `inputs` with concurrency limit.
        """
        if inspect.iscoroutinefunction(func):
            tasks: list[ExecutorTask[R]] = [func(item, **kwargs) for item in inputs]
        else:
            tasks = [functools.partial(func, item, **kwargs) for item in inputs]
        return await self.execute(*tasks)

    async def signal(
        self,
        signal_name: str,
        payload: SignalValueT | None = None,
        signal_description: str | None = None,
    ) -> None:
        """
        Emit a signal.
        """
        if self.signal_bus is None:
            raise RuntimeError("No signal bus configured")
        sig: Signal[SignalValueT] = Signal(
            name=signal_name, payload=payload, description=signal_description
        )
        await self.signal_bus.signal(sig)

    async def wait_for_signal(
        self,
        signal_name: str,
        request_id: str | None = None,
        workflow_id: str | None = None,
        signal_description: str | None = None,
        timeout_seconds: int | None = None,
        signal_type: type[Any] | None = None,
    ) -> Any:
        """
        Wait until a signal with signal_name is emitted (or timeout).
        Return the signal's payload when triggered, or raise on timeout.
        """
        if self.signal_bus is None:
            raise RuntimeError("No signal bus configured")

        # Notify any callbacks that the workflow is about to be paused waiting for a signal
        if self.context.signal_notification:
            await self.context.signal_notification(
                signal_name=signal_name,
                request_id=request_id,
                workflow_id=workflow_id,
                metadata={
                    "description": signal_description,
                    "timeout_seconds": timeout_seconds,
                    "signal_type": signal_type or str,
                },
            )

        sig: Signal[Any] = Signal(
            name=signal_name, description=signal_description, workflow_id=workflow_id
        )
        return await self.signal_bus.wait_for_signal(sig)


class AsyncioExecutor(Executor):
    """Default executor using asyncio"""

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        signal_bus: SignalHandler | None = None,
    ) -> None:
        signal_bus = signal_bus or AsyncioSignalHandler()
        super().__init__(engine="asyncio", config=config, signal_bus=signal_bus)

        self._activity_semaphore: asyncio.Semaphore | None = None
        if self.config.max_concurrent_activities is not None:
            self._activity_semaphore = asyncio.Semaphore(self.config.max_concurrent_activities)

    @asynccontextmanager
    async def _activity_limit(self) -> AsyncIterator[None]:
        if self._activity_semaphore is None:
            yield
            return
        async with self._activity_semaphore:
            yield

    async def _run_awaitable(self, task: Awaitable[R], kwargs: dict[str, Any]) -> R:
        if kwargs:
            if inspect.iscoroutine(task):
                task.close()
            raise TypeError("Keyword arguments cannot be passed to awaitable tasks")
        return await task

    async def _run_callable(self, task: Callable[..., R], kwargs: dict[str, Any]) -> R:
        if inspect.iscoroutinefunction(task):
            raise TypeError(f"Pass coroutine objects, not coroutine functions: {task}")
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        fn = functools.partial(task, **kwargs) if kwargs else task
        result = await loop.run_in_executor(None, lambda: ctx.run(fn))
        if inspect.isawaitable(result):
            raise TypeError(
                "Executor callables must return plain values. "
                "Pass async work as an awaitable task instead."
            )
        return result

    async def _run_task_once(self, task: ExecutorTask[R], kwargs: dict[str, Any]) -> R:
        if _is_awaitable_task(task):
            return await self._run_awaitable(task, kwargs)
        if _is_callable_task(task):
            return await self._run_callable(task, kwargs)
        raise TypeError(f"Task must be an awaitable or callable: {task}")

    async def _execute_task(
        self,
        task: ExecutorTask[R],
        kwargs: dict[str, Any],
    ) -> R | BaseException:
        try:
            async with self._activity_limit():
                if self.config.timeout_seconds is None:
                    return await self._run_task_once(task, kwargs)

                timeout = self.config.timeout_seconds.total_seconds()
                async with asyncio.timeout(timeout):
                    return await self._run_task_once(task, kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return e

    async def execute(
        self,
        *tasks: ExecutorTask[R],
        **kwargs: Any,
    ) -> list[R | BaseException]:
        running = [asyncio.create_task(self._execute_task(task, kwargs)) for task in tasks]
        results: list[R | BaseException] = []

        try:
            results.extend([await task for task in running])
        except BaseException:
            for task in running:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise

        return results

    def execute_streaming(
        self,
        *tasks: ExecutorTask[R],
        **kwargs: Any,
    ) -> AsyncGenerator[R | BaseException, None]:
        async def stream() -> AsyncGenerator[R | BaseException, None]:
            pending = {asyncio.create_task(self._execute_task(task, kwargs)) for task in tasks}
            try:
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for future in done:
                        yield await future
            finally:
                for future in pending:
                    future.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

        return stream()

    async def signal(
        self,
        signal_name: str,
        payload: SignalValueT | None = None,
        signal_description: str | None = None,
    ) -> None:
        await super().signal(signal_name, payload, signal_description)

    async def wait_for_signal(
        self,
        signal_name: str,
        request_id: str | None = None,
        workflow_id: str | None = None,
        signal_description: str | None = None,
        timeout_seconds: int | None = None,
        signal_type: type[Any] | None = None,
    ) -> Any:
        return await super().wait_for_signal(
            signal_name,
            request_id,
            workflow_id,
            signal_description,
            timeout_seconds,
            signal_type,
        )
