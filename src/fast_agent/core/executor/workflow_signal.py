import asyncio
import inspect
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from fast_agent.utils.async_utils import gather_with_cancel

SignalValueT = TypeVar("SignalValueT")


class Signal(BaseModel, Generic[SignalValueT]):
    """Represents a signal that can be sent to a workflow."""

    name: str
    description: str | None = "Workflow Signal"
    payload: SignalValueT | None = None
    metadata: dict[str, Any] | None = None
    workflow_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SignalRegistration(BaseModel):
    """Tracks registration of a signal handler."""

    signal_name: str
    unique_name: str
    workflow_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SignalHandler(Protocol, Generic[SignalValueT]):
    """Protocol for handling signals."""

    @abstractmethod
    async def signal(self, signal: Signal[SignalValueT]) -> None:
        """Emit a signal to all waiting handlers and registered callbacks."""

    @abstractmethod
    async def wait_for_signal(
        self,
        signal: Signal[SignalValueT],
        timeout_seconds: int | None = None,
    ) -> SignalValueT:
        """Wait for a signal to be emitted."""

    def on_signal(self, signal_name: str) -> Callable:
        """
        Decorator to register a handler for a signal.

        Example:
            @signal_handler.on_signal("approval_needed")
            async def handle_approval(value: str):
                print(f"Got approval signal with value: {value}")
        """


class PendingSignal(BaseModel, Generic[SignalValueT]):
    """Tracks a waiting signal handler and its event."""

    registration: SignalRegistration
    event: asyncio.Event | None = None
    value: SignalValueT | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseSignalHandler(ABC, Generic[SignalValueT]):
    """Base class implementing common signal handling functionality."""

    def __init__(self) -> None:
        # Map signal_name -> list of PendingSignal objects
        self._pending_signals: dict[str, list[PendingSignal]] = {}
        # Map signal_name -> list of (unique_name, handler) tuples
        self._handlers: dict[str, list[tuple[str, Callable]]] = {}
        self._lock = asyncio.Lock()

    async def cleanup(self, signal_name: str | None = None) -> None:
        """Clean up handlers and registrations for a signal or all signals."""
        async with self._lock:
            if signal_name:
                if signal_name in self._handlers:
                    del self._handlers[signal_name]
                if signal_name in self._pending_signals:
                    del self._pending_signals[signal_name]
            else:
                self._handlers.clear()
                self._pending_signals.clear()

    def on_signal(self, signal_name: str) -> Callable:
        """Register a handler for a signal."""

        def decorator(func: Callable) -> Callable:
            unique_name = f"{signal_name}_{uuid.uuid4()}"

            async def wrapped(value: SignalValueT) -> None:
                try:
                    if inspect.iscoroutinefunction(func):
                        await func(value)
                    else:
                        func(value)
                except Exception as e:
                    # Log the error but don't fail the entire signal handling
                    print(f"Error in signal handler {signal_name}: {e!s}")

            self._handlers.setdefault(signal_name, []).append((unique_name, wrapped))
            return wrapped

        return decorator

    @abstractmethod
    async def signal(self, signal: Signal[SignalValueT]) -> None:
        """Emit a signal to all waiting handlers and registered callbacks."""

    @abstractmethod
    async def wait_for_signal(
        self,
        signal: Signal[SignalValueT],
        timeout_seconds: int | None = None,
    ) -> SignalValueT:
        """Wait for a signal to be emitted."""


class AsyncioSignalHandler(BaseSignalHandler[SignalValueT]):
    """
    Asyncio-based signal handling using an internal dictionary of asyncio Events.
    """

    async def wait_for_signal(self, signal, timeout_seconds: int | None = None) -> SignalValueT:
        event = asyncio.Event()
        unique_name = str(uuid.uuid4())

        registration = SignalRegistration(
            signal_name=signal.name,
            unique_name=unique_name,
            workflow_id=signal.workflow_id,
        )

        pending_signal: PendingSignal[SignalValueT] = PendingSignal(
            registration=registration, event=event
        )

        async with self._lock:
            # Add to pending signals
            self._pending_signals.setdefault(signal.name, []).append(pending_signal)

        try:
            # Wait for signal
            if timeout_seconds is not None:
                await asyncio.wait_for(event.wait(), timeout_seconds)
            else:
                await event.wait()

            # After event is set, value should be populated
            if pending_signal.value is None:
                raise ValueError(f"Signal {signal.name} was received but value is None")
            return pending_signal.value
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Timeout waiting for signal {signal.name}") from e
        finally:
            async with self._lock:
                # Remove from pending signals
                if signal.name in self._pending_signals:
                    self._pending_signals[signal.name] = [
                        ps
                        for ps in self._pending_signals[signal.name]
                        if ps.registration.unique_name != unique_name
                    ]
                    if not self._pending_signals[signal.name]:
                        del self._pending_signals[signal.name]

    def on_signal(self, signal_name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            unique_name = f"{signal_name}_{uuid.uuid4()}"

            async def wrapped(value: SignalValueT) -> None:
                if inspect.iscoroutinefunction(func):
                    await func(value)
                else:
                    func(value)

            self._handlers.setdefault(signal_name, []).append((unique_name, wrapped))
            return wrapped

        return decorator

    async def signal(self, signal) -> None:
        async with self._lock:
            # Notify any waiting coroutines
            if signal.name in self._pending_signals:
                pending = self._pending_signals[signal.name]
                for ps in pending:
                    ps.value = signal.payload
                    if ps.event is not None:
                        ps.event.set()

        # Notify any registered handler functions
        tasks = []
        handlers = self._handlers.get(signal.name, [])
        for _, handler in handlers:
            tasks.append(handler(signal))

        await gather_with_cancel(tasks)


class SignalWaitCallback(Protocol):
    """Protocol for callbacks that are triggered when a workflow pauses waiting for a given signal."""

    async def __call__(
        self,
        signal_name: str,
        request_id: str | None = None,
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Receive a notification that a workflow is pausing on a signal.

        Args:
            signal_name: The name of the signal the workflow is pausing on.
            workflow_id: The ID of the workflow that is pausing (if using a workflow engine).
            metadata: Additional metadata about the signal.
        """
        ...
