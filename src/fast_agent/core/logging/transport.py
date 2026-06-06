"""
Transports for the Logger module for MCP Agent, including:
- Local + optional remote event transport
- Async event bus
"""

import asyncio
import json
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from contextlib import suppress
from pathlib import Path
from typing import Protocol

import aiohttp
from opentelemetry import trace
from rich import print
from rich.json import JSON
from rich.text import Text

from fast_agent.config import LoggerSettings
from fast_agent.core.logging.events import Event, EventFilter
from fast_agent.core.logging.json_serializer import JSONSerializer
from fast_agent.core.logging.listeners import EventListener, LifecycleAwareListener
from fast_agent.ui.console import console
from fast_agent.utils.async_utils import ensure_event_loop, gather_with_cancel


class EventTransport(Protocol):
    """
    Pluggable interface for sending events to a remote or external system
    (Kafka, RabbitMQ, REST, etc.).
    """

    async def send_event(self, event: Event) -> None:
        """
        Send an event to the external system.
        Args:
            event: Event to send.
        """
        ...


class FilteredEventTransport(EventTransport, ABC):
    """
    Event transport that filters events based on a filter before sending.
    """

    def __init__(self, event_filter: EventFilter | None = None) -> None:
        self.filter = event_filter

    async def send_event(self, event: Event) -> None:
        if not self.filter or self.filter.matches(event):
            await self.send_matched_event(event)

    @abstractmethod
    async def send_matched_event(self, event: Event):
        """Send an event to the external system."""


class NoOpTransport(FilteredEventTransport):
    """Default transport that does nothing (purely local)."""

    async def send_matched_event(self, event) -> None:
        """Do nothing."""


class ConsoleTransport(FilteredEventTransport):
    """Simple transport that prints events to console."""

    def __init__(self, event_filter: EventFilter | None = None) -> None:
        super().__init__(event_filter=event_filter)
        # Use shared console instances
        self._serializer = JSONSerializer()
        self.log_level_styles: dict[str, str] = {
            "info": "bold green",
            "debug": "dim white",
            "warning": "bold yellow",
            "error": "bold red",
        }

    async def send_matched_event(self, event: Event) -> None:
        # Map log levels to styles
        style = self.log_level_styles.get(event.type, "white")

        # Use the appropriate console based on event type
        #        output_console = error_console if event.type == "error" else console
        output_console = console

        # Create namespace without None
        namespace = event.namespace
        if event.name:
            namespace = f"{namespace}.{event.name}"

        log_text = Text.assemble(
            (f"[{event.type.upper()}] ", style),
            (f"{event.timestamp.replace(microsecond=0).isoformat()} ", "cyan"),
            (f"{namespace} ", "magenta"),
            (f"- {event.message}", "white"),
        )
        output_console.print(log_text)

        # Print additional data as JSON if available
        if event.data:
            serialized_data = self._serializer(event.data)
            output_console.print(JSON.from_data(serialized_data))


class FileTransport(FilteredEventTransport):
    """Transport that writes events to a file with proper formatting."""

    def __init__(
        self,
        filepath: str | Path,
        event_filter: EventFilter | None = None,
        mode: str = "a",
        encoding: str = "utf-8",
    ) -> None:
        """Initialize FileTransport.

        Args:
            filepath: Path to the log file. If relative, the current working directory will be used
            event_filter: Optional filter for events
            mode: File open mode ('a' for append, 'w' for write)
            encoding: File encoding to use
        """
        super().__init__(event_filter=event_filter)
        self.filepath = Path(filepath)
        self.mode = mode
        self.encoding = encoding
        self._serializer = JSONSerializer()

        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    async def send_matched_event(self, event: Event) -> None:
        """Write matched event to log file asynchronously.

        Args:
            event: Event to write to file
        """
        # Format the log entry
        namespace = event.namespace
        if event.name:
            namespace = f"{namespace}.{event.name}"

        log_entry = {
            "level": event.type.upper(),
            "timestamp": event.timestamp.isoformat(),
            "namespace": namespace,
            "message": event.message,
        }

        # Add event data if present
        if event.data:
            log_entry["data"] = self._serializer(event.data)

        try:
            with self.filepath.open(mode=self.mode, encoding=self.encoding) as f:
                # Write the log entry as compact JSON (JSONL format)
                f.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
                f.flush()  # Ensure writing to disk
        except IOError as e:
            # Log error without recursion
            print(f"Error writing to log file {self.filepath}: {e}")

    async def close(self) -> None:
        """Clean up resources if needed."""
        # File handles are automatically closed after each write.

    @property
    def is_closed(self) -> bool:
        """Check if transport is closed."""
        return False  # Since we open/close per write


class HTTPTransport(FilteredEventTransport):
    """
    Sends events to an HTTP endpoint in batches.
    Useful for sending to remote logging services like Elasticsearch, etc.
    """

    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        batch_size: int = 100,
        timeout: float = 5.0,
        event_filter: EventFilter | None = None,
    ) -> None:
        super().__init__(event_filter=event_filter)
        self.endpoint = endpoint
        self.headers = headers or {}
        self.batch_size = batch_size
        self.timeout = timeout

        self.batch: list[Event] = []
        self.lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._serializer = JSONSerializer()

    async def start(self) -> None:
        """Initialize HTTP session."""
        if not self._session:
            self._session = aiohttp.ClientSession(
                headers=self.headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    async def stop(self) -> None:
        """Close HTTP session and flush any remaining events."""
        if self.batch:
            await self._flush()
        if self._session:
            await self._session.close()
            self._session = None

    async def send_matched_event(self, event: Event) -> None:
        """Add event to batch, flush if batch is full."""
        async with self.lock:
            self.batch.append(event)
            if len(self.batch) >= self.batch_size:
                await self._flush()

    async def _flush(self) -> None:
        """Send batch of events to HTTP endpoint."""
        if not self.batch:
            return

        if not self._session:
            await self.start()
        assert self._session is not None

        try:
            # Convert events to JSON-serializable dicts
            events_data = [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.type,
                    "name": event.name,
                    "namespace": event.namespace,
                    "message": event.message,
                    "data": self._serializer(event.data),
                    "trace_id": event.trace_id,
                    "span_id": event.span_id,
                    "context": event.context.model_dump() if event.context else None,
                }
                for event in self.batch
            ]

            async with self._session.post(self.endpoint, json=events_data) as response:
                if response.status >= 400:
                    text = await response.text()
                    print(
                        f"Error sending log events to {self.endpoint}. "
                        f"Status: {response.status}, Response: {text}"
                    )
        except Exception as e:
            print(f"Error sending log events to {self.endpoint}: {e}")
        finally:
            self.batch.clear()


class AsyncEventBus:
    """
    Async event bus with local in-process listeners + optional remote transport.
    Also injects distributed tracing (trace_id, span_id) if there's a current span.
    """

    _instance = None

    def __init__(self, transport: EventTransport | None = None) -> None:
        self.transport: EventTransport = transport or NoOpTransport()
        self.listeners: dict[str, EventListener] = {}
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    @classmethod
    def get(cls, transport: EventTransport | None = None) -> "AsyncEventBus":
        """Get the singleton instance of the event bus."""
        if cls._instance is not None:
            task = cls._instance._task
            # Drop stale singleton instances whose worker task loop is gone.
            if task is not None:
                try:
                    if task.get_loop().is_closed():
                        cls._instance = None
                except Exception:
                    cls._instance = None

        if cls._instance is None:
            cls._instance = cls(transport=transport)
        elif transport is not None:
            # Update transport if provided
            cls._instance.transport = transport
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.
        This is primarily useful for testing scenarios where you need to ensure
        a clean state between tests.
        """
        if cls._instance:
            # Signal shutdown
            cls._instance._running = False

            # Best-effort task cancellation to avoid pending-task warnings in tests.
            task = cls._instance._task
            if task is not None and not task.done():
                with suppress(Exception):
                    task.cancel()

            # Clear the singleton instance
            cls._instance = None

    async def start(self) -> None:
        """Start the event bus and all lifecycle-aware listeners."""
        if self._running:
            return

        ensure_event_loop()

        self._queue = asyncio.Queue()

        # Start each lifecycle-aware listener
        for listener in self.listeners.values():
            if isinstance(listener, LifecycleAwareListener):
                await listener.start()

        # Start processing
        self._running = True
        self._task = asyncio.create_task(self._process_events())

    @staticmethod
    def _is_task_on_current_loop(task: asyncio.Task | None) -> bool:
        if task is None:
            return True
        try:
            return task.get_loop() is asyncio.get_running_loop()
        except Exception:
            return False

    async def stop(self) -> None:
        """Stop the event bus and all lifecycle-aware listeners."""
        if not self._running:
            await self._cancel_process_task()
            return

        # Signal processing to stop
        self._running = False

        same_loop_task = self._is_task_on_current_loop(self._task)
        await self._drain_queue_before_stop(same_loop_task)
        await self._cancel_process_task(same_loop_task=same_loop_task)
        await self._stop_lifecycle_listeners()

    async def _cancel_process_task(self, *, same_loop_task: bool | None = None) -> None:
        task = self._task
        if task is None:
            return

        if task.done():
            self._task = None
            return

        task.cancel()
        should_await = (
            self._is_task_on_current_loop(task) if same_loop_task is None else same_loop_task
        )
        if should_await:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Task was cancelled or timed out.
            except Exception as e:
                print(f"Error cancelling process task: {e}")
        self._task = None

    async def _drain_queue_before_stop(self, same_loop_task: bool) -> None:
        queue = self._queue
        if not same_loop_task or queue is None:
            self._queue = None
            return

        if not queue.empty():
            try:
                # Give some time for remaining items to be processed.
                await asyncio.wait_for(queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                self._discard_queued_events(queue)
            except Exception as e:
                print(f"Error during queue cleanup: {e}")

        self._queue = None

    @staticmethod
    def _discard_queued_events(queue: asyncio.Queue) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _stop_lifecycle_listeners(self) -> None:
        for listener in self.listeners.values():
            if isinstance(listener, LifecycleAwareListener):
                await self._stop_lifecycle_listener(listener)

    @staticmethod
    async def _stop_lifecycle_listener(listener: LifecycleAwareListener) -> None:
        try:
            await asyncio.wait_for(listener.stop(), timeout=3.0)
        except asyncio.TimeoutError:
            print(f"Timeout stopping listener: {listener}")
        except Exception as e:
            print(f"Error stopping listener: {e}")

    async def emit(self, event: Event) -> None:
        """Emit an event to all listeners and transport."""
        if not self._running:
            return

        # Inject current tracing info if available
        span = trace.get_current_span()
        if span.is_recording():
            ctx = span.get_span_context()
            event.trace_id = f"{ctx.trace_id:032x}"
            event.span_id = f"{ctx.span_id:016x}"

        # Forward to transport first (immediate processing)
        try:
            await self.transport.send_event(event)
        except Exception as e:
            print(f"Error in transport.send_event: {e}")

        # Then queue for listeners
        if self._queue is not None:
            try:
                await self._queue.put(event)
            except RuntimeError:
                # Event loop may be closing during shutdown/test teardown.
                return

    def add_listener(self, name: str, listener: EventListener) -> None:
        """Add a listener to the event bus."""
        self.listeners[name] = listener

    def remove_listener(self, name: str) -> None:
        """Remove a listener from the event bus."""
        self.listeners.pop(name, None)

    async def _process_events(self) -> None:
        """Process events from the queue until stopped."""
        while self._running:
            event: Event | None = None
            try:
                # Use wait_for with a timeout to allow checking running state
                event = await self._next_event()
                if event is None:
                    continue

                # Process the event through all listeners
                await self._dispatch_event_to_listeners(event)

                # Mark the event as processed so queue.join() can complete
                self._mark_event_done(event)

            except asyncio.CancelledError:
                # TODO -- added _queue assertion; is that necessary?
                self._mark_event_done(event)
                raise
            except Exception as e:
                print(f"Error in event processing loop: {e}")
                # Mark task done for this event
                self._mark_event_done(event)

        # Process remaining events in queue
        await self._drain_remaining_events()

    async def _next_event(self) -> Event | None:
        queue = self._queue
        if queue is None:
            await asyncio.sleep(0)
            return None
        try:
            return await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    async def _dispatch_event_to_listeners(self, event: Event) -> None:
        tasks = self._listener_tasks(event)
        if not tasks:
            return

        results = await gather_with_cancel(tasks)
        for result in results:
            if isinstance(result, Exception):
                self._print_listener_error(result)

    def _listener_tasks(self, event: Event) -> list[Awaitable[None]]:
        tasks: list[Awaitable[None]] = []
        for listener in self.listeners.values():
            try:
                tasks.append(listener.handle_event(event))
            except Exception as e:
                print(f"Error creating listener task: {e}")
        return tasks

    @staticmethod
    def _print_listener_error(error: Exception) -> None:
        print(f"Error in listener: {error}")
        print(
            "Stacktrace: "
            f"{''.join(traceback.format_exception(type(error), error, error.__traceback__))}"
        )

    def _mark_event_done(self, event: Event | None) -> None:
        if event is not None and self._queue is not None:
            self._queue.task_done()

    async def _drain_remaining_events(self) -> None:
        queue = self._queue
        if queue is None:
            return

        while not queue.empty():
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            with suppress(Exception):
                await self._dispatch_event_to_listeners(event)
            queue.task_done()


def create_transport(
    settings: LoggerSettings, event_filter: EventFilter | None = None
) -> EventTransport:
    """Create event transport based on settings."""
    if settings.type == "none":
        return NoOpTransport(event_filter=event_filter)
    if settings.type == "console":
        return ConsoleTransport(event_filter=event_filter)
    if settings.type == "file":
        if not settings.path:
            raise ValueError("File path required for file transport")
        return FileTransport(
            filepath=settings.path,
            event_filter=event_filter,
        )
    if settings.type == "http":
        if not settings.http_endpoint:
            raise ValueError("HTTP endpoint required for HTTP transport")
        return HTTPTransport(
            endpoint=settings.http_endpoint,
            headers=settings.http_headers,
            batch_size=settings.batch_size,
            timeout=settings.http_timeout,
            event_filter=event_filter,
        )
    raise ValueError(f"Unsupported transport type: {settings.type}")
