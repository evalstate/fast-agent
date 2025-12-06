"""
Logger module for the MCP Agent, which provides:
- Local + optional remote event transport
- Async event bus
- OpenTelemetry tracing decorators (for distributed tracing)
- Automatic injection of trace_id/span_id into events
- Developer-friendly Logger that can be used anywhere
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from fast_agent.core.logging.events import Event, EventContext, EventFilter, EventType
from fast_agent.core.logging.listeners import (
    BatchingListener,
    LoggingListener,
    ProgressListener,
)
from fast_agent.core.logging.transport import AsyncEventBus, EventTransport


class Logger:
    """
    Developer-friendly logger that sends events to the AsyncEventBus.
    - `type` is a broad category (INFO, ERROR, etc.).
    - `name` can be a custom domain-specific event name, e.g. "ORDER_PLACED".
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.event_bus = AsyncEventBus.get()

    def _ensure_event_loop(self):
        """Ensure we have an event loop we can use."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            # If no loop is running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _emit_event(self, event: Event) -> None:
        """Emit an event by running it in the event loop."""
        loop = self._ensure_event_loop()
        if loop.is_running():
            # If we're in a thread with a running loop, schedule the coroutine
            asyncio.create_task(self.event_bus.emit(event))
        else:
            # If no loop is running, run it until the emit completes
            loop.run_until_complete(self.event_bus.emit(event))

    def event(
        self,
        etype: EventType,
        ename: str | None,
        message: str,
        context: EventContext | None,
        data: dict,
    ) -> None:
        """Create and emit an event."""
        evt = Event(
            type=etype,
            name=ename,
            namespace=self.namespace,
            message=message,
            context=context,
            data=data,
        )
        self._emit_event(evt)

    def debug(
        self,
        message: str,
        name: str | None = None,
        context: EventContext | None = None,
        **data,
    ) -> None:
        """Log a debug message."""
        self.event("debug", name, message, context, data)

    def info(
        self,
        message: str,
        name: str | None = None,
        context: EventContext | None = None,
        **data,
    ) -> None:
        """Log an info message."""
        self.event("info", name, message, context, data)

    def warning(
        self,
        message: str,
        name: str | None = None,
        context: EventContext | None = None,
        **data,
    ) -> None:
        """Log a warning message."""
        self.event("warning", name, message, context, data)

    def error(
        self,
        message: str,
        name: str | None = None,
        context: EventContext | None = None,
        **data,
    ) -> None:
        """Log an error message."""
        self.event("error", name, message, context, data)

    def progress(
        self,
        message: str,
        name: str | None = None,
        percentage: float | None = None,
        context: EventContext | None = None,
        **data,
    ) -> None:
        """Log a progress message."""
        merged_data = dict(percentage=percentage, **data)
        self.event("progress", name, message, context, merged_data)


@contextmanager
def event_context(
    logger: Logger,
    message: str,
    event_type: EventType = "info",
    name: str | None = None,
    **data,
):
    """
    Times a synchronous block, logs an event after completion.
    Because logger methods are async, we schedule the final log.
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time

        logger.event(
            event_type,
            name,
            f"{message} finished in {duration:.3f}s",
            None,
            {"duration": duration, **data},
        )


# TODO: saqadri - check if we need this
@asynccontextmanager
async def async_event_context(
    logger: Logger,
    message: str,
    event_type: EventType = "info",
    name: str | None = None,
    **data,
):
    """
    Times an asynchronous block, logs an event after completion.
    Because logger methods are async, we schedule the final log.
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.event(
            event_type,
            name,
            f"{message} finished in {duration:.3f}s",
            None,
            {"duration": duration, **data},
        )


class LoggingConfig:
    """Global configuration for the logging system."""

    _initialized = False

    @classmethod
    async def configure(
        cls,
        event_filter: EventFilter | None = None,
        transport: EventTransport | None = None,
        batch_size: int = 100,
        flush_interval: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """
        Configure the logging system.

        Args:
            event_filter: Default filter for all loggers
            transport: Transport for sending events to external systems
            batch_size: Default batch size for batching listener
            flush_interval: Default flush interval for batching listener
            **kwargs: Additional configuration options
        """
        if cls._initialized:
            return

        # Suppress boto3/botocore logging to prevent flooding
        logging.getLogger("boto3").setLevel(logging.WARNING)
        logging.getLogger("botocore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("s3transfer").setLevel(logging.WARNING)

        bus = AsyncEventBus.get(transport=transport)

        # Add standard listeners
        if "logging" not in bus.listeners:
            bus.add_listener("logging", LoggingListener(event_filter=event_filter))

        # Only add progress listener if enabled in settings
        if "progress" not in bus.listeners and kwargs.get("progress_display", True):
            bus.add_listener("progress", ProgressListener())

        if "batching" not in bus.listeners:
            bus.add_listener(
                "batching",
                BatchingListener(
                    event_filter=event_filter,
                    batch_size=batch_size,
                    flush_interval=flush_interval,
                ),
            )

        await bus.start()
        cls._initialized = True

    @classmethod
    async def shutdown(cls) -> None:
        """Shutdown the logging system gracefully."""
        if not cls._initialized:
            return
        bus = AsyncEventBus.get()
        await bus.stop()
        cls._initialized = False

    @classmethod
    @asynccontextmanager
    async def managed(cls, **config_kwargs):
        """Context manager for the logging system lifecycle."""
        try:
            await cls.configure(**config_kwargs)
            yield
        finally:
            await cls.shutdown()


_logger_lock = threading.Lock()
_loggers: dict[str, Logger] = {}


def get_logger(namespace: str) -> Logger:
    """
    Get a logger instance for a given namespace.
    Creates a new logger if one doesn't exist for this namespace.

    Args:
        namespace: The namespace for the logger (e.g. "agent.helper", "workflow.demo")

    Returns:
        A Logger instance for the given namespace
    """

    with _logger_lock:
        if namespace not in _loggers:
            _loggers[namespace] = Logger(namespace)
        return _loggers[namespace]
