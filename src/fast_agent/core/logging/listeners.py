"""
Listeners for the logger module of MCP Agent.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from fast_agent.event_progress import ProgressEvent

from fast_agent.core.logging.events import Event, EventFilter, EventType


def convert_log_event(event: Event) -> "ProgressEvent | None":
    """Convert a log event to a progress event if applicable."""

    # Import at runtime to avoid circular imports
    from fast_agent.event_progress import ProgressAction, ProgressEvent

    # Check to see if there is any additional data
    if not event.data:
        return None

    event_data = event.data.get("data")
    if not isinstance(event_data, dict):
        return None

    progress_action = event_data.get("progress_action")
    if not progress_action:
        return None

    # Build target string based on the event type.
    # Progress display is currently [time] [event] --- [target] [details]
    namespace = event.namespace
    agent_name = event_data.get("agent_name")
    target = agent_name
    details = ""
    if progress_action == ProgressAction.FATAL_ERROR:
        details = event_data.get("error_message", "An error occurred")
    elif "mcp_aggregator" in namespace:
        server_name = event_data.get("server_name", "")
        tool_name = event_data.get("tool_name")
        if tool_name:
            # fetch(fetch)
            details = f"{server_name} ({tool_name})"
        else:
            details = f"{server_name}"

        # For TOOL_PROGRESS, use progress message if available, otherwise keep default
        if progress_action == ProgressAction.TOOL_PROGRESS:
            progress_message = event_data.get("details", "")
            if progress_message:  # Only override if message is non-empty
                details = progress_message

    # TODO: there must be a better way :D?!
    elif "llm" in namespace:
        model = event_data.get("model", "")

        # For all augmented_llm events, put model info in details column
        details = f"{model}"
        chat_turn = event_data.get("chat_turn")
        if chat_turn is not None:
            details = f"{model} turn {chat_turn}"

        tool_name = event_data.get("tool_name")
        tool_event = event_data.get("tool_event")
        if tool_name:
            tool_suffix = tool_name
            if tool_event:
                tool_suffix = f"{tool_suffix} ({tool_event})"
            details = f"{details} • {tool_suffix}".strip()
    else:
        if not target:
            target = event_data.get("target", "unknown")

    # Extract streaming token count for STREAMING actions
    streaming_tokens = None
    if progress_action == ProgressAction.STREAMING or progress_action == ProgressAction.THINKING:
        streaming_tokens = event_data.get("details", "")

    # Extract progress data for TOOL_PROGRESS actions
    progress = None
    total = None
    if progress_action == ProgressAction.TOOL_PROGRESS:
        progress = event_data.get("progress")
        total = event_data.get("total")

    return ProgressEvent(
        action=ProgressAction(progress_action),
        target=target or "unknown",
        details=details,
        agent_name=event_data.get("agent_name"),
        streaming_tokens=streaming_tokens,
        progress=progress,
        total=total,
    )


class EventListener(ABC):
    """Base async listener that processes events."""

    @abstractmethod
    async def handle_event(self, event: Event):
        """Process an incoming event."""


class LifecycleAwareListener(EventListener):
    """
    Optionally override start()/stop() for setup/teardown.
    The event bus calls these at bus start/stop time.
    """

    async def start(self) -> None:
        """Start an event listener, usually when the event bus is set up."""
        pass

    async def stop(self) -> None:
        """Stop an event listener, usually when the event bus is shutting down."""
        pass


class FilteredListener(LifecycleAwareListener):
    """
    Only processes events that pass the given filter.
    Subclasses override _handle_matched_event().
    """

    def __init__(self, event_filter: EventFilter | None = None) -> None:
        """
        Initialize the listener.
        Args:
            filter: Event filter to apply to incoming events.
        """
        self.filter = event_filter

    async def handle_event(self, event) -> None:
        if not self.filter or self.filter.matches(event):
            await self.handle_matched_event(event)

    async def handle_matched_event(self, event: Event) -> None:
        """Process an event that matches the filter."""
        pass


class LoggingListener(FilteredListener):
    """
    Routes events to Python's logging facility with appropriate severity level.
    """

    def __init__(
        self,
        event_filter: EventFilter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the listener.
        Args:
            logger: Logger to use for event processing. Defaults to 'fast_agent'.
        """
        super().__init__(event_filter=event_filter)
        self.logger = logger or logging.getLogger("fast_agent")

    async def handle_matched_event(self, event) -> None:
        level_map: Dict[EventType, int] = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        level = level_map.get(event.type, logging.INFO)

        # Check if this is a server stderr message and format accordingly
        if event.name == "mcpserver.stderr":
            message = f"MCP Server: {event.message}"
        else:
            message = event.message

        self.logger.log(
            level,
            "[%s] %s",
            event.namespace,
            message,
            extra={
                "event_data": event.data,
                "span_id": event.span_id,
                "trace_id": event.trace_id,
                "event_name": event.name,
            },
        )


class ProgressListener(LifecycleAwareListener):
    """
    Listens for all events pre-filtering and converts them to progress events
    for display. By inheriting directly from LifecycleAwareListener instead of
    FilteredListener, we get events before any filtering occurs.
    """

    def __init__(self, display=None) -> None:
        """Initialize the progress listener.
        Args:
            display: Optional display handler. If None, the shared progress_display will be used.
        """
        from fast_agent.ui.progress_display import progress_display

        self.display = display or progress_display

    async def start(self) -> None:
        """Start the progress display."""
        self.display.start()

    async def stop(self) -> None:
        """Stop the progress display."""
        self.display.stop()

    async def handle_event(self, event: Event) -> None:
        """Process an incoming event and display progress if relevant."""

        if event.data:
            progress_event = convert_log_event(event)
            if progress_event:
                self.display.update(progress_event)


class BatchingListener(FilteredListener):
    """
    Accumulates events in memory, flushes them in batches.
    Here we just print the batch size, but you might store or forward them.
    """

    def __init__(
        self,
        event_filter: EventFilter | None = None,
        batch_size: int = 5,
        flush_interval: float = 2.0,
    ) -> None:
        """
        Initialize the listener.
        Args:
            batch_size: Number of events to accumulate before flushing.
            flush_interval: Time in seconds to wait before flushing events.
        """
        super().__init__(event_filter=event_filter)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch: List[Event] = []
        self.last_flush: float = time.time()  # Time of last flush
        self._flush_task: asyncio.Task | None = None  # Task for periodic flush loop
        self._stop_event = None  # Event to signal flush task to stop

    async def start(self, loop=None) -> None:
        """Spawn a periodic flush loop."""
        self._stop_event = asyncio.Event()
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        """Stop flush loop and flush any remaining events."""
        if self._stop_event:
            self._stop_event.set()

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            await self._flush_task
            self._flush_task = None
        await self.flush()

    async def _periodic_flush(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.flush_interval)
                except asyncio.TimeoutError:
                    await self.flush()
        # except asyncio.CancelledError:
        #     break
        finally:
            await self.flush()  # Final flush

    async def handle_matched_event(self, event) -> None:
        self.batch.append(event)
        if len(self.batch) >= self.batch_size:
            await self.flush()

    async def flush(self) -> None:
        """Flush the current batch of events."""
        if not self.batch:
            return
        to_process = self.batch[:]
        self.batch.clear()
        self.last_flush = time.time()
        await self._process_batch(to_process)

    async def _process_batch(self, events: List[Event]) -> None:
        pass
