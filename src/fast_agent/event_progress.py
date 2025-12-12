"""Module for converting log events to progress events."""

from enum import Enum

from pydantic import BaseModel


class ProgressAction(str, Enum):
    """Progress actions available in the system."""

    STARTING = "Starting"
    LOADED = "Loaded"
    INITIALIZED = "Initialized"
    CHATTING = "Chatting"
    STREAMING = "Streaming"  # Special action for real-time streaming updates
    THINKING = "Thinking"  # Special action for real-time thinking updates
    ROUTING = "Routing"
    PLANNING = "Planning"
    READY = "Ready"
    CALLING_TOOL = "Calling Tool"
    TOOL_PROGRESS = "Tool Progress"
    UPDATED = "Updated"
    FINISHED = "Finished"
    SHUTDOWN = "Shutdown"
    AGGREGATOR_INITIALIZED = "Running"
    SERVER_OFFLINE = "Offline"
    SERVER_RECONNECTING = "Reconnecting"
    SERVER_ONLINE = "Online"
    FATAL_ERROR = "Error"


class ProgressEvent(BaseModel):
    """Represents a progress event converted from a log event."""

    action: ProgressAction
    target: str
    details: str | None = None
    agent_name: str | None = None
    correlation_id: str | None = None
    instance_name: str | None = None
    tool_name: str | None = None
    streaming_tokens: str | None = None  # Special field for streaming token count
    progress: float | None = None  # Current progress value
    total: float | None = None  # Total value for progress calculation

    def __str__(self) -> str:
        """Format the progress event for display."""
        # Special handling for streaming - show token count in action position
        if self.action == ProgressAction.STREAMING and self.streaming_tokens:
            # For streaming, show just the token count instead of "Streaming"
            action_display = self.streaming_tokens.ljust(11)
            base = f"{action_display}. {self.target}"
            if self.details:
                base += f" - {self.details}"
        else:
            base = f"{self.action.ljust(11)}. {self.target}"
            if self.details:
                base += f" - {self.details}"

        if self.agent_name:
            base = f"[{self.agent_name}] {base}"
        return base
