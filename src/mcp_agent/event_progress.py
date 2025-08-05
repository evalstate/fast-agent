"""Module for converting log events to progress events."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel

from mcp_agent.logging.events import Event


class ProgressAction(str, Enum):
    """Progress actions available in the system."""

    STARTING = "Starting"
    LOADED = "Loaded"
    INITIALIZED = "Initialized"
    CHATTING = "Chatting"
    STREAMING = "Streaming"  # Special action for real-time streaming updates
    ROUTING = "Routing"
    PLANNING = "Planning"
    READY = "Ready"
    CALLING_TOOL = "Calling Tool"
    TOOL_PROGRESS = "Tool Progress"
    UPDATED = "Updated"
    FINISHED = "Finished"
    SHUTDOWN = "Shutdown"
    AGGREGATOR_INITIALIZED = "Running"
    FATAL_ERROR = "Error"


class ProgressEvent(BaseModel):
    """Represents a progress event converted from a log event."""

    action: ProgressAction
    target: str
    details: Optional[str] = None
    agent_name: Optional[str] = None
    streaming_tokens: Optional[str] = None  # Special field for streaming token count
    progress: Optional[float] = None  # Current progress value
    total: Optional[float] = None  # Total value for progress calculation

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


def convert_log_event(event: Event) -> Optional[ProgressEvent]:
    """Convert a log event to a progress event if applicable."""

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

    elif "augmented_llm" in namespace:
        model = event_data.get("model", "")
        
        # For all augmented_llm events, put model info in details column
        details = f"{model}"
        chat_turn = event_data.get("chat_turn")
        if chat_turn is not None:
            details = f"{model} turn {chat_turn}"
    else:
        if not target:
            target = event_data.get("target", "unknown")

    # Extract streaming token count for STREAMING actions
    streaming_tokens = None
    if progress_action == ProgressAction.STREAMING:
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
