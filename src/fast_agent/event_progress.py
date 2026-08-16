"""Module for converting log events to progress events."""

from enum import Enum

from pydantic import BaseModel


class ProgressAction(str, Enum):
    """Progress actions available in the system."""

    STARTING = "Starting"
    CONNECTING = "Connecting"
    LOADED = "Loaded"
    INITIALIZED = "Initialized"
    SENDING = "Sending"
    STREAMING = "Streaming"  # Special action for real-time streaming updates
    THINKING = "Thinking"  # Special action for real-time thinking updates
    COMPACTING = "Compacting"
    ROUTING = "Routing"
    PLANNING = "Planning"
    MONITORING = "Monitoring"
    RUNNING = "Subagent Running"
    READY = "Ready"
    CALLING_TOOL = "Calling Tool"
    READING_RESOURCE = "Reading Resource"
    RESOURCE_READ = "Resource Read"
    TOOL_PROGRESS = "Tool Progress"
    UPDATED = "Updated"
    FINISHED = "Finished"
    SHUTDOWN = "Shutdown"
    AGGREGATOR_INITIALIZED = "Running"
    FATAL_ERROR = "Error"


class SubagentMonitorSnapshot(BaseModel):
    """Structured state for one live subagent monitor row."""

    model: str | None = None
    context_percentage: float | None = None
    state: str
    turn: int
    input_tokens: int
    cache_percentage: float | None = None
    output_tokens: int
    output_estimated: bool = False


class ProgressEvent(BaseModel):
    """Represents a progress event converted from a log event."""

    action: ProgressAction
    target: str
    details: str | None = None
    agent_name: str | None = None
    correlation_id: str | None = None
    instance_name: str | None = None
    server_name: str | None = None
    tool_name: str | None = None
    tool_event: str | None = None
    tool_state: str | None = None
    activity: str | None = None
    tool_terminal: bool = False
    process_elapsed_seconds: float | None = None
    process_command: str | None = None
    process_id: str | None = None
    process_wait_seconds: int | None = None
    process_yield_reason: str | None = None
    process_has_observed_output: bool | None = None
    process_seconds_since_last_output: float | None = None
    process_total_output_bytes: int | None = None
    process_seconds_since_last_stdout: float | None = None
    process_seconds_since_last_stderr: float | None = None
    process_stdout_bytes: int | None = None
    process_stderr_bytes: int | None = None
    elapsed_seconds: float | None = None
    subagent_monitor: SubagentMonitorSnapshot | None = None
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
