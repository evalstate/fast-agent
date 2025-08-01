"""Rich-based progress display for MCP Agent."""

import time
from contextlib import contextmanager
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mcp_agent.console import console as default_console
from mcp_agent.event_progress import ProgressAction, ProgressEvent


class RichProgressDisplay:
    """Rich-based display for progress events."""

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize the progress display."""
        self.console = console or default_console
        self._taskmap = {}
        self._progress = Progress(
            SpinnerColumn(spinner_name="simpleDotsScrolling"),
            TextColumn(
                "[progress.description]{task.description}▎",
                #                table_column=Column(max_width=16),
            ),
            TextColumn(text_format="{task.fields[target]:<16}", style="Bold Blue"),
            TextColumn(text_format="{task.fields[details]}", style="dim white"),
            console=self.console,
            transient=False,
        )
        self._paused = False

    def start(self) -> None:
        """start"""

        self._progress.start()

    def stop(self) -> None:
        """Stop and clear the progress display."""
        # Hide all tasks before stopping (like pause does)
        for task in self._progress.tasks:
            task.visible = False
        self._progress.stop()
        self._paused = True

    def pause(self) -> None:
        """Pause the progress display."""
        if not self._paused:
            self._paused = True

            for task in self._progress.tasks:
                task.visible = False
            self._progress.stop()

    def resume(self) -> None:
        """Resume the progress display."""
        if self._paused:
            for task in self._progress.tasks:
                task.visible = True
            self._paused = False
            self._progress.start()

    @contextmanager
    def paused(self):
        """Context manager for temporarily pausing the display."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    def _get_action_style(self, action: ProgressAction) -> str:
        """Map actions to appropriate styles."""
        return {
            ProgressAction.STARTING: "bold yellow",
            ProgressAction.LOADED: "dim green",
            ProgressAction.INITIALIZED: "dim green",
            ProgressAction.CHATTING: "bold blue",
            ProgressAction.STREAMING: "bold green",  # Assistant Colour
            ProgressAction.ROUTING: "bold blue",
            ProgressAction.PLANNING: "bold blue",
            ProgressAction.READY: "dim green",
            ProgressAction.CALLING_TOOL: "bold magenta",
            ProgressAction.TOOL_PROGRESS: "bold magenta",
            ProgressAction.FINISHED: "black on green",
            ProgressAction.SHUTDOWN: "black on red",
            ProgressAction.AGGREGATOR_INITIALIZED: "bold green",
            ProgressAction.FATAL_ERROR: "black on red",
        }.get(action, "white")

    def update(self, event: ProgressEvent) -> None:
        """Update the progress display with a new event."""
        task_name = event.agent_name or "default"

        # Create new task if needed
        if task_name not in self._taskmap:
            task_id = self._progress.add_task(
                "",
                total=None,
                target=f"{event.target or task_name}",  # Use task_name as fallback for target
                details=f"{event.agent_name or ''}",
            )
            self._taskmap[task_name] = task_id
        else:
            task_id = self._taskmap[task_name]

        # Ensure no None values in the update
        # For streaming, use custom description immediately to avoid flashing
        if event.action == ProgressAction.STREAMING and event.streaming_tokens:
            # Account for [dim][/dim] tags (11 characters) in padding calculation
            formatted_tokens = f"▎[dim]◀[/dim] {event.streaming_tokens.strip()}".ljust(17 + 11)
            description = f"[{self._get_action_style(event.action)}]{formatted_tokens}"
        elif event.action == ProgressAction.CHATTING:
            # Add special formatting for chatting with dimmed arrow
            formatted_text = f"▎[dim]▶[/dim] {event.action.value.strip()}".ljust(17 + 11)
            description = f"[{self._get_action_style(event.action)}]{formatted_text}"
        elif event.action == ProgressAction.CALLING_TOOL:
            # Add special formatting for calling tool with dimmed arrow
            formatted_text = f"▎[dim]◀[/dim] {event.action.value}".ljust(17 + 11)
            description = f"[{self._get_action_style(event.action)}]{formatted_text}"
        elif event.action == ProgressAction.TOOL_PROGRESS:
            # Format similar to streaming - show progress numbers
            if event.progress is not None:
                if event.total is not None:
                    progress_display = f"{int(event.progress)}/{int(event.total)}"
                else:
                    progress_display = str(int(event.progress))
            else:
                progress_display = "Processing"
            formatted_text = f"▎[dim]▶[/dim] {progress_display}".ljust(17 + 11)
            description = f"[{self._get_action_style(event.action)}]{formatted_text}"
        else:
            description = f"[{self._get_action_style(event.action)}]▎ {event.action.value:<15}"

        # Update basic task information
        update_kwargs = {
            "description": description,
            "target": event.target or task_name,  # Use task_name as fallback for target
            "details": event.details or "",
            "task_name": task_name,
        }
        
        # For TOOL_PROGRESS events, update progress if available
        if event.action == ProgressAction.TOOL_PROGRESS and event.progress is not None:
            if event.total is not None:
                update_kwargs["completed"] = event.progress
                update_kwargs["total"] = event.total
            else:
                # If no total, just show as indeterminate progress
                self._progress.reset(task_id)
        
        self._progress.update(task_id, **update_kwargs)

        if (
            event.action == ProgressAction.INITIALIZED
            or event.action == ProgressAction.READY
            or event.action == ProgressAction.LOADED
        ):
            self._progress.update(task_id, completed=100, total=100)
        elif event.action == ProgressAction.FINISHED:
            self._progress.update(
                task_id,
                completed=100,
                total=100,
                details=f" / Elapsed Time {time.strftime('%H:%M:%S', time.gmtime(self._progress.tasks[task_id].elapsed))}",
            )
            for task in self._progress.tasks:
                if task.id != task_id:
                    task.visible = False
        elif event.action == ProgressAction.FATAL_ERROR:
            self._progress.update(
                task_id,
                completed=100,
                total=100,
                details=f" / {event.details}",
            )
            for task in self._progress.tasks:
                if task.id != task_id:
                    task.visible = False
        else:
            self._progress.reset(task_id)
