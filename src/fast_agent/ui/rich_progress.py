"""Rich-based progress display for MCP Agent."""

import time
from contextlib import contextmanager
from threading import RLock
from typing import Any

from rich.console import Console
from rich.progress import Progress, ProgressColumn, Task, TaskID, TextColumn
from rich.spinner import Spinner
from rich.table import Column
from rich.text import Text

from fast_agent.event_progress import ProgressAction, ProgressEvent
from fast_agent.ui.console import console as default_console
from fast_agent.ui.console import ensure_blocking_console


class SpinnerDescriptionColumn(ProgressColumn):
    """Render the task description with an inline spinner (no column padding gap)."""

    def __init__(
        self,
        *,
        spinner_name: str = "dots",
        spinner_style: str | None = "progress.spinner",
        speed: float = 1.0,
        finished_text: str = " ",
        description_style: str = "progress.description",
        markup: bool = True,
        table_column: Column | None = None,
    ) -> None:
        self.spinner = Spinner(spinner_name, style=spinner_style, speed=speed)
        self.finished_text = Text.from_markup(finished_text)
        self.description_style = description_style
        self.markup = markup
        super().__init__(table_column=table_column or Column(no_wrap=True))

    def render(self, task: "Task") -> Text:
        description_markup = f"[{self.description_style}]{task.description}▎"
        if self.markup:
            description_text = Text.from_markup(description_markup)
        else:
            description_text = Text(description_markup, style=self.description_style)

        if task.finished:
            spinner_text = self.finished_text
        else:
            rendered = self.spinner.render(task.get_time())
            spinner_text = rendered if isinstance(rendered, Text) else Text(str(rendered))

        return Text.assemble(description_text, spinner_text)


class RichProgressDisplay:
    """Rich-based display for progress events."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the progress display."""
        self.console = console or default_console
        self._lock = RLock()
        self._taskmap: dict[str, TaskID] = {}
        self._description_spinner = SpinnerDescriptionColumn(spinner_name="dots3")
        self._progress = Progress(
            self._description_spinner,
            TextColumn(text_format="{task.fields[target]:<16}", style="Bold Blue"),
            TextColumn(text_format="{task.fields[details]}", style="dim white"),
            console=self.console,
            transient=False,
        )
        self._paused = False
        self._stopped = False

    def start(self) -> None:
        """start"""
        with self._lock:
            ensure_blocking_console()
            self._stopped = False
            self._progress.start()

    def stop(self) -> None:
        """Stop and clear the progress display permanently."""
        with self._lock:
            ensure_blocking_console()
            # Mark as permanently stopped — resume() will be a no-op after this
            self._stopped = True
            self._paused = True
            # Hide all tasks before stopping (like pause does)
            for task in self._progress.tasks:
                task.visible = False
            self._progress.stop()

    def pause(self) -> None:
        """Pause the progress display."""
        with self._lock:
            if self._stopped or self._paused:
                return
            ensure_blocking_console()
            self._paused = True
            for task in self._progress.tasks:
                task.visible = False
            self._progress.stop()

    def resume(self) -> None:
        """Resume the progress display."""
        with self._lock:
            # Never resume after a permanent stop()
            if self._stopped or not self._paused:
                return
            try:
                live_stack = getattr(self.console, "_live_stack", [])
            except Exception:
                live_stack = []

            if live_stack and any(live is not self._progress.live for live in live_stack):
                return
            ensure_blocking_console()
            if getattr(self._progress.live, "_nested", False):
                self._progress.live._nested = False
            for task in self._progress.tasks:
                task.visible = True
            # Start the Live display before clearing the flag so that
            # update() never runs against an un-started Progress.
            self._progress.start()
            self._paused = False

    def hide_task(self, task_name: str) -> None:
        """Hide an existing task from the progress display by name."""
        with self._lock:
            task_id = self._taskmap.get(task_name)
            if task_id is None:
                return
            for task in self._progress.tasks:
                if task.id == task_id:
                    task.visible = False
                    break

    @contextmanager
    def paused(self):
        """Context manager for temporarily pausing the display."""
        with self._lock:
            was_paused = self._paused
        self.pause()
        try:
            yield
        finally:
            if not was_paused:
                self.resume()

    def _get_action_style(self, action: ProgressAction) -> str:
        """Map actions to appropriate styles."""
        return {
            ProgressAction.STARTING: "bold yellow",
            ProgressAction.LOADED: "dim green",
            ProgressAction.INITIALIZED: "dim green",
            ProgressAction.CHATTING: "bold blue",
            ProgressAction.STREAMING: "bold green",  # Assistant Colour
            ProgressAction.THINKING: "bold yellow",  # Assistant Colour
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

    def _drop_task(self, task_name: str, task_id: TaskID) -> None:
        """Remove a task from visible progress tracking and internal maps."""
        self._taskmap.pop(task_name, None)

        remove_task = getattr(self._progress, "remove_task", None)
        if callable(remove_task):
            try:
                remove_task(task_id)
                return
            except Exception:
                # Fall back to hiding when remove_task is unavailable/unsupported.
                pass

        for task in self._progress.tasks:
            if task.id == task_id:
                task.visible = False
                break

    def update(self, event: ProgressEvent) -> None:
        """Update the progress display with a new event."""
        with self._lock:
            # Skip updates when display is paused or stopped
            if self._paused or self._stopped:
                return

            task_name = event.agent_name or "default"
            is_correlated_tool_event = (
                event.action in {ProgressAction.CALLING_TOOL, ProgressAction.TOOL_PROGRESS}
                and event.correlation_id is not None
            )
            if (
                is_correlated_tool_event
                and event.correlation_id
            ):
                task_name = f"{task_name}::{event.correlation_id}"

            should_drop_completed_tool_task = (
                is_correlated_tool_event
                and event.action == ProgressAction.TOOL_PROGRESS
                and event.progress is not None
                and event.total is not None
                and event.total > 0
                and event.progress >= event.total
            )
            should_drop_stopped_tool_task = (
                is_correlated_tool_event
                and event.action == ProgressAction.CALLING_TOOL
                and (event.tool_event or "").lower()
                in {"stop", "stopped", "done", "complete", "completed"}
            )

            # Create new task if needed
            if task_name not in self._taskmap:
                task_id = self._progress.add_task(
                    "",
                    total=None,
                    target=event.target or task_name,
                    details=event.details or "",
                    task_name=task_name,
                )
                self._taskmap[task_name] = task_id
            else:
                task_id = self._taskmap[task_name]

            # Ensure no None values in the update
            # For streaming, use custom description immediately to avoid flashing
            if (
                event.action == ProgressAction.STREAMING
                or event.action == ProgressAction.THINKING
            ) and event.streaming_tokens:
                # Account for [dim][/dim] tags (11 characters) in padding calculation
                formatted_tokens = (
                    f"▎[dim]◀[/dim] {event.streaming_tokens.strip()}".ljust(17 + 11)
                )
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
                formatted_text = f"▎[dim]•[/dim] {event.action.value}".ljust(17 + 11)
                description = f"[{self._get_action_style(event.action)}]{formatted_text}"

            # Update basic task information
            update_kwargs: dict[str, Any] = {
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
                    # Reset to indeterminate in a single update call to avoid
                    # an intermediate render of the cleared state.
                    update_kwargs["completed"] = 0
                    update_kwargs["total"] = None

            self._progress.update(task_id, **update_kwargs)

            if (
                event.action == ProgressAction.INITIALIZED
                or event.action == ProgressAction.READY
                or event.action == ProgressAction.LOADED
            ):
                self._progress.update(task_id, completed=100, total=100)
            elif event.action == ProgressAction.FINISHED:
                elapsed = self._progress.tasks[task_id].elapsed
                elapsed_str = time.strftime(
                    "%H:%M:%S", time.gmtime(elapsed if elapsed is not None else 0)
                )
                self._progress.update(
                    task_id,
                    completed=100,
                    total=100,
                    target=event.target or task_name,
                    details=f" / Elapsed Time {elapsed_str}",
                    task_name=task_name,
                )
            elif event.action == ProgressAction.FATAL_ERROR:
                self._progress.update(
                    task_id,
                    completed=100,
                    total=100,
                    target=event.target or task_name,
                    details=f" / {event.details}",
                    task_name=task_name,
                )
            elif should_drop_completed_tool_task:
                self._drop_task(task_name, task_id)
            elif should_drop_stopped_tool_task:
                self._drop_task(task_name, task_id)
            else:
                if event.action != ProgressAction.TOOL_PROGRESS:
                    self._progress.reset(task_id)
