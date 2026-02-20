"""Rich-based progress display for MCP Agent."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
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
        self._task_kind: dict[str, str] = {}
        self._description_spinner = SpinnerDescriptionColumn(spinner_name="dots3")
        self._progress = Progress(
            self._description_spinner,
            TextColumn(
                text_format="{task.fields[target]}",
                style="Bold Blue",
                table_column=Column(
                    min_width=10,
                    max_width=16,
                    overflow="ellipsis",
                    no_wrap=True,
                ),
            ),
            TextColumn(
                text_format="{task.fields[details]}",
                style="dim white",
                table_column=Column(
                    ratio=1,
                    min_width=24,
                    overflow="ellipsis",
                    no_wrap=True,
                ),
            ),
            console=self.console,
            transient=False,
        )
        self._paused = False
        self._stopped = False
        self._deferred_resume_at: float | None = None
        trace_path_raw = os.getenv("FAST_AGENT_PROGRESS_DEBUG_TRACE", "").strip()
        self._trace_path: Path | None = Path(trace_path_raw).expanduser() if trace_path_raw else None

    def start(self) -> None:
        """start"""
        with self._lock:
            ensure_blocking_console()
            self._stopped = False
            self._paused = False
            self._deferred_resume_at = None
            self._progress.start()
            self._trace("start")

    def stop(self) -> None:
        """Stop and clear the progress display permanently."""
        with self._lock:
            ensure_blocking_console()
            # Mark as permanently stopped — resume() will be a no-op after this
            self._stopped = True
            self._paused = True
            self._deferred_resume_at = None
            # Hide all tasks before stopping (like pause does)
            for task in self._progress.tasks:
                task.visible = False
            self._progress.stop()
            self._trace("stop")

    def pause(self, *, cancel_deferred_on_noop: bool = False) -> None:
        """Pause the progress display."""
        with self._lock:
            if self._stopped or self._paused:
                if cancel_deferred_on_noop:
                    self._deferred_resume_at = None
                self._trace(
                    "pause.noop",
                    stopped=self._stopped,
                    paused=self._paused,
                    cancel_deferred_on_noop=cancel_deferred_on_noop,
                )
                return

            self._deferred_resume_at = None
            ensure_blocking_console()
            self._paused = True
            for task in self._progress.tasks:
                task.visible = False
            self._progress.stop()
            self._trace("pause")

    def _resume_locked(self) -> None:
        """Resume live rendering while lock is held."""
        # Never resume after a permanent stop()
        if self._stopped or not self._paused:
            self._trace("resume_locked.noop", stopped=self._stopped, paused=self._paused)
            return
        try:
            live_stack = getattr(self.console, "_live_stack", [])
        except Exception:
            live_stack = []

        if live_stack and any(live is not self._progress.live for live in live_stack):
            self._trace("resume_locked.blocked_nested_live", live_stack=len(live_stack))
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
        self._trace("resume_locked.applied")

    def resume(self, *, debounce_seconds: float = 0.0) -> None:
        """Resume the progress display."""
        with self._lock:
            if self._stopped or not self._paused:
                self._deferred_resume_at = None
                self._trace(
                    "resume.noop",
                    debounce_seconds=debounce_seconds,
                    stopped=self._stopped,
                    paused=self._paused,
                )
                return

            if debounce_seconds > 0:
                self._deferred_resume_at = time.monotonic() + debounce_seconds
                self._trace(
                    "resume.deferred",
                    debounce_seconds=debounce_seconds,
                    resume_at=self._deferred_resume_at,
                )
                return

            self._deferred_resume_at = None
            self._trace("resume.immediate")
            self._resume_locked()

    def _flush_deferred_resume_locked(self, *, force: bool = False) -> None:
        """Apply deferred resume request if debounce window has elapsed."""
        resume_at = self._deferred_resume_at
        if resume_at is None:
            return
        if not force and time.monotonic() < resume_at:
            self._trace("resume.deferred_pending", force=force, resume_at=resume_at)
            return
        self._deferred_resume_at = None
        self._trace("resume.deferred_flushed", force=force)
        self._resume_locked()

    def _trace(self, event: str, **fields: Any) -> None:
        """Write optional debug traces when FAST_AGENT_PROGRESS_DEBUG_TRACE is set."""
        if self._trace_path is None:
            return

        payload = {
            "ts": time.time(),
            "mono": time.monotonic(),
            "event": event,
            "paused": self._paused,
            "stopped": self._stopped,
            "deferred_resume_at": self._deferred_resume_at,
            **fields,
        }
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            # Never let diagnostics interfere with user-facing progress behavior.
            pass

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
            ProgressAction.CONNECTING: "bold yellow",
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
        self._task_kind.pop(task_name, None)

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

    @staticmethod
    def _is_internal_shell_tool(tool_name: str | None, server_name: str | None) -> bool:
        """Return True when event is for the built-in ACP shell tool."""
        if not tool_name or not server_name:
            return False
        return tool_name.lower() == "execute" and server_name == "acp_terminal"

    @staticmethod
    def _short_correlation_id(correlation_id: str) -> str:
        """Return a concise correlation identifier for display."""
        normalized = correlation_id.strip()
        if len(normalized) <= 16:
            return normalized
        return f"{normalized[:13]}…"

    def _max_details_chars(self) -> int:
        """Estimate available width for the details column."""
        console_width = getattr(self.console, "width", 100)
        # Rough budget: action+spinner (~22), target (<=16), table/padding (~10).
        # Keep a sane lower bound so details remain informative.
        return max(24, int(console_width) - 48)

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        """Truncate with ellipsis when text exceeds max display chars."""
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        if max_chars == 1:
            return "…"
        return f"{value[: max_chars - 1]}…"

    def _count_correlated_tool_rows(self, agent_name: str) -> int:
        """Return number of active correlated tool rows for an agent."""
        prefix = f"{agent_name}::"
        return sum(
            1
            for task_name, task_kind in self._task_kind.items()
            if task_kind == "tool" and task_name.startswith(prefix)
        )

    def _format_correlated_details(
        self,
        *,
        details: str,
        correlation_id: str | None,
        force_show_id: bool,
    ) -> str:
        """Format details for correlated tool events with width-aware id handling."""
        if not correlation_id:
            return details

        correlation_marker = f"id:{self._short_correlation_id(correlation_id)}"
        if not force_show_id:
            budget = self._max_details_chars()
            if len(details) + 3 + len(correlation_marker) > budget:
                return self._truncate_text(details, budget)
            return f"{details} • {correlation_marker}" if details else correlation_marker

        budget = self._max_details_chars()
        # Reserve room for separator and id marker when id must be shown.
        message_budget = max(0, budget - (3 + len(correlation_marker)))
        details_part = self._truncate_text(details, message_budget)
        if details_part:
            return f"{details_part} • {correlation_marker}"
        return correlation_marker

    @staticmethod
    def _task_kind_for_event(event: ProgressEvent, *, is_correlated_tool_event: bool) -> str:
        """Classify the row kind for task lifecycle handling."""
        if is_correlated_tool_event:
            return "tool"
        if event.action in {ProgressAction.STREAMING, ProgressAction.THINKING}:
            return "stream"
        return "agent"

    @staticmethod
    def _is_terminal_tool_event(tool_event: str | None) -> bool:
        """Return True for tool lifecycle events that should close correlated rows."""
        normalized = (tool_event or "").strip().lower()
        if not normalized:
            return False
        return normalized in {
            "stop",
            "stopped",
            "done",
            "complete",
            "completed",
            "cancel",
            "canceled",
            "cancelled",
            "error",
            "fail",
            "failed",
        }

    @staticmethod
    def _is_terminal_tool_progress(
        *, progress: float | None, total: float | None, details: str | None
    ) -> bool:
        """Return True when progress payload indicates terminal completion/failure."""
        if progress is not None and total is not None and total > 0 and progress >= total:
            return True

        normalized_details = (details or "").strip().lower()
        if not normalized_details:
            return False

        terminal_details = {
            "stop",
            "stopped",
            "done",
            "complete",
            "completed",
            "cancel",
            "canceled",
            "cancelled",
            "error",
            "fail",
            "failed",
        }
        if normalized_details in terminal_details:
            return True

        return normalized_details.startswith(("failed:", "error:"))

    def update(self, event: ProgressEvent) -> None:
        """Update the progress display with a new event."""
        with self._lock:
            # Skip updates when display is stopped
            if self._stopped:
                self._trace("update.skipped_stopped", action=event.action.value)
                return

            self._flush_deferred_resume_locked(force=True)
            self._trace(
                "update",
                action=event.action.value,
                agent_name=event.agent_name,
                target=event.target,
            )

            self._apply_update_locked(event)

    def _apply_update_locked(self, event: ProgressEvent) -> None:
        """Apply an update while holding lock and with active display state."""
        if (
            event.action == ProgressAction.AGGREGATOR_INITIALIZED
            and not (event.details or "").strip()
        ):
            # "Running" without additional details is redundant noise because
            # active rows already imply the app is running.
            return

        task_name = event.agent_name or "default"
        is_correlated_tool_event = (
            event.action in {ProgressAction.CALLING_TOOL, ProgressAction.TOOL_PROGRESS}
            and event.correlation_id is not None
            and not self._is_internal_shell_tool(event.tool_name, event.server_name)
        )
        if (
            is_correlated_tool_event
            and event.correlation_id
        ):
            task_name = f"{task_name}::{event.correlation_id}"

        should_drop_tool_task = (
            is_correlated_tool_event
            and (
                (
                    event.action == ProgressAction.CALLING_TOOL
                    and self._is_terminal_tool_event(event.tool_event)
                )
                or (
                    event.action == ProgressAction.TOOL_PROGRESS
                    and self._is_terminal_tool_progress(
                        progress=event.progress,
                        total=event.total,
                        details=event.details,
                    )
                )
            )
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

        self._task_kind[task_name] = self._task_kind_for_event(
            event,
            is_correlated_tool_event=is_correlated_tool_event,
        )

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
        details_value = event.details or ""
        if is_correlated_tool_event:
            active_correlated = self._count_correlated_tool_rows(event.agent_name or "default")
            details_value = self._format_correlated_details(
                details=details_value.strip(),
                correlation_id=event.correlation_id,
                force_show_id=active_correlated > 1,
            )

        update_kwargs: dict[str, Any] = {
            "description": description,
            "target": event.target or task_name,  # Use task_name as fallback for target
            "details": details_value,
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

        if event.action in {
            ProgressAction.INITIALIZED,
            ProgressAction.READY,
            ProgressAction.LOADED,
        }:
            # Keep lifecycle transitions visible very briefly, then clear them.
            # This prevents idle/inactive agents from permanently cluttering the board.
            self._drop_task(task_name, task_id)
        elif event.action == ProgressAction.FINISHED:
            finished_task = next((task for task in self._progress.tasks if task.id == task_id), None)
            elapsed = finished_task.elapsed if finished_task is not None else None
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
            # Keep fatal errors visible via command output/logging, but avoid
            # permanently pinning stale "Error" rows in the live progress board.
            self._drop_task(task_name, task_id)
        elif should_drop_tool_task:
            self._drop_task(task_name, task_id)
        else:
            if event.action != ProgressAction.TOOL_PROGRESS:
                self._progress.reset(task_id)
