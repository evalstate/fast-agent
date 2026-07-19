"""Rich-based progress display for MCP Agent."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, Timer
from typing import Any

from rich.console import Console
from rich.markup import escape as escape_markup
from rich.progress import Progress, ProgressColumn, Task, TaskID, TextColumn
from rich.spinner import SPINNERS, Spinner
from rich.table import Column
from rich.text import Text

from fast_agent.event_progress import ProgressAction, ProgressEvent
from fast_agent.tools.tool_sources import ACP_TERMINAL_TOOL_SOURCE
from fast_agent.ui.agent_identity import is_default_agent_name
from fast_agent.ui.console import console as default_console
from fast_agent.ui.console import ensure_blocking_console
from fast_agent.ui.process_poll_display import (
    format_process_output_activity,
    format_process_output_size,
    format_process_poll_countdown_track,
)
from fast_agent.ui.tool_call_ids import format_tool_call_id
from fast_agent.utils.time import format_process_elapsed
from fast_agent.utils.tool_names import (
    EXECUTE_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    matches_tool_name,
)

# Braille pulses moving through a 3-cell track. Registered on first use via
# _ensure_spinners().
PROGRESS_SPINNER_NAME = "braille_dense"
_BRAILLE_DENSE = {
    "interval": 110,
    "frames": [
        "   ",
        "⡇  ",
        "⣿  ",
        "⢸⡇ ",
        " ⣿ ",
        " ⢸⡇",
        "  ⣿",
        "  ⢸",
        "   ",
    ],
}


def _ensure_spinners() -> None:
    SPINNERS.setdefault(PROGRESS_SPINNER_NAME, _BRAILLE_DENSE)


_ACTION_DESCRIPTION_ICONS = {
    ProgressAction.SENDING: "▶",
    ProgressAction.CALLING_TOOL: "◀",
    ProgressAction.READING_RESOURCE: "◀",
    ProgressAction.RESOURCE_READ: "✓",
    ProgressAction.TOOL_PROGRESS: "▶",
}
_ACTION_STYLES = {
    ProgressAction.STARTING: "bold yellow",
    ProgressAction.CONNECTING: "bold yellow",
    ProgressAction.LOADED: "dim green",
    ProgressAction.INITIALIZED: "dim green",
    ProgressAction.SENDING: "blue",
    ProgressAction.STREAMING: "green",  # Assistant Colour
    ProgressAction.THINKING: "yellow",  # Assistant Colour
    ProgressAction.COMPACTING: "cyan",
    ProgressAction.ROUTING: "blue",
    ProgressAction.PLANNING: "blue",
    ProgressAction.READY: "dim green",
    ProgressAction.CALLING_TOOL: "magenta",
    ProgressAction.READING_RESOURCE: "magenta",
    ProgressAction.RESOURCE_READ: "dim green",
    ProgressAction.TOOL_PROGRESS: "magenta",
    ProgressAction.FINISHED: "black on green",
    ProgressAction.SHUTDOWN: "black on red",
    ProgressAction.AGGREGATOR_INITIALIZED: "bold green",
    ProgressAction.FATAL_ERROR: "black on red",
}


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
        if self.markup:
            description_text = Text.from_markup(
                task.description,
                style=self.description_style,
            )
        else:
            description_text = Text(task.description, style=self.description_style)

        if task.finished:
            spinner_text = self.finished_text
        else:
            spinner_text = self._render_activity_glyph(task)

        # Glyph trails the padded label (historical layout). Narrow terminals may
        # clip it; countdown still reads from colour/shape when visible.
        return Text.assemble(description_text, spinner_text)

    def _render_activity_glyph(self, task: "Task") -> Text:
        """Pulse spinner, or a depleting braille track while a process poll waits."""
        if bool(task.fields.get("is_process_poll")):
            wait_seconds = task.fields.get("process_wait_seconds")
            countdown = format_process_poll_countdown_track(
                wait_seconds=wait_seconds if type(wait_seconds) is int else None,
                elapsed_seconds=task.elapsed or 0.0,
            )
            if countdown is not None:
                return Text(countdown, style="magenta")

        rendered = self.spinner.render(task.get_time())
        return rendered if isinstance(rendered, Text) else Text(str(rendered))


class DynamicDetailsColumn(ProgressColumn):
    """Render optional process elapsed time without emitting timer events."""

    def __init__(
        self,
        *,
        style: str = "dim white",
        table_column: Column | None = None,
    ) -> None:
        self.style = style
        super().__init__(table_column=table_column)

    def render(self, task: "Task") -> Text:
        details = str(task.fields.get("details") or "").strip()
        is_process_poll = bool(task.fields.get("is_process_poll"))
        parts: list[str | Text] = []
        if details:
            parts.append(details)
        local_tick = self._local_tick_seconds(task)
        elapsed_base = task.fields.get("process_elapsed_seconds")
        if isinstance(elapsed_base, (int, float)) and not isinstance(elapsed_base, bool):
            parts.append(format_process_elapsed(float(elapsed_base) + local_tick))
        if is_process_poll:
            output_age = task.fields.get("process_seconds_since_last_output")
            if isinstance(output_age, (int, float)) and not isinstance(output_age, bool):
                output_age = float(output_age) + local_tick
            else:
                output_age = None
            output_activity = format_process_output_activity(
                has_observed_output=task.fields.get("process_has_observed_output"),
                seconds_since_last_output=output_age,
            )
            if output_activity is not None:
                if output_activity.style:
                    parts.append(Text(output_activity.text, style=output_activity.style))
                else:
                    parts.append(output_activity.text)
            output_size = format_process_output_size(
                task.fields.get("process_total_output_bytes")
            )
            if output_size:
                parts.append(output_size)
        command = task.fields.get("process_command")
        if isinstance(command, str) and command:
            parts.append(command)
        return self._join_detail_parts(parts)

    @staticmethod
    def _local_tick_seconds(task: "Task") -> float:
        """Seconds since the latest process field snapshot was applied."""
        task_elapsed = task.elapsed or 0.0
        snapshot = task.fields.get("process_snapshot_task_elapsed")
        if isinstance(snapshot, (int, float)) and not isinstance(snapshot, bool):
            return max(task_elapsed - float(snapshot), 0.0)
        return task_elapsed

    def _join_detail_parts(self, parts: list[str | Text]) -> Text:
        line = Text(style=self.style)
        first = True
        for part in parts:
            if not part:
                continue
            if not first:
                line.append(" · ", style=self.style)
            first = False
            if isinstance(part, Text):
                line.append_text(part)
            else:
                line.append(part, style=self.style)
        return line


class RichProgressDisplay:
    """Rich-based display for progress events."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        default_agent_name: str | None = None,
    ) -> None:
        """Initialize the progress display."""
        self.console = console or default_console
        self.default_agent_name = default_agent_name
        self._lock = RLock()
        self._taskmap: dict[str, TaskID] = {}
        self._task_kind: dict[str, str] = {}
        _ensure_spinners()
        self._description_spinner = SpinnerDescriptionColumn(spinner_name=PROGRESS_SPINNER_NAME)
        self._progress = Progress(
            self._description_spinner,
            TextColumn(
                text_format="{task.fields[target]}",
                style="Blue",
                table_column=Column(
                    min_width=0,
                    max_width=16,
                    overflow="ellipsis",
                    no_wrap=True,
                ),
            ),
            DynamicDetailsColumn(
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
        self._trace_path: Path | None = (
            Path(trace_path_raw).expanduser() if trace_path_raw else None
        )

    def set_default_agent_name(self, default_agent_name: str | None) -> None:
        self.default_agent_name = default_agent_name

    def is_default_agent_name(self, agent_name: str | None) -> bool:
        return is_default_agent_name(
            agent_name,
            default_agent_name=self.default_agent_name,
        )

    def _live_started(self) -> bool:
        """Return whether Rich's Live renderer has been started."""
        return bool(getattr(self._progress.live, "is_started", False))

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
            if self._live_started():
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
            if self._live_started():
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
            # A concurrent Rich Live (for example, a streaming renderer that is
            # still unwinding after cancellation) temporarily blocks the shared
            # progress display from resuming. Keep an immediate deferred retry
            # armed so the next update() can re-attempt resume once the other
            # Live is gone instead of silently remaining paused for the turn.
            self._deferred_resume_at = time.monotonic()
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

    def clear_agent_tasks(self, agent_name: str | None) -> None:
        """Remove all progress rows for an agent, including correlated tool rows."""
        normalized_agent = (agent_name or "").strip()
        if not normalized_agent:
            return

        prefix = f"{normalized_agent}::"
        with self._lock:
            task_names = [
                task_name
                for task_name in list(self._taskmap)
                if task_name == normalized_agent or task_name.startswith(prefix)
            ]
            if not task_names:
                return

            for task_name in task_names:
                task_id = self._taskmap.get(task_name)
                if task_id is None:
                    continue
                self._drop_task(task_name, task_id)

            self._trace(
                "clear_agent_tasks",
                agent_name=normalized_agent,
                cleared=len(task_names),
            )

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
        return _ACTION_STYLES.get(action, "white")

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
        return (
            matches_tool_name(tool_name, EXECUTE_TOOL_NAME)
            and server_name == ACP_TERMINAL_TOOL_SOURCE
        )

    @staticmethod
    def _short_correlation_id(correlation_id: str) -> str:
        """Return a concise correlation identifier for display."""
        return format_tool_call_id(correlation_id.strip()) or ""

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

        correlation_marker = f"id: {self._short_correlation_id(correlation_id)}"
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

    def _task_name_for_event(
        self,
        event: ProgressEvent,
        *,
        is_correlated_tool_event: bool,
    ) -> str:
        task_name = event.agent_name or "default"
        if is_correlated_tool_event and event.correlation_id:
            return f"{task_name}::{event.correlation_id}"
        return task_name

    def _task_id_for_event(self, event: ProgressEvent, task_name: str) -> TaskID:
        if task_name in self._taskmap:
            return self._taskmap[task_name]

        task_id = self._progress.add_task(
            "",
            total=None,
            target=event.target or task_name,
            details=event.details or "",
            task_name=task_name,
        )
        self._taskmap[task_name] = task_id
        return task_id

    def _description_for_event(self, event: ProgressEvent) -> str:
        action_style = self._get_action_style(event.action)
        if event.action in (ProgressAction.STREAMING, ProgressAction.THINKING):
            tokens = event.streaming_tokens
            if tokens:
                formatted_tokens = f"▎[dim]◀[/dim] {escape_markup(tokens.strip())}".ljust(17 + 11)
                return f"[{action_style}]{formatted_tokens}"

        icon = _ACTION_DESCRIPTION_ICONS.get(event.action, "•")
        label = (
            self._tool_progress_label(event)
            if event.action == ProgressAction.TOOL_PROGRESS
            else self._action_label(event)
        )
        formatted_text = f"▎[dim]{icon}[/dim] {label}".ljust(17 + 11)
        return f"[{action_style}]{formatted_text}"

    @staticmethod
    def _is_process_poll_event(event: ProgressEvent) -> bool:
        return matches_tool_name(event.tool_name, POLL_PROCESS_TOOL_NAME) and event.action in {
            ProgressAction.CALLING_TOOL,
            ProgressAction.TOOL_PROGRESS,
        }

    @classmethod
    def _action_label(cls, event: ProgressEvent) -> str:
        if cls._is_process_poll_event(event):
            return "Monitoring"
        return event.action.value.strip()

    @staticmethod
    def _tool_progress_label(event: ProgressEvent) -> str:
        if event.progress is None:
            return "Processing"
        if event.total is None:
            return str(int(event.progress))
        return f"{int(event.progress)}/{int(event.total)}"

    def _details_for_event(
        self,
        event: ProgressEvent,
        *,
        is_correlated_tool_event: bool,
    ) -> str:
        details_value = event.details or ""
        if not is_correlated_tool_event:
            return details_value
        if self._is_process_poll_event(event):
            if self.is_default_agent_name(event.agent_name):
                return ""
            return (event.process_id or details_value).strip()

        active_correlated = self._count_correlated_tool_rows(event.agent_name or "default")
        return self._format_correlated_details(
            details=details_value.strip(),
            correlation_id=event.correlation_id,
            force_show_id=active_correlated > 1,
        )

    def _update_kwargs_for_event(
        self,
        event: ProgressEvent,
        *,
        task_name: str,
        is_correlated_tool_event: bool,
    ) -> dict[str, Any]:
        is_process_poll = self._is_process_poll_event(event)
        target = event.target or task_name
        if is_process_poll and self.is_default_agent_name(event.agent_name):
            target = event.process_id or ""
        update_kwargs: dict[str, Any] = {
            "description": self._description_for_event(event),
            "target": target,
            "details": self._details_for_event(
                event,
                is_correlated_tool_event=is_correlated_tool_event,
            ),
            "task_name": task_name,
            "is_process_poll": is_process_poll,
        }
        if event.process_elapsed_seconds is not None:
            update_kwargs["process_elapsed_seconds"] = event.process_elapsed_seconds
        if event.process_command is not None:
            update_kwargs["process_command"] = event.process_command
        if event.process_wait_seconds is not None:
            update_kwargs["process_wait_seconds"] = event.process_wait_seconds
        if event.process_has_observed_output is not None:
            update_kwargs["process_has_observed_output"] = (
                event.process_has_observed_output
            )
        if event.process_seconds_since_last_output is not None:
            update_kwargs["process_seconds_since_last_output"] = (
                event.process_seconds_since_last_output
            )
        if event.process_total_output_bytes is not None:
            update_kwargs["process_total_output_bytes"] = (
                event.process_total_output_bytes
            )
        if event.action == ProgressAction.TOOL_PROGRESS and event.progress is not None:
            self._add_tool_progress_update_kwargs(event, update_kwargs)
        return update_kwargs

    @staticmethod
    def _add_tool_progress_update_kwargs(
        event: ProgressEvent,
        update_kwargs: dict[str, Any],
    ) -> None:
        if event.total is not None:
            update_kwargs["completed"] = event.progress
            update_kwargs["total"] = event.total
            return

        # Reset to indeterminate in a single update call to avoid
        # an intermediate render of the cleared state.
        update_kwargs["completed"] = 0
        update_kwargs["total"] = None

    def _apply_post_update_lifecycle(
        self,
        event: ProgressEvent,
        *,
        task_name: str,
        task_id: TaskID,
        should_drop_tool_task: bool,
    ) -> None:
        if event.action in {
            ProgressAction.INITIALIZED,
            ProgressAction.READY,
            ProgressAction.LOADED,
            ProgressAction.RESOURCE_READ,
        }:
            # Keep lifecycle transitions visible very briefly, then clear them.
            # This prevents idle/inactive agents from permanently cluttering the board.
            self._drop_task(task_name, task_id)
        elif event.action == ProgressAction.FINISHED:
            self._mark_finished_task(event, task_name=task_name, task_id=task_id)
        elif event.action == ProgressAction.FATAL_ERROR:
            self._mark_fatal_error_task(event, task_name=task_name, task_id=task_id)
        elif should_drop_tool_task:
            self._drop_task(task_name, task_id)
        elif event.action != ProgressAction.TOOL_PROGRESS and not self._is_process_poll_event(
            event
        ):
            # Process polls keep a stable start_time so the braille countdown
            # track can drain across refresh updates.
            self._progress.reset(task_id)

    def _finish_process_poll_task(
        self,
        task_name: str,
        task_id: TaskID,
        task: Task,
    ) -> None:
        """Show an empty countdown track briefly, then remove the poll row."""
        wait_seconds = task.fields.get("process_wait_seconds")
        if type(wait_seconds) is not int or wait_seconds <= 0:
            wait_seconds = 1
        # Freeze elapsed past the wait budget so the glyph stays blank.
        now = time.time()
        if task.start_time is not None:
            task.start_time = now - float(wait_seconds) - 0.05
        task.stop_time = now
        self._progress.update(
            task_id,
            is_process_poll=True,
            process_wait_seconds=wait_seconds,
            process_snapshot_task_elapsed=float(wait_seconds),
            description=task.description,
        )
        # Force one Live refresh so the empty track paints before teardown.
        if self._live_started() and not self._paused:
            self._progress.refresh()

        def _drop_later() -> None:
            with self._lock:
                if self._taskmap.get(task_name) != task_id:
                    return
                self._drop_task(task_name, task_id)

        timer = Timer(0.7, _drop_later)
        timer.daemon = True
        timer.start()

    def _mark_finished_task(
        self,
        event: ProgressEvent,
        *,
        task_name: str,
        task_id: TaskID,
    ) -> None:
        finished_task = next((task for task in self._progress.tasks if task.id == task_id), None)
        elapsed = finished_task.elapsed if finished_task is not None else None
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed if elapsed is not None else 0))
        self._progress.update(
            task_id,
            completed=100,
            total=100,
            target=event.target or task_name,
            details=f" / Elapsed Time {elapsed_str}",
            task_name=task_name,
        )

    def _mark_fatal_error_task(
        self,
        event: ProgressEvent,
        *,
        task_name: str,
        task_id: TaskID,
    ) -> None:
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

        is_correlated_tool_event = (
            event.action in {ProgressAction.CALLING_TOOL, ProgressAction.TOOL_PROGRESS}
            and event.correlation_id is not None
            and not self._is_internal_shell_tool(event.tool_name, event.server_name)
        )
        task_name = self._task_name_for_event(
            event,
            is_correlated_tool_event=is_correlated_tool_event,
        )
        should_drop_tool_task = is_correlated_tool_event and event.tool_terminal
        task_id = self._task_id_for_event(event, task_name)

        self._task_kind[task_name] = self._task_kind_for_event(
            event,
            is_correlated_tool_event=is_correlated_tool_event,
        )

        existing = next((item for item in self._progress.tasks if item.id == task_id), None)
        finishing_process_poll = (
            should_drop_tool_task
            and existing is not None
            and bool(existing.fields.get("is_process_poll"))
            and event.process_yield_reason == "deadline"
        )
        if finishing_process_poll:
            # Hold the monitoring row with an empty countdown; skip overwriting
            # the description with a terminal "Processing" label.
            self._finish_process_poll_task(task_name, task_id, existing)
            return

        self._progress.update(
            task_id,
            **self._update_kwargs_for_event(
                event,
                task_name=task_name,
                is_correlated_tool_event=is_correlated_tool_event,
            ),
        )
        if self._is_process_poll_event(event):
            # Anchor field baselines to the current task clock so local ticks
            # between refresh events do not double-count process age.
            task = next(
                (item for item in self._progress.tasks if item.id == task_id),
                None,
            )
            if task is not None:
                self._progress.update(
                    task_id,
                    process_snapshot_task_elapsed=task.elapsed or 0.0,
                )
        self._apply_post_update_lifecycle(
            event,
            task_name=task_name,
            task_id=task_id,
            should_drop_tool_task=should_drop_tool_task,
        )
