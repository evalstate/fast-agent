"""Tests for RichProgressDisplay focusing on state machine correctness."""

import io
import json
import threading
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.text import Text

from fast_agent.event_progress import ProgressAction, ProgressEvent
from fast_agent.ui.rich_progress import (
    DynamicDetailsColumn,
    RichProgressDisplay,
)
from fast_agent.utils.time import format_process_elapsed


def _make_event(
    action: ProgressAction = ProgressAction.SENDING,
    agent_name: str | None = "test-agent",
    target: str = "test-agent",
    details: str = "",
    **kwargs,
) -> ProgressEvent:
    return ProgressEvent(
        action=action,
        target=target,
        details=details,
        agent_name=agent_name,
        **kwargs,
    )


def _make_display() -> RichProgressDisplay:
    """Create a display backed by a non-interactive string console."""
    console = Console(file=open("/dev/null", "w"), force_terminal=True)
    return RichProgressDisplay(console=console)


def _make_buffered_display() -> tuple[RichProgressDisplay, io.StringIO]:
    """Create a display backed by an in-memory console for output assertions."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False)
    return RichProgressDisplay(console=console), buffer


def _task_fields(display: RichProgressDisplay, task_name: str) -> dict[str, Any]:
    task_id = display._taskmap[task_name]
    for task in display._progress.tasks:
        if task.id == task_id:
            return task.fields
    raise AssertionError(f"Task not found for {task_name}")


class TestStopPreventsResume:
    """Issue #4: stop() must permanently disable the display so resume() is a no-op."""

    def test_resume_after_stop_is_noop(self) -> None:
        display = _make_display()
        display.start()
        display.stop()
        assert display._stopped is True
        assert display._paused is True

        # resume() should not restart the display
        display.resume()
        assert display._stopped is True
        assert display._paused is True

    def test_paused_context_manager_after_stop_does_not_resume(self) -> None:
        display = _make_display()
        display.start()
        display.stop()

        with display.paused():
            pass

        # Still stopped after the context manager exits
        assert display._stopped is True
        assert display._paused is True

    def test_update_after_stop_is_noop(self) -> None:
        display = _make_display()
        display.start()
        display.stop()

        event = _make_event()
        display.update(event)
        # No task should have been created
        assert len(display._taskmap) == 0

    def test_start_after_stop_resets_stopped_flag(self) -> None:
        display = _make_display()
        display.start()
        display.stop()
        assert display._stopped is True

        display.start()
        assert display._stopped is False
        assert display._paused is False

    def test_stop_before_start_does_not_emit_blank_line(self) -> None:
        display, buffer = _make_buffered_display()

        display.stop()

        assert display._stopped is True
        assert display._paused is True
        assert buffer.getvalue() == ""


class TestPauseResumeOrdering:
    """Issue #3: resume() must call start() before clearing _paused."""

    def test_resume_ordering_start_before_flag(self) -> None:
        display = _make_display()
        display.start()
        display.pause()
        assert display._paused is True

        # After resume, _paused should be False and display should be active
        display.resume()
        assert display._paused is False

    def test_pause_when_already_paused_is_noop(self) -> None:
        display = _make_display()
        display.start()
        display.pause()
        assert display._paused is True

        # Second pause should be a no-op (no crash)
        display.pause()
        assert display._paused is True

    def test_resume_when_not_paused_is_noop(self) -> None:
        display = _make_display()
        display.start()
        assert display._paused is False

        # resume() when not paused should be a no-op
        display.resume()
        assert display._paused is False

    def test_pause_when_stopped_is_noop(self) -> None:
        display = _make_display()
        display.start()
        display.stop()

        # pause() when stopped should be a no-op
        display.pause()
        assert display._stopped is True
        assert display._paused is True

    def test_resume_retries_after_nested_live_unwinds(self) -> None:
        display = _make_display()
        display.start()
        display.pause()

        with Live(Text("streaming"), console=display.console, auto_refresh=False):
            display.resume()
            assert display._paused is True
            assert display._deferred_resume_at is not None

        display.update(_make_event())
        assert display._paused is False

        display.stop()


class TestDebouncedResume:
    """Debounced resume should coalesce rapid pause/resume transitions."""

    def test_debounced_resume_waits_until_update_after_window(self) -> None:
        display = _make_display()
        display.start()
        display.pause()

        display.resume(debounce_seconds=0.02)
        assert display._paused is True

        time.sleep(0.03)
        display.update(_make_event())
        assert display._paused is False

        display.stop()

    def test_pause_noop_keeps_pending_debounced_resume(self) -> None:
        display = _make_display()
        display.start()
        display.pause()

        display.resume(debounce_seconds=0.05)
        display.pause()

        display.update(_make_event())
        assert display._paused is False

        display.stop()

    def test_pause_noop_can_cancel_pending_debounced_resume(self) -> None:
        display = _make_display()
        display.start()
        display.pause()

        display.resume(debounce_seconds=0.05)
        display.pause(cancel_deferred_on_noop=True)

        display.update(_make_event())
        assert display._paused is True

        display.stop()

    def test_debug_trace_records_deferred_resume_lifecycle(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        trace_path = tmp_path / "progress-trace.jsonl"
        monkeypatch.setenv("FAST_AGENT_PROGRESS_DEBUG_TRACE", str(trace_path))

        display = _make_display()
        display.start()
        display.pause()
        display.resume(debounce_seconds=0.02)
        time.sleep(0.03)
        display.update(_make_event())
        display.stop()

        records = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
        events = [record.get("event") for record in records]

        assert "start" in events
        assert "pause" in events
        assert "resume.deferred" in events
        assert "resume.deferred_flushed" in events
        assert "resume_locked.applied" in events
        assert "stop" in events


class TestUpdateSkipsWhenInactive:
    """Issue #2: update() must skip when stopped and continue state while paused."""

    def test_update_applies_while_paused(self) -> None:
        display = _make_display()
        display.start()
        display.pause()

        event = _make_event()
        display.update(event)

        # Pause suppresses rendering, but state still advances.
        assert "test-agent" in display._taskmap

        display.resume()
        assert "test-agent" in display._taskmap

    def test_update_skipped_when_stopped(self) -> None:
        display = _make_display()
        display.start()
        display.stop()

        event = _make_event()
        display.update(event)
        assert len(display._taskmap) == 0

    def test_update_works_when_active(self) -> None:
        display = _make_display()
        display.start()

        event = _make_event()
        display.update(event)
        assert "test-agent" in display._taskmap

        display.stop()


class TestToolProgressNoDoubleRender:
    """Issue #5: TOOL_PROGRESS without total should not call reset() then update()."""

    def test_tool_progress_without_total_does_not_reset(self) -> None:
        display = _make_display()
        display.start()

        # First create the task
        event = _make_event(action=ProgressAction.SENDING)
        display.update(event)
        assert "test-agent" in display._taskmap

        # Now send TOOL_PROGRESS with progress but no total
        event = _make_event(
            action=ProgressAction.TOOL_PROGRESS,
            progress=5.0,
            total=None,
        )
        display.update(event)
        # Should succeed without error (no intermediate reset)

        display.stop()


class TestParallelToolProgress:
    """Tool progress events with correlation IDs should get distinct rows."""

    def test_parallel_tool_progress_creates_separate_tasks(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                progress=1.0,
                total=3.0,
                correlation_id="tool-call-1",
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                progress=2.0,
                total=3.0,
                correlation_id="tool-call-2",
            )
        )

        assert "test-agent::tool-call-1" in display._taskmap
        assert "test-agent::tool-call-2" in display._taskmap
        assert len(display._taskmap) == 2

        display.stop()

    def test_correlated_tool_details_keep_id_prefix_and_suffix(self) -> None:
        display = _make_display()
        display.start()

        correlation_id = "call_DTwuI86WabcdefK6AdYX"
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                details="execute",
                correlation_id=correlation_id,
            )
        )

        fields = _task_fields(display, f"test-agent::{correlation_id}")
        assert fields["details"].endswith("id: call_…K6AdYX")

        display.stop()


class TestAggregatorInitializedVisibility:
    """Running status should only render when it carries meaningful details."""

    def test_running_event_without_details_is_suppressed(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.AGGREGATOR_INITIALIZED,
                details="",
            )
        )

        assert len(display._taskmap) == 0

        display.stop()

    def test_running_event_with_details_is_rendered(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.AGGREGATOR_INITIALIZED,
                details="warming up tool registry",
            )
        )

        assert "test-agent" in display._taskmap

        display.stop()

    def test_completed_tool_progress_row_is_removed(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-1",
            )
        )
        assert "test-agent::tool-call-1" in display._taskmap

        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                correlation_id="tool-call-1",
                tool_state="completed",
                tool_terminal=True,
            )
        )

        assert "test-agent::tool-call-1" not in display._taskmap

        display.stop()

    def test_poll_process_uses_dense_braille_spinner(self) -> None:
        display = _make_display()
        event = _make_event(
            action=ProgressAction.CALLING_TOOL,
            correlation_id="tool-call-poll",
            tool_name="poll_process",
            details="pid 4321 · ≤30s",
        )

        description = display._description_for_event(event)

        assert "Monitoring" in description
        spinner = display._description_spinner.spinner
        assert spinner.name == "braille_dense"
        assert "⢸⡇ " in spinner.frames

    def test_process_elapsed_time_ticks_during_rendering(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=open("/dev/null", "w"), force_terminal=True, width=120),
            default_agent_name="test-agent",
        )
        display.start()
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="call_abcdef0123456789",
                tool_name="poll_process",
                details="process-4",
                process_id="process-4",
                process_elapsed_seconds=65,
                process_command="uv run worker.py",
                process_wait_seconds=30,
                process_has_observed_output=True,
                process_seconds_since_last_output=4,
                process_total_output_bytes=12_500,
            )
        )
        task_id = display._taskmap["test-agent::call_abcdef0123456789"]
        task = next(task for task in display._progress.tasks if task.id == task_id)
        assert task.fields["target"] == "process-4"
        assert task.start_time is not None
        task.start_time -= 5

        rendered = DynamicDetailsColumn().render(task).plain

        assert rendered == (
            "running · 1m10s · last output 9s ago · poll 25s · "
            "12.5KB · uv run worker.py"
        )
        display.stop()

    def test_process_poll_countdown_reports_finishing_at_deadline(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=open("/dev/null", "w"), force_terminal=True),
            default_agent_name="test-agent",
        )
        display.start()
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="call-poll",
                tool_name="poll_process",
                details="process-4",
                process_id="process-4",
                process_elapsed_seconds=65,
                process_wait_seconds=5,
            )
        )
        task_id = display._taskmap["test-agent::call-poll"]
        task = next(task for task in display._progress.tasks if task.id == task_id)
        assert task.start_time is not None
        task.start_time -= 5

        assert DynamicDetailsColumn().render(task).plain == (
            "running · 1m10s · poll finishing…"
        )
        display.stop()

    def test_process_output_progress_refreshes_live_poll_baselines(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=open("/dev/null", "w"), force_terminal=True),
            default_agent_name="test-agent",
        )
        display.start()
        initial = _make_event(
            action=ProgressAction.CALLING_TOOL,
            correlation_id="call-poll",
            tool_name="poll_process",
            details="process-4",
            process_id="process-4",
            process_elapsed_seconds=65,
            process_command="uv run worker.py",
            process_wait_seconds=30,
            process_has_observed_output=False,
            process_seconds_since_last_output=65,
            process_total_output_bytes=0,
        )
        display.update(initial)
        display.update(
            initial.model_copy(
                update={
                    "tool_event": "progress",
                    "process_elapsed_seconds": 70,
                    "process_wait_seconds": 25,
                    "process_has_observed_output": True,
                    "process_seconds_since_last_output": 0,
                    "process_total_output_bytes": 25_000,
                }
            )
        )
        task_id = display._taskmap["test-agent::call-poll"]
        task = next(task for task in display._progress.tasks if task.id == task_id)

        assert DynamicDetailsColumn().render(task).plain == (
            "running · 1m10s · output now · poll 25s · 25.0KB · uv run worker.py"
        )
        display.stop()

    def test_poll_process_keeps_non_default_agent_name(self) -> None:
        display = RichProgressDisplay(default_agent_name="default-agent")
        event = _make_event(
            action=ProgressAction.CALLING_TOOL,
            agent_name="reviewer",
            target="reviewer",
            correlation_id="tool-call-poll",
            tool_name="poll_process",
            details="process-4",
        )

        update = display._update_kwargs_for_event(
            event,
            task_name="reviewer::tool-call-poll",
            is_correlated_tool_event=True,
        )

        assert update["target"] == "reviewer"

    def test_process_elapsed_uses_aligned_minutes_and_seconds(self) -> None:
        assert format_process_elapsed(49) == "0m49s"
        assert format_process_elapsed(600) == "10m00s"
        assert format_process_elapsed(3700) == "1h01m"

    def test_full_progress_without_terminal_state_keeps_row(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-progress",
            )
        )
        assert "test-agent::tool-call-progress" in display._taskmap

        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                progress=1.0,
                total=1.0,
                correlation_id="tool-call-progress",
            )
        )

        assert "test-agent::tool-call-progress" in display._taskmap

        display.stop()

    def test_calling_tool_stop_event_row_is_removed(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-2",
                tool_event="start",
            )
        )
        assert "test-agent::tool-call-2" in display._taskmap

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-2",
                tool_event="stop",
                tool_terminal=True,
            )
        )

        assert "test-agent::tool-call-2" not in display._taskmap
        assert "test-agent::tool-call-2" not in display._task_kind

        display.stop()

    def test_calling_tool_failed_event_row_is_removed(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-failed",
                tool_event="start",
            )
        )
        assert "test-agent::tool-call-failed" in display._taskmap

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-failed",
                tool_event="failed",
                tool_terminal=True,
            )
        )

        assert "test-agent::tool-call-failed" not in display._taskmap

        display.stop()

    def test_tool_progress_failed_final_state_removes_row(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-failed-details",
                tool_event="start",
            )
        )
        assert "test-agent::tool-call-failed-details" in display._taskmap

        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                correlation_id="tool-call-failed-details",
                details="failed: boom",
                tool_state="failed",
                tool_terminal=True,
            )
        )

        assert "test-agent::tool-call-failed-details" not in display._taskmap

        display.stop()

    def test_calling_tool_stop_while_paused_removes_row(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-3",
                tool_event="start",
            )
        )
        assert "test-agent::tool-call-3" in display._taskmap

        display.pause()
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-3",
                tool_event="stop",
                tool_terminal=True,
            )
        )
        # State advances while paused, so terminal events can clean rows.
        assert "test-agent::tool-call-3" not in display._taskmap

        display.resume()
        assert "test-agent::tool-call-3" not in display._taskmap

        display.stop()

    def test_completed_tool_progress_while_paused_removes_row(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-call-4",
            )
        )
        assert "test-agent::tool-call-4" in display._taskmap

        display.pause()
        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                correlation_id="tool-call-4",
                tool_state="completed",
                tool_terminal=True,
            )
        )
        # State advances while paused, so terminal events can clean rows.
        assert "test-agent::tool-call-4" not in display._taskmap

        display.resume()
        assert "test-agent::tool-call-4" not in display._taskmap

        display.stop()

    def test_tool_progress_with_total_sets_completed(self) -> None:
        display = _make_display()
        display.start()

        event = _make_event(action=ProgressAction.SENDING)
        display.update(event)

        event = _make_event(
            action=ProgressAction.TOOL_PROGRESS,
            progress=50.0,
            total=100.0,
        )
        display.update(event)
        # Should succeed without error

        display.stop()

    def test_internal_execute_tool_does_not_create_correlated_parallel_rows(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="exec-call-1",
                tool_name="execute",
                server_name="acp_terminal",
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="exec-call-2",
                tool_name="execute",
                server_name="acp_terminal",
            )
        )

        assert "test-agent" in display._taskmap
        assert "test-agent::exec-call-1" not in display._taskmap
        assert "test-agent::exec-call-2" not in display._taskmap
        assert len(display._taskmap) == 1

        display.stop()

    def test_non_shell_execute_tool_uses_correlated_parallel_rows(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="exec-call-1",
                tool_name="execute",
                server_name="codex",
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="exec-call-2",
                tool_name="execute",
                server_name="codex",
            )
        )

        assert "test-agent::exec-call-1" in display._taskmap
        assert "test-agent::exec-call-2" in display._taskmap
        assert len(display._taskmap) == 2

        display.stop()

    def test_fatal_error_row_is_removed_after_update(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.FATAL_ERROR,
                agent_name=None,
                target="127-0-0-1",
                details="Connection refused",
            )
        )

        # Fatal errors should not leave sticky rows in future turns.
        assert "default" not in display._taskmap

        display.stop()


class TestFinishedEventHandlesNoneElapsed:
    """Issue #8: FINISHED event must handle None elapsed without crashing."""

    def test_finished_event_with_no_prior_start(self) -> None:
        display = _make_display()
        display.start()

        # Send FINISHED as the very first event for an agent
        event = _make_event(action=ProgressAction.FINISHED)
        # This should not raise TypeError from time.gmtime(None)
        display.update(event)

        display.stop()

    def test_finished_event_handles_sparse_task_ids(self) -> None:
        display = _make_display()
        display.start()

        # Create and drop a lifecycle row so the next task id is no longer
        # aligned with progress.tasks list indices.
        display.update(
            _make_event(
                action=ProgressAction.STARTING,
                agent_name="agent-1",
                target="agent-1",
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.READY,
                agent_name="agent-1",
                target="agent-1",
            )
        )
        assert "agent-1" not in display._taskmap

        # This should not raise IndexError even when the new task id is sparse.
        display.update(
            _make_event(
                action=ProgressAction.FINISHED,
                agent_name="agent-2",
                target="agent-2",
            )
        )
        assert "agent-2" in display._taskmap

        display.stop()


class TestAgentLifecycleRows:
    """Startup lifecycle rows should not linger in the progress board."""

    def test_resource_read_completion_clears_reading_row(self) -> None:
        display = _make_display()
        display.start()

        display.update(_make_event(action=ProgressAction.READING_RESOURCE))
        assert "test-agent" in display._taskmap

        display.update(_make_event(action=ProgressAction.RESOURCE_READ))
        assert "test-agent" not in display._taskmap

        display.stop()

    def test_ready_event_row_is_cleared(self) -> None:
        display = _make_display()
        display.start()

        display.update(_make_event(action=ProgressAction.STARTING))
        assert "test-agent" in display._taskmap

        display.update(_make_event(action=ProgressAction.READY))
        assert "test-agent" not in display._taskmap

        display.stop()

    def test_ready_event_while_paused_still_clears_row(self) -> None:
        display = _make_display()
        display.start()

        display.update(_make_event(action=ProgressAction.STARTING))
        assert "test-agent" in display._taskmap

        display.pause()
        display.update(_make_event(action=ProgressAction.READY))
        assert "test-agent" not in display._taskmap

        display.resume()
        assert "test-agent" not in display._taskmap

        display.stop()


class TestAgentTaskClearing:
    """Interrupted sends should be able to clear stale rows for one agent."""

    def test_clear_agent_tasks_removes_base_and_correlated_rows(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.SENDING,
                agent_name="agent-a",
                target="agent-a",
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                agent_name="agent-a",
                target="agent-a",
                correlation_id="tool-a-1",
                tool_event="start",
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.SENDING,
                agent_name="agent-b",
                target="agent-b",
            )
        )

        assert "agent-a" in display._taskmap
        assert "agent-a::tool-a-1" in display._taskmap
        assert "agent-b" in display._taskmap

        display.clear_agent_tasks("agent-a")

        assert "agent-a" not in display._taskmap
        assert "agent-a::tool-a-1" not in display._taskmap
        assert "agent-a::tool-a-1" not in display._task_kind
        assert "agent-b" in display._taskmap

        display.stop()


class TestTaskKindClassification:
    """Internal row-kind metadata should reflect the latest event class."""

    def test_stream_rows_are_classified_as_stream(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.STREAMING,
                streaming_tokens="42",
            )
        )

        assert display._task_kind.get("test-agent") == "stream"
        display.stop()

    def test_correlated_tool_rows_are_classified_as_tool(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-kind",
                tool_event="start",
            )
        )

        assert display._task_kind.get("test-agent::tool-kind") == "tool"
        display.stop()


class TestCorrelationIdDetails:
    """Correlated tool rows should surface a short id in details for clarity."""

    def test_correlated_tool_row_appends_short_id_to_details(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="call_abcdef0123456789",
                details="web (search)",
            )
        )

        fields = _task_fields(display, "test-agent::call_abcdef0123456789")
        details = str(fields.get("details", ""))
        assert "web (search)" in details
        assert "id: call_…456789" in details

        display.stop()

    def test_correlated_tool_row_without_details_still_shows_id(self) -> None:
        display = _make_display()
        display.start()

        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="tool-id",
            )
        )

        fields = _task_fields(display, "test-agent::tool-id")
        details = str(fields.get("details", ""))
        assert details == "id: tool-id"

        display.stop()


class TestThreadSafety:
    """Verify that concurrent pause/resume/update don't crash."""

    def test_concurrent_pause_resume_update(self) -> None:
        display = _make_display()
        display.start()

        errors: list[Exception] = []
        start_event = threading.Event()

        def updater() -> None:
            start_event.wait()
            for _ in range(25):
                try:
                    display.update(
                        _make_event(
                            action=ProgressAction.STREAMING,
                            streaming_tokens="42",
                        )
                    )
                except Exception as e:
                    errors.append(e)

        def pauser() -> None:
            start_event.wait()
            for _ in range(25):
                try:
                    display.pause()
                    display.resume()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=pauser),
            threading.Thread(target=updater),
        ]
        for t in threads:
            t.start()

        start_event.set()

        for t in threads:
            t.join(timeout=2)
            assert not t.is_alive()

        display.stop()
        assert errors == [], f"Concurrent operations raised: {errors}"
