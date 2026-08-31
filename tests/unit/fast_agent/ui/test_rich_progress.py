"""Tests for RichProgressDisplay focusing on state machine correctness."""

import io
import json
import threading
import time
from collections.abc import Callable
from typing import Any

from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from fast_agent.event_progress import ProgressAction, ProgressEvent, SubagentMonitorSnapshot
from fast_agent.ui.display_suppression import suppress_interactive_display
from fast_agent.ui.progress.display import (
    DynamicDetailsColumn,
    RichProgressDisplay,
    SpinnerDescriptionColumn,
    _format_compacting_track,
)
from fast_agent.ui.progress.subagent_table import _cache_percentage_text
from fast_agent.utils.time import format_process_elapsed


class _CountingSpinner(Spinner):
    def __init__(self) -> None:
        super().__init__("dots")
        self.render_count = 0

    def render(self, time: float) -> RenderableType:
        self.render_count += 1
        return Text("abc")


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
    console = Console(file=io.StringIO(), force_terminal=True)
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


def _subagent_event(
    *,
    action: ProgressAction = ProgressAction.RUNNING,
    agent_name: str = "parent[reviewer]",
    label: str = "reviewer",
    row_id: str = "parent::subagent::outer-call",
    state: str = "Thinking",
    turn: int = 2,
    input_tokens: int = 100,
    cache_percentage: float | None = None,
    output_tokens: int = 20,
    output_estimated: bool = False,
    model: str | None = "gpt-5.6-terra",
    context_percentage: float | None = None,
    details: str = "",
) -> ProgressEvent:
    return _make_event(
        action=action,
        agent_name=agent_name,
        target=label,
        instance_name=row_id,
        tool_name="subagent",
        tool_event="subagent_monitor",
        activity=state,
        details=details,
        subagent_monitor=SubagentMonitorSnapshot(
            model=model,
            context_percentage=context_percentage,
            state=state,
            turn=turn,
            input_tokens=input_tokens,
            cache_percentage=cache_percentage,
            output_tokens=output_tokens,
            output_estimated=output_estimated,
        ),
    )


def test_compacting_track_drains_by_braille_row_then_repeats() -> None:
    frames = [_format_compacting_track((index + 0.1) * 0.18) for index in range(6)]
    assert frames == ["⣿⣿⣿", "⣶⣶⣶", "⣤⣤⣤", "⣀⣀⣀", "   ", "⣿⣿⣿"]


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

    def test_subagents_use_sine_spinner_without_changing_ordinary_progress(self) -> None:
        display = _make_display()

        assert display._description_spinner.spinner.name == "braille_dense"
        assert display._progress._subagent_spinner.name == "braille_sine"
        assert "⡼⢷⣤" in display._progress._subagent_spinner.frames
        assert all(frame.strip() for frame in display._progress._subagent_spinner.frames)

    def test_process_poll_countdown_track_replaces_pulse_spinner(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=True),
            default_agent_name="test-agent",
        )
        display.start()
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="call-poll-countdown",
                tool_name="poll_process",
                details="process-4",
                process_id="process-4",
                process_elapsed_seconds=10,
                process_wait_seconds=30,
            )
        )
        task_id = display._taskmap["test-agent::call-poll-countdown"]
        task = next(task for task in display._progress.tasks if task.id == task_id)
        assert task.start_time is not None
        task.start_time -= 10  # 10s into a 30s wait → ~2/3 remaining track

        column = SpinnerDescriptionColumn(spinner_name="braille_dense")
        rendered = column.render(task)
        assert "Monitoring" in rendered.plain
        # The countdown immediately follows the compact monitoring label.
        prefix = "▎◀ Monitoring "
        assert rendered.plain.startswith(prefix)
        assert len(rendered.plain) == len(prefix) + 3
        display.stop()

    def test_process_poll_heartbeats_toggle_next_dot_blink(self) -> None:
        display = _make_display()
        event = _make_event(
            action=ProgressAction.CALLING_TOOL,
            correlation_id="call-poll-blink",
            tool_name="poll_process",
            process_wait_seconds=50,
        )

        display.update(event)
        fields = _task_fields(display, "test-agent::call-poll-blink")
        assert fields["process_poll_blink_next"] is False

        display.update(event)
        fields = _task_fields(display, "test-agent::call-poll-blink")
        assert fields["process_poll_blink_next"] is True

        display.update(event)
        fields = _task_fields(display, "test-agent::call-poll-blink")
        assert fields["process_poll_blink_next"] is False

    def test_process_poll_completion_snaps_countdown_empty_before_drop(self, monkeypatch) -> None:
        deferred_callbacks: list[Callable[[], None]] = []

        class _DeferredTimer:
            def __init__(self, _delay: float, callback: Callable[[], None]) -> None:
                self._callback = callback
                self.daemon = False

            def start(self) -> None:
                deferred_callbacks.append(self._callback)

        monkeypatch.setattr("fast_agent.ui.progress.display.Timer", _DeferredTimer)
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=True),
            default_agent_name="test-agent",
        )
        display.start()
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="call-poll-finish",
                tool_name="poll_process",
                details="process-4",
                process_id="process-4",
                process_elapsed_seconds=5,
                process_wait_seconds=30,
            )
        )
        task_id = display._taskmap["test-agent::call-poll-finish"]
        task = next(task for task in display._progress.tasks if task.id == task_id)
        assert task.start_time is not None
        task.start_time -= 10

        mid = SpinnerDescriptionColumn(spinner_name="braille_dense").render(task)
        assert "Monitoring" in mid.plain
        assert not mid.plain.endswith("   ")

        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                correlation_id="call-poll-finish",
                tool_name="poll_process",
                tool_state="completed",
                tool_terminal=True,
                process_yield_reason="deadline",
            )
        )
        # Row is held briefly with an empty track before drop.
        assert "test-agent::call-poll-finish" in display._taskmap
        finished = next(task for task in display._progress.tasks if task.id == task_id)
        empty = SpinnerDescriptionColumn(spinner_name="braille_dense").render(finished)
        assert "Monitoring" in empty.plain
        assert empty.plain.endswith("   ")
        assert len(empty.plain) == len("▎◀ Monitoring ") + 3
        assert len(deferred_callbacks) == 1
        deferred_callbacks[0]()
        assert "test-agent::call-poll-finish" not in display._taskmap
        display.stop()

    def test_process_poll_early_completion_drops_without_fake_empty_frame(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=True),
            default_agent_name="test-agent",
        )
        display.start()
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="call-poll-early",
                tool_name="poll_process",
                process_id="process-4",
                process_wait_seconds=30,
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                correlation_id="call-poll-early",
                tool_name="poll_process",
                tool_state="completed",
                tool_terminal=True,
                process_yield_reason="completion",
            )
        )

        assert "test-agent::call-poll-early" not in display._taskmap
        display.stop()

    def test_process_poll_refresh_keeps_countdown_start_time(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=True),
            default_agent_name="test-agent",
        )
        display.start()
        event = _make_event(
            action=ProgressAction.CALLING_TOOL,
            correlation_id="call-poll-stable",
            tool_name="poll_process",
            details="process-4",
            process_id="process-4",
            process_elapsed_seconds=0,
            process_wait_seconds=30,
        )
        display.update(event)
        task_id = display._taskmap["test-agent::call-poll-stable"]
        task = next(task for task in display._progress.tasks if task.id == task_id)
        assert task.start_time is not None
        original_start = task.start_time

        display.update(
            event.model_copy(
                update={
                    "process_elapsed_seconds": 5,
                    "process_has_observed_output": True,
                    "process_seconds_since_last_output": 1,
                }
            )
        )
        refreshed = next(task for task in display._progress.tasks if task.id == task_id)
        assert refreshed.start_time == original_start
        display.stop()

    def test_process_elapsed_time_ticks_during_rendering(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=True, width=120),
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
                process_seconds_since_last_stdout=4,
                process_stdout_bytes=12_000,
                process_stderr_bytes=500,
            )
        )
        task_id = display._taskmap["test-agent::call_abcdef0123456789"]
        task = next(task for task in display._progress.tasks if task.id == task_id)
        assert task.fields["target"] == "process-4"
        assert task.start_time is not None
        task.start_time -= 5  # local tick only; process baselines stay fixed

        rendered = DynamicDetailsColumn().render(task)
        # 65s base + 5s local tick; 4s-old output ages into the warm window.
        assert rendered.plain == "out  9s · err   — · time 1m10s · size 12.5KB · uv run worker.py"
        assert any(str(span.style) == "green" for span in rendered.spans)
        display.stop()

    def test_process_monitor_row_prioritizes_progress_over_command(self) -> None:
        command = (
            "uv run python -m fast_agent.cli.__main__ serve "
            "--transport streamable-http --host 127.0.0.1 --port 8000"
        )
        rendered_by_width: dict[int, str] = {}

        for width in (50, 60, 70, 74, 84, 98, 120, 160):
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, width=width)
            display = RichProgressDisplay(
                console=console,
                default_agent_name="test-agent",
            )
            display.update(
                _make_event(
                    action=ProgressAction.CALLING_TOOL,
                    correlation_id=f"poll-{width}",
                    tool_name="poll_process",
                    process_id="process-4",
                    process_elapsed_seconds=3700,
                    process_wait_seconds=120,
                    process_command=command,
                    process_seconds_since_last_stdout=3,
                    process_seconds_since_last_stderr=47,
                    process_stdout_bytes=1_200_000,
                    process_stderr_bytes=34_567,
                    process_total_output_bytes=1_234_567,
                )
            )

            console.print(*display._progress.get_renderables())
            lines = buffer.getvalue().splitlines()
            assert len(lines) == 1
            assert len(lines[0]) <= width
            assert lines[0] == lines[0].rstrip()
            rendered_by_width[width] = lines[0]

        for rendered in rendered_by_width.values():
            assert "▎◀ Monitoring ⣿⣿ " in rendered
            assert "process-4" in rendered
            assert "out" in rendered
            assert "err" in rendered

        assert "time" not in rendered_by_width[60]
        assert "1h01m" in rendered_by_width[60]
        assert "1.2MB" not in rendered_by_width[60]
        assert "1.2MB" in rendered_by_width[70]
        assert "time" not in rendered_by_width[74]
        assert "1.2MB" in rendered_by_width[74]
        assert "time 1h01m" in rendered_by_width[84]
        assert "size  1.2MB" in rendered_by_width[84]
        assert "uv run" not in rendered_by_width[84]
        assert "uv run" in rendered_by_width[98]
        assert rendered_by_width[98].endswith("…")
        assert rendered_by_width[120].endswith("…")
        assert rendered_by_width[160].endswith("…")
        assert "--transport streamable-http" not in rendered_by_width[160]
        assert len(rendered_by_width[160]) < 160

    def test_process_stats_do_not_regress_with_a_wide_target(self) -> None:
        command = "uv run worker.py " + "x" * 80
        for width in (70, 74, 84, 98):
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, width=width)
            display = RichProgressDisplay(
                console=console,
                default_agent_name="test-agent",
            )
            display.update(
                _make_event(
                    action=ProgressAction.READING_RESOURCE,
                    target="server",
                    details="generic-" + "x" * 120,
                )
            )
            display.update(
                _make_event(
                    action=ProgressAction.CALLING_TOOL,
                    correlation_id=f"poll-wide-target-{width}",
                    tool_name="poll_process",
                    process_id="process-12345678",
                    process_elapsed_seconds=3700,
                    process_wait_seconds=120,
                    process_command=command,
                    process_seconds_since_last_stdout=3,
                    process_seconds_since_last_stderr=47,
                    process_stdout_bytes=1_200_000,
                    process_stderr_bytes=34_567,
                )
            )

            console.print(*display._progress.get_renderables())
            rendered = next(line for line in buffer.getvalue().splitlines() if "Monitoring" in line)

            assert "process-12345678" in rendered
            assert "out" in rendered
            assert "err" in rendered
            assert "1h01m" in rendered
            assert "1.2MB" in rendered
            assert rendered == rendered.rstrip()
            if width == 98:
                assert "uv run" in rendered
                assert rendered.endswith("…")

    def test_process_command_preview_collapses_whitespace_and_is_bounded(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=False, width=160),
            default_agent_name="test-agent",
        )
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="poll-command-preview",
                tool_name="poll_process",
                process_command=f"uv   run\nworker.py {'x' * 80}",
            )
        )
        task_id = display._taskmap["test-agent::poll-command-preview"]
        task = next(task for task in display._progress.tasks if task.id == task_id)

        rendered = DynamicDetailsColumn().render(task)

        assert "\n" not in rendered.plain
        assert "uv run worker.py" in rendered.plain
        assert rendered.plain.endswith("…")
        assert cell_len(rendered.plain.rsplit(" · ", maxsplit=1)[-1]) == 48

        task.fields["process_command"] = f"uv run {'界' * 80}"
        rendered = DynamicDetailsColumn().render(task)
        command_preview = rendered.plain.rsplit(" · ", maxsplit=1)[-1]
        assert command_preview.endswith("…")
        assert cell_len(command_preview) <= 48

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=98)
        display = RichProgressDisplay(
            console=console,
            default_agent_name="test-agent",
        )
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="poll-wide-command-preview",
                tool_name="poll_process",
                process_id="process-12345678",
                process_command=f"uv run {'界' * 80}",
            )
        )
        console.print(*display._progress.get_renderables())
        line = buffer.getvalue().rstrip()
        assert cell_len(line) <= 98
        assert line.endswith("…")

    def test_wide_generic_progress_keeps_detail_and_activity_glyph(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=200)
        display = RichProgressDisplay(console=console)
        display._description_spinner.spinner = _CountingSpinner()
        details = f"detail-{'x' * 120}-end"
        display.update(
            _make_event(
                action=ProgressAction.READING_RESOURCE,
                target="server",
                details=details,
            )
        )

        console.print(*display._progress.get_renderables())
        rendered = buffer.getvalue()

        assert "Reading Resourceabc" in rendered
        assert details in rendered

    def test_process_output_progress_refreshes_live_poll_baselines(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=True),
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

        rendered = DynamicDetailsColumn().render(task)
        assert rendered.plain == "out   — · err   — · time 1m10s · size 25.0KB · uv run worker.py"
        display.stop()

    def test_subagent_elapsed_time_ticks_between_monitor_events(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=False, width=100)
        )
        display.start()
        display.update(
            _make_event(
                action=ProgressAction.RUNNING,
                instance_name="test-agent::subagent::call-1",
                tool_event="subagent_monitor",
                activity="Thinking",
                subagent_monitor=SubagentMonitorSnapshot(
                    state="Thinking",
                    turn=1,
                    input_tokens=58_662,
                    output_tokens=7_095,
                ),
                elapsed_seconds=2,
            )
        )
        task_id = display._taskmap["test-agent::subagent::call-1"]
        task = next(task for task in display._progress.tasks if task.id == task_id)
        assert task.start_time is not None
        task.start_time -= 5.5

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=100)
        console.print(*display._progress.get_renderables())
        rendered = buffer.getvalue()

        assert "Thinking · 7s" in rendered
        assert "58,662" in rendered
        assert "7,095" in rendered
        display.stop()

    def test_process_output_activity_fades_then_goes_quiet(self) -> None:
        display = RichProgressDisplay(
            console=Console(file=io.StringIO(), force_terminal=True),
            default_agent_name="test-agent",
        )
        display.start()
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                correlation_id="call-poll-quiet",
                tool_name="poll_process",
                details="process-4",
                process_id="process-4",
                process_elapsed_seconds=90,
                process_has_observed_output=True,
                process_seconds_since_last_output=12,
                process_total_output_bytes=12_500,
            )
        )
        task_id = display._taskmap["test-agent::call-poll-quiet"]
        task = next(task for task in display._progress.tasks if task.id == task_id)

        warm = DynamicDetailsColumn().render(task)
        assert warm.plain == "out   — · err   — · time 1m30s · size 12.5KB"

        task.fields["process_seconds_since_last_output"] = 90
        quiet = DynamicDetailsColumn().render(task)
        assert quiet.plain == "out   — · err   — · time 1m30s · size 12.5KB"
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


class TestSubagentMonitoringRows:
    """Built-in subagent monitoring stays visible while a parent awaits children."""

    def test_registered_child_folds_delayed_generic_progress(self) -> None:
        display = _make_display()
        display.start()
        child_name = "parent[reviewer]"
        display.fold_agent_progress(child_name)

        display.update(
            _make_event(
                action=ProgressAction.SENDING,
                agent_name=child_name,
                target="reviewer",
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                agent_name=child_name,
                target="reviewer",
                tool_name="read_text_file",
                correlation_id="inner-call",
            )
        )
        display.update(
            _subagent_event(
                agent_name=child_name,
                state="tool: read_text_file",
            )
        )

        assert set(display._taskmap) == {"parent::subagent::outer-call"}
        snapshot = _task_fields(display, "parent::subagent::outer-call")["subagent_monitor"]
        assert snapshot.state == "tool: read_text_file"
        display.stop()

    def test_monitor_only_scope_folds_generic_child_progress(self) -> None:
        display = _make_display()
        display.start()

        with suppress_interactive_display("monitor_only"):
            display.update(
                _make_event(
                    action=ProgressAction.CALLING_TOOL,
                    agent_name="parent",
                    target="parent",
                    tool_name="agent__ripgrep_spark",
                    correlation_id="inner-call",
                )
            )
            display.update(_subagent_event())

        assert set(display._taskmap) == {"parent::subagent::outer-call"}
        display.stop()

    def test_paused_parallel_children_restore_running_row_and_clean_up(self) -> None:
        display = _make_display()
        display.start()
        display.pause()

        display.update(
            _subagent_event(
                agent_name="parent[scout]",
                label="scout",
                row_id="parent::subagent::scout-call",
                turn=1,
                input_tokens=3,
                output_tokens=2,
            )
        )
        display.update(
            _subagent_event(
                agent_name="parent[verifier]",
                label="verifier",
                row_id="parent::subagent::verifier-call",
                state="tool: lookup",
                turn=1,
                input_tokens=5,
                output_tokens=3,
            )
        )

        assert set(display._taskmap) == {
            "parent::subagent::scout-call",
            "parent::subagent::verifier-call",
        }

        # Clone initialization owns the logical agent row, not the durable
        # subagent monitor row.
        display.update(
            _make_event(
                action=ProgressAction.READY,
                agent_name="parent[scout]",
                target="scout",
            )
        )
        assert "parent::subagent::scout-call" in display._taskmap

        display.resume()
        display.update(
            _make_event(
                action=ProgressAction.STREAMING,
                agent_name="parent[scout]",
                target="scout",
                streaming_tokens="18",
                instance_name="parent::subagent::scout-call",
                tool_event="subagent_monitor",
            )
        )
        assert "18" in next(
            task.description
            for task in display._progress.tasks
            if task.id == display._taskmap["parent::subagent::scout-call"]
        )

        display.update(
            _subagent_event(
                agent_name="parent[scout]",
                label="scout",
                row_id="parent::subagent::scout-call",
                state="Thinking",
                turn=1,
                input_tokens=3,
                output_tokens=2,
            )
        )
        assert (
            _task_fields(display, "parent::subagent::scout-call")["subagent_monitor"].state
            == "Thinking"
        )

        display.update(
            _subagent_event(
                action=ProgressAction.READY,
                agent_name="parent[scout]",
                label="scout",
                row_id="parent::subagent::scout-call",
            )
        )
        assert set(display._taskmap) == {"parent::subagent::verifier-call"}

        display.update(
            _subagent_event(
                action=ProgressAction.READY,
                agent_name="parent[verifier]",
                label="verifier",
                row_id="parent::subagent::verifier-call",
            )
        )
        assert display._taskmap == {}
        display.stop()

    def test_subagent_process_is_folded_then_promoted(self) -> None:
        display = _make_display()
        display.start()
        row_id = "parent::subagent::outer-call"
        child_name = "parent[reviewer]"
        display.fold_agent_progress(child_name)
        display.update(_subagent_event(agent_name=child_name, row_id=row_id))

        with suppress_interactive_display("monitor_only"):
            display.update(
                _make_event(
                    action=ProgressAction.CALLING_TOOL,
                    agent_name=child_name,
                    target="reviewer",
                    tool_name="poll_process",
                    correlation_id="poll-1",
                    process_id="process-41",
                    process_elapsed_seconds=42,
                )
            )

        process_task_name = f"{child_name}::poll-1"
        process_fields = _task_fields(display, process_task_name)
        assert process_fields["process_owner_row"] == row_id
        process_task = next(
            task
            for task in display._progress.tasks
            if task.id == display._taskmap[process_task_name]
        )
        assert process_task.visible is False

        display.update(
            _subagent_event(
                action=ProgressAction.READY,
                agent_name=child_name,
                row_id=row_id,
            )
        )

        assert row_id not in display._taskmap
        assert process_task_name in display._taskmap
        assert process_task.fields["process_owner_row"] is None
        assert process_task.visible is True
        display.stop()

    def test_cancelled_subagent_discards_hidden_process_monitor(self) -> None:
        display = _make_display()
        display.start()
        row_id = "parent::subagent::outer-call"
        child_name = "parent[reviewer]"
        display.fold_agent_progress(child_name)
        display.update(_subagent_event(agent_name=child_name, row_id=row_id))
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                agent_name=child_name,
                target="reviewer",
                tool_name="poll_process",
                correlation_id="poll-1",
                process_id="process-41",
                process_elapsed_seconds=42,
            )
        )
        process_task_name = f"{child_name}::poll-1"

        display.update(
            _subagent_event(
                action=ProgressAction.READY,
                agent_name=child_name,
                row_id=row_id,
                details="cancelled",
            )
        )

        assert row_id not in display._taskmap
        assert process_task_name not in display._taskmap
        display.stop()

    def test_terminal_process_poll_is_not_promoted_with_completed_subagent(self) -> None:
        display = _make_display()
        display.start()
        row_id = "parent::subagent::outer-call"
        child_name = "parent[reviewer]"
        display.fold_agent_progress(child_name)
        display.update(_subagent_event(agent_name=child_name, row_id=row_id))
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                agent_name=child_name,
                target="reviewer",
                tool_name="poll_process",
                correlation_id="poll-1",
                process_id="process-41",
                process_elapsed_seconds=42,
                process_wait_seconds=30,
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.TOOL_PROGRESS,
                agent_name=child_name,
                target="reviewer",
                tool_name="poll_process",
                correlation_id="poll-1",
                process_id="process-41",
                process_elapsed_seconds=42,
                process_wait_seconds=30,
                tool_terminal=True,
                process_yield_reason="deadline",
            )
        )
        process_task_name = f"{child_name}::poll-1"
        process_task = next(
            task
            for task in display._progress.tasks
            if task.id == display._taskmap[process_task_name]
        )
        assert process_task.stop_time is not None

        display.update(
            _subagent_event(
                action=ProgressAction.READY,
                agent_name=child_name,
                row_id=row_id,
                details="completed",
            )
        )

        assert process_task.fields["process_owner_row"] == row_id
        assert process_task.visible is False
        display.stop()

    def test_subagent_table_renders_headers_metrics_and_process_summary(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=98)
        display = RichProgressDisplay(console=console)
        spinner = _CountingSpinner()
        display._description_spinner.spinner = spinner
        display._progress._subagent_spinner = spinner
        child_name = "parent[reviewer]"
        row_id = "parent::subagent::outer-call"
        display.fold_agent_progress(child_name)
        display.fold_agent_progress("parent[verifier]")
        display.update(
            _subagent_event(
                agent_name=child_name,
                label="Review SDK",
                row_id=row_id,
                state="tool: read_text_file",
                turn=22,
                input_tokens=2_100,
                cache_percentage=100 / 3,
                output_tokens=128_000,
                output_estimated=True,
                context_percentage=18.0,
            )
        )
        display.update(
            _subagent_event(
                agent_name="parent[verifier]",
                label="Verify tests",
                row_id="parent::subagent::verify-call",
                state="Thinking",
                turn=2,
                input_tokens=1_700,
                output_tokens=403,
                model="gpt-5.3-codex-spark-with-long-suffix",
                context_percentage=11.0,
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                agent_name=child_name,
                target="reviewer",
                tool_name="poll_process",
                correlation_id="poll-1",
                process_id="process-41",
                process_elapsed_seconds=42,
            )
        )
        display.update(
            _make_event(
                action=ProgressAction.SENDING,
                agent_name="parent",
                target="ripgrep_spark [1]",
            )
        )

        console.print(*display._progress.get_renderables())
        rendered = buffer.getvalue()

        assert "subagent" in rendered
        assert "model" in rendered
        assert "detail" in rendered
        assert "turn" not in rendered
        assert "in" in rendered
        assert "cache" in rendered
        assert "out" in rendered
        assert "processes" in rendered
        assert "Review SDK" in rendered
        assert "Verify tests" in rendered
        assert "gpt-5.6-terra" in rendered
        assert "gpt-5.3-cod" in rendered
        assert "(11%)" in rendered
        spinner_columns = [line.index("abc") for line in rendered.splitlines() if "abc" in line]
        assert len(spinner_columns) == 3
        assert len(set(spinner_columns)) == 1
        assert spinner.render_count == 2
        assert "tool: read_text_" in rendered
        assert "gpt-5.6-terra · 22 (18%)" in rendered
        assert "gpt-5.3-codex-… · 2 (11%)  Thinking" in rendered
        review_row = next(line for line in rendered.splitlines() if "Review SDK" in line)
        verify_row = next(line for line in rendered.splitlines() if "Verify tests" in line)
        header_row = next(line for line in rendered.splitlines() if "detail" in line)
        assert header_row.index("detail") == verify_row.index("Thinking")
        assert review_row.index("tool:") == verify_row.index("Thinking")
        assert verify_row.index("(11%)") < verify_row.index("Thinking")
        assert "gpt-5.6-terra · 22 (18%)" in review_row
        assert "2,100" in rendered
        assert "33%" in rendered
        assert "~128,000" in rendered
        assert review_row.index("2,100") < review_row.index("33%") < review_row.index("~128,000")
        assert "1 · 42s" in rendered

    def test_narrow_subagent_table_keeps_core_columns_and_drops_metrics(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=60)
        display = RichProgressDisplay(console=console)
        display.update(
            _subagent_event(
                label="Review TypeScript SDK",
                state="tool: read_text_file",
                turn=3,
                input_tokens=2_100,
                cache_percentage=100 / 3,
                output_tokens=812,
            )
        )

        console.print(*display._progress.get_renderables())
        rendered = buffer.getvalue()

        assert "de" in rendered
        assert "gp" in rendered
        assert "turn" not in rendered
        assert "processes" not in rendered
        assert "2,100" not in rendered
        assert "33%" not in rendered
        assert "812" not in rendered

    def test_narrow_model_metadata_precedes_detail_when_turn_is_hidden(self) -> None:
        for context_percentage, expected_model in (
            (18.0, "gpt-5.6-terra · (18%)"),
            (None, "gpt-5.6-terra"),
        ):
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, width=60)
            display = RichProgressDisplay(console=console)
            display.update(_subagent_event(context_percentage=context_percentage))

            console.print(*display._progress.get_renderables())
            header, row = buffer.getvalue().splitlines()[:2]

            assert expected_model in row
            assert header.index("detail") == row.index("Thinking")

    def test_input_breakpoint_shows_adjacent_cache_column(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=74)
        display = RichProgressDisplay(console=console)
        display.update(_subagent_event(input_tokens=2_100, cache_percentage=100 / 3))

        console.print(*display._progress.get_renderables())
        rendered = buffer.getvalue()

        assert "in" in rendered
        assert "cache" in rendered
        assert "2,100" in rendered
        assert "33%" in rendered
        assert "out" not in rendered

    def test_subagent_table_responsive_boundaries_fit_without_squeezing(self) -> None:
        for width in (60, 64, 74, 84, 97, 98):
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, width=width)
            display = RichProgressDisplay(console=console)
            spinner = _CountingSpinner()
            display._progress._subagent_spinner = spinner
            display.update(
                _subagent_event(
                    turn=123_456,
                    input_tokens=999_999,
                    cache_percentage=100,
                    output_tokens=128_000,
                    output_estimated=True,
                )
            )

            console.print(*display._progress.get_renderables())
            lines = buffer.getvalue().splitlines()
            header, row = lines[:2]

            assert all(len(line) <= width for line in lines)
            assert row.index("abc") == 17
            assert ("cache" in header) is (width >= 74)
            assert ("out" in header) is (width >= 84)
            assert ("processes" in header) is (width >= 98)
            assert (" · 12… " in row) is (width >= 64)
            if width >= 64:
                assert "Thi" in row
            if width >= 74:
                assert "999,999" in row
                assert "100%" in row
            if width >= 84:
                assert "~128,000" in row
            if width == 97:
                assert row.endswith("~128,000")
            if width == 98:
                assert row.endswith("—")

    def test_parent_process_remains_standalone_while_subagent_is_active(self) -> None:
        display = _make_display()
        display.start()
        display.update(_subagent_event())
        display.update(
            _make_event(
                action=ProgressAction.CALLING_TOOL,
                agent_name="parent",
                target="parent",
                tool_name="poll_process",
                correlation_id="parent-poll",
                process_id="process-9",
                process_elapsed_seconds=5,
            )
        )

        process_task_name = "parent::parent-poll"
        fields = _task_fields(display, process_task_name)
        process_task = next(
            task
            for task in display._progress.tasks
            if task.id == display._taskmap[process_task_name]
        )

        assert fields["process_owner_row"] is None
        assert process_task.visible is True
        display.stop()


def test_subagent_cache_percentage_formatting_and_schema_default() -> None:
    assert _cache_percentage_text(None) == "—"
    assert _cache_percentage_text(0) == "0%"
    assert _cache_percentage_text(99.9) == ">99%"
    assert _cache_percentage_text(100) == "100%"
    assert _cache_percentage_text(float("nan")) == "—"

    snapshot = SubagentMonitorSnapshot.model_validate(
        {
            "state": "Thinking",
            "turn": 1,
            "input_tokens": 100,
            "output_tokens": 20,
        }
    )
    assert snapshot.cache_percentage is None


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
        display.update(
            _make_event(
                action=ProgressAction.SENDING,
                agent_name="agent-a[old-clone]",
                target="old-clone",
            )
        )

        assert "agent-a" in display._taskmap
        assert "agent-a::tool-a-1" in display._taskmap
        assert "agent-a[old-clone]" in display._taskmap
        assert "agent-b" in display._taskmap

        display.clear_agent_tasks("agent-a")

        assert "agent-a" not in display._taskmap
        assert "agent-a::tool-a-1" not in display._taskmap
        assert "agent-a[old-clone]" not in display._taskmap
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
