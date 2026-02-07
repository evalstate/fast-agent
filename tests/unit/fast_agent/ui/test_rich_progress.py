"""Tests for RichProgressDisplay focusing on state machine correctness."""

import threading
import time

from rich.console import Console

from fast_agent.event_progress import ProgressAction, ProgressEvent
from fast_agent.ui.rich_progress import RichProgressDisplay


def _make_event(
    action: ProgressAction = ProgressAction.CHATTING,
    agent_name: str = "test-agent",
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


class TestUpdateSkipsWhenInactive:
    """Issue #2: update() must check both _paused and _stopped."""

    def test_update_skipped_when_paused(self) -> None:
        display = _make_display()
        display.start()
        display.pause()

        event = _make_event()
        display.update(event)
        assert len(display._taskmap) == 0

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
        event = _make_event(action=ProgressAction.CHATTING)
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

    def test_tool_progress_with_total_sets_completed(self) -> None:
        display = _make_display()
        display.start()

        event = _make_event(action=ProgressAction.CHATTING)
        display.update(event)

        event = _make_event(
            action=ProgressAction.TOOL_PROGRESS,
            progress=50.0,
            total=100.0,
        )
        display.update(event)
        # Should succeed without error

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


class TestThreadSafety:
    """Verify that concurrent pause/resume/update don't crash."""

    def test_concurrent_pause_resume_update(self) -> None:
        display = _make_display()
        display.start()

        errors: list[Exception] = []
        stop_event = threading.Event()

        def updater() -> None:
            while not stop_event.is_set():
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
            while not stop_event.is_set():
                try:
                    display.pause()
                    time.sleep(0.001)
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

        time.sleep(0.1)
        stop_event.set()

        for t in threads:
            t.join(timeout=2)

        display.stop()
        assert errors == [], f"Concurrent operations raised: {errors}"
