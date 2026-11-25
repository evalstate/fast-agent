"""Best-effort terminal lifecycle reporting via OSC 7501."""

from __future__ import annotations

import atexit
import os
import threading
from contextlib import contextmanager
from typing import Literal

from fast_agent.integrations.program_status import (
    BlockedKind,
    ProgramStatusReporter,
    program_status_enabled,
)

type HerdrBaseState = Literal["idle", "working", "error", "unknown"]
type HerdrAgentState = Literal["idle", "working", "blocked", "error", "unknown"]


class _HerdrLifecycleReporter:
    def __init__(self) -> None:
        self._program_status = ProgramStatusReporter()
        self._creator_pid = os.getpid()
        self._lock = threading.Lock()
        self._base_state: HerdrBaseState = "unknown"
        self._working_scope_depth = 0
        self._prompt_working = False
        self._blocked_depth = 0
        self._startup_hold_depth = 0
        self._closing = False

    @property
    def creator_pid(self) -> int:
        return self._creator_pid

    def report_state(self, state: HerdrBaseState) -> None:
        if state not in ("idle", "working", "error", "unknown"):
            return
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._base_state = state
            if (
                self._blocked_depth == 0
                and self._working_scope_depth == 0
                and not self._prompt_working
            ):
                self._report_program_status_locked(state=state)

    def enter_working(self) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._working_scope_depth += 1
            if (
                self._working_scope_depth == 1
                and self._blocked_depth == 0
                and not self._prompt_working
            ):
                self._report_program_status_locked(state="working")

    def exit_working(self) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing or self._working_scope_depth == 0:
                return
            self._working_scope_depth -= 1
            if (
                self._working_scope_depth == 0
                and self._blocked_depth == 0
                and not self._prompt_working
            ):
                self._report_program_status_locked(state=self._base_state)

    def enter_startup_hold(self) -> None:
        """Keep OSC 7501 silent until interactive startup recovery finishes."""
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._startup_hold_depth += 1

    def exit_startup_hold(self) -> None:
        """Release one startup hold level."""
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing or self._startup_hold_depth == 0:
                return
            self._startup_hold_depth -= 1

    def report_prompt_working(
        self, working: bool, *, prompt_start: bool = False, failed: bool = False
    ) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            if self._startup_hold_depth == 0:
                self._program_status.activate()
            # OSC 133 A may clear the terminal's record. Republish a retained
            # failure rather than immediately erasing it when the prompt returns.
            if not prompt_start or self._base_state != "error":
                self._base_state = "error" if failed else "idle"
            self._prompt_working = working
            if self._blocked_depth > 0 or self._working_scope_depth > 0:
                if prompt_start:
                    self._program_status.refresh()
                return
            self._report_program_status_locked(
                state="working" if working else self._base_state,
                force_program_status=prompt_start,
            )

    def enter_blocked(self, *, kind: BlockedKind | None = None, activate: bool = False) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            if activate and self._startup_hold_depth == 0:
                # Before the first prompt (startup questions) nothing has
                # activated status output yet; callers check for a TTY session.
                self._program_status.activate()
            self._enter_blocked_locked(kind)

    def exit_blocked(self) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing or self._blocked_depth == 0:
                return
            self._blocked_depth -= 1
            if self._blocked_depth == 0:
                state: HerdrAgentState = (
                    "working"
                    if self._working_scope_depth > 0 or self._prompt_working
                    else self._base_state
                )
                if state == "unknown" and self._startup_hold_depth == 0:
                    # A startup question was answered before any prompt exists.
                    # Stay "working" rather than clearing the status record so
                    # herdr-core holds queued notices/sends until prompt idle.
                    self._program_status.report_state("working")
                else:
                    self._report_program_status_locked(state=state)

    def report_session_metadata(
        self,
        *,
        session_id: str | None,
        title: str | None,
        model: str | None,
        agent_name: str | None,
        pinned: bool,
        forked_from: str | None,
        context_usage: str | None,
        token_usage: str | None,
        prompt: str | None = None,
    ) -> None:
        del model, pinned, forked_from, prompt
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._program_status.report_metadata(
                session_id=session_id,
                agent_name=agent_name,
                title=title,
                context_usage=context_usage,
                token_usage=token_usage,
            )

    def report_session_usage(self, usage: str) -> None:
        if os.getpid() != self._creator_pid or not usage:
            return
        with self._lock:
            if self._closing:
                return
            self._program_status.report_usage(usage)

    def report_queued_input(self, count: int) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._program_status.report_queued(count)

    def report_attached(self, summary: str | None) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._program_status.report_attached(summary)

    def report_warm(self, summary: str | None) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._program_status.report_warm(summary)

    def release(self) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._program_status.release()

    def _enter_blocked_locked(self, kind: BlockedKind | None = None) -> None:
        self._blocked_depth += 1
        if self._blocked_depth == 1:
            self._report_program_status_locked(state="blocked", kind=kind)

    def _report_program_status_locked(
        self,
        *,
        state: HerdrAgentState | None = None,
        force_program_status: bool = False,
        kind: BlockedKind | None = None,
    ) -> None:
        if self._startup_hold_depth == 0 and state in ("idle", "working", "blocked", "error"):
            self._program_status.report_state(state, force=force_program_status, kind=kind)


_reporter_lock = threading.Lock()
_reporter: _HerdrLifecycleReporter | None = None
_reporting_closed = False


def _after_fork_child() -> None:
    global _reporter_lock, _reporter, _reporting_closed

    _reporter_lock = threading.Lock()
    _reporter = None
    _reporting_closed = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _active_reporter() -> _HerdrLifecycleReporter | None:
    global _reporter

    if not program_status_enabled():
        return None

    current_pid = os.getpid()
    with _reporter_lock:
        if _reporting_closed:
            return None
        if _reporter is None or _reporter.creator_pid != current_pid:
            _reporter = _HerdrLifecycleReporter()
        return _reporter


def session_reporting_enabled() -> bool:
    """Whether session presentation has a potential consumer."""
    return program_status_enabled()


def report_agent_state(state: HerdrBaseState) -> None:
    """Report the pane-level lifecycle state via OSC 7501."""
    if state not in ("idle", "working", "error", "unknown"):
        return
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_state(state)
    except Exception:
        pass


def report_session_metadata(
    *,
    session_id: str | None,
    title: str | None,
    model: str | None,
    agent_name: str | None,
    pinned: bool = False,
    forked_from: str | None = None,
    context_usage: str | None = None,
    token_usage: str | None = None,
    prompt: str | None = None,
) -> None:
    """Report current session presentation metadata via OSC 7501."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_session_metadata(
                session_id=session_id,
                title=title,
                model=model,
                agent_name=agent_name,
                pinned=pinned,
                forked_from=forked_from,
                context_usage=context_usage,
                token_usage=token_usage,
                prompt=prompt,
            )
    except Exception:
        pass


def report_session_usage(usage: str) -> None:
    """Report a compact usage token (cost or tokens) for the current turn."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_session_usage(usage)
    except Exception:
        pass


def report_queued_input(count: int) -> None:
    """Report how many interactive turns are queued locally behind the active turn."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_queued_input(count)
    except Exception:
        pass


def report_session_attached(summary: str | None) -> None:
    """Report attached children or background processes in the idle status message."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_attached(summary)
    except Exception:
        pass


def report_session_warm(summary: str | None) -> None:
    """Report active cache-warming countdown in the idle status message."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_warm(summary)
    except Exception:
        pass


def report_prompt_mark(code: str) -> None:
    """Map OSC 133 semantic prompt marks to OSC 7501 lifecycle states."""
    command, _, status = code.partition(";")
    if command not in ("A", "C", "D"):
        return
    try:
        reporter = _active_reporter()
        if reporter is None:
            return
        if command == "A":
            reporter.report_prompt_working(False, prompt_start=True)
        elif command == "C":
            reporter.report_prompt_working(True)
        elif command == "D":
            try:
                exit_code = int(status)
            except ValueError:
                exit_code = 0  # Bare/unspecified completion is not evidence of failure.
            # Conventional interrupt/termination statuses are cancellation, not failure.
            failed = exit_code not in (0, 130, 143, -2, -15)
            reporter.report_prompt_working(False, failed=failed)
    except Exception:
        pass


@contextmanager
def herdr_working():
    """Keep lifecycle state reported as working while a turn or task runs."""
    reporter: _HerdrLifecycleReporter | None = None
    try:
        try:
            reporter = _active_reporter()
            if reporter is not None:
                reporter.enter_working()
        except Exception:
            reporter = None
        yield
    finally:
        if reporter is not None:
            try:
                reporter.exit_working()
            except Exception:
                pass


def enter_startup_hold() -> None:
    """Keep OSC 7501 silent until interactive startup recovery finishes."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.enter_startup_hold()
    except Exception:
        pass


def exit_startup_hold() -> None:
    """Release a prior ``enter_startup_hold()`` call."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.exit_startup_hold()
    except Exception:
        pass


@contextmanager
def herdr_startup_hold():
    """Keep OSC 7501 silent across startup recovery and cleanup."""
    enter_startup_hold()
    try:
        yield
    finally:
        exit_startup_hold()


@contextmanager
def herdr_blocked(*, kind: BlockedKind | None = None, activate: bool = False):
    """Report blocked while fast-agent is waiting for interactive human input.

    ``kind`` says what is needed (``question``, ``permission`` or ``auth``).
    ``activate`` enables status output before the first prompt, for startup
    questions in an interactive terminal session.
    """
    reporter: _HerdrLifecycleReporter | None = None
    try:
        try:
            reporter = _active_reporter()
            if reporter is not None:
                reporter.enter_blocked(kind=kind, activate=activate)
        except Exception:
            reporter = None
        yield
    finally:
        if reporter is not None:
            try:
                reporter.exit_blocked()
            except Exception:
                pass


def release_agent() -> None:
    """Release lifecycle authority without creating a reporter solely for shutdown."""
    global _reporter, _reporting_closed

    try:
        with _reporter_lock:
            _reporting_closed = True
            reporter = _reporter
            _reporter = None
        if reporter is not None:
            reporter.release()
    except Exception:
        pass


def _reset_for_tests() -> None:
    global _reporting_closed

    release_agent()
    with _reporter_lock:
        _reporting_closed = False


atexit.register(release_agent)
