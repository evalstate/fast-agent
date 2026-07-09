from __future__ import annotations

import asyncio
import os

import pytest

import fast_agent.config as config_module
from fast_agent.constants import FAST_AGENT_RUNTIME_HOME
from fast_agent.session import reset_session_manager


@pytest.fixture(autouse=True)
def shorten_logging_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests from waiting on full production logging shutdown timeouts."""

    from fast_agent.core.logging.transport import AsyncEventBus

    async def drain_queue_before_stop(self: AsyncEventBus, same_loop_task: bool) -> None:
        queue = self._queue
        if queue is not None:
            self._discard_queued_events(queue)
        self._queue = None

    async def cancel_process_task(
        self: AsyncEventBus, *, same_loop_task: bool | None = None
    ) -> None:
        task = self._task
        if task is None:
            return
        if task.done():
            self._task = None
            return

        task.cancel()
        should_await = (
            self._is_task_on_current_loop(task) if same_loop_task is None else same_loop_task
        )
        if should_await:
            try:
                await asyncio.wait_for(task, timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._task = None

    monkeypatch.setattr(AsyncEventBus, "_drain_queue_before_stop", drain_queue_before_stop)
    monkeypatch.setattr(AsyncEventBus, "_cancel_process_task", cancel_process_task)


@pytest.fixture(autouse=True)
def isolate_home(tmp_path):
    """Ensure unit tests never write sessions/skills into a real home.

    Unit tests are sometimes run from within an interactive fast-agent process where
    ``FAST_AGENT_HOME`` or ``FAST_AGENT_HOME`` may already point at a real user
    environment. Force an isolated temporary environment path per test to avoid
    reading from or writing to developer session storage.
    """

    original_runtime_environment = os.environ.get(FAST_AGENT_RUNTIME_HOME)
    original_fast_agent_home = os.environ.get("FAST_AGENT_HOME")
    original_home = os.environ.get("FAST_AGENT_HOME")
    isolated_home = tmp_path / ".fast-agent-test-env"
    os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
    os.environ.pop("FAST_AGENT_HOME", None)
    os.environ["FAST_AGENT_HOME"] = str(isolated_home)
    # Ensure cached global settings never leak across tests.
    config_module._settings = None
    reset_session_manager()

    try:
        yield
    finally:
        reset_session_manager()
        config_module._settings = None
        if original_runtime_environment is None:
            os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_HOME] = original_runtime_environment
        if original_fast_agent_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = original_fast_agent_home
        if original_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = original_home
