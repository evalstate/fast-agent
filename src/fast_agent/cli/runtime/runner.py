"""Synchronous event-loop runner for CLI requests."""

from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from fast_agent.cli.asyncio_utils import set_asyncio_exception_handler
from fast_agent.ui.interactive_diagnostics import write_interactive_trace
from fast_agent.utils.async_utils import configure_uvloop, create_event_loop, ensure_event_loop

from .agent_setup import run_agent_request

if TYPE_CHECKING:
    from .run_request import AgentRunRequest


def _should_convert_keyboard_interrupt_to_task_cancel(request: "AgentRunRequest") -> bool:
    """Return True when runner-level Ctrl+C should cancel the interactive task.

    In interactive REPL mode, a single Ctrl+C should cancel the active turn and
    let the session recover. If ``run_until_complete`` surfaces a raw
    ``KeyboardInterrupt`` first, convert that signal into ``main_task.cancel()``
    so the interactive stack can handle it as an in-flight turn cancellation
    instead of immediately tearing down the entire loop.
    """
    return request.mode == "interactive" and request.is_repl


def run_request(request: AgentRunRequest) -> None:
    """Run an agent request with CLI-compatible loop lifecycle semantics."""
    configure_uvloop()

    loop = ensure_event_loop()
    if loop.is_running():
        loop = create_event_loop()
    set_asyncio_exception_handler(loop)

    exit_code: int | None = None
    main_task = loop.create_task(run_agent_request(request))
    try:
        write_interactive_trace("cli.runner.start")
        while True:
            try:
                loop.run_until_complete(main_task)
                break
            except KeyboardInterrupt:
                convert_to_cancel = (
                    _should_convert_keyboard_interrupt_to_task_cancel(request)
                    and not main_task.done()
                )
                write_interactive_trace(
                    "cli.runner.keyboard_interrupt",
                    task_done=main_task.done(),
                    converted_to_cancel=convert_to_cancel,
                )
                if convert_to_cancel:
                    main_task.cancel()
                    continue
                raise
    except SystemExit as exc:
        write_interactive_trace("cli.runner.system_exit", code=exc.code)
        exit_code = exc.code if isinstance(exc.code, int) else None
    finally:
        with suppress(BaseException):
            if not main_task.done():
                loop.run_until_complete(asyncio.gather(main_task, return_exceptions=True))

        tasks = set()
        with suppress(BaseException):
            tasks = {task for task in asyncio.all_tasks(loop) if task is not main_task}
        write_interactive_trace("cli.runner.finally", task_count=len(tasks))

        for task in tasks:
            with suppress(BaseException):
                task.cancel()

        if sys.version_info >= (3, 7) and tasks:
            with suppress(BaseException):
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        with suppress(BaseException):
            loop.run_until_complete(loop.shutdown_asyncgens())
        with suppress(BaseException):
            loop.close()

    if exit_code not in (None, 0):
        raise SystemExit(exit_code)
