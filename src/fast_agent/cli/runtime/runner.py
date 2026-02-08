"""Synchronous event-loop runner for CLI requests."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from fast_agent.cli.asyncio_utils import set_asyncio_exception_handler
from fast_agent.utils.async_utils import configure_uvloop, create_event_loop, ensure_event_loop

from .agent_setup import run_agent_request

if TYPE_CHECKING:
    from .run_request import AgentRunRequest


def run_request(request: AgentRunRequest) -> None:
    """Run an agent request with CLI-compatible loop lifecycle semantics."""
    configure_uvloop()

    loop = ensure_event_loop()
    if loop.is_running():
        loop = create_event_loop()
    set_asyncio_exception_handler(loop)

    exit_code: int | None = None
    try:
        loop.run_until_complete(run_agent_request(request))
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else None
    finally:
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            if sys.version_info >= (3, 7):
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception:
            pass

    if exit_code not in (None, 0):
        raise SystemExit(exit_code)
