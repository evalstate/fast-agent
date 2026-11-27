"""Prompt-cache status and warming while the interactive prompt waits for input."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, nullcontext, suppress
from typing import TYPE_CHECKING

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.application.current import set_app
from rich.text import Text

from fast_agent.core.cache_warmer import CacheWarmer, format_status_summary
from fast_agent.integrations.hc_children import current_hc_attachments
from fast_agent.integrations.herdr_lifecycle import report_session_warm
from fast_agent.ui.console import rich_print

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine

    from prompt_toolkit import PromptSession

# How often an idle prompt waiting on hc children redraws its elapsed wait.
_WAITING_REDRAW_SECONDS = 30.0


async def _redraw_when_cache_lapses(
    warmer: CacheWarmer, report: Callable[[str | None], None]
) -> None:
    """Refresh the display at the moment the cache is expected to expire."""
    while (lapses_at := warmer.status().lapses_at) is not None and lapses_at > time.time():
        await asyncio.sleep(lapses_at - time.time() + 1)
        report(None)


async def _redraw_while_waiting(session: "PromptSession") -> None:
    """Keep the toolbar's elapsed wait on hc children current while idle."""
    while True:
        await asyncio.sleep(_WAITING_REDRAW_SECONDS)
        session.app.invalidate()


@asynccontextmanager
async def _running(coroutine: Coroutine[object, object, None]) -> AsyncGenerator[None]:
    """Run a redraw loop for the scope; leaving it cancels the loop."""
    task = asyncio.create_task(coroutine)
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@asynccontextmanager
async def idle_cache_warming(
    *,
    session: "PromptSession",
    warmer: CacheWarmer | None,
    has_event_waits: bool = False,
) -> AsyncGenerator[None]:
    """Show the cache's expected lifetime while the prompt is idle, and keep it warm if asked.

    A ``/warm`` window is one reason to warm; herdr-core children or wake-enabled
    MCP event subscriptions that will resume this session are the other. Leaving
    the scope (any submitted input) stops warming before the next turn makes a
    request of its own.
    """
    hc_attached = bool(current_hc_attachments().resuming)
    attached = hc_attached or has_event_waits
    waiting = _running(_redraw_while_waiting(session)) if attached else nullcontext()
    if warmer is None:
        async with waiting:
            yield
        return

    def report(line: str | None) -> None:
        report_session_warm(format_status_summary(warmer.status()))
        app = session.app
        if line is not None:
            # The warming task did not start inside the prompt application, so
            # name it: the line is then printed above the prompt, not over it.
            with set_app(app):
                run_in_terminal(lambda: rich_print(Text(line, style="dim")))
        app.invalidate()

    report(None)
    armed = warmer.armed_until is not None
    reason = "/warm" if armed else ("hc" if hc_attached else "events")
    warming = (
        warmer.keep_warm(reason, automatic=attached, armed=armed, report=report)
        if armed or attached
        else nullcontext()
    )
    async with waiting, _running(_redraw_when_cache_lapses(warmer, report)), warming:
        yield
