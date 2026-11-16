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
    from collections.abc import AsyncGenerator, Callable

    from prompt_toolkit import PromptSession


async def _redraw_when_cache_lapses(
    warmer: CacheWarmer, report: Callable[[str | None], None]
) -> None:
    """Refresh the display at the moment the cache is expected to expire."""
    while (lapses_at := warmer.status().lapses_at) is not None and lapses_at > time.time():
        await asyncio.sleep(lapses_at - time.time() + 1)
        report(None)


@asynccontextmanager
async def idle_cache_warming(
    *, session: "PromptSession", warmer: CacheWarmer | None
) -> AsyncGenerator[None]:
    """Show the cache's expected lifetime while the prompt is idle, and keep it warm if asked.

    A ``/warm`` window is one reason to warm; herdr-core children that will
    resume this session are the other. Leaving the scope (any submitted input)
    stops warming before the next turn makes a request of its own.
    """
    if warmer is None:
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
    attached = bool(current_hc_attachments().resuming)
    warming = (
        warmer.keep_warm("/warm" if armed else "hc", automatic=attached, armed=armed, report=report)
        if armed or attached
        else nullcontext()
    )
    redraw = asyncio.create_task(_redraw_when_cache_lapses(warmer, report))
    try:
        async with warming:
            yield
    finally:
        redraw.cancel()
        with suppress(asyncio.CancelledError):
            await redraw
