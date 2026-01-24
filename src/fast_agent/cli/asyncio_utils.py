from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio


def set_asyncio_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Attach a detailed exception handler to the provided event loop."""
    logger = logging.getLogger("fast_agent.asyncio")

    def _handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        message = context.get("message", "(no message)")
        task = context.get("task")
        future = context.get("future")
        handle = context.get("handle")
        source_traceback = context.get("source_traceback")
        exception = context.get("exception")

        details = {
            "message": message,
            "task": repr(task) if task else None,
            "future": repr(future) if future else None,
            "handle": repr(handle) if handle else None,
            "source_traceback": [str(frame) for frame in source_traceback]
            if source_traceback
            else None,
        }

        logger.error("Unhandled asyncio error: %s", message)
        logger.error("Asyncio context: %s", json.dumps(details, indent=2))

        if exception:
            logger.exception("Asyncio exception", exc_info=exception)

    try:
        loop.set_exception_handler(_handler)
    except Exception:
        logger.exception("Failed to set asyncio exception handler")
