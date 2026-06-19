from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

T = TypeVar("T")


async def await_stream_start(
    awaitable: "Awaitable[T]",
    *,
    timeout_seconds: float | None,
    timeout_message: str,
) -> T:
    """Await stream startup with the same timeout semantics as stream idleness."""

    if timeout_seconds is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(timeout_message) from exc


@asynccontextmanager
async def enter_stream_with_timeout(
    stream_context: Any,
    *,
    timeout_seconds: float | None,
    timeout_message: str,
) -> "AsyncIterator[Any]":
    """Enter an async stream context with a startup timeout."""

    stream = await await_stream_start(
        stream_context.__aenter__(),
        timeout_seconds=timeout_seconds,
        timeout_message=timeout_message,
    )
    try:
        yield stream
    except BaseException as exc:
        suppress = await stream_context.__aexit__(type(exc), exc, exc.__traceback__)
        if not suppress:
            raise
    else:
        await stream_context.__aexit__(None, None, None)
