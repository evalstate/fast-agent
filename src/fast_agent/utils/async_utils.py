from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

T = TypeVar("T")


async def gather_with_cancel(aws: Iterable[Awaitable[T]]) -> list[T | BaseException]:
    """
    Gather results while keeping per-task exceptions, but propagate cancellation.

    This mirrors asyncio.gather(..., return_exceptions=True) except that
    asyncio.CancelledError is re-raised so cancellation never gets swallowed.
    """

    results = await asyncio.gather(*aws, return_exceptions=True)
    for item in results:
        if isinstance(item, asyncio.CancelledError):
            raise item
    return results
