import asyncio
import concurrent.futures
import os
import sys
from collections.abc import Awaitable, Callable, Iterable
from typing import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

_UVLOOP_REQUESTED: bool | None = None
_UVLOOP_CONFIGURED: bool | None = None


def _env_value(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_uvloop(
    env_var: str = "FAST_AGENT_UVLOOP",
    disable_env_var: str = "FAST_AGENT_DISABLE_UV_LOOP",
) -> tuple[bool, bool]:
    """
    Configure uvloop via an env var toggle.

    Returns a tuple of (requested, enabled).
    """
    global _UVLOOP_REQUESTED, _UVLOOP_CONFIGURED
    if _UVLOOP_REQUESTED is not None and _UVLOOP_CONFIGURED is not None:
        return _UVLOOP_REQUESTED, _UVLOOP_CONFIGURED

    explicit_enable = _env_value(env_var)
    explicit_disable = _env_value(disable_env_var)
    requested = explicit_enable is True and explicit_disable is not True
    enabled = False

    if explicit_disable is True or explicit_enable is False:
        enabled = False
    elif not sys.platform.startswith("win"):
        try:
            import uvloop
        except Exception:
            enabled = False
        else:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            enabled = True

    _UVLOOP_REQUESTED = requested
    _UVLOOP_CONFIGURED = enabled
    return requested, enabled


def create_event_loop() -> asyncio.AbstractEventLoop:
    """Create and set a new event loop using the configured policy."""
    configure_uvloop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Return a usable event loop, creating one if needed."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return create_event_loop()
        if loop.is_closed():
            return create_event_loop()
        return loop


def run_sync(
    func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs
) -> T | None:
    """
    Run an async callable from sync code using the shared loop policy.

    If a loop is already running in this thread, we run the coroutine in a new thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = ensure_event_loop()
        if loop.is_running():
            return _run_in_new_loop(func, *args, **kwargs)
        return loop.run_until_complete(func(*args, **kwargs))
    return _run_in_new_loop(func, *args, **kwargs)


def _run_in_new_loop(func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
    def runner() -> T:
        loop = create_event_loop()
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(runner).result()


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
