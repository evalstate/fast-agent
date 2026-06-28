import asyncio
import concurrent.futures
import functools
import sys
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from importlib.util import find_spec
from typing import Any, ParamSpec, TypeVar

from anyio import to_thread

from fast_agent.utils.env import optional_env_flag

T = TypeVar("T")
P = ParamSpec("P")

_UVLOOP_REQUESTED: bool | None = None
_UVLOOP_CONFIGURED: bool | None = None


def configure_uvloop(
    env_var: str = "FAST_AGENT_UVLOOP",
    disable_env_var: str = "FAST_AGENT_DISABLE_UV_LOOP",
) -> tuple[bool, bool]:
    """
    Resolve the uvloop env var toggle for fast-agent-created event loops.

    Returns a tuple of (requested, enabled).
    """
    global _UVLOOP_REQUESTED, _UVLOOP_CONFIGURED
    if _UVLOOP_REQUESTED is not None and _UVLOOP_CONFIGURED is not None:
        return _UVLOOP_REQUESTED, _UVLOOP_CONFIGURED

    explicit_enable = optional_env_flag(env_var)
    explicit_disable = optional_env_flag(disable_env_var)
    requested = explicit_enable is True and explicit_disable is not True
    enabled = False

    if explicit_disable is True or explicit_enable is False:
        enabled = False
    elif not sys.platform.startswith("win"):
        enabled = find_spec("uvloop") is not None

    _UVLOOP_REQUESTED = requested
    _UVLOOP_CONFIGURED = enabled
    return requested, enabled


def create_event_loop() -> asyncio.AbstractEventLoop:
    """Create and set a new event loop using uvloop when enabled."""
    _, enabled = configure_uvloop()
    if enabled:
        try:
            import uvloop

            loop = uvloop.new_event_loop()
        except Exception:
            global _UVLOOP_CONFIGURED
            _UVLOOP_CONFIGURED = False
            loop = asyncio.new_event_loop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def run_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine with the fast-agent event-loop factory."""
    with asyncio.Runner(loop_factory=create_event_loop) as runner:
        return runner.run(coro)


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


def run_sync(func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T | None:
    """
    Run an async callable from sync code using the fast-agent loop factory.

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


async def run_in_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a synchronous callable in a worker thread without uvloop deprecation noise."""
    if kwargs:
        return await to_thread.run_sync(functools.partial(func, *args, **kwargs))
    return await to_thread.run_sync(func, *args)


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
