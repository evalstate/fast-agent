import asyncio
import concurrent.futures
import contextlib
import os
import sys
import warnings
from collections.abc import Awaitable, Callable, Iterable
from typing import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

_UVLOOP_REQUESTED: bool | None = None
_UVLOOP_CONFIGURED: bool | None = None

_UVLOOP_PROMPT_TOOLKIT_DEPRECATION_MESSAGE = (
    r"'asyncio\.iscoroutinefunction' is deprecated and slated for removal in Python 3\.16; "
    r"use inspect\.iscoroutinefunction\(\) instead"
)


def install_known_runtime_warning_filters(
    *,
    version_info: tuple[int, ...] | None = None,
) -> None:
    """Install targeted runtime warning filters for supported fast-agent runtimes."""
    _suppress_known_uvloop_prompt_toolkit_deprecation(version_info=version_info)


@contextlib.contextmanager
def suppress_known_runtime_warnings(
    *,
    version_info: tuple[int, ...] | None = None,
):
    """Apply targeted warning filters for a narrow runtime scope."""
    install_known_runtime_warning_filters(version_info=version_info)
    with warnings.catch_warnings():
        install_known_runtime_warning_filters(version_info=version_info)
        yield


def _suppress_known_uvloop_prompt_toolkit_deprecation(
    *,
    version_info: tuple[int, ...] | None = None,
) -> None:
    """Hide the known Python 3.14 uvloop/prompt-toolkit startup warning."""
    current_version = sys.version_info if version_info is None else version_info
    if current_version < (3, 14):
        return

    # uvloop 0.22.1 still calls `asyncio.iscoroutinefunction()` internally from
    # its signal-handler path, which Python 3.14 surfaces as a DeprecationWarning
    # during prompt-toolkit startup. Remove this filter when uvloop ships a
    # release that switches to `inspect.iscoroutinefunction()` and we adopt it.
    warnings.filterwarnings(
        "ignore",
        message=_UVLOOP_PROMPT_TOOLKIT_DEPRECATION_MESSAGE,
        category=DeprecationWarning,
    )


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

    install_known_runtime_warning_filters()

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
        policy = asyncio.get_event_loop_policy()
        local = getattr(policy, "_local", None)
        loop = getattr(local, "_loop", None) if local is not None else None
        if isinstance(loop, asyncio.AbstractEventLoop):
            if loop.is_closed():
                return create_event_loop()
            return loop
        return create_event_loop()


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
