import asyncio
from datetime import timedelta

import pytest
from pydantic import ValidationError

from fast_agent.core.executor.executor import AsyncioExecutor, ExecutorConfig


def add(value: int, *, increment: int = 0) -> int:
    return value + increment


async def async_value(value: int) -> int:
    await asyncio.sleep(0)
    return value


@pytest.mark.asyncio
async def test_execute_accepts_sync_callables_and_shared_kwargs() -> None:
    executor = AsyncioExecutor()

    results = await executor.execute(
        lambda *, increment: add(1, increment=increment),
        lambda *, increment: add(2, increment=increment),
        increment=3,
    )

    assert results == [4, 5]


@pytest.mark.asyncio
async def test_execute_accepts_awaitables() -> None:
    executor = AsyncioExecutor()

    results = await executor.execute(async_value(1), async_value(2))

    assert results == [1, 2]


@pytest.mark.asyncio
async def test_execute_rejects_kwargs_for_awaitables() -> None:
    executor = AsyncioExecutor()

    results = await executor.execute(async_value(1), unused=True)

    assert len(results) == 1
    assert isinstance(results[0], TypeError)


@pytest.mark.asyncio
async def test_execute_rejects_coroutine_functions() -> None:
    executor = AsyncioExecutor()

    results = await executor.execute(async_value)

    assert len(results) == 1
    assert isinstance(results[0], TypeError)


class _AwaitableResult:
    def __await__(self):
        async def value() -> int:
            return 1

        return value().__await__()


def returns_awaitable() -> _AwaitableResult:
    return _AwaitableResult()


@pytest.mark.asyncio
async def test_execute_rejects_sync_callables_returning_awaitables() -> None:
    executor = AsyncioExecutor()

    results = await executor.execute(returns_awaitable)

    assert len(results) == 1
    assert isinstance(results[0], TypeError)


@pytest.mark.asyncio
async def test_execute_returns_task_exceptions() -> None:
    executor = AsyncioExecutor()

    def fail() -> int:
        raise ValueError("boom")

    results = await executor.execute(fail)

    assert len(results) == 1
    assert isinstance(results[0], ValueError)


@pytest.mark.asyncio
async def test_execute_timeout_returns_timeout_error() -> None:
    executor = AsyncioExecutor(ExecutorConfig(timeout_seconds=timedelta(milliseconds=10)))

    results = await executor.execute(asyncio.sleep(1))

    assert len(results) == 1
    assert isinstance(results[0], TimeoutError)


@pytest.mark.asyncio
async def test_execute_limits_concurrent_activities() -> None:
    executor = AsyncioExecutor(ExecutorConfig(max_concurrent_activities=1))
    active = 0
    max_active = 0

    async def tracked(value: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value

    results = await executor.map(tracked, [1, 2, 3])

    assert results == [1, 2, 3]
    assert max_active == 1


def test_executor_config_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValidationError):
        ExecutorConfig(max_concurrent_activities=0)


def test_executor_config_rejects_invalid_timeout() -> None:
    with pytest.raises(ValidationError):
        ExecutorConfig(timeout_seconds=timedelta())


def test_executor_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExecutorConfig.model_validate({"retry_policy": {}})


@pytest.mark.asyncio
async def test_execute_propagates_cancellation() -> None:
    executor = AsyncioExecutor()

    sleeper = asyncio.create_task(asyncio.sleep(1))
    run = asyncio.create_task(executor.execute(sleeper))
    await asyncio.sleep(0)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run


@pytest.mark.asyncio
async def test_map_runs_one_task_per_input() -> None:
    executor = AsyncioExecutor()

    results = await executor.map(add, [1, 2, 3], increment=10)

    assert results == [11, 12, 13]


@pytest.mark.asyncio
async def test_map_accepts_async_mappers() -> None:
    executor = AsyncioExecutor()

    async def add_async(value: int, *, increment: int) -> int:
        await asyncio.sleep(0)
        return value + increment

    results = await executor.map(add_async, [1, 2, 3], increment=10)

    assert results == [11, 12, 13]


@pytest.mark.asyncio
async def test_execute_streaming_yields_as_tasks_complete() -> None:
    executor = AsyncioExecutor()

    async def later(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    results = [result async for result in executor.execute_streaming(later(1, 0.02), later(2, 0))]

    assert results == [2, 1]


@pytest.mark.asyncio
async def test_execute_streaming_cancels_pending_tasks_when_closed() -> None:
    executor = AsyncioExecutor()
    cancelled = asyncio.Event()

    async def completes() -> int:
        await asyncio.sleep(0)
        return 1

    async def waits() -> int:
        try:
            await asyncio.sleep(10)
            return 2
        except asyncio.CancelledError:
            cancelled.set()
            raise

    stream = executor.execute_streaming(completes(), waits())
    assert await stream.__anext__() == 1

    await stream.aclose()

    await asyncio.wait_for(cancelled.wait(), timeout=1)
