import asyncio
import contextvars
import threading

import pytest
from opentelemetry import trace

from fast_agent.utils.async_utils import run_sync


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_error", [False, True])
async def test_run_sync_drains_background_tasks_from_worker_loop(raise_error: bool) -> None:
    cancelled = threading.Event()
    completed = threading.Event()

    async def operation() -> int:
        started = asyncio.Event()

        async def background() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                completed.set()

        asyncio.create_task(background())
        await started.wait()
        if raise_error:
            raise ValueError("operation failed")
        return 7

    if raise_error:
        with pytest.raises(ValueError, match="operation failed"):
            run_sync(operation)
    else:
        assert run_sync(operation) == 7
    assert cancelled.is_set()
    assert completed.is_set()


@pytest.mark.asyncio
async def test_run_sync_accepts_future_returned_by_callable() -> None:
    parent_loop = asyncio.get_running_loop()

    def ready_future() -> asyncio.Future[int]:
        loop = asyncio.get_running_loop()
        assert loop is not parent_loop
        future = loop.create_future()
        future.set_result(7)
        return future

    assert run_sync(ready_future) == 7


@pytest.mark.asyncio
async def test_run_sync_preserves_caller_context_without_leaking_changes() -> None:
    request_id = contextvars.ContextVar("request_id", default="unset")
    token = request_id.set("caller-request")
    parent_context = trace.SpanContext(trace_id=123, span_id=456, is_remote=False)

    async def operation() -> tuple[str, trace.SpanContext]:
        inherited = request_id.get()
        request_id.set("worker-request")
        return inherited, trace.get_current_span().get_span_context()

    try:
        with trace.use_span(trace.NonRecordingSpan(parent_context)):
            assert run_sync(operation) == ("caller-request", parent_context)
        assert request_id.get() == "caller-request"
    finally:
        request_id.reset(token)
