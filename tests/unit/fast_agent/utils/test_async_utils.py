"""Tests for asyncio runtime helpers."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from fast_agent.utils import async_utils


def test_uvloop_disable_env_prevents_uvloop_creation(monkeypatch) -> None:
    monkeypatch.setenv("FAST_AGENT_DISABLE_UV_LOOP", "1")
    async_utils._UVLOOP_REQUESTED = None
    async_utils._UVLOOP_CONFIGURED = None

    requested, enabled = async_utils.configure_uvloop()
    loop = async_utils.create_event_loop()
    try:
        assert not requested
        assert not enabled
        assert type(loop).__module__.startswith("asyncio.")
    finally:
        loop.close()
        async_utils._UVLOOP_REQUESTED = None
        async_utils._UVLOOP_CONFIGURED = None


def test_run_coroutine_uses_fast_agent_loop_factory() -> None:
    async def value() -> int:
        return 7

    assert async_utils.run_coroutine(value()) == 7


def test_uvloop_creation_failure_falls_back_to_asyncio(monkeypatch) -> None:
    def broken_new_event_loop():
        raise RuntimeError("broken uvloop wheel")

    monkeypatch.delenv("FAST_AGENT_DISABLE_UV_LOOP", raising=False)
    monkeypatch.delenv("FAST_AGENT_UVLOOP", raising=False)
    monkeypatch.setattr(async_utils, "find_spec", lambda name: object())
    monkeypatch.setitem(
        sys.modules, "uvloop", SimpleNamespace(new_event_loop=broken_new_event_loop)
    )
    async_utils._UVLOOP_REQUESTED = None
    async_utils._UVLOOP_CONFIGURED = None

    requested, enabled = async_utils.configure_uvloop()
    loop = async_utils.create_event_loop()
    try:
        assert not requested
        assert enabled
        assert async_utils._UVLOOP_CONFIGURED is False
        assert type(loop).__module__.startswith("asyncio.")
    finally:
        loop.close()
        async_utils._UVLOOP_REQUESTED = None
        async_utils._UVLOOP_CONFIGURED = None
        sys.modules.pop("uvloop", None)


@pytest.mark.asyncio
async def test_gather_with_cancel_preserves_results_and_exceptions() -> None:
    async def return_value() -> int:
        return 7

    async def fail() -> int:
        raise RuntimeError("failed")

    results = await async_utils.gather_with_cancel([return_value(), fail()])

    assert results[0] == 7
    assert isinstance(results[1], RuntimeError)


@pytest.mark.asyncio
async def test_gather_with_cancel_propagates_child_cancellation_and_cancels_sibling() -> None:
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def cancel() -> None:
        await sibling_started.wait()
        raise asyncio.CancelledError("child cancelled")

    async def wait_forever() -> None:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    with pytest.raises(asyncio.CancelledError, match="child cancelled"):
        await asyncio.wait_for(
            async_utils.gather_with_cancel([cancel(), wait_forever()]),
            timeout=1,
        )

    assert sibling_cancelled.is_set()
