"""Tests for asyncio runtime helpers."""

import sys
from types import SimpleNamespace

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
