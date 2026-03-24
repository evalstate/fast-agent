"""Unit tests for AgentChannel — Unix Domain Socket IPC abstraction."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from fast_agent.spawn.agent_channel import (
    AgentChannel,
    _get_sock_dir,
    _resolve_channel_dir,
    _sanitize_name,
)


@pytest.fixture
def short_tmp():
    """Short temp directory for Unix sockets (path must be <108 chars)."""
    d = tempfile.mkdtemp(prefix="ac_", dir="/tmp")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ─── Helper: name sanitization ───


class TestSanitizeName:
    def test_simple_name(self):
        assert _sanitize_name("dev") == "dev"

    def test_name_with_spaces(self):
        assert _sanitize_name("Minh - Dev") == "Minh_-_Dev"

    def test_name_with_slashes(self):
        assert _sanitize_name("team/lead") == "team_lead"

    def test_name_mixed(self):
        assert _sanitize_name("Hoa - BA / Lead") == "Hoa_-_BA___Lead"


# ─── Helper: sock dir resolution ───


class TestGetSockDir:
    def test_from_spawn_project_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path))
        monkeypatch.delenv("TEAM_WORKSPACE", raising=False)
        result = _get_sock_dir()
        # Returns a symlink jch_<hash> in system tmpdir
        assert result.name.startswith("jch_")
        assert result.is_symlink()

    def test_from_team_workspace(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPAWN_PROJECT_DIR", raising=False)
        monkeypatch.setenv("TEAM_WORKSPACE", str(tmp_path / "workspace"))
        result = _get_sock_dir()
        assert result.name.startswith("jch_")

    def test_same_input_same_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path))
        monkeypatch.delenv("TEAM_WORKSPACE", raising=False)
        r1 = _get_sock_dir()
        r2 = _get_sock_dir()
        assert r1 == r2

    def test_different_inputs_different_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path / "a"))
        monkeypatch.delenv("TEAM_WORKSPACE", raising=False)
        r1 = _get_sock_dir()
        monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path / "b"))
        r2 = _get_sock_dir()
        assert r1 != r2

    def test_explicit_channel_dir(self, short_tmp):
        result = _get_sock_dir(channel_dir=short_tmp)
        assert result.name.startswith("jch_")
        assert result.is_symlink()
        # Symlink points to the actual channel_dir
        assert result.resolve() == short_tmp.resolve()

    def test_path_always_under_104_chars(self, monkeypatch, tmp_path):
        """Ensure even with long project paths, socket path stays short."""
        long_path = str(tmp_path / ("a" * 80))
        monkeypatch.setenv("SPAWN_PROJECT_DIR", long_path)
        monkeypatch.delenv("TEAM_WORKSPACE", raising=False)
        result = _get_sock_dir()
        # Even with a long agent name, total path must be < 104
        full = result / "Trang_-_Designer.sock"
        assert len(str(full)) < 104

    def test_symlink_target_is_project_dir(self, tmp_path, monkeypatch):
        """Symlink should point to the resolved channel dir in the project."""
        monkeypatch.setenv("SPAWN_PROJECT_DIR", str(tmp_path))
        monkeypatch.delenv("TEAM_WORKSPACE", raising=False)
        result = _get_sock_dir()
        expected_real = _resolve_channel_dir()
        assert result.resolve() == expected_real


# ─── AgentChannel: init + properties ───


class TestAgentChannelInit:
    def test_creates_with_name(self, short_tmp):
        ch = AgentChannel("test-agent", channel_dir=short_tmp)
        assert ch.agent_name == "test-agent"
        assert ch.socket_path.name == "test-agent.sock"
        assert ch.socket_path.parent.name.startswith("jch_")

    def test_creates_with_spaces_in_name(self, short_tmp):
        ch = AgentChannel("Minh - Dev", channel_dir=short_tmp)
        assert ch.socket_path.name == "Minh_-_Dev.sock"
        assert ch.socket_path.parent.name.startswith("jch_")

    def test_socket_path_property(self, short_tmp):
        ch = AgentChannel("demo", channel_dir=short_tmp)
        assert ch.socket_path.name == "demo.sock"
        assert ch.socket_path.parent.name.startswith("jch_")


# ─── AgentChannel: is_alive (static) ───


class TestIsAlive:
    def test_not_alive_when_no_socket(self, short_tmp):
        assert AgentChannel.is_alive("ghost", channel_dir=short_tmp) is False

    def test_alive_when_socket_exists(self, short_tmp):
        # Create socket in the hashed directory where is_alive looks
        sock_dir = _get_sock_dir(short_tmp)
        sock_dir.mkdir(parents=True, exist_ok=True)
        (sock_dir / "agent-x.sock").touch()
        assert AgentChannel.is_alive("agent-x", channel_dir=short_tmp) is True

    def test_sanitized_name_lookup(self, short_tmp):
        sock_dir = _get_sock_dir(short_tmp)
        sock_dir.mkdir(parents=True, exist_ok=True)
        (sock_dir / "Minh_-_Dev.sock").touch()
        assert AgentChannel.is_alive("Minh - Dev", channel_dir=short_tmp) is True


# ─── AgentChannel: send_signal (static) ───


class TestSendSignal:
    def test_send_to_nonexistent_returns_false(self, short_tmp):
        result = AgentChannel.send_signal(
            "no-agent", signal="wake", channel_dir=short_tmp
        )
        assert result is False

    def test_send_to_stale_socket_returns_false(self, short_tmp):
        sock_file = short_tmp / "stale.sock"
        sock_file.touch()
        result = AgentChannel.send_signal(
            "stale", signal="wake", channel_dir=short_tmp
        )
        assert result is False


# ─── AgentChannel: server lifecycle (async) ───


class TestServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_socket(self, short_tmp):
        ch = AgentChannel("lifecycle", channel_dir=short_tmp)
        await ch.start_server()
        try:
            assert ch.socket_path.exists()
        finally:
            await ch.stop()

    @pytest.mark.asyncio
    async def test_stop_removes_socket(self, short_tmp):
        ch = AgentChannel("cleanup", channel_dir=short_tmp)
        await ch.start_server()
        assert ch.socket_path.exists()
        await ch.stop()
        assert not ch.socket_path.exists()

    @pytest.mark.asyncio
    async def test_start_cleans_stale_socket(self, short_tmp):
        sock_path = short_tmp / "stale2.sock"
        sock_path.touch()
        assert sock_path.exists()

        ch = AgentChannel("stale2", channel_dir=short_tmp)
        await ch.start_server()
        try:
            assert sock_path.exists()
        finally:
            await ch.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, short_tmp):
        ch = AgentChannel("dblstop", channel_dir=short_tmp)
        await ch.start_server()
        await ch.stop()
        await ch.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_creates_channel_dir_if_missing(self, short_tmp):
        deep_dir = short_tmp / "a" / "b"
        ch = AgentChannel("nested", channel_dir=deep_dir)
        await ch.start_server()
        try:
            # The hashed sock dir is created, not the raw deep_dir
            assert ch.socket_path.parent.exists()
            assert ch.socket_path.exists()
        finally:
            await ch.stop()


# ─── AgentChannel: listen + signal round-trip ───


class TestSignalRoundTrip:
    @pytest.mark.asyncio
    async def test_listen_timeout_returns_none(self, short_tmp):
        ch = AgentChannel("tmout", channel_dir=short_tmp)
        await ch.start_server()
        try:
            result = await ch.listen(timeout=0.3)
            assert result is None
        finally:
            await ch.stop()

    @pytest.mark.asyncio
    async def test_signal_round_trip(self, short_tmp):
        """Send signal from a thread (simulating another process)."""
        ch = AgentChannel("rt", channel_dir=short_tmp)
        await ch.start_server()
        try:

            def send_delayed():
                time.sleep(0.2)
                AgentChannel.send_signal(
                    "rt", signal="wake|email", channel_dir=short_tmp
                )

            t = threading.Thread(target=send_delayed, daemon=True)
            t.start()

            result = await ch.listen(timeout=3.0)
            assert result == "wake|email"
            t.join(timeout=2)
        finally:
            await ch.stop()

    @pytest.mark.asyncio
    async def test_signal_before_listen_not_lost(self, short_tmp):
        """Signal sent before listen() should be returned immediately."""
        ch = AgentChannel("presig", channel_dir=short_tmp)
        await ch.start_server()
        try:

            def send_now():
                AgentChannel.send_signal(
                    "presig", signal="early", channel_dir=short_tmp
                )

            t = threading.Thread(target=send_now, daemon=True)
            t.start()
            t.join(timeout=3)

            # Give event loop time to process the callback
            await asyncio.sleep(0.2)

            result = await ch.listen(timeout=1.0)
            assert result == "early"
        finally:
            await ch.stop()

    @pytest.mark.asyncio
    async def test_multiple_signals_sequential(self, short_tmp):
        """Multiple listen() calls should each receive a signal."""
        ch = AgentChannel("multi", channel_dir=short_tmp)
        await ch.start_server()
        try:
            signals_received = []

            for i in range(3):

                def send_one(idx=i):
                    time.sleep(0.1)
                    AgentChannel.send_signal(
                        "multi",
                        signal=f"sig-{idx}",
                        channel_dir=short_tmp,
                    )

                t = threading.Thread(target=send_one, daemon=True)
                t.start()
                result = await ch.listen(timeout=2.0)
                signals_received.append(result)
                t.join(timeout=2)

            assert len(signals_received) == 3
            assert all(s is not None for s in signals_received)
            assert all(s.startswith("sig-") for s in signals_received)
        finally:
            await ch.stop()

    @pytest.mark.asyncio
    async def test_default_signal_is_wake(self, short_tmp):
        """Sending empty signal body should default to 'wake'."""
        ch = AgentChannel("defsig", channel_dir=short_tmp)
        await ch.start_server()
        try:

            def send_empty():
                time.sleep(0.1)
                AgentChannel.send_signal(
                    "defsig", signal="", channel_dir=short_tmp
                )

            t = threading.Thread(target=send_empty, daemon=True)
            t.start()
            result = await ch.listen(timeout=2.0)
            assert result == "wake"
            t.join(timeout=2)
        finally:
            await ch.stop()


# ─── AgentChannel: is_alive integration with server ───


class TestIsAliveWithServer:
    @pytest.mark.asyncio
    async def test_is_alive_tracks_server_state(self, short_tmp):
        ch = AgentChannel("alive2", channel_dir=short_tmp)
        assert AgentChannel.is_alive("alive2", channel_dir=short_tmp) is False
        await ch.start_server()
        assert AgentChannel.is_alive("alive2", channel_dir=short_tmp) is True
        await ch.stop()
        assert AgentChannel.is_alive("alive2", channel_dir=short_tmp) is False
