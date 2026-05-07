"""AgentChannel — cross-process async IPC via Unix Domain Socket.

Provides a reusable, zero-latency communication channel between agent
subprocesses and MCP servers. Replaces polling loops with event-driven
wake signals.

Server side (agent subprocess):
    channel = AgentChannel("Dev", channel_dir)
    await channel.start_server()
    signal = await channel.listen()  # blocks indefinitely, zero CPU

Client side (any process):
    AgentChannel.send_signal("Dev", "wake", channel_dir)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import socket
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Socket path strategy ──
# macOS AF_UNIX limit is 104 bytes.  Workspace paths can easily exceed this.
# Solution: actual socket files stay in the project directory, but we create
# a short symlink  $TMPDIR/jch_<hash8> → <project_channel_dir>  so the
# AF_UNIX path used by bind()/connect() is always ~40 chars.
import tempfile as _tempfile


def _resolve_channel_dir(channel_dir: Path | None = None) -> Path:
    """Return the *logical* channel directory inside the project tree."""
    if channel_dir:
        return channel_dir.resolve()
    project_dir = os.environ.get("SPAWN_PROJECT_DIR", "")
    workspace = os.environ.get("TEAM_WORKSPACE", "")
    base = Path(project_dir or workspace or os.getcwd())
    return (base / ".runtime" / "state" / "channels").resolve()


def _get_sock_dir(channel_dir: Path | None = None) -> Path:
    """Return a short symlink path suitable for AF_UNIX sockets.

    *   Actual socket files live inside the project tree (``_resolve_channel_dir``).
    *   A symlink ``$TMPDIR/jch_<hash8>`` → that directory keeps the path
        that ``bind()`` / ``connect()`` see well under 104 bytes.
    """
    real_dir = _resolve_channel_dir(channel_dir)
    real_dir.mkdir(parents=True, exist_ok=True)

    dir_hash = hashlib.sha256(str(real_dir).encode()).hexdigest()[:8]
    link = Path(_tempfile.gettempdir()) / f"jch_{dir_hash}"

    # Create or update symlink
    if link.is_symlink():
        if link.resolve() != real_dir:
            link.unlink()
            link.symlink_to(real_dir)
    elif link.exists():
        # Collision with a real dir/file — remove and recreate
        import shutil
        shutil.rmtree(link, ignore_errors=True)
        link.symlink_to(real_dir)
    else:
        link.symlink_to(real_dir)

    return link


def _sanitize_name(agent_name: str) -> str:
    """Convert agent name to safe filename (short)."""
    return agent_name.replace(" ", "_").replace("/", "_")


class AgentChannel:
    """Cross-process async IPC channel via Unix Domain Socket.

    Usage — server side (agent subprocess)::

        channel = AgentChannel("Agent A - Dev", channel_dir)
        await channel.start_server()
        try:
            while True:
                signal = await channel.listen()
                if signal is None:
                    break  # channel closed
                # handle signal...
        finally:
            await channel.stop()

    Usage — client side (any process)::

        AgentChannel.send_signal("Agent A - Dev", "wake")
    """

    def __init__(self, agent_name: str, channel_dir: Path | None = None) -> None:
        self.agent_name = agent_name
        self._channel_dir = _get_sock_dir(channel_dir)
        self._sock_path = self._channel_dir / f"{_sanitize_name(agent_name)}.sock"
        self._server: asyncio.AbstractServer | None = None
        self._wake_event = asyncio.Event()
        self._last_signal: str = ""

    @property
    def socket_path(self) -> Path:
        return self._sock_path

    async def start_server(self) -> None:
        """Start Unix socket server. Call once at agent startup."""
        self._channel_dir.mkdir(parents=True, exist_ok=True)

        # Clean up stale socket
        if self._sock_path.exists():
            self._sock_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._sock_path),
        )
        logger.info(
            "📡 AgentChannel started for %s at %s",
            self.agent_name,
            self._sock_path,
        )

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming socket connections."""
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            signal = data.decode().strip() if data else "wake"
            self._last_signal = signal
            self._wake_event.set()

            writer.write(b"ok\n")
            await writer.drain()
        except Exception as exc:
            logger.debug("AgentChannel client error: %s", exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def listen(self, timeout: float | None = None) -> str | None:
        """Block until a signal arrives.

        Returns the signal string, or None if timeout expires.
        Pass timeout=None (default) for indefinite wait.
        Zero CPU while waiting.
        """
        # If a signal arrived before listen() was called, return it
        if self._wake_event.is_set():
            signal = self._last_signal or "wake"
            self._wake_event.clear()
            self._last_signal = ""
            return signal

        self._last_signal = ""
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
            signal = self._last_signal or "wake"
            self._wake_event.clear()
            self._last_signal = ""
            return signal
        except asyncio.TimeoutError:
            return None

    async def stop(self) -> None:
        """Shutdown server, cleanup socket file."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._sock_path.exists():
            try:
                self._sock_path.unlink()
            except OSError:
                pass
        logger.info("📡 AgentChannel stopped for %s", self.agent_name)

    # ── Static client API (usable from any process) ──

    @staticmethod
    def send_signal(
        agent_name: str,
        signal: str = "wake",
        channel_dir: Path | None = None,
    ) -> bool:
        """Send a signal to an agent's channel. Non-blocking.

        Returns True if signal was delivered, False otherwise.
        Safe to call from synchronous code.
        """
        cdir = _get_sock_dir(channel_dir)
        sock_path = cdir / f"{_sanitize_name(agent_name)}.sock"

        if not sock_path.exists():
            logger.debug(
                "AgentChannel: no socket for %s at %s", agent_name, sock_path
            )
            return False

        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(str(sock_path))
            s.sendall(f"{signal}\n".encode())
            # Read ack (optional, best-effort)
            try:
                s.recv(64)
            except Exception:
                pass
            s.close()
            logger.info(
                "📡 Signal '%s' sent to %s", signal, agent_name
            )
            return True
        except Exception as exc:
            logger.debug(
                "AgentChannel: failed to send signal to %s: %s",
                agent_name,
                exc,
            )
            return False

    @staticmethod
    def is_alive(
        agent_name: str,
        channel_dir: Path | None = None,
    ) -> bool:
        """Check if an agent's channel is active (socket exists).

        This is a quick check — it only verifies the socket file
        exists, not that the server is actively listening.
        """
        cdir = _get_sock_dir(channel_dir)
        sock_path = cdir / f"{_sanitize_name(agent_name)}.sock"
        return sock_path.exists()
