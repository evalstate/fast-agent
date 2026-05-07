"""Spawn Events — JSON Lines IPC protocol between child agent processes and parent.

Child writes events to stderr as JSON lines. Parent parses and forwards
to the display manager. Prefix ``__SPAWN_EVENT__`` distinguishes spawn
events from regular stderr logging.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

EVENT_PREFIX = "__SPAWN_EVENT__"


@dataclass
class SpawnEvent:
    """A structured event from a spawned agent process."""

    event: str  # started | mcp_connected | thinking | response | tool_call | tool_result | result | error | agent_completed
    run_id: str
    role: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        """Serialize to a tagged JSON line for stderr."""
        return f"{EVENT_PREFIX}{json.dumps(asdict(self))}"

    @classmethod
    def from_line(cls, line: str) -> SpawnEvent | None:
        """Parse a tagged JSON line from stderr. Returns None if not a spawn event."""
        stripped = line.strip()
        if not stripped.startswith(EVENT_PREFIX):
            return None
        try:
            payload = json.loads(stripped[len(EVENT_PREFIX) :])
            return cls(**payload)
        except (json.JSONDecodeError, TypeError):
            return None


# ── Direct UDS socket for bypassing broken stderr pipe ──
import os as _os
import socket as _socket

_direct_socket: _socket.socket | None = None


def _get_direct_socket() -> _socket.socket | None:
    """Connect to the backend event socket. Always retries on failure."""
    global _direct_socket  # noqa: PLW0603
    if _direct_socket is not None:
        return _direct_socket
    sock_path = _os.environ.get("SPAWN_EVENT_SOCKET")
    if not sock_path:
        return None
    try:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(5.0)  # Don't hang on connect
        sock.connect(sock_path)
        sock.settimeout(None)  # Back to blocking for sendall
        _direct_socket = sock
        return sock
    except Exception as e:
        sys.stderr.write(f"[emit_event] Socket connect failed ({sock_path}): {e}\n")
        sys.stderr.flush()
        return None


def emit_event(event: str, run_id: str, role: str, **data: Any) -> None:
    """Emit a spawn event via UDS socket to the backend bridge.

    No stderr fallback — if the socket fails, it logs a warning.
    """
    evt = SpawnEvent(event=event, run_id=run_id, role=role, data=data)

    # Send directly via UDS socket to backend
    socket_ok = False
    socket_err = None
    try:
        sock = _get_direct_socket()
        if sock:
            bridge_msg = json.dumps({
                "timestamp": evt.timestamp,
                "agent_name": role,
                "event_type": event,
                "run_id": run_id,
                "data": data,
            })
            sock.sendall((bridge_msg + "\n").encode("utf-8"))
            socket_ok = True
        else:
            socket_err = "no socket (SPAWN_EVENT_SOCKET not set or connect failed)"
    except (BrokenPipeError, OSError, ConnectionResetError) as e:
        socket_err = str(e)
        global _direct_socket  # noqa: PLW0603
        _direct_socket = None  # Force reconnect on next emit

    if not socket_ok:
        # Log warning to stderr — don't swallow failures
        sys.stderr.write(f"[emit_event] FAILED {role}/{event}: {socket_err}\n")
        sys.stderr.flush()


# ─── Event constructors ───


def evt_started(
    run_id: str, role: str, model: str = "", servers: list[str] | None = None
) -> SpawnEvent:
    """Agent process has started."""
    return SpawnEvent(
        event="started",
        run_id=run_id,
        role=role,
        data={"model": model, "servers": servers or []},
    )


def evt_mcp_connected(run_id: str, role: str, server_name: str, status: str = "ok") -> SpawnEvent:
    """An MCP server connection succeeded or failed."""
    return SpawnEvent(
        event="mcp_connected",
        run_id=run_id,
        role=role,
        data={"server_name": server_name, "status": status},
    )


def evt_thinking(run_id: str, role: str, model: str = "") -> SpawnEvent:
    """Agent is waiting for LLM response."""
    return SpawnEvent(
        event="thinking",
        run_id=run_id,
        role=role,
        data={"model": model},
    )


def evt_response(
    run_id: str,
    role: str,
    text: str = "",
    reasoning: str = "",
    stop_reason: str = "",
) -> SpawnEvent:
    """LLM response received, with optional reasoning content."""
    data: dict[str, Any] = {}
    if text:
        data["text"] = text[:1000]
    if reasoning:
        data["reasoning"] = reasoning[:2000]
    if stop_reason:
        data["stop_reason"] = stop_reason
    return SpawnEvent(event="response", run_id=run_id, role=role, data=data)


def evt_tool_call(
    run_id: str,
    role: str,
    tool_name: str,
    args_preview: str = "",
    args_full: dict[str, Any] | None = None,
) -> SpawnEvent:
    """Agent is calling a tool."""
    data: dict[str, Any] = {
        "tool_name": tool_name,
        "args_preview": args_preview[:200],
    }
    if args_full is not None:
        data["args_full"] = args_full
    return SpawnEvent(event="tool_call", run_id=run_id, role=role, data=data)


def evt_tool_result(
    run_id: str,
    role: str,
    tool_name: str,
    status: str = "ok",
    duration_ms: float = 0,
    result_preview: str = "",
    is_error: bool = False,
) -> SpawnEvent:
    """Tool call completed."""
    data: dict[str, Any] = {
        "tool_name": tool_name,
        "status": status,
        "duration_ms": round(duration_ms, 1),
    }
    if result_preview:
        data["result_preview"] = result_preview[:500]
    if is_error:
        data["is_error"] = True
    return SpawnEvent(event="tool_result", run_id=run_id, role=role, data=data)


def evt_result(
    run_id: str, role: str, summary: str = "", duration_seconds: float = 0
) -> SpawnEvent:
    """Agent completed successfully."""
    return SpawnEvent(
        event="result",
        run_id=run_id,
        role=role,
        data={
            "summary": summary[:200],
            "duration_seconds": round(duration_seconds, 1),
        },
    )


def evt_error(run_id: str, role: str, message: str = "") -> SpawnEvent:
    """Agent encountered an error."""
    return SpawnEvent(
        event="error",
        run_id=run_id,
        role=role,
        data={"message": message[:500]},
    )


def evt_agent_completed(
    run_id: str,
    role: str,
    agent_name: str = "",
    status: str = "idle",
    result_summary: str = "",
    duration_seconds: float = 0,
) -> SpawnEvent:
    """Agent completed its task and is going idle.

    Emitted by the parent process (isolated_spawner) after the child exits.
    Used by the bridge to notify PM/stakeholder of completion.
    """
    return SpawnEvent(
        event="agent_completed",
        run_id=run_id,
        role=role,
        data={
            "agent_name": agent_name,
            "status": status,
            "result_summary": result_summary[:500],
            "duration_seconds": round(duration_seconds, 1),
        },
    )


def evt_agent_ready(
    run_id: str,
    role: str,
    agent_name: str = "",
) -> SpawnEvent:
    """Agent's MCP servers are loaded and it is ready to receive tasks.

    Emitted by the child process immediately after ``fast.run()`` context
    is entered and tool hooks are installed.  The parent process uses this
    to update the registry so meeting creators know the agent can accept
    join_meeting instructions.
    """
    return SpawnEvent(
        event="agent_ready",
        run_id=run_id,
        role=role,
        data={"agent_name": agent_name or role},
    )

