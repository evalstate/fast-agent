"""Spawn Display Manager — Real-time TUI visualization for spawned agents.

Prints formatted lines directly to the user's terminal to bypass MCP
subprocess stdout/stderr capture. Uses ancestor process tree discovery
to find the controlling TTY.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from fast_agent.spawn.spawn_events import SpawnEvent


# ─── TTY Writer ───


def _find_ancestor_tty() -> str | None:
    """Walk up the process tree to find the first ancestor with a TTY.

    MCP SDK's get_default_environment() filters out custom env vars,
    and /dev/tty is unavailable in MCP server subprocesses. Instead
    we use ``ps`` to find which TTY the ancestor (main.py) is attached to.
    """
    import os
    import subprocess

    pid = os.getpid()
    for _ in range(10):
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "tty=,ppid="],
                capture_output=True,
                text=True,
                timeout=2,
            )
            parts = result.stdout.strip().split()
            if len(parts) < 2:
                break
            tty_name, ppid = parts[0], parts[1]
            if tty_name and tty_name not in ("??", "-"):
                return f"/dev/{tty_name}"
            pid = int(ppid)
            if pid <= 1:
                break
        except Exception:
            break
    return None


def _open_tty() -> TextIO | None:
    """Open the terminal for direct output.

    Strategy:
    1. Check SPAWN_TUI_TTY env var (if available)
    2. Walk ancestor process tree to find TTY
    3. Try /dev/tty as last resort
    """
    import os

    tty_path = os.environ.get("SPAWN_TUI_TTY")
    if tty_path:
        try:
            return open(tty_path, "w")  # noqa: SIM115
        except OSError:
            pass

    ancestor_tty = _find_ancestor_tty()
    if ancestor_tty:
        try:
            return open(ancestor_tty, "w")  # noqa: SIM115
        except OSError:
            pass

    try:
        return open("/dev/tty", "w")  # noqa: SIM115
    except OSError:
        pass

    return None


# ─── Panel State ───


@dataclass
class SpawnPanel:
    """Tracks the display state of a single spawned agent."""

    run_id: str
    role: str
    task: str
    lifecycle: str = "oneshot"
    agent_name: str = ""
    status: str = "starting"
    model: str = ""
    servers: list[str] = field(default_factory=list)
    mcp_status: dict[str, str] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        return self.agent_name or self.role
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    current_tool: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    error_message: str = ""
    result_summary: str = ""

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "error")

    @property
    def status_icon(self) -> str:
        return {
            "starting": "🚀",
            "thinking": "🤔",
            "tool_call": "🔧",
            "completed": "✅",
            "error": "❌",
        }.get(self.status, "⏳")

    @property
    def lifecycle_badge(self) -> str:
        # Legacy "persistent" rows display as "resumable" — the lifecycle was
        # merged on 2026-05-20.
        return {
            "oneshot": "oneshot",
            "persistent": "resumable",
            "resumable": "resumable",
        }.get(self.lifecycle, self.lifecycle)


@dataclass
class TeamPanel:
    """Tracks the display state of a team workflow."""

    session_id: str
    template_name: str
    workspace: str = ""
    steps: list[dict[str, object]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


# ─── Display Manager ───


class SpawnDisplayManager:
    """Print spawn agent events directly to /dev/tty as formatted lines.

    Each event prints immediately — no Rich Live display needed, no
    conflicts with fast-agent's console. Falls back to a no-op if
    /dev/tty is unavailable.

    Usage::

        mgr = SpawnDisplayManager()
        mgr.add_spawn(run_id, agent_name, task, lifecycle)
        mgr.handle_event(event)   # from child stderr
        mgr.remove_spawn(run_id)
    """

    def __init__(self, tty: TextIO | None = None, event_callback=None) -> None:
        self._tty = tty
        self._tty_tried = tty is not None
        self._event_callback = event_callback
        self._panels: dict[str, SpawnPanel] = {}
        self._teams: dict[str, TeamPanel] = {}

    def _get_tty(self) -> TextIO | None:
        """Lazy-open TTY on first use."""
        if not self._tty_tried:
            self._tty_tried = True
            self._tty = _open_tty()
        return self._tty

    def _write(self, msg: str) -> None:
        """Write a line to the terminal."""
        tty = self._get_tty()
        if tty:
            try:
                tty.write("\n" + msg)
                tty.flush()
            except Exception:
                pass

    # ─── Spawn Management ───

    def start(self) -> None:
        """No-op (kept for API compatibility)."""

    def stop(self) -> None:
        """No-op (kept for API compatibility)."""

    def add_spawn(
        self,
        run_id: str,
        agent_name: str,
        task: str,
        lifecycle: str = "oneshot",
    ) -> None:
        """Register a new spawn for display."""
        self._panels[run_id] = SpawnPanel(
            run_id=run_id,
            role=agent_name,
            task=task[:80],
            lifecycle=lifecycle,
            agent_name=agent_name,
        )
        short_id = run_id[:8]
        self._write(f"  🔀 ── {agent_name} [{short_id}] ({lifecycle}) ──────────────────")
        self._write(f"  📋 Task: {task[:80]}")
        self._write("  🚀 Starting...")

    def remove_spawn(self, run_id: str) -> None:
        """Remove a spawn panel."""
        self._panels.pop(run_id, None)

    def set_event_callback(self, callback) -> None:
        """Set a callback to receive all spawn events (for external integrations)."""
        self._event_callback = callback

    def handle_event(self, event: SpawnEvent) -> None:
        """Process a spawn event from a child and print to terminal."""
        # Forward to callback (e.g. SSE progress bridge) regardless of panel state
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                import logging as _logging
                _logging.getLogger("spawn_display").warning(
                    "Event callback failed for event=%s: %s",
                    event.event, e,
                )

        # File-based debug — trace every event through display
        try:
            import os, pathlib, time
            prj = os.environ.get("SPAWN_PROJECT_DIR", ".")
            dbg = pathlib.Path(prj) / ".runtime" / "cache" / "logs" / "display_handle_debug.log"
            dbg.parent.mkdir(parents=True, exist_ok=True)
            has_cb = "CB_YES" if self._event_callback else "CB_NO"
            with open(dbg, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} DISPLAY: {event.role}/{event.event} run={event.run_id[:8]} {has_cb}\n")
        except Exception as e:
            import logging as _log_dbg
            _log_dbg.getLogger("spawn_display").warning("Debug write failed: %s", e)

        panel = self._panels.get(event.run_id)
        if not panel:
            return

        data = event.data or {}
        short_id = event.run_id[:8]

        if event.event == "started":
            panel.status = "starting"
            panel.model = data.get("model", "")
            panel.servers = data.get("servers", [])
            for s in panel.servers:
                panel.mcp_status[s] = "⏳"
            if panel.servers:
                self._write(f"  📡 MCP: {', '.join(panel.servers)}")

        elif event.event == "mcp_connected":
            server = data.get("server_name", "")
            status = data.get("status", "ok")
            icon = "✓" if status == "ok" else "✗"
            panel.mcp_status[server] = icon
            self._write(f"  📡 {server} {icon}")

        elif event.event == "thinking":
            panel.status = "thinking"
            panel.current_tool = ""
            model = data.get("model", panel.model)
            self._write(f"  🤔 Thinking... ({model})")

        elif event.event == "response":
            panel.status = "response"
            text = data.get("text", "")[:80]
            has_reasoning = bool(data.get("reasoning"))
            prefix = "🧠💬" if has_reasoning else "💬"
            self._write(f"  {prefix} [{panel.display_name}] {text}...")

        elif event.event == "tool_call":
            panel.status = "tool_call"
            tool_name = data.get("tool_name", "unknown")
            args_preview = data.get("args_preview", "")
            panel.current_tool = tool_name
            panel.tool_calls.append(
                {
                    "name": tool_name,
                    "args": args_preview,
                    "status": "running",
                    "start_time": time.time(),
                    "duration_ms": 0,
                }
            )
            args_str = f" ({args_preview[:60]})" if args_preview else ""
            self._write(f"  🔧 [{panel.display_name}] {tool_name}{args_str}")

        elif event.event == "tool_result":
            tool_name = data.get("tool_name", "")
            duration_ms = data.get("duration_ms", 0)
            status = data.get("status", "ok")
            panel.current_tool = ""
            for tc in reversed(panel.tool_calls):
                if tc["name"] == tool_name and tc["status"] == "running":
                    tc["status"] = status
                    tc["duration_ms"] = duration_ms
                    break
            icon = "✓" if status == "ok" else "✗"
            self._write(f"     {icon} [{panel.display_name}] {tool_name} ({duration_ms:.0f}ms)")

        elif event.event == "result":
            panel.status = "completed"
            panel.end_time = time.time()
            panel.result_summary = data.get("summary", "")
            dur = f"{panel.duration:.1f}s"
            n_tools = len(panel.tool_calls)
            self._write(f"  ✅ Completed ({dur}, {n_tools} tool calls)")
            self._write(f"  ── {panel.display_name} [{short_id}] done ──────────────────\n")

        elif event.event == "error":
            panel.status = "error"
            panel.end_time = time.time()
            panel.error_message = data.get("message", "unknown error")
            dur = f"{panel.duration:.1f}s"
            self._write(f"  ❌ Error ({dur}): {panel.error_message[:100]}")
            self._write(f"  ── {panel.display_name} [{short_id}] failed ────────────────\n")

        elif event.event == "idle":
            panel.status = "idle"
            self._write(f"  💤 [{panel.display_name}] Idle — waiting for messages")

        elif event.event == "resumed":
            panel.status = "running"
            msg_count = data.get("message_count", 0)
            self._write(f"  ⚡ [{panel.display_name}] Resumed ({msg_count} messages)")

    # ─── Team Management ───

    def add_team(
        self,
        session_id: str,
        template_name: str,
        steps: list[dict[str, object]],
        workspace: str = "",
    ) -> None:
        """Register a team workflow for display."""
        self._teams[session_id] = TeamPanel(
            session_id=session_id,
            template_name=template_name,
            workspace=workspace,
            steps=steps,
        )
        short_id = session_id[:8]
        self._write(f"\n  🏢 ── Team: {template_name} [{short_id}] ──────────────")
        for i, step in enumerate(steps, 1):
            agent = step.get("agent", "")
            self._write(f"     Step {i}: {step['name']} ({agent}) ⏸")

    def update_team_step(
        self,
        session_id: str,
        step_name: str,
        status: str,
        duration: float = 0,
    ) -> None:
        """Update the status of a team step."""
        team = self._teams.get(session_id)
        if not team:
            return
        for step in team.steps:
            if step["name"] == step_name:
                step["status"] = status
                step["duration"] = duration
                break
        icon = {
            "completed": "✅",
            "running": "⏳",
            "error": "❌",
        }.get(status, "⏸")
        dur_str = f" ({duration:.1f}s)" if duration else ""
        self._write(f"     {icon} {step_name}{dur_str}")

    def remove_team(self, session_id: str) -> None:
        """Remove a team panel."""
        team = self._teams.pop(session_id, None)
        if team:
            short_id = session_id[:8]
            self._write(f"  🏢 ── Team [{short_id}] done ──────────────────\n")


# Global instance
_display_manager: SpawnDisplayManager | None = None


def get_display_manager() -> SpawnDisplayManager:
    """Get or create the global display manager."""
    global _display_manager  # noqa: PLW0603
    if _display_manager is None:
        _display_manager = SpawnDisplayManager()
    return _display_manager
