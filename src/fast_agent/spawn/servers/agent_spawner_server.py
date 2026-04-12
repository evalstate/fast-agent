"""Agent Spawner MCP Server — full lifecycle management for spawned agents.

Tools:
1.  spawn_and_run_isolated — blocking spawn for short tasks
2.  spawn_and_run_background — non-blocking spawn for long tasks
3.  spawn_agent — create persistent agent card
4.  check_spawn_status — poll status of background spawn
5.  get_spawn_result — retrieve completed spawn result
6.  list_active_spawns — list all active spawns
7.  cancel_spawn_tool — cancel a background spawn
8.  cleanup_spawn — remove spawn record
9.  list_spawned_agents — list agent card files
10. remove_spawned_agent — remove agent card file
11. list_available_servers — list MCP servers from config
12. delegate_task_to_spawned_agent — send message to a runtime-spawned agent
13. read_spawned_agent_inbox — read a spawned agent's inbox
14. wait_for_spawned_agent — block until a spawned agent completes
15. restart_spawn — re-run a persistent/resumable spawn
16. resume_spawn — continue a resumable spawn with follow-up
17. spawn_team_tool — spawn a full team from a template
18. get_team_status — get team session status
19. get_team_result — get consolidated team result
20. list_team_templates_tool — list available templates
21. trigger_retrospective — run team retrospective
22. send_team_message — send directive to team PM
23. resume_team_tool — resume completed/idle team (restarts same agents)

All paths are resolved from environment variables set by the host
process, not from module-level globals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from fast_agent.spawn.card_generator import (
    generate_agent_card,
    get_agent_card_content,
    list_agent_cards,
    remove_agent_card,
)
from fast_agent.spawn.config_reader import get_available_servers
from fast_agent.spawn.runtime_paths import get_runtime_paths
from fast_agent.spawn.isolated_spawner import (
    _check_and_resume_on_inbox,
    _find_latest_history,
    cancel_spawn,
    run_isolated_agent,
    run_isolated_agent_background,
)
from fast_agent.spawn.message_bus import MessageBus

from fast_agent.spawn.spawn_display import get_display_manager
from fast_agent.spawn.spawn_registry import SpawnRegistry
from fast_agent.spawn.team_spawner import (
    get_team_session,
    list_team_sessions,
)
from fast_agent.spawn.team_spawner import (
    list_team_templates as _list_templates,
)
from fast_agent.spawn.team_spawner import (
    spawn_team as _spawn_team,
)
from fast_agent.spawn.workspace_manager import (
    get_workspace_summary,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("agent-spawner")


# ───────────────────────────────────────────────────────────
# Shared state — resolved from environment at import time
# ───────────────────────────────────────────────────────────

_PROJECT_DIR = Path(os.environ.get("SPAWN_PROJECT_DIR", os.getcwd()))
# CRITICAL: Export back to os.environ so child processes inherit it.
# Without this, _PROJECT_DIR is only a Python variable and children
# (Level 1) and their MCP servers (Level 2) can't find team sessions
# or spawn registry.
os.environ["SPAWN_PROJECT_DIR"] = str(_PROJECT_DIR)
_SKILLS_DIR = Path(
    os.environ.get(
        "SPAWN_SKILLS_DIR",
        str(_PROJECT_DIR / ".fast-agent" / "skills"),
    )
)
_SERVERS_LIST = ", ".join(get_available_servers(project_dir=str(_PROJECT_DIR)))
_registry = SpawnRegistry(
    registry_file=str(_PROJECT_DIR / ".runtime" / "state" / "spawn_registry.json"),
)
_bus = MessageBus(messages_dir=str(_PROJECT_DIR / ".runtime" / "state" / "messages"))

_display = get_display_manager()

# ── Wire socket-based event forwarding ──
# The MCP server runs in a subprocess — in-memory callbacks cannot
# reach the main backend process. Events are sent as JSON lines over
# a Unix domain socket (path from SPAWN_EVENT_SOCKET env var).
import socket as _socket

_event_socket: _socket.socket | None = None

# ── File-based diagnostic logger (MCP stderr is consumed by protocol) ──
_dbg_file = None
_event_counters = {"sent": 0, "dropped": 0, "callback": 0}


def _dbg(msg: str) -> None:
    """Write diagnostic line to file (MCP stderr is consumed by protocol)."""
    global _dbg_file  # noqa: PLW0603
    try:
        if _dbg_file is None:
            log_dir = _PROJECT_DIR / ".runtime" / "cache" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            _dbg_file = open(log_dir / "event_socket_debug.log", "a", buffering=1)  # line-buffered

        import time as _t
        _dbg_file.write(f"{_t.strftime('%H:%M:%S')} {msg}\n")
    except Exception as e:
        import sys
        sys.stderr.write(f"[_dbg WARN] Cannot write debug: {e}\n")


def _connect_event_socket() -> _socket.socket | None:
    """Connect to the main process event socket."""
    socket_path = os.environ.get("SPAWN_EVENT_SOCKET")
    if not socket_path:
        _dbg("CONNECT: SPAWN_EVENT_SOCKET not set")
        return None
    try:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.connect(socket_path)
        _dbg(f"CONNECT: OK → {socket_path}")
        return sock
    except Exception as e:
        _dbg(f"CONNECT: FAILED {socket_path} → {e}")
        return None


def _send_event_line(line: str) -> None:
    """Send a JSON line to the main process via socket."""
    global _event_socket  # noqa: PLW0603
    if _event_socket is None:
        _event_socket = _connect_event_socket()
    if _event_socket is None:
        _event_counters["dropped"] += 1
        _dbg(f"SEND: NO SOCKET — dropped (total={_event_counters['dropped']})")
        return
    try:
        _event_socket.sendall((line + "\n").encode("utf-8"))
        _event_counters["sent"] += 1
    except (BrokenPipeError, OSError, ConnectionResetError) as e:
        _dbg(f"SEND: FAILED ({e}), reconnecting...")
        _event_socket = _connect_event_socket()
        if _event_socket:
            try:
                _event_socket.sendall((line + "\n").encode("utf-8"))
                _event_counters["sent"] += 1
                _dbg("SEND: RETRY OK")
            except Exception as e2:
                _dbg(f"SEND: RETRY FAILED ({e2})")
                _event_counters["dropped"] += 1
                _event_socket = None
        else:
            _event_counters["dropped"] += 1
            _dbg("SEND: RECONNECT FAILED — events lost!")


def _write_spawn_event_to_socket(event: Any) -> None:
    """Forward a SpawnEvent to the main process via socket."""
    _event_counters["callback"] += 1
    evt_type = getattr(event, "event", "?")
    agent = getattr(event, "agent_name", "") or getattr(event, "role", "")
    _dbg(f"CALLBACK #{_event_counters['callback']}: {agent}/{evt_type}")
    try:
        import time as _time

        line = json.dumps(
            {
                "timestamp": _time.time(),
                "agent_name": agent,
                "event_type": evt_type,
                "run_id": getattr(event, "run_id", ""),
                "data": getattr(event, "data", {}),
            }
        )
        _send_event_line(line)
    except Exception as e:
        _dbg(f"CALLBACK ERROR: {evt_type} → {e}")


_display.set_event_callback(_write_spawn_event_to_socket)


def _emit_removal_event(agent_names: list[str], run_ids: list[str]) -> None:
    """Send a removal event so the backend bridge can clean DB records."""
    try:
        import time as _time

        line = json.dumps({
            "timestamp": _time.time(),
            "role": "system",
            "event_type": "removed",
            "run_id": "",
            "data": {
                "agent_names": agent_names,
                "run_ids": run_ids,
            },
        })
        _send_event_line(line)
    except Exception as e:
        _dbg(f"REMOVAL EVENT FAILED: {e}")


# ── Lifecycle hooks — send lifecycle events via socket ──
class _SocketSpawnLifecycleHooks:
    """SpawnLifecycleHooks that sends lifecycle events via socket for bridge consumption."""

    def _emit(self, event_type: str, run_id: str, agent_name: str, data: dict | None = None) -> None:
        try:
            import time as _time

            line = json.dumps({
                "timestamp": _time.time(),
                "agent_name": agent_name or "system",
                "event_type": event_type,
                "run_id": run_id,
                "data": data or {},
            })
            _send_event_line(line)
        except Exception as e:
            _dbg(f"LIFECYCLE EMIT FAILED: {event_type} → {e}")

    async def on_pre_spawn(self, run_id: str, agent_name: str, config: dict) -> None:
        self._emit("lifecycle_pre_spawn", run_id, agent_name, {"config_keys": list(config.keys())})

    async def on_registered(self, run_id: str, agent_name: str, record: Any) -> None:
        self._emit("lifecycle_registered", run_id, agent_name, {
            "lifecycle": getattr(record, "lifecycle", ""),
            "role": getattr(record, "role", ""),
            "team_name": getattr(record, "team_name", ""),
        })

    async def on_process_started(self, run_id: str, agent_name: str, pid: int) -> None:
        self._emit("lifecycle_process_started", run_id, agent_name, {"pid": pid})

    async def on_agent_ready(self, run_id: str, agent_name: str) -> None:
        self._emit("lifecycle_agent_ready", run_id, agent_name)

    async def on_completed(self, run_id: str, agent_name: str, result: str) -> None:
        self._emit("lifecycle_completed", run_id, agent_name, {"result_preview": result[:200]})

    async def on_error(self, run_id: str, agent_name: str, error: str) -> None:
        self._emit("lifecycle_error", run_id, agent_name, {"error": error[:500]})

    async def on_idle(self, run_id: str, agent_name: str) -> None:
        self._emit("lifecycle_idle", run_id, agent_name)

    async def on_pre_cleanup(self, run_id: str, agent_name: str) -> None:
        self._emit("lifecycle_pre_cleanup", run_id, agent_name)

    async def on_after_cleanup(self, run_id: str, agent_name: str) -> None:
        self._emit("lifecycle_after_cleanup", run_id, agent_name)

    async def on_auto_resume(self, run_id: str, agent_name: str, new_run_id: str, reason: str) -> None:
        self._emit("lifecycle_auto_resume", run_id, agent_name, {"new_run_id": new_run_id, "reason": reason})

    async def on_cancelled(self, run_id: str, agent_name: str) -> None:
        self._emit("lifecycle_cancelled", run_id, agent_name)


_spawn_hooks = _SocketSpawnLifecycleHooks()


def _resolve_skills_for_spawn(
    skills_csv: str,
) -> list[str]:
    """Convert comma-separated skill names to a temp dir.

    Returns a list with a single path (temp dir containing
    symlinks to skills), or an empty list if no valid skills.
    """
    if not skills_csv or not skills_csv.strip():
        return []

    skill_names = [s.strip() for s in skills_csv.split(",") if s.strip()]
    valid_skills: list[tuple[str, Path]] = []
    for name in skill_names:
        skill_dir = _SKILLS_DIR / name
        if skill_dir.exists() and (skill_dir / "SKILL.md").exists():
            valid_skills.append((name, skill_dir))
        else:
            logger.warning("Skill '%s' not found at %s", name, skill_dir)

    if not valid_skills:
        return []

    role_skills_dir = Path(tempfile.mkdtemp(prefix="fastagent_skills_"))
    for name, skill_dir in valid_skills:
        symlink = role_skills_dir / name
        try:
            symlink.symlink_to(skill_dir)
        except OSError:
            import shutil

            shutil.copytree(skill_dir, symlink)

    logger.info(
        "[SKILLS] Resolved %d skills: %s -> %s",
        len(valid_skills),
        [n for n, _ in valid_skills],
        role_skills_dir,
    )
    return [str(role_skills_dir)]


# ───────────────────────────────────────────────────────────
# Spawn Tools (blocking + background)
# ───────────────────────────────────────────────────────────


@mcp.tool()
async def spawn_and_run_isolated(
    task: str,
    instruction: str = "",
    context: str = "",
    servers: str = "",
    model: str = "",
    timeout_seconds: int = 120,
    role: str = "",
    lifecycle: str = "oneshot",
    skills: str = "",
) -> str:
    """Spawn an ISOLATED sub-agent (BLOCKING).

    Use for short tasks (< 2 min). Result returns in the same turn.

    Args:
        task: What the sub-agent should do (be specific).
        instruction: Custom system instruction.
        context: Relevant context from conversation.
        servers: Comma-separated MCP server names.
        model: Override LLM model.
        timeout_seconds: Max execution time (default 120).
        role: Role label for tracking.
        lifecycle: "oneshot" | "persistent".
        skills: Comma-separated skill names.
    """
    server_list = [s.strip() for s in servers.split(",") if s.strip()] if servers else []
    skill_paths = _resolve_skills_for_spawn(skills)

    result = await run_isolated_agent(
        task=task,
        project_dir=str(_PROJECT_DIR),
        instruction=instruction,
        context=context,
        servers=server_list,
        model=model,
        timeout_seconds=timeout_seconds,
        role=role,
        lifecycle=lifecycle,
        registry=_registry,
        display_manager=_display,
        skills=skill_paths,
    )

    run_id = result.get("run_id", "")
    formatted = result.get("formatted_result", json.dumps(result))
    # Prepend run_id so the caller can resume this agent later
    if run_id:
        return f"[run_id: {run_id}]\n{formatted}"
    return formatted


@mcp.tool()
async def spawn_and_run_background(
    task: str,
    instruction: str = "",
    context: str = "",
    servers: str = "",
    model: str = "",
    timeout_seconds: int = 600,
    role: str = "",
    lifecycle: str = "oneshot",
    skills: str = "",
) -> str:
    """Spawn an agent in the BACKGROUND (non-blocking).

    Returns a run_id immediately. Use ``check_spawn_status``
    and ``get_spawn_result`` to poll.

    Args:
        task: What the sub-agent should do.
        instruction: Custom system instruction.
        context: Relevant context from conversation.
        servers: Comma-separated MCP server names.
        model: Override LLM model.
        timeout_seconds: Max execution time (default 600).
        role: Role label for tracking.
        lifecycle: "oneshot" | "persistent".
        skills: Comma-separated skill names.
    """
    server_list = [s.strip() for s in servers.split(",") if s.strip()] if servers else []
    skill_paths = _resolve_skills_for_spawn(skills)

    # Non-oneshot agents should not be killed by outer timeout
    effective_timeout = timeout_seconds
    if lifecycle in ("persistent", "resumable") and timeout_seconds > 0:
        logger.info(
            "Overriding timeout_seconds=%d → 0 for lifecycle=%s agent",
            timeout_seconds,
            lifecycle,
        )
        effective_timeout = 0

    run_id = await run_isolated_agent_background(
        task=task,
        project_dir=str(_PROJECT_DIR),
        instruction=instruction,
        context=context,
        servers=server_list,
        model=model,
        timeout_seconds=effective_timeout,
        role=role,
        agent_name=role,
        lifecycle=lifecycle,
        registry=_registry,
        display_manager=_display,
        skills=skill_paths,
        spawn_lifecycle_hooks=_spawn_hooks,
    )

    return json.dumps(
        {
            "status": "spawned",
            "run_id": run_id,
            "message": (
                f"Agent spawned in background. Use check_spawn_status(run_id='{run_id}') to poll."
            ),
        }
    )


# ───────────────────────────────────────────────────────────
# Spawn Management Tools
# ───────────────────────────────────────────────────────────


@mcp.tool()
def check_spawn_status(run_id: str) -> str:
    """Check the status and result of a background-spawned agent.

    Returns status, progress info, and result if completed.
    For oneshot spawns, auto-cleans after reading result.

    Args:
        run_id: The run_id from spawn_and_run_background.
    """
    record = _registry.get_latest(run_id)
    if not record:
        return json.dumps({"error": f"No spawn found with run_id '{run_id}'"})

    info: dict[str, Any] = {
        "run_id": record.run_id,
        "role": record.role,
        "status": record.status,
        "lifecycle": record.lifecycle,
        "task": record.task,
        "duration_seconds": record.duration_seconds,
        "error": record.error or None,
    }

    # Include result if terminal
    if record.is_terminal and record.result:
        info["result"] = record.result
        if record.lifecycle == "oneshot":
            _registry.remove(run_id)

    return json.dumps(info)


@mcp.tool()
def list_active_spawns() -> str:
    """List all tracked spawns and their status."""
    summaries = _registry.to_summary()
    return json.dumps({"count": len(summaries), "spawns": summaries})


@mcp.tool()
async def cancel_spawn_tool(run_id: str, cleanup: bool = False) -> str:
    """Cancel a running background spawn and optionally remove its record.

    Args:
        run_id: The run_id to cancel.
        cleanup: If True, also remove the spawn record from registry.
    """
    cancelled = await cancel_spawn(run_id, registry=_registry, spawn_lifecycle_hooks=_spawn_hooks)
    if cancelled:
        if cleanup:
            _registry.remove(run_id)
        return json.dumps({"status": "cancelled", "run_id": run_id, "cleaned_up": cleanup})
    # If not running, try cleanup only
    if cleanup:
        removed = _registry.remove(run_id)
        if removed:
            return json.dumps({"status": "removed", "run_id": run_id})
    return json.dumps({"error": (f"Could not cancel '{run_id}' — not running or not found.")})


# ───────────────────────────────────────────────────────────
# Lifecycle Management (restart / resume)
# ───────────────────────────────────────────────────────────


@mcp.tool()
async def restart_spawn(run_id: str) -> str:
    """Re-run a completed persistent/resumable spawn.

    Creates a new background spawn with the original config.

    Args:
        run_id: The run_id of the completed spawn to restart.
    """
    record = _registry.get(run_id)
    if not record:
        return json.dumps({"error": f"No spawn found with run_id '{run_id}'"})

    if not record.is_terminal:
        return json.dumps(
            {"error": (f"Spawn '{run_id}' is still running (status: {record.status}).")}
        )

    if record.lifecycle == "oneshot":
        return json.dumps(
            {"error": ("Cannot restart a oneshot spawn. Use persistent or resumable lifecycle.")}
        )

    cfg = record.original_config
    if not cfg:
        return json.dumps({"error": (f"No saved config for spawn '{run_id}'. Cannot restart.")})

    new_run_id = await run_isolated_agent_background(
        task=cfg.get("task", record.task),
        project_dir=str(_PROJECT_DIR),
        instruction=cfg.get("instruction", ""),
        context=cfg.get("context", ""),
        servers=cfg.get("servers", []),
        model=cfg.get("model", ""),
        timeout_seconds=cfg.get("timeout_seconds", 600),
        role=cfg.get("role", record.role),
        agent_name=cfg.get("agent_name", record.agent_name),
        team_name=cfg.get("team_name", record.team_name),
        lifecycle=record.lifecycle,
        registry=_registry,
        display_manager=_display,
        spawn_lifecycle_hooks=_spawn_hooks,
    )

    _registry._load()
    if run_id in _registry._data:
        _registry._data[run_id]["restart_count"] = record.restart_count + 1
        _registry._data[run_id].setdefault("metadata", {})["latest_restart_run_id"] = new_run_id
        _registry._save()

    return json.dumps(
        {
            "status": "restarted",
            "original_run_id": run_id,
            "new_run_id": new_run_id,
            "restart_count": record.restart_count + 1,
            "message": (f"Spawn restarted. Use check_spawn_status(run_id='{new_run_id}') to poll."),
        }
    )


@mcp.tool()
async def resume_spawn(run_id: str, follow_up_task: str) -> str:
    """Continue a completed resumable spawn with follow-up.

    The agent restarts with full conversation history from its previous
    session. Agent name, workspace, and context are preserved.

    Args:
        run_id: The run_id of the completed resumable spawn.
        follow_up_task: New task building on previous results.
    """
    record = _registry.get(run_id)
    if not record:
        return json.dumps({"error": f"No spawn found with run_id '{run_id}'"})

    if not record.is_terminal:
        return json.dumps({"error": (f"Spawn '{run_id}' is still running. Wait for completion.")})

    if record.lifecycle != "resumable":
        return json.dumps(
            {
                "error": (
                    f"Spawn '{run_id}' has lifecycle "
                    f"'{record.lifecycle}'. "
                    "Only 'resumable' spawns can be resumed."
                )
            }
        )

    cfg = record.original_config
    if not cfg:
        return json.dumps({"error": (f"No saved config for spawn '{run_id}'. Cannot resume.")})

    # Find previous session history for native FastAgent resume
    # (same pattern as auto-resume in _check_and_resume_on_inbox)
    workspace_dir = cfg.get("workspace_dir", "")
    history_file = _find_latest_history(workspace_dir) if workspace_dir else None

    agent_name = cfg.get("agent_name", record.agent_name)
    if history_file:
        # History file found — agent gets full conversation via
        # load_history_into_agent(). Only pass follow-up as task.
        enriched_context = follow_up_task
        logger.info(
            "📂 resume_spawn: found history for %s: %s", agent_name, history_file,
        )
    else:
        # No history file — fall back to text-based context
        prev_context = cfg.get("context", "")
        prev_result = record.result or ""
        enriched_context = ""
        if prev_context:
            enriched_context += f"## Original Context\n{prev_context}\n\n"
        if prev_result:
            enriched_context += f"## Previous Agent Result\n{prev_result}\n\n"
        enriched_context += f"## Follow-Up Task\n{follow_up_task}"
        logger.info(
            "⚠️ resume_spawn: no history for %s — using text-based context", agent_name,
        )

    # Determine correct project_dir from original_config
    project_dir = cfg.get("project_dir", str(_PROJECT_DIR))

    # Re-inject team context if this is a team agent
    env_vars = cfg.get("env_vars") or None
    team_session_id = (env_vars or {}).get("TEAM_SESSION_ID", "")
    if team_session_id:
        try:
            session = get_team_session(team_session_id)
            if session:
                role = cfg.get("role", "")
                roster_ctx = session.roster_context(for_role=role)
                enriched_context = roster_ctx + "\n\n" + enriched_context
                logger.info(
                    "📋 Re-injected team context for %s (session %s)",
                    agent_name, team_session_id,
                )
        except Exception as e:
            logger.warning("Failed to re-inject team context: %s", e)

    new_run_id = await run_isolated_agent_background(
        task=follow_up_task,
        project_dir=project_dir,
        instruction=cfg.get("instruction", ""),
        context=enriched_context,
        servers=cfg.get("servers", []),
        model=cfg.get("model", ""),
        timeout_seconds=cfg.get("timeout_seconds", 0),
        role=cfg.get("role", record.role),
        agent_name=agent_name,
        team_name=cfg.get("team_name", record.team_name),
        workspace_dir=cfg.get("workspace_dir") or None,
        lifecycle="resumable",
        registry=_registry,
        display_manager=_display,
        env_vars=env_vars,
        skills=cfg.get("skills", []),
        history_file=history_file,
        spawn_lifecycle_hooks=_spawn_hooks,
        session_id=team_session_id,
    )

    _registry._load()
    if run_id in _registry._data:
        _registry._data[run_id]["restart_count"] = record.restart_count + 1
        _registry._data[run_id].setdefault("metadata", {})["latest_resume_run_id"] = new_run_id
        _registry._save()

    return json.dumps(
        {
            "status": "resumed",
            "original_run_id": run_id,
            "new_run_id": new_run_id,
            "agent_name": agent_name,
            "message": (
                f"Agent '{agent_name}' resumed. Use check_spawn_status(run_id='{new_run_id}') to poll."
            ),
        }
    )





# ───────────────────────────────────────────────────────────
# Agent Card Management (persistent agents)
# ───────────────────────────────────────────────────────────


@mcp.tool()
def spawn_agent(
    name: str,
    instruction: str,
    servers: str = "",
    model: str = "",
    extra_instruction: str = "",
) -> str:
    """Create a PERSISTENT sub-agent (agent card).

    For one-shot tasks, use spawn_and_run_isolated instead.

    Args:
        name: Unique name (e.g. "web_researcher").
        instruction: Defining the agent's role and behavior.
        servers: Comma-separated MCP server names.
        model: Override model.
        extra_instruction: Additional instruction text.
    """
    try:
        server_list = [s.strip() for s in servers.split(",") if s.strip()] if servers else []
        # Resolve agent_cards dir from runtime paths
        agent_cards_dir = get_runtime_paths(str(_PROJECT_DIR))["agent_cards"]
        agent_cards_dir.mkdir(parents=True, exist_ok=True)

        # Also try the legacy .fast-agent/agent_cards location
        legacy_cards = _PROJECT_DIR / ".fast-agent" / "agent_cards"
        if legacy_cards.exists():
            agent_cards_dir = legacy_cards

        card_path = generate_agent_card(
            name=name,
            instruction=instruction,
            agent_cards_dir=str(agent_cards_dir),
            servers=server_list,
            model=model if model else None,
            extra_instruction=extra_instruction,
        )

        reload_signal = Path(get_runtime_paths(str(_PROJECT_DIR))["reload_signal"])
        reload_signal.touch()

        return json.dumps(
            {
                "status": "success",
                "agent_name": card_path.stem,
                "card_file": str(card_path),
                "servers": server_list,
                "message": (
                    f"Agent '{card_path.stem}' created. Available at the start of the NEXT turn."
                ),
            }
        )
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Unexpected error: {e}",
            }
        )





@mcp.tool()
def remove_spawned_agent(name: str) -> str:
    """Remove a persistent agent or an entire team by exact name.

    IMPORTANT: Call list_spawned_agents first to get the exact agent
    or team name. Use the exact "name" or "team_name" value from
    list_spawned_agents — do NOT guess or modify the name.

    First tries to remove an individual agent by exact name.
    If not found, checks if the name matches a team_name
    and removes all agents in that team.

    Args:
        name: Exact name of the agent or team_name to remove.
              Must match exactly as returned by list_spawned_agents.
    """
    # Resolve agent_cards dir (same logic as spawn_agent)
    agent_cards_dir = get_runtime_paths(str(_PROJECT_DIR))["agent_cards"]
    legacy_cards = _PROJECT_DIR / ".fast-agent" / "agent_cards"
    if legacy_cards.exists():
        agent_cards_dir = legacy_cards

    card_removed = remove_agent_card(name, agent_cards_dir=str(agent_cards_dir))

    # Try individual agent removal from registry
    registry_removed = False
    removed_run_ids: list[str] = []
    record = _registry.find_by_name(name)
    if record:
        removed_run_ids.append(record.run_id)
        registry_removed = _registry.remove(record.run_id)

    if card_removed or registry_removed:
        parts = []
        if card_removed:
            parts.append("card removed")
        if registry_removed:
            parts.append("registry cleaned")
        # Emit removal event so backend bridge cleans DB
        _emit_removal_event([name], removed_run_ids)
        return json.dumps(
            {
                "status": "success",
                "message": f"Agent '{name}' removed ({', '.join(parts)}).",
            }
        )

    # Try team-level removal — name might match a team_name
    team_members = _registry.find_by_team(name)
    if team_members:
        # Collect run_ids for DB cleanup
        team_run_ids = [m.run_id for m in team_members if m.run_id]
        team_agent_names = [m.agent_name for m in team_members]

        # Remove all agent cards for team members
        cards_removed = 0
        for member in team_members:
            if remove_agent_card(member.agent_name, agent_cards_dir=str(agent_cards_dir)):
                cards_removed += 1

        # Remove all registry entries for the team
        registry_count = _registry.remove_team(name)

        # Emit removal event so backend bridge cleans DB
        _emit_removal_event(team_agent_names, team_run_ids)

        return json.dumps(
            {
                "status": "success",
                "message": (
                    f"Team '{name}' fully cleaned up: "
                    f"{registry_count} registry entries."
                ),
                "removed_agents": team_agent_names,
                "cleaned": [f"{registry_count} registry entries"],
            }
        )

    # Collect known teams for hint
    all_teams = {d.get("team_name", "") for d in _registry._data.values() if d.get("team_name")}
    result: dict[str, Any] = {
        "status": "error",
        "message": f"Agent or team '{name}' not found.",
    }
    if all_teams:
        result["available_teams"] = sorted(all_teams)
    return json.dumps(result)


@mcp.tool()
def list_available_servers_tool() -> str:
    """List all MCP servers available for agent assignment."""
    available = get_available_servers(project_dir=str(_PROJECT_DIR))
    return json.dumps(
        {
            "servers": available,
            "message": ("Pass these names as comma-separated values."),
        }
    )


# ───────────────────────────────────────────────────────────
# Team Management
# ───────────────────────────────────────────────────────────


@mcp.tool()
async def spawn_team_tool(
    template: str,
    project_brief: str,
    team_name: str,
    mode: str = "background",
    timeout_seconds: int = 300,
) -> str:
    """Spawn a team from a template. Only the orchestrator (PM) starts first.

    The orchestrator agent spawns immediately with the full team roster
    as context. Other team members are NOT spawned yet — the orchestrator
    uses spawn_team_members() to bring in specific roles on demand.

    Args:
        template: Team template name (e.g. "agile-team").
        project_brief: Description for the team.
        team_name: Unique name for this team instance.
                   Name should reflect the task purpose
                   (e.g. "notes-cli-dev", "payment-redesign").
                   This name is used for display and removal.
        mode: "blocking" (wait for ALL agents to complete) or
              "background" (return immediately). Default: background.
        timeout_seconds: Max seconds to wait in blocking mode (default 300).
                         If timeout is reached, returns current progress so you
                         can decide to call get_team_status/get_team_result later,
                         or call spawn_team_tool again with more time.
    """
    try:
        session = await _spawn_team(
            template_name=template,
            project_brief=project_brief,
            registry=_registry,
            display_manager=_display,
            project_dir=str(_PROJECT_DIR),
            mode=mode,
            team_name=team_name,
            spawn_lifecycle_hooks=_spawn_hooks,
        )

        if mode == "blocking":
            # Wait for ALL agents (orchestrator + members) to reach terminal state
            poll_interval = 5
            elapsed = 0
            terminal_statuses = {"completed", "error", "cancelled", "failed", "idle"}
            while elapsed < timeout_seconds:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                # Check all agents in session
                all_terminal = True
                for agent_name, info in session.agents.items():
                    status = info.get("status", "unknown")
                    # Also check registry for latest status
                    run_id = info.get("run_id", "")
                    if run_id:
                        record = _registry.get_latest(run_id)
                        if record and record.status in terminal_statuses:
                            info["status"] = record.status
                            if record.result:
                                info["result"] = record.result
                            continue
                    if status not in terminal_statuses and status != "available":
                        all_terminal = False

                if all_terminal:
                    break

            # Build per-agent info
            agents_info = {}
            still_running = []
            for agent_name, info in session.agents.items():
                agent_status = info.get("status", "unknown")
                agents_info[agent_name] = {
                    "run_id": info.get("run_id", ""),
                    "role": agent_name,
                    "status": agent_status,
                    "result": (info.get("result") or "")[:2000],
                }
                if agent_status not in terminal_statuses and agent_status != "available":
                    still_running.append(agent_name)

            timed_out = elapsed >= timeout_seconds and len(still_running) > 0
            session.sprint_status = "timeout" if timed_out else "completed"

            result: dict[str, Any] = {
                "status": session.sprint_status,
                "session_id": session.session_id,
                "team_name": team_name,
                "template": session.template.get("name", template),
                "workspace": str(session.workspace),
                "agents": agents_info,
                "elapsed_seconds": elapsed,
            }

            if timed_out:
                result["still_running"] = still_running
                result["message"] = (
                    f"Timeout after {elapsed}s. {len(still_running)} agents still running: "
                    f"{', '.join(still_running)}. "
                    f"Use get_team_status(session_id='{session.session_id}') to check later, "
                    f"or get_team_result(session_id='{session.session_id}') for partial results."
                )

            return json.dumps(result)

        # Background mode — return immediately
        agents_info = {
            role: {
                "run_id": info.get("run_id", ""),
                "role": role,
                "status": info.get("status", "unknown"),
            }
            for role, info in session.agents.items()
        }

        # Collect available (not yet spawned) roles
        available_roles = [
            role for role, info in session.agents.items()
            if info.get("status") == "available"
        ]

        result: dict[str, Any] = {
            "status": "orchestrator_spawned",
            "session_id": session.session_id,
            "team_name": team_name,
            "template": session.template.get("name", template),
            "workspace": str(session.workspace),
            "agents": agents_info,
            "available_roles": available_roles,
        }

        result["message"] = (
            "Orchestrator spawned. Other roles are available but not yet "
            "running. The orchestrator will use spawn_team_members() "
            "to bring in specific roles as needed. "
            f"session_id='{session.session_id}'"
        )

        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Team spawn failed: {e}"})


@mcp.tool()
async def spawn_team_members(
    roles: str,
    team_session_id: str = "",
) -> str:
    """Spawn specific team members from an active team session.

    Only the orchestrator (PM) should call this. Each role is spawned
    with its predefined skills, servers, and instruction from the template.

    Args:
        roles: Comma-separated role keys to spawn (e.g. "ba,dev,qe").
        team_session_id: The session_id from spawn_team_tool.
                         Auto-detected from TEAM_SESSION_ID env if empty.
    """
    from fast_agent.spawn.team_spawner import spawn_team_members_for_session

    team_session_id = team_session_id or os.environ.get("TEAM_SESSION_ID", "")
    if not team_session_id:
        return json.dumps({"error": "team_session_id required. Not in a team session?"})

    try:
        role_list = [r.strip() for r in roles.split(",") if r.strip()]
        if not role_list:
            return json.dumps({"error": "No roles specified."})

        results = await spawn_team_members_for_session(
            session_id=team_session_id,
            roles=role_list,
            registry=_registry,
            display_manager=_display,
            project_dir=str(_PROJECT_DIR),
            spawn_lifecycle_hooks=_spawn_hooks,
        )

        return json.dumps({
            "status": "spawned",
            "session_id": team_session_id,
            "spawned": results,
            "message": (
                f"Spawned {len(results)} team members. "
                "Use check_spawn_status(run_id) to monitor progress."
            ),
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Failed to spawn team members: {e}"})


@mcp.tool()
def get_team_status(session_id: str = "") -> str:
    """Get team session status. If session_id is empty, lists ALL sessions (including past ones from disk).

    Use without session_id to discover old sessions after a server restart,
    then use the returned session_id with resume_team_tool or send_team_message.

    Args:
        session_id: The session_id from spawn_team_tool. If empty, lists all sessions.
    """
    # --- List all sessions mode ---
    if not session_id:
        sessions = list_team_sessions()
        sessions.sort(key=lambda s: s.get("session_id", ""), reverse=True)
        return json.dumps({
            "count": len(sessions),
            "sessions": [
                {
                    "session_id": s.get("session_id"),
                    "team_name": s.get("team_name"),
                    "sprint_status": s.get("sprint_status"),
                    "agents": {
                        name: info.get("status")
                        for name, info in s.get("agents", {}).items()
                    },
                }
                for s in sessions
            ],
        })

    # --- Single session mode ---
    session = get_team_session(session_id)
    if not session:
        return json.dumps({"error": f"Team session '{session_id}' not found."})

    # Sync session agent statuses from registry (session may be stale)
    _done_statuses = {"completed", "idle", "error", "timeout", "cancelled"}
    for role, info in session.agents.items():
        run_id = info.get("run_id", "")
        if run_id:
            record = _registry.get_latest(run_id)
            if record:
                info["status"] = record.status
                if record.result:
                    info["result"] = record.result

    agents = session.get_roster()

    # Separate spawned vs available (not yet spawned)
    spawned = {k: v for k, v in session.agents.items()
               if v.get("status") != "available"}
    available = {k: v for k, v in session.agents.items()
                 if v.get("status") == "available"}

    total_spawned = len(spawned)
    done = sum(1 for a in spawned.values()
               if a.get("status") in _done_statuses)
    errored = sum(1 for a in spawned.values()
                  if a.get("status") == "error")

    # Determine sprint status
    if total_spawned == 0:
        sprint_status = "not_started"
    elif done == total_spawned:
        sprint_status = "completed"
    else:
        sprint_status = "running"

    return json.dumps(
        {
            "session_id": session_id,
            "template": session.template.get("name", "unknown"),
            "workspace": str(session.workspace),
            "sprint_status": sprint_status,
            "progress": f"{done}/{total_spawned} agents done, {errored} errors",
            "agents": agents,
            "available_roles": [k for k in available],
        }
    )


@mcp.tool()
def get_team_result(session_id: str) -> str:
    """Get the consolidated result of a completed team.

    Includes per-agent results and workspace contents.

    Args:
        session_id: The session_id of the team session.
    """
    session = get_team_session(session_id)
    if not session:
        return json.dumps({"error": f"Team session '{session_id}' not found."})

    # Sync agent statuses/results from registry (same pattern as get_team_status)
    for role, info in session.agents.items():
        run_id = info.get("run_id", "")
        if run_id and _registry:
            record = _registry.get_latest(run_id)
            if record:
                info["status"] = record.status
                if record.result:
                    info["result"] = record.result

    agents_results: dict[str, dict[str, Any]] = {}
    for role, info in session.agents.items():
        agents_results[role] = {
            "run_id": info.get("run_id", ""),
            "role": role,
            "status": info.get("status", "unknown"),
            "result": info.get("result", "")[:3000],
        }

    ws_summary = get_workspace_summary(session.workspace)

    return json.dumps(
        {
            "session_id": session_id,
            "template": session.template.get("name", "unknown"),
            "workspace": str(session.workspace),
            "workspace_contents": ws_summary.get("directories", {}),
            "agents": agents_results,
        }
    )


@mcp.tool()
def list_team_templates_tool() -> str:
    """List all available team templates."""
    templates = _list_templates(template_dir=str(_PROJECT_DIR / "team_templates"))
    return json.dumps({"count": len(templates), "templates": templates})




# ───────────────────────────────────────────────────────────
# Phase 2: User → PM Communication Bridge
# ───────────────────────────────────────────────────────────


@mcp.tool()
async def send_team_message(
    session_id: str,
    message: str,
    priority: str = "normal",
) -> str:
    """Send a directive message to the team's PM (orchestrator).

    This is the ONLY way for Jarvis to communicate with a running team.
    Messages always route to the PM — never directly to team members.
    If PM is idle, it will be auto-woken to process the directive.

    Args:
        session_id: The team session_id from spawn_team_tool.
        message: Directive or feedback for the PM.
        priority: Message priority: low | normal | high | urgent.
    """
    session = get_team_session(session_id)
    if not session:
        return json.dumps({"error": f"Team session '{session_id}' not found."})

    # Find the orchestrator (PM) agent name
    orchestrator_role = session.template.get("orchestrator", "")
    pm_agent_name = ""
    pm_run_id = ""
    for name, info in session.agents.items():
        if info.get("role") == orchestrator_role:
            pm_agent_name = name
            pm_run_id = info.get("run_id", "")
            break

    if not pm_agent_name:
        return json.dumps({"error": "No PM/orchestrator found in this team session."})

    # Send message via MessageBus
    msg = _bus.send(
        from_name="Jarvis",
        to_name=pm_agent_name,
        content=message,
        message_type="directive",
        priority=priority,
        context={"session_id": session_id},
    )

    # Auto-wake PM if idle
    woke = False
    if pm_run_id:
        record = _registry.get_latest(pm_run_id)
        if record and record.status in ("idle", "completed"):
            try:
                await _check_and_resume_on_inbox(
                    run_id=pm_run_id,
                    registry=_registry,
                    display_manager=_display,
                    project_dir=str(_PROJECT_DIR),
                )
                woke = True
            except Exception as e:
                logger.warning("Failed to auto-wake PM: %s", e)

    return json.dumps({
        "status": "sent",
        "message_id": msg.message_id,
        "to": pm_agent_name,
        "priority": priority,
        "auto_woke_pm": woke,
        "message": (
            f"Directive sent to {pm_agent_name}."
            + (" PM was idle and has been woken." if woke else "")
        ),
    })


# ───────────────────────────────────────────────────────────
# Phase 4: Resume Team Session
# ───────────────────────────────────────────────────────────


def _resolve_latest_run_id(run_id: str) -> str:
    """Follow the resume chain to find the latest run_id.

    When an agent is resumed (via inbox messages or manual resume),
    each resume creates a new run_id. The original run_id stored in
    the team session may be stale. This follows the chain:
        original → resume_1 → resume_2 → ... → latest

    DB (registry) is the single source of truth.
    """
    record = _registry.get(run_id)
    if not record:
        return run_id

    metadata = getattr(record, "metadata", None) or {}
    latest = metadata.get("latest_resume_run_id")
    if latest:
        return _resolve_latest_run_id(latest)
    return run_id


@mcp.tool()
async def resume_team_tool(
    session_id: str,
    follow_up_task: str,
) -> str:
    """Resume a completed/idle team with a follow-up task.

    Restarts the SAME agents from the previous session — same agent
    names, workspace, and full conversation history. Does NOT create
    a new team.

    Each agent is resumed individually via resume_spawn(), which loads
    their previous conversation history and restores context.

    Args:
        session_id: The session_id of the team to resume.
        follow_up_task: New task/brief for the team.
    """
    session = get_team_session(session_id)
    if not session:
        return json.dumps({"error": f"Team session '{session_id}' not found."})

    results: dict[str, Any] = {}
    resumed_count = 0
    skipped_count = 0

    for agent_name, info in session.agents.items():
        original_run_id = info.get("run_id", "")
        if not original_run_id:
            results[agent_name] = {"status": "skipped", "reason": "no run_id"}
            skipped_count += 1
            continue

        # Follow resume chain to find the latest run_id (DB is source of truth)
        latest_run_id = _resolve_latest_run_id(original_run_id)

        record = _registry.get(latest_run_id)
        if not record:
            results[agent_name] = {"status": "skipped", "reason": f"run_id '{latest_run_id}' not found in registry"}
            skipped_count += 1
            continue

        if not record.is_terminal:
            results[agent_name] = {"status": "skipped", "reason": f"still running (status={record.status})"}
            skipped_count += 1
            continue

        # Resume this agent — keeps same name, workspace, loads history
        try:
            result_json = await resume_spawn(latest_run_id, follow_up_task)
            result = json.loads(result_json)

            if result.get("status") == "resumed":
                new_run_id = result.get("new_run_id", "")
                # Update session with new run_id so future queries find it
                session.update_agent_run_id(agent_name, new_run_id)
                results[agent_name] = {
                    "status": "resumed",
                    "new_run_id": new_run_id,
                }
                resumed_count += 1
            else:
                results[agent_name] = result
                skipped_count += 1
        except Exception as e:
            logger.error("Failed to resume agent %s: %s", agent_name, e, exc_info=True)
            results[agent_name] = {"status": "error", "reason": str(e)}
            skipped_count += 1

    # Update session status
    session.sprint_status = "running"
    paths = get_runtime_paths(str(_PROJECT_DIR))
    session.save(sessions_dir=paths["workspaces"] / "team_sessions")

    logger.info(
        "Team %s resumed: %d agents restarted, %d skipped",
        session_id, resumed_count, skipped_count,
    )

    return json.dumps({
        "status": "resumed",
        "session_id": session_id,
        "team_name": session.team_name,
        "resumed_agents": resumed_count,
        "skipped_agents": skipped_count,
        "agents": results,
        "message": (
            f"Team '{session.team_name}' resumed with {resumed_count} agents. "
            f"Use get_team_status(session_id='{session_id}') to monitor."
        ),
    })


if __name__ == "__main__":
    mcp.run()
