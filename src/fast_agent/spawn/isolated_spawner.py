"""Isolated Agent Spawner — subprocess lifecycle manager + handoff protocol.

Manages the full lifecycle of spawning an isolated FastAgent child process:

1. Write handoff config JSON (Layer 1)
2. Spawn subprocess with isolated_runner
3. Wait with timeout (graceful SIGTERM → SIGKILL)
4. Read result JSON (Layer 3)
5. Format tool_result for orchestrator LLM
6. Cleanup temp files

Supports both blocking and background (fire-and-forget) spawn modes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fast_agent.spawn.config_reader import get_available_servers
from fast_agent.spawn.runtime_paths import get_runtime_paths
from fast_agent.spawn.spawn_events import SpawnEvent

logger = logging.getLogger(__name__)

# Recursion limits
DEFAULT_MAX_DEPTH = 6
DEFAULT_TIMEOUT_SECONDS = 3600

# Track background tasks and their subprocesses
_background_tasks: dict[str, asyncio.Task[None]] = {}
_background_processes: dict[str, asyncio.subprocess.Process] = {}


def _find_latest_history(workspace_dir: str) -> str | None:
    """Find the latest history_child.json from FastAgent sessions.

    FastAgent saves conversation history at:
      {workspace}/.fast-agent/sessions/{session_id}/history_child.json

    Returns the path to the most recent history file, or None.
    """
    sessions_dir = Path(workspace_dir) / ".fast-agent" / "sessions"
    if not sessions_dir.is_dir():
        return None

    best_file: Path | None = None
    best_mtime: float = 0.0

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        history_file = session_dir / "history_child.json"
        if history_file.exists():
            mtime = history_file.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best_file = history_file

    return str(best_file) if best_file else None


def _build_handoff_config(
    run_id: str,
    task: str,
    instruction: str,
    project_dir: str | Path,
    context: str = "",
    servers: list[str] | None = None,
    model: str = "",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    depth: int = 1,
    max_depth: int = DEFAULT_MAX_DEPTH,
    workspace_dir: str | None = None,
    role: str = "agent",
    skills: list[str] | None = None,
    history_file: str | None = None,
) -> dict[str, Any]:
    """Build Layer 1 handoff config for the child agent."""
    servers = servers or []
    available = get_available_servers(project_dir)

    unknown = [s for s in servers if s not in available]
    if unknown:
        raise ValueError(f"Unknown MCP servers: {unknown}. Available: {available}")

    paths = get_runtime_paths(project_dir)
    result_file = str(paths["runs"] / f"run_{run_id}_result.json")
    ws_dir = workspace_dir or str(Path(project_dir).resolve())

    cfg: dict[str, Any] = {
        "run_id": run_id,
        "parent_run_id": run_id,
        "task": task,
        "context": context,
        "instruction": instruction,
        "servers": servers,
        "skills": skills or [],
        "model": model,
        "timeout_seconds": timeout_seconds,
        "depth": depth,
        "max_depth": max_depth,
        "workspace_dir": ws_dir,
        "result_file": result_file,
        "role": role,
    }
    if history_file:
        cfg["history_file"] = history_file
    return cfg


def _format_tool_result(result: dict[str, Any]) -> str:
    """Format the child's result for the orchestrator LLM."""
    status = result.get("status", "unknown")
    task_summary = result.get("summary", "")
    main_result = result.get("result", "(no output)")
    error = result.get("error", "")
    metadata = result.get("metadata", {})
    artifacts = result.get("artifacts", [])
    duration = metadata.get("duration_seconds", "?")

    lines = ["[Subagent Result]", f"Status: {status} ({duration}s)"]

    run_id = result.get("run_id")
    if run_id:
        lines.append(f"Run ID: {run_id}")

    if task_summary:
        lines.append(f"Summary: {task_summary}")

    lines.append("")

    if status == "completed":
        lines.append("Result:")
        lines.append(main_result)
    elif status == "error":
        lines.append(f"Error: {error}")
        if main_result and main_result != "(no output)":
            lines.extend(["", "Partial output:", main_result])
    elif status == "timeout":
        lines.append(f"Error: Agent timed out after {duration}s")
        if main_result and main_result != "(no output)":
            lines.extend(["", "Partial output before timeout:", main_result])

    if artifacts:
        lines.extend(["", "Artifacts created:"])
        for a in artifacts:
            lines.append(f"  - {a}")

    return "\n".join(lines)


async def _run_subprocess(
    run_id: str,
    config: dict[str, Any],
    config_file: str,
    result_file: str,
    timeout_seconds: int,
    start_time: float,
    project_dir: str | Path,
    display_manager: Any | None = None,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute the subprocess and return the result dict.

    When a display_manager is provided, child stderr is read in
    real-time so spawn events can be forwarded to the TUI.
    """
    project_path = Path(project_dir).resolve()
    runner_module = "fast_agent.spawn.isolated_runner"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        runner_module,
        "--config",
        config_file,
        "--project-dir",
        str(project_path),
    ]

    subprocess_env = {
        **os.environ,
        "PYTHONPATH": str(project_path),
    }
    if env_vars:
        subprocess_env.update(env_vars)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(project_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=subprocess_env,
    )

    _background_processes[run_id] = process

    # Store PID in registry for cross-process cleanup
    try:
        from fast_agent.spawn.spawn_registry import SpawnRegistry
        project_dir_str = str(project_path)
        pid_registry_path = Path(project_dir_str) / ".runtime" / "state" / "spawn_registry.json"
        if pid_registry_path.exists():
            pid_reg = SpawnRegistry(registry_file=str(pid_registry_path))
            pid_reg._load()
            if run_id in pid_reg._data:
                pid_reg._data[run_id]["pid"] = process.pid
                pid_reg._save()
    except Exception:
        pass  # Best-effort PID tracking

    stderr_lines: list[str] = []

    async def _read_stderr() -> None:
        assert process.stderr is not None
        while True:
            raw = await process.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            evt = SpawnEvent.from_line(line)
            if evt and display_manager:
                display_manager.handle_event(evt)
            elif not evt:
                stderr_lines.append(line)

    try:
        stderr_task = asyncio.create_task(_read_stderr())
        assert process.stdout is not None
        stdout = await asyncio.wait_for(
            process.stdout.read(),
            timeout=timeout_seconds,
        )
        await asyncio.wait_for(stderr_task, timeout=5)
        await process.wait()
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        agent_name = config.get("agent_name", config.get("role", run_id))
        role = config.get("role", "unknown")
        logger.error(
            "⏰ [TIMEOUT] Agent '%s' (role=%s, run_id=%s) KILLED after %.0fs "
            "(limit=%ds). The agent's work was interrupted mid-execution. "
            "Consider increasing timeout_seconds in the team template.",
            agent_name,
            role,
            run_id,
            duration,
            timeout_seconds,
        )
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(
                    "Agent '%s' (run_id=%s) didn't stop after SIGTERM, sending SIGKILL...",
                    agent_name,
                    run_id,
                )
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass

        return {
            "status": "timeout",
            "result": "",
            "summary": f"Agent '{agent_name}' timed out after {timeout_seconds}s",
            "error": f"Subprocess timed out after {timeout_seconds}s",
            "metadata": {"duration_seconds": round(duration, 1)},
        }

    _background_processes.pop(run_id, None)

    duration = time.time() - start_time

    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        result.setdefault("metadata", {})["duration_seconds"] = round(duration, 1)
    else:
        stdout_text = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
        stderr_text = "\n".join(stderr_lines).strip()

        if process.returncode != 0:
            result = {
                "status": "error",
                "result": stdout_text,
                "summary": "Child process failed",
                "error": (stderr_text or f"Process exited with code {process.returncode}"),
                "metadata": {"duration_seconds": round(duration, 1)},
            }
        else:
            result = {
                "status": "completed",
                "result": stdout_text or "(no output)",
                "summary": "Task completed",
                "metadata": {"duration_seconds": round(duration, 1)},
            }

    return result


async def run_isolated_agent(
    task: str,
    project_dir: str | Path,
    instruction: str = "",
    context: str = "",
    servers: list[str] | None = None,
    model: str = "",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    depth: int = 1,
    max_depth: int = DEFAULT_MAX_DEPTH,
    workspace_dir: str | None = None,
    role: str = "",
    agent_name: str = "",
    team_name: str = "",
    lifecycle: str = "oneshot",
    registry: Any | None = None,
    display_manager: Any | None = None,
    run_id: str = "",
    env_vars: dict[str, str] | None = None,
    skills: list[str] | None = None,
    history_file: str | None = None,
) -> dict[str, Any]:
    """Spawn and run an isolated FastAgent child process (BLOCKING).

    Returns dict with keys: status, result, formatted_result, metadata.
    """
    if depth >= max_depth:
        return {
            "status": "error",
            "result": "",
            "formatted_result": (
                "[Subagent Result]\n"
                "Status: error\n"
                f"Error: Max spawn depth reached ({depth}/{max_depth}). "
                "Cannot spawn more sub-agents."
            ),
            "error": f"Max spawn depth {max_depth} reached",
        }

    if not instruction.strip():
        instruction = (
            "You are a helpful sub-agent. Complete the given task thoroughly and concisely."
        )

    run_id = run_id or str(uuid.uuid4())
    start_time = time.time()
    paths = get_runtime_paths(project_dir)
    paths["runs"].mkdir(parents=True, exist_ok=True)

    config = _build_handoff_config(
        run_id=run_id,
        task=task,
        instruction=instruction,
        project_dir=project_dir,
        context=context,
        servers=servers,
        model=model,
        timeout_seconds=timeout_seconds,
        depth=depth,
        max_depth=max_depth,
        workspace_dir=workspace_dir,
        role=role or "agent",
        skills=skills,
        history_file=history_file,
    )

    config_file = str(paths["runs"] / f"run_{run_id}.json")
    result_file = config["result_file"]

    # Register with registry if provided
    if registry:
        from fast_agent.spawn.spawn_registry import (
            Lifecycle,
            SpawnRecord,
        )

        orig_cfg: dict[str, Any] = {}
        if lifecycle in (
            Lifecycle.PERSISTENT.value,
            Lifecycle.RESUMABLE.value,
        ):
            orig_cfg = {
                "task": task,
                "instruction": instruction,
                "context": context,
                "servers": servers or [],
                "model": model,
                "timeout_seconds": timeout_seconds,
                "role": role or "agent",
                "project_dir": str(Path(project_dir).resolve()),
            }
        record = SpawnRecord(
            run_id=run_id,
            agent_name=agent_name or role or "agent",
            role=role or "agent",
            team_name=team_name,
            task=task[:200],
            lifecycle=lifecycle,
            status="running",
            original_config=orig_cfg,
        )
        registry.register(record)

    if display_manager:
        display_manager.add_spawn(run_id, agent_name or role or "agent", task[:80], lifecycle)

    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(
            "Spawning isolated agent run_id=%s task=%s...",
            run_id,
            task[:80],
        )

        result = await _run_subprocess(
            run_id=run_id,
            config=config,
            config_file=config_file,
            result_file=result_file,
            timeout_seconds=timeout_seconds,
            start_time=start_time,
            project_dir=project_dir,
            display_manager=display_manager,
            env_vars=env_vars,
        )

        result["formatted_result"] = _format_tool_result(result)
        result["run_id"] = run_id
        logger.info(
            "Isolated agent %s finished: status=%s",
            run_id,
            result["status"],
        )

        if registry:
            from fast_agent.spawn.spawn_registry import (
                Lifecycle,
                SpawnStatus,
            )

            if result["status"] != "completed":
                status_enum = SpawnStatus.ERROR
            elif lifecycle == Lifecycle.RESUMABLE.value:
                # Team agents go idle (not completed) — still reachable
                status_enum = SpawnStatus.IDLE
            else:
                status_enum = SpawnStatus.COMPLETED

            registry.update_status(
                run_id,
                status_enum,
                result=result.get("result", ""),
                error=result.get("error", ""),
            )
            if lifecycle == Lifecycle.ONESHOT.value:
                registry.remove(run_id)

        if display_manager:

            async def _delayed_remove() -> None:
                await asyncio.sleep(3)
                display_manager.remove_spawn(run_id)

            asyncio.create_task(_delayed_remove())

        return result

    finally:
        for fp in [config_file, result_file]:
            try:
                if os.path.exists(fp):
                    os.unlink(fp)
            except OSError:
                pass


async def _check_and_resume_on_inbox(
    run_id: str,
    agent_name: str,
    registry: Any | None = None,
    display_manager: Any | None = None,
    env_vars: dict[str, str] | None = None,
) -> None:
    """Check inbox for unread messages and auto-resume agent if any.

    Called after an agent completes its task. If the agent has unread
    messages in its MessageBus inbox, it is automatically resumed with
    those messages as the follow-up task — preserving full conversation
    context.
    """
    if not agent_name or not registry:
        return

    # Guard: skip if agent already has a running instance
    if registry.has_running_resume(agent_name):
        logger.info(
            "📬 %s already has a running instance — skipping auto-resume",
            agent_name,
        )
        return

    # Determine messages dir from env or registry
    workspace_dir = ""
    if env_vars:
        workspace_dir = env_vars.get("TEAM_WORKSPACE", "")

    if not workspace_dir:
        record = registry.get(run_id)
        if record and record.original_config:
            ctx = record.original_config.get("context", "")
            # Try to extract workspace path from context
            for line in ctx.split("\n"):
                if "Shared Workspace" in line or "workspaces/" in line:
                    import re
                    match = re.search(r"(/\S+workspaces/\S+)", line)
                    if match:
                        workspace_dir = match.group(1)
                        break

    if not workspace_dir:
        return

    from pathlib import Path
    # Walk up to find .runtime root
    cur = Path(workspace_dir)
    messages_dir = None
    while cur != cur.parent:
        if cur.name == ".runtime":
            messages_dir = cur / "state" / "messages"
            break
        cur = cur.parent
    if not messages_dir or not messages_dir.exists():
        return

    from fast_agent.spawn.message_bus import MessageBus
    bus = MessageBus(messages_dir=str(messages_dir))
    unread = bus.read_unread(agent_name)

    if not unread:
        return

    logger.info(
        "📬 %s has %d unread message(s) — auto-resuming",
        agent_name, len(unread),
    )

    # Format inbox messages as follow-up task
    inbox_lines = [f"## New Messages ({len(unread)} unread)\n"]
    for msg in unread:
        inbox_lines.append(
            f"### From {msg.from_name} [{msg.message_type}] (id: {msg.message_id})\n"
            f"{msg.content}\n"
        )
    inbox_lines.append(
        "\n## Instructions\n"
        "You have new messages. Follow these steps:\n"
        "1. Read ALL messages above to understand the overall situation\n"
        "2. Prioritize: bugs > tasks > questions > responses\n"
        "3. For each message: take action if needed, then reply using "
        "`reply_to_message(to=sender, message=your_response, original_message_id=id)`\n"
        "4. When all messages are handled, finish your work"
    )
    follow_up = "\n".join(inbox_lines)

    # Mark all as done (agent will process them in the resumed session)
    bus.mark_all_done(agent_name)

    # Resume the agent
    record = registry.get(run_id)
    if not record or not record.original_config:
        return

    cfg = record.original_config
    prev_result = record.result or ""

    # Find previous session history for native FastAgent resume
    workspace_dir = cfg.get("workspace_dir", "")
    history_file = _find_latest_history(workspace_dir) if workspace_dir else None

    if history_file:
        # History file found — agent will get full conversation via
        # load_history_into_agent(). Resume task is just the new messages.
        enriched_context = follow_up
        logger.info(
            "📂 Found previous history for %s: %s", agent_name, history_file,
        )
    else:
        # No history file — fall back to text-based context
        enriched_parts: list[str] = []
        original_task = cfg.get("task", "")
        if original_task:
            enriched_parts.append(f"## Your Original Task\n{original_task}")
        original_context = cfg.get("context", "")
        if original_context:
            enriched_parts.append(f"## Project Context\n{original_context}")
        if prev_result:
            enriched_parts.append(f"## Your Previous Work Summary\n{prev_result}")
        enriched_parts.append(follow_up)
        enriched_context = "\n\n".join(enriched_parts)
        logger.info(
            "⚠️ No history file for %s — using text-based context", agent_name,
        )

    # Determine correct project_dir from original_config (fixes Unknown MCP servers)
    project_dir = cfg.get("project_dir", "")
    if not project_dir and env_vars:
        project_dir = env_vars.get("SPAWN_PROJECT_DIR", "")
    if not project_dir:
        project_dir = "."

    # Re-inject team context if this is a team agent
    team_session_id = (env_vars or {}).get("TEAM_SESSION_ID", "")
    if team_session_id:
        try:
            from fast_agent.spawn.team_spawner import get_team_session
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
        task=follow_up,
        project_dir=project_dir,
        instruction=cfg.get("instruction", ""),
        context=enriched_context,
        servers=cfg.get("servers", []),
        model=cfg.get("model", ""),
        timeout_seconds=cfg.get("timeout_seconds", 600),
        role=cfg.get("role", ""),
        agent_name=agent_name,
        lifecycle="resumable",
        registry=registry,
        display_manager=display_manager,
        env_vars=env_vars,
        history_file=history_file,
    )

    # Track the resume chain
    registry._load()
    if run_id in registry._data:
        restart_count = registry._data[run_id].get("restart_count", 0)
        registry._data[run_id]["restart_count"] = restart_count + 1
        registry._data[run_id].setdefault("metadata", {})["latest_resume_run_id"] = new_run_id
        registry._data[run_id].setdefault("metadata", {})["resume_reason"] = "inbox_messages"
        registry._save()

    logger.info(
        "📬 %s auto-resumed as %s to process %d message(s)",
        agent_name, new_run_id, len(unread),
    )


async def run_isolated_agent_background(
    task: str,
    project_dir: str | Path,
    instruction: str = "",
    context: str = "",
    servers: list[str] | None = None,
    model: str = "",
    timeout_seconds: int = 600,
    depth: int = 1,
    max_depth: int = DEFAULT_MAX_DEPTH,
    workspace_dir: str | None = None,
    role: str = "",
    agent_name: str = "",
    team_name: str = "",
    lifecycle: str = "oneshot",
    registry: Any | None = None,
    display_manager: Any | None = None,
    env_vars: dict[str, str] | None = None,
    skills: list[str] | None = None,
    history_file: str | None = None,
) -> str:
    """Spawn an isolated agent in the BACKGROUND (fire-and-forget).

    Returns the run_id immediately. Use check_spawn_status to poll.
    """
    run_id = str(uuid.uuid4())

    if registry:
        from fast_agent.spawn.spawn_registry import (
            Lifecycle,
            SpawnRecord,
        )

        orig_cfg: dict[str, Any] = {}
        if lifecycle in (
            Lifecycle.PERSISTENT.value,
            Lifecycle.RESUMABLE.value,
        ):
            orig_cfg = {
                "task": task,
                "instruction": instruction,
                "context": context,
                "servers": servers or [],
                "model": model,
                "timeout_seconds": timeout_seconds,
                "role": role or "agent",
                "agent_name": agent_name or role or "agent",
                "team_name": team_name,
                "workspace_dir": workspace_dir or "",
                "env_vars": env_vars or {},
                "project_dir": str(Path(project_dir).resolve()),
            }
        record = SpawnRecord(
            run_id=run_id,
            agent_name=agent_name or role or "agent",
            role=role or "agent",
            team_name=team_name,
            task=task[:200],
            lifecycle=lifecycle,
            status="running",
            original_config=orig_cfg,
        )
        registry.register(record)

    async def _bg_task() -> None:
        try:
            result = await run_isolated_agent(
                task=task,
                project_dir=project_dir,
                instruction=instruction,
                context=context,
                servers=servers,
                model=model,
                timeout_seconds=timeout_seconds,
                depth=depth,
                max_depth=max_depth,
                workspace_dir=workspace_dir,
                role=role,
                agent_name=agent_name,
                lifecycle=lifecycle,
                registry=None,
                display_manager=display_manager,
                run_id=run_id,
                env_vars=env_vars,
                skills=skills,
                history_file=history_file,
            )
            if registry:
                from fast_agent.spawn.spawn_registry import (
                    Lifecycle,
                    SpawnStatus,
                )

                if result.get("status") != "completed":
                    status_enum = SpawnStatus.ERROR
                elif lifecycle == Lifecycle.RESUMABLE.value:
                    # Team agents go idle (not completed) — still reachable
                    status_enum = SpawnStatus.IDLE
                else:
                    status_enum = SpawnStatus.COMPLETED

                registry.update_status(
                    run_id,
                    status_enum,
                    result=result.get("result", ""),
                    error=result.get("error", ""),
                )

            # ── Auto-resume on inbox messages ──
            await _check_and_resume_on_inbox(
                run_id=run_id,
                agent_name=agent_name,
                registry=registry,
                display_manager=display_manager,
                env_vars=env_vars,
            )

        except asyncio.CancelledError:
            if registry:
                from fast_agent.spawn.spawn_registry import SpawnStatus

                registry.update_status(run_id, SpawnStatus.CANCELLED)
            logger.info("Background spawn %s was cancelled", run_id)
        except BaseException as e:
            if registry:
                from fast_agent.spawn.spawn_registry import SpawnStatus

                registry.update_status(run_id, SpawnStatus.ERROR, error=str(e))
            logger.error("Background spawn %s failed: %s", run_id, e)
        finally:
            _background_tasks.pop(run_id, None)

    task_obj = asyncio.create_task(_bg_task())
    _background_tasks[run_id] = task_obj

    return run_id


async def cancel_spawn(run_id: str, registry: Any | None = None) -> bool:
    """Cancel a background spawn by run_id."""
    cancelled = False

    proc = _background_processes.pop(run_id, None)
    if proc and proc.returncode is None:
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            cancelled = True
            logger.info("Subprocess for %s terminated", run_id)
        except ProcessLookupError:
            pass

    task_obj = _background_tasks.pop(run_id, None)
    if task_obj and not task_obj.done():
        task_obj.cancel()
        cancelled = True

    if cancelled and registry:
        from fast_agent.spawn.spawn_registry import SpawnStatus

        registry.update_status(run_id, SpawnStatus.CANCELLED)

    return cancelled


def cleanup_all_spawns() -> None:
    """Terminate all tracked background processes.

    Called during shutdown (atexit, SIGTERM) to prevent orphaned
    agent subprocesses after the parent process exits.
    """
    import signal as _signal

    killed = 0
    for run_id, proc in list(_background_processes.items()):
        if proc.returncode is None:  # Still running
            try:
                proc.terminate()
                killed += 1
                logger.info("Terminated spawned agent %s (pid=%s)", run_id, proc.pid)
            except ProcessLookupError:
                pass

    # Cancel asyncio tasks
    for run_id, task_obj in list(_background_tasks.items()):
        if not task_obj.done():
            task_obj.cancel()

    if killed:
        # Give processes a moment to exit gracefully
        import time as _time
        _time.sleep(1)

        # Force-kill any survivors
        for run_id, proc in list(_background_processes.items()):
            if proc.returncode is None:
                try:
                    proc.kill()
                    logger.warning("Force-killed spawned agent %s (pid=%s)", run_id, proc.pid)
                except ProcessLookupError:
                    pass

    _background_processes.clear()
    _background_tasks.clear()
    logger.info("Spawn cleanup complete (%d processes terminated)", killed)
