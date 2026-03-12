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
DEFAULT_TIMEOUT_SECONDS = 600

# Track background tasks and their subprocesses
_background_tasks: dict[str, asyncio.Task[None]] = {}
_background_processes: dict[str, asyncio.subprocess.Process] = {}


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

    return {
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
        logger.warning("Isolated agent %s timed out, sending SIGTERM...", run_id)
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(
                    "Isolated agent %s didn't stop, sending SIGKILL...",
                    run_id,
                )
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass

        duration = time.time() - start_time
        return {
            "status": "timeout",
            "result": "",
            "summary": f"Agent timed out after {timeout_seconds}s",
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
    lifecycle: str = "oneshot",
    registry: Any | None = None,
    display_manager: Any | None = None,
    run_id: str = "",
    env_vars: dict[str, str] | None = None,
    skills: list[str] | None = None,
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
            }
        record = SpawnRecord(
            run_id=run_id,
            role=role or "agent",
            task=task[:200],
            lifecycle=lifecycle,
            status="running",
            original_config=orig_cfg,
        )
        registry.register(record)

    if display_manager:
        display_manager.add_spawn(run_id, role or "agent", task[:80], lifecycle)

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

            status_enum = (
                SpawnStatus.COMPLETED if result["status"] == "completed" else SpawnStatus.ERROR
            )
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
    lifecycle: str = "oneshot",
    registry: Any | None = None,
    display_manager: Any | None = None,
    env_vars: dict[str, str] | None = None,
    skills: list[str] | None = None,
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
            }
        record = SpawnRecord(
            run_id=run_id,
            role=role or "agent",
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
                lifecycle=lifecycle,
                registry=None,
                display_manager=display_manager,
                run_id=run_id,
                env_vars=env_vars,
                skills=skills,
            )
            if registry:
                from fast_agent.spawn.spawn_registry import SpawnStatus

                status_enum = (
                    SpawnStatus.COMPLETED
                    if result.get("status") == "completed"
                    else SpawnStatus.ERROR
                )
                registry.update_status(
                    run_id,
                    status_enum,
                    result=result.get("result", ""),
                    error=result.get("error", ""),
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
