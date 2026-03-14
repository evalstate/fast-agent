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
12. send_message_to_agent — send message between agents
13. read_agent_inbox — read an agent's inbox
14. wait_for_agent — block until agent completes
15. restart_spawn — re-run a persistent/resumable spawn
16. resume_spawn — continue a resumable spawn with follow-up
17. spawn_team_tool — spawn a full team from a template
18. get_team_status — get team session status
19. get_team_result — get consolidated team result
20. list_team_templates_tool — list available templates
21. trigger_retrospective — run team retrospective

All paths are resolved from environment variables set by the host
process, not from module-level globals.
"""

from __future__ import annotations

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
    cancel_spawn,
    run_isolated_agent,
    run_isolated_agent_background,
)
from fast_agent.spawn.message_bus import MessageBus

from fast_agent.spawn.spawn_display import get_display_manager
from fast_agent.spawn.spawn_registry import SpawnRegistry
from fast_agent.spawn.team_spawner import (
    get_team_session,
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
_SKILLS_DIR = Path(
    os.environ.get(
        "SPAWN_SKILLS_DIR",
        str(_PROJECT_DIR / "skills"),
    )
)
_SERVERS_LIST = ", ".join(get_available_servers(project_dir=str(_PROJECT_DIR)))
_registry = SpawnRegistry(
    registry_file=str(_PROJECT_DIR / ".runtime" / "state" / "spawn_registry.json"),
)
_bus = MessageBus(messages_dir=str(_PROJECT_DIR / ".runtime" / "state" / "messages"))

_display = get_display_manager()

# ── Wire file-based event forwarding ──
# The MCP server runs in a subprocess — in-memory callbacks cannot
# reach the main backend process. Instead, write events as JSON lines
# to a shared file that the main process can tail-follow.
_SPAWN_EVENTS_FILE = _PROJECT_DIR / ".runtime" / "state" / "spawn_events.jsonl"


def _write_spawn_event_to_file(event: Any) -> None:
    """Append a SpawnEvent as JSON line to the shared events file."""
    try:
        import time as _time

        line = json.dumps(
            {
                "timestamp": _time.time(),
                "role": getattr(event, "role", ""),
                "event_type": getattr(event, "event", ""),
                "run_id": getattr(event, "run_id", ""),
                "data": getattr(event, "data", {}),
            }
        )
        _SPAWN_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SPAWN_EVENTS_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass  # Never crash the MCP server for a display event


_display.set_event_callback(_write_spawn_event_to_file)



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

    run_id = await run_isolated_agent_background(
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
    """Check the status of a background-spawned agent.

    Args:
        run_id: The run_id from spawn_and_run_background.
    """
    record = _registry.get_latest(run_id)
    if not record:
        return json.dumps({"error": f"No spawn found with run_id '{run_id}'"})

    return json.dumps(
        {
            "run_id": record.run_id,
            "role": record.role,
            "status": record.status,
            "lifecycle": record.lifecycle,
            "task": record.task,
            "duration_seconds": record.duration_seconds,
            "error": record.error or None,
        }
    )


@mcp.tool()
def get_spawn_result(run_id: str) -> str:
    """Get the result of a completed background spawn.

    Args:
        run_id: The run_id of the spawn.
    """
    record = _registry.get(run_id)
    if not record:
        return json.dumps({"error": f"No spawn found with run_id '{run_id}'"})
    if not record.is_terminal:
        return json.dumps(
            {
                "status": record.status,
                "message": "Spawn still running. Use check_spawn_status.",
            }
        )
    result = json.dumps(
        {
            "run_id": record.run_id,
            "status": record.status,
            "result": record.result,
            "error": record.error or None,
            "duration_seconds": record.duration_seconds,
        }
    )
    if record.lifecycle == "oneshot":
        _registry.remove(run_id)
    return result


@mcp.tool()
def list_active_spawns() -> str:
    """List all tracked spawns and their status."""
    summaries = _registry.to_summary()
    return json.dumps({"count": len(summaries), "spawns": summaries})


@mcp.tool()
async def cancel_spawn_tool(run_id: str) -> str:
    """Cancel a running background spawn.

    Args:
        run_id: The run_id to cancel.
    """
    cancelled = await cancel_spawn(run_id, registry=_registry)
    if cancelled:
        return json.dumps({"status": "cancelled", "run_id": run_id})
    return json.dumps({"error": (f"Could not cancel '{run_id}' — not running or not found.")})


@mcp.tool()
def cleanup_spawn(run_id: str) -> str:
    """Remove a spawn record from the registry.

    Args:
        run_id: The run_id to remove.
    """
    removed = _registry.remove(run_id)
    if removed:
        return json.dumps({"status": "removed", "run_id": run_id})
    return json.dumps({"error": f"Spawn '{run_id}' not found."})


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
        lifecycle=record.lifecycle,
        registry=_registry,
        display_manager=_display,
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

    The new agent receives the previous result as context.

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

    prev_context = cfg.get("context", "")
    prev_result = record.result or ""
    enriched_context = ""
    if prev_context:
        enriched_context += f"## Original Context\n{prev_context}\n\n"
    if prev_result:
        enriched_context += f"## Previous Agent Result\n{prev_result}\n\n"
    enriched_context += f"## Follow-Up Task\n{follow_up_task}"

    new_run_id = await run_isolated_agent_background(
        task=follow_up_task,
        project_dir=str(_PROJECT_DIR),
        instruction=cfg.get("instruction", ""),
        context=enriched_context,
        servers=cfg.get("servers", []),
        model=cfg.get("model", ""),
        timeout_seconds=cfg.get("timeout_seconds", 600),
        role=cfg.get("role", record.role),
        agent_name=cfg.get("agent_name", record.agent_name),
        workspace_dir=cfg.get("workspace_dir") or None,
        lifecycle="resumable",
        registry=_registry,
        display_manager=_display,
        env_vars=cfg.get("env_vars") or None,
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
            "message": (
                f"Resumable spawn continued. Use check_spawn_status(run_id='{new_run_id}') to poll."
            ),
        }
    )


# ───────────────────────────────────────────────────────────
# Inter-Agent Communication
# ───────────────────────────────────────────────────────────


@mcp.tool()
def send_message_to_agent(
    to: str,
    message: str,
    message_type: str = "task",
    priority: str = "normal",
) -> str:
    """Send a message to another agent's inbox by name.

    If the target agent is idle, it will be automatically woken up
    to process the message.

    Args:
        to: Target agent name (e.g. "Minh - Dev").
        message: Message content.
        message_type: "task" | "question" | "response" | "notification"
        priority: "low" | "normal" | "high" | "urgent"
    """
    my_name = os.environ.get("TEAM_MY_NAME", os.environ.get("TEAM_MY_ROLE", "orchestrator"))
    msg = _bus.send(
        from_name=my_name,
        to_name=to,
        content=message,
        message_type=message_type,
        priority=priority,
    )

    # Auto-wake idle agent
    record = _registry.find_by_name(to)
    if (
        record
        and record.status == "idle"
        and not _registry.has_running_resume(to)
    ):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    _check_and_resume_on_inbox(
                        run_id=record.run_id,
                        agent_name=to,
                        registry=_registry,
                        display_manager=_display,
                        env_vars=record.original_config.get("env_vars") if record.original_config else None,
                    )
                )
        except RuntimeError:
            pass  # No event loop — skip auto-wake

    return json.dumps(
        {
            "status": "sent",
            "message_id": msg.message_id,
            "from": my_name,
            "to": to,
        }
    )


@mcp.tool()
def read_agent_inbox(agent_name: str) -> str:
    """Read all messages in an agent's inbox.

    Args:
        agent_name: Agent name whose inbox to read.
    """
    return _bus.read_inbox_formatted(agent_name)


@mcp.tool()
def wait_for_agent(agent_name: str, timeout_seconds: int = 300) -> str:
    """Wait (block) until a named agent completes.

    Polls the spawn registry by agent_name — unique lookup, no ambiguity.

    Args:
        agent_name: Name of the agent to wait for (e.g. "Minh - Dev").
        timeout_seconds: Max wait time (default 300s).
    """
    import time as _time

    poll_interval = 2.0
    start = _time.time()

    while _time.time() - start < timeout_seconds:
        record = _registry.find_by_name(agent_name)
        if (
            record
            and record.status in ("completed", "error", "timeout", "cancelled")
            and not _registry.has_running_resume(agent_name)
        ):
            return json.dumps(
                {
                    "status": record.status,
                    "agent_name": record.agent_name,
                    "role": record.role,
                    "run_id": record.run_id,
                    "result_summary": record.result[:500] if record.result else "",
                    "error": record.error or "",
                }
            )
        _time.sleep(poll_interval)

    return json.dumps(
        {
            "status": "timeout",
            "message": f"Agent '{agent_name}' did not complete within {timeout_seconds}s",
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
def list_spawned_agents() -> str:
    """List all dynamically spawned persistent agents."""
    # Resolve agent_cards dir (same logic as spawn_agent / remove_spawned_agent)
    agent_cards_dir = get_runtime_paths(str(_PROJECT_DIR))["agent_cards"]
    legacy_cards = _PROJECT_DIR / ".fast-agent" / "agent_cards"
    if legacy_cards.exists():
        agent_cards_dir = legacy_cards

    cards = list_agent_cards(agent_cards_dir=str(agent_cards_dir))
    if not cards:
        return json.dumps(
            {
                "status": "success",
                "count": 0,
                "agents": [],
                "message": "No spawned agents found.",
            }
        )

    enriched = []
    for card in cards:
        content = get_agent_card_content(card["name"], agent_cards_dir=str(agent_cards_dir))
        enriched.append(
            {
                "name": card["name"],
                "file": card["file"],
                "preview": content[:200] if content else "",
            }
        )
    return json.dumps(
        {
            "status": "success",
            "count": len(enriched),
            "agents": enriched,
        }
    )


@mcp.tool()
def remove_spawned_agent(name: str) -> str:
    """Remove a persistent agent by name.

    Removes the agent card file AND the spawn registry entry,
    so the agent disappears from both the runtime and the UI.

    Args:
        name: Name of the agent to remove.
    """
    # Resolve agent_cards dir (same logic as spawn_agent)
    agent_cards_dir = get_runtime_paths(str(_PROJECT_DIR))["agent_cards"]
    legacy_cards = _PROJECT_DIR / ".fast-agent" / "agent_cards"
    if legacy_cards.exists():
        agent_cards_dir = legacy_cards

    card_removed = remove_agent_card(name, agent_cards_dir=str(agent_cards_dir))

    # Also remove from spawn registry (by agent_name)
    registry_removed = False
    record = _registry.find_by_name(name)
    if record:
        registry_removed = _registry.remove(record.run_id)

    if card_removed or registry_removed:
        parts = []
        if card_removed:
            parts.append("card removed")
        if registry_removed:
            parts.append("registry cleaned")
        return json.dumps(
            {
                "status": "success",
                "message": f"Agent '{name}' removed ({', '.join(parts)}).",
            }
        )
    return json.dumps(
        {
            "status": "error",
            "message": f"Agent '{name}' not found.",
        }
    )


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
    mode: str = "blocking",
) -> str:
    """Spawn a full team of agents from a template.

    Each agent receives the team roster (who's on the team) and
    coordinates via skills + MCP tools. No hardcoded flow control.

    Args:
        template: Team template name (e.g. "agile-team").
        project_brief: Description for the team.
        mode: "blocking" (wait for all to complete) or
              "background" (return agent IDs immediately).
    """
    try:
        session = await _spawn_team(
            template_name=template,
            project_brief=project_brief,
            registry=_registry,
            display_manager=_display,
            project_dir=str(_PROJECT_DIR),
            mode=mode,
        )

        agents_info = {
            role: {
                "run_id": info.get("run_id", ""),
                "role": role,
                "status": info.get("status", "unknown"),
            }
            for role, info in session.agents.items()
        }

        status = "spawning_background" if mode == "background" else "completed"

        result: dict[str, Any] = {
            "status": status,
            "session_id": session.session_id,
            "template": session.template.get("name", template),
            "workspace": str(session.workspace),
            "agents": agents_info,
        }

        if mode == "background":
            result["message"] = (
                "Team spawning in background. Use "
                "check_spawn_status(run_id) for individual "
                "agents, or get_team_status(session_id="
                f"'{session.session_id}') for team overview."
            )
        else:
            ws_summary = get_workspace_summary(session.workspace)
            result["workspace_contents"] = ws_summary.get("directories", {})
            result["message"] = (
                "Team completed. Use resume_spawn(run_id, "
                "follow_up_task) to continue any agent, or "
                "get_team_result(session_id="
                f"'{session.session_id}') for results."
            )

        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Team spawn failed: {e}"})


@mcp.tool()
def get_team_status(session_id: str) -> str:
    """Get the status of a team session.

    Returns agent roster with run_ids and statuses.

    Args:
        session_id: The session_id from spawn_team_tool.
    """
    session = get_team_session(session_id)
    if not session:
        return json.dumps({"error": f"Team session '{session_id}' not found."})

    agents = session.get_roster()
    total = len(agents)
    completed = sum(1 for a in session.agents.values() if a.get("status") == "completed")
    errored = sum(1 for a in session.agents.values() if a.get("status") == "error")

    return json.dumps(
        {
            "session_id": session_id,
            "template": session.template.get("name", "unknown"),
            "workspace": str(session.workspace),
            "sprint_status": session.sprint_status,
            "progress": f"{completed}/{total} agents completed, {errored} errors",
            "agents": agents,
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


if __name__ == "__main__":
    mcp.run()
