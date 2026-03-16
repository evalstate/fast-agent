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
        str(_PROJECT_DIR / ".fast-agent" / "skills"),
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
        agent_name=role,
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
    cancelled = await cancel_spawn(run_id, registry=_registry)
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
        team_name=cfg.get("team_name", record.team_name),
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

    # Try team-level removal — name might match a team_name
    team_members = _registry.find_by_team(name)
    if team_members:
        # Remove all agent cards for team members
        cards_removed = 0
        for member in team_members:
            if remove_agent_card(member.agent_name, agent_cards_dir=str(agent_cards_dir)):
                cards_removed += 1

        # Remove all registry entries for the team
        registry_count = _registry.remove_team(name)

        return json.dumps(
            {
                "status": "success",
                "message": (
                    f"Team '{name}' removed: {registry_count} agents "
                    f"({cards_removed} cards removed)."
                ),
                "removed_agents": [m.agent_name for m in team_members],
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
        mode: "blocking" (wait for orchestrator to complete) or
              "background" (return immediately). Default: background.
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
        )

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
