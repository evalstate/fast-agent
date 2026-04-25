"""Team Spawner — spawn and manage teams of agents from a template.

Provides **only primitives**: template loading, workspace creation,
agent spawning, and roster management. All workflow intelligence
lives in **skills** assigned to agents.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from fast_agent.spawn.isolated_spawner import (
    run_isolated_agent,
    run_isolated_agent_background,
)
from fast_agent.spawn.spawn_hooks import SpawnLifecycleHooks
from fast_agent.spawn.runtime_paths import get_runtime_paths
from fast_agent.spawn.workspace_manager import (
    create_workspace,
)

if TYPE_CHECKING:
    from fast_agent.spawn.spawn_registry import SpawnRegistry

logger = logging.getLogger(__name__)

# Global team sessions store
_team_sessions: dict[str, TeamSession] = {}

# ───────────────────────────────────────────────────────────
# Random Agent Naming
# ───────────────────────────────────────────────────────────

# Pool of gender-neutral first names (ASCII only, internationally readable).
_AGENT_NAME_POOL = [
    "Alex", "Avery", "Bailey", "Blake", "Cameron", "Casey", "Charlie",
    "Dakota", "Drew", "Eden", "Elliot", "Emerson", "Finley", "Frankie",
    "Hayden", "Jamie", "Jesse", "Jordan", "Kai", "Kendall", "Logan",
    "Mason", "Morgan", "Parker", "Peyton", "Phoenix", "Quinn", "Reagan",
    "Reese", "Riley", "River", "Robin", "Rowan", "Ryan", "Sage", "Sam",
    "Sasha", "Sawyer", "Skyler", "Sydney", "Taylor", "Toby", "Tyler",
    "Wren", "Adrian", "Ash", "Bennett", "Carson", "Devon", "Emery",
]


def _generate_unique_agent_name(
    role_display: str,
    registry: "SpawnRegistry",
) -> str:
    """Generate a unique agent name: '{RandomName} [{Role}]'.

    Checks active agents in registry to avoid name collision.
    Falls back to name+number if pool is exhausted.
    """
    import random

    # Collect all active agent names from registry
    try:
        active_names = {r.agent_name for r in registry.list_active()}
    except Exception:
        active_names = set()

    # Also check names already used in current in-memory team sessions
    for session in _team_sessions.values():
        for agent_info in session.agents.values():
            active_names.add(agent_info.get("agent_name", ""))

    available = [
        n for n in _AGENT_NAME_POOL
        if f"{n} [{role_display}]" not in active_names
    ]
    if available:
        name = random.choice(available)
        return f"{name} [{role_display}]"

    # Fallback: append number
    for i in range(1, 100):
        candidate = f"{random.choice(_AGENT_NAME_POOL)}{i} [{role_display}]"
        if candidate not in active_names:
            return candidate

    # Ultimate fallback
    return f"Agent{uuid.uuid4().hex[:4]} [{role_display}]"


# ───────────────────────────────────────────────────────────
# Template I/O
# ───────────────────────────────────────────────────────────


def load_team_template(
    template_name: str,
    template_dir: str | Path,
) -> dict[str, Any]:
    """Load a team template YAML by name.

    Raises:
        ValueError: If template not found.
    """
    tdir = Path(template_dir)
    candidates = [
        tdir / f"{template_name}.yaml",
        tdir / f"{template_name}_team.yaml",
        tdir / f"{template_name.replace('-', '_')}.yaml",
        tdir / f"{template_name.replace('-', '_')}_team.yaml",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:  # noqa: SIM115
                return yaml.safe_load(f)

    available = [f.stem for f in tdir.glob("*.yaml")]
    raise ValueError(f"Team template '{template_name}' not found. Available: {available}")


def list_team_templates(
    template_dir: str | Path,
) -> list[dict[str, Any]]:
    """List all available team templates."""
    templates: list[dict[str, Any]] = []
    tdir = Path(template_dir)
    for path in tdir.glob("*.yaml"):
        try:
            with open(path, encoding="utf-8") as f:  # noqa: SIM115
                data = yaml.safe_load(f)
            templates.append(
                {
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "roles": list(data.get("roles", {}).keys()),
                }
            )
        except Exception as e:
            logger.warning("Failed to load template %s: %s", path, e)
    return templates


# ───────────────────────────────────────────────────────────
# TeamSession
# ───────────────────────────────────────────────────────────


class TeamSession:
    """Tracks the execution state of a team."""

    def __init__(
        self,
        session_id: str,
        template: dict[str, Any],
        workspace: Path,
        parent_session_id: str = "",
        team_name: str = "",
        conversation_id: str = "",
    ) -> None:
        self.session_id = session_id
        self.template = template
        self.workspace = workspace
        self.parent_session_id = parent_session_id
        self.team_name = team_name or template.get("name", "team")
        self.conversation_id = conversation_id  # originating chat session
        self.agents: dict[str, dict[str, Any]] = {}  # agent_name → {run_id, role, status, ...}
        self.sprint_status = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "template": self.template,
            "workspace": str(self.workspace),
            "parent_session_id": self.parent_session_id,
            "team_name": self.team_name,
            "conversation_id": self.conversation_id,
            "agents": self.agents,
            "sprint_status": self.sprint_status,
        }

    # ── Roster ───────────────────────────────────────────

    def get_roster(self) -> dict[str, dict[str, Any]]:
        """Return team roster: agent_name → {run_id, role, status}."""
        return {
            name: {
                "run_id": info.get("run_id", ""),
                "agent_name": name,
                "role": info.get("role", ""),
                "status": info.get("status", "unknown"),
            }
            for name, info in self.agents.items()
        }

    def write_roster(self) -> Path:
        """Write team_roster.json to workspace."""
        path = self.workspace / "team_roster.json"
        path.write_text(json.dumps(self.get_roster(), indent=2))
        return path

    def update_agent_run_id(self, agent_name: str, new_run_id: str) -> None:
        """Update run_id for an agent after resume.

        Keeps session in sync with the latest run_id from registry (DB).
        """
        if agent_name in self.agents:
            self.agents[agent_name]["run_id"] = new_run_id
            self.agents[agent_name]["status"] = "running"

    def roster_context(self, for_role: str = "") -> str:
        """Build roster context string for agent injection.

        If for_role is the orchestrator, include spawn_team_members instructions.
        """
        orchestrator_role = self.template.get("orchestrator", "")
        is_orchestrator = for_role == orchestrator_role

        lines = ["## Your Team"]
        active_agents = []
        available_roles = []

        for agent_name, info in self.agents.items():
            run_id = info.get("run_id", "?")
            role = info.get("role", "")
            status = info.get("status", "unknown")

            if status == "available":
                available_roles.append((agent_name, role))
                lines.append(f"- **{agent_name}** (role: {role}) — ⏸️ Available (not yet spawned)")
            else:
                active_agents.append((agent_name, role))
                lines.append(f"- **{agent_name}** (role: {role}, run_id: {run_id}) — ▶️ {status}")

        lines.append("")
        lines.append("## Communication Tools")
        lines.append("Use `send_email(to=\"Agent Name\", body=\"...\", subject=\"...\")` to contact teammates.")
        lines.append("Emails from teammates are **auto-delivered** to your context — no need to poll.")
        lines.append("Team member results are **auto-delivered** to your inbox when ALL members finish. No polling needed.")
        lines.append("Use `create_meeting(participants=\"Agent Name 1, Agent Name 2\", agenda=\"...\")` for real-time discussions (use agent names, not role keys).")
        lines.append("")
        lines.append("## Waiting for Dependencies")
        lines.append("If you need output from a teammate, send a request email or create a meeting:")
        lines.append("`send_email(to=\"Agent Name\", body=\"Please send me [deliverable] when ready\", subject=\"[WAITING] ...\")`")
        lines.append("Then continue other work. Results will be auto-delivered to your inbox when all members complete.")

        if is_orchestrator and available_roles:
            lines.append("")
            lines.append("## Team Management (Orchestrator Only)")
            roles_str = ",".join(r for _, r in available_roles)
            lines.append(f"Use `spawn_team_members(roles=\"{roles_str}\", team_session_id=\"{self.session_id}\")` to bring in team members.")
            lines.append("Use `resume_spawn(run_id=\"...\", follow_up_task=\"...\")` to ask a completed agent to revise their work.")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────

    def save(self, sessions_dir: str | Path | None = None) -> Path:
        """Persist session state to JSON file."""
        if sessions_dir:
            sdir = Path(sessions_dir)
        else:
            sdir = self.workspace.parent / "team_sessions"
        sdir.mkdir(parents=True, exist_ok=True)
        path = sdir / f"{self.session_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(
        cls,
        session_id: str,
        sessions_dir: str | Path,
    ) -> TeamSession | None:
        """Load a previously saved team session."""
        path = Path(sessions_dir) / f"{session_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        # Support both old format (template as string) and new (template as dict)
        template_data = data.get("template", {})
        if isinstance(template_data, str):
            template_data = {"name": template_data}
        session = cls(
            session_id=data["session_id"],
            template=template_data,
            workspace=Path(data["workspace"]),
            parent_session_id=data.get("parent_session_id", ""),
            team_name=data.get("team_name", ""),
            conversation_id=data.get("conversation_id", ""),
        )
        session.agents = data.get("agents", {})
        session.sprint_status = data.get("sprint_status", "unknown")
        return session


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────


def _build_team_env(
    workspace: Path,
    roles: dict[str, Any],
    my_role: str,
    my_name: str = "",
    session_id: str = "",
    session: "TeamSession | None" = None,
) -> dict[str, str]:
    """Build environment variables for team agent."""
    env = {
        "TEAM_WORKSPACE": str(workspace),
        "TEAM_MY_ROLE": my_role,
    }
    if my_name:
        env["TEAM_MY_NAME"] = my_name
    if session_id:
        env["TEAM_SESSION_ID"] = session_id
        # Session-scoped messages directory for isolation between sessions
        runtime_dir = workspace
        cur = workspace
        while cur != cur.parent:
            if cur.name == ".runtime":
                runtime_dir = cur
                break
            cur = cur.parent
        messages_dir = runtime_dir / "state" / "messages" / session_id
        messages_dir.mkdir(parents=True, exist_ok=True)
        env["TEAM_MESSAGES_DIR"] = str(messages_dir)

    # Provide team config with resolved agent_name → role mapping
    team_config: dict[str, Any] = {}
    if session:
        # Use session's resolved names (random unique names)
        for agent_name, agent_info in session.agents.items():
            rname = agent_info.get("role", "")
            rcfg = roles.get(rname, {}) if isinstance(roles.get(rname), dict) else {}
            team_config[rname] = {
                "agent_name": agent_name,
                "instruction": rcfg.get("instruction", ""),
                "servers": rcfg.get("servers", []),
            }
    else:
        for rname, rcfg in roles.items():
            if isinstance(rcfg, dict):
                team_config[rname] = {
                    "agent_name": rcfg.get("agent_name", f"Agent - {rname.upper()}"),
                    "instruction": rcfg.get("instruction", ""),
                    "servers": rcfg.get("servers", []),
                }
    env["TEAM_ROLES_CONFIG"] = json.dumps(team_config)
    return env


def _resolve_role_skills(
    role_config: dict[str, Any],
    skills_dir: str | Path,
) -> list[str]:
    """Validate and return skill names from role config.

    Returns a list of skill names (strings) that exist in the skills directory.
    The child process will resolve these names to SkillManifest objects using
    the shared get_skills() helper from config_reader.
    """
    skill_names = role_config.get("skills", [])
    if not skill_names:
        return []

    sdir = Path(skills_dir)
    valid: list[str] = []
    for name in skill_names:
        if (sdir / name / "SKILL.md").exists():
            valid.append(name)
        else:
            logger.warning("Skill '%s' not found at %s", name, sdir / name)

    return valid


# ───────────────────────────────────────────────────────────
# Main: spawn_team
# ───────────────────────────────────────────────────────────


async def spawn_team(
    template_name: str,
    project_brief: str,
    registry: SpawnRegistry,
    project_dir: str | Path,
    team_name: str,
    template_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
    workspace_root: Path | None = None,
    display_manager: Any | None = None,
    parent_session_id: str = "",
    mode: str = "background",
    spawn_lifecycle_hooks: SpawnLifecycleHooks | None = None,
    conversation_id: str = "",
) -> TeamSession:
    """Spawn a team — only the orchestrator starts immediately.

    Other roles are registered as 'available' and can be spawned
    on-demand by the orchestrator using spawn_team_members_for_session().

    Args:
        template_name: Name of the team template.
        project_brief: Project description for agents.
        registry: SpawnRegistry for tracking agents.
        project_dir: Root directory of host application.
        team_name: Unique name for this team instance.
        template_dir: Directory with team template YAMLs.
        skills_dir: Directory with skill subdirectories.
        workspace_root: Override workspace root directory.
        display_manager: TUI display manager instance.
        parent_session_id: Optional fast-agent session ID.
        mode: "blocking" (wait for orchestrator) or
              "background" (return immediately).

    Returns:
        A TeamSession with orchestrator running, others available.
    """
    pdir = Path(project_dir).resolve()
    tdir = Path(template_dir) if template_dir else pdir / "team_templates"
    # Use SPAWN_SKILLS_DIR env var (same convention as agent_spawner_server)
    sdir = Path(skills_dir) if skills_dir else Path(
        os.environ.get("SPAWN_SKILLS_DIR", str(pdir / ".fast-agent" / "skills"))
    )
    paths = get_runtime_paths(project_dir)

    template = load_team_template(template_name, tdir)
    session_id = str(uuid.uuid4())[:8]

    # Clean up previous session artifacts
    template_prefix = team_name.lower().replace(" ", "_")[:50]
    workspaces_base = workspace_root or paths["workspaces"]
    if Path(workspaces_base).exists():
        import shutil
        for old_ws in Path(workspaces_base).iterdir():
            if old_ws.is_dir() and old_ws.name.startswith(template_prefix):
                logger.info("Cleaning old workspace: %s", old_ws)
                shutil.rmtree(old_ws, ignore_errors=True)
    child_configs = paths["tmp"] / "child_configs"
    if child_configs.exists():
        import shutil
        shutil.rmtree(child_configs, ignore_errors=True)

    project_name = f"{template.get('name', template_name)}_{session_id}"
    workspace = create_workspace(
        project_name,
        workspaces_dir=paths["workspaces"],
        root=workspace_root,
    )

    # Auto-capture conversation_id from shared_state if not provided
    if not conversation_id:
        try:
            import services.shared_state as _sstate
            conversation_id = _sstate.current_conversation_id or ""
        except ImportError:
            pass

    session = TeamSession(
        session_id=session_id,
        template=template,
        workspace=workspace,
        parent_session_id=parent_session_id,
        team_name=team_name,
        conversation_id=conversation_id,
    )
    _team_sessions[session_id] = session

    roles = template.get("roles", {})
    orchestrator_role = template.get("orchestrator", "")

    # If no orchestrator specified, use the first role
    if not orchestrator_role and roles:
        orchestrator_role = next(iter(roles))

    # Pre-register all agents: orchestrator as 'pending', others as 'available'
    for role_name, role_config in roles.items():
        role_display = role_config.get("role_display", role_name.upper())
        agent_name = _generate_unique_agent_name(role_display, registry)
        run_id = f"team_{session_id}_{role_name}_{uuid.uuid4().hex[:6]}"
        status = "pending" if role_name == orchestrator_role else "available"
        session.agents[agent_name] = {
            "run_id": run_id,
            "role": role_name,
            "agent_name": agent_name,
            "role_display": role_display,
            "instruction": role_config.get("instruction", ""),
            "status": status,
        }

    session.write_roster()

    # Spawn only the orchestrator
    if orchestrator_role not in roles:
        raise ValueError(
            f"Orchestrator role '{orchestrator_role}' not found in template. "
            f"Available: {list(roles.keys())}"
        )

    orchestrator_config = roles[orchestrator_role]
    roster_ctx = session.roster_context(for_role=orchestrator_role)

    run_id = await _spawn_single_agent(
        session=session,
        role_name=orchestrator_role,
        role_config=orchestrator_config,
        workspace=workspace,
        roster_ctx=roster_ctx,
        project_brief=project_brief,
        registry=registry,
        project_dir=pdir,
        skills_dir=sdir,
        display_manager=display_manager,
        team_name=team_name,
        spawn_lifecycle_hooks=spawn_lifecycle_hooks,
    )

    logger.info(
        "Team %s: orchestrator '%s' spawned → run_id=%s. "
        "Other roles available: %s",
        session_id,
        orchestrator_role,
        run_id,
        [r for r in roles if r != orchestrator_role],
    )

    session.sprint_status = "orchestrator_running"
    session.save(sessions_dir=paths["workspaces"] / "team_sessions")
    return session


async def _spawn_single_agent(
    session: TeamSession,
    role_name: str,
    role_config: dict[str, Any],
    workspace: Path,
    roster_ctx: str,
    project_brief: str,
    registry: SpawnRegistry,
    project_dir: str | Path,
    skills_dir: str | Path,
    team_name: str,
    display_manager: Any | None = None,
    first_task: str = "",
    spawn_lifecycle_hooks: SpawnLifecycleHooks | None = None,
) -> str:
    """Spawn a single team agent. Returns run_id."""
    # Use the pre-generated unique name from the session
    # (assigned during spawn_team pre-registration)
    agent_name = None
    for name, info in session.agents.items():
        if info.get("role") == role_name:
            agent_name = name
            break
    if not agent_name:
        role_display = role_config.get("role_display", role_name.upper())
        agent_name = _generate_unique_agent_name(role_display, registry)

    task = role_config.get("task", project_brief)

    # If first_task provided, prepend it as agents' #1 priority
    if first_task:
        task = (
            f"⚠️ FIRST TASK — Execute IMMEDIATELY before anything else:\n"
            f"{first_task}\n"
            f"Do NOT explore the workspace before completing this task.\n\n"
            f"--- MAIN TASK (after first task is done) ---\n"
            + task
        )
    else:
        # No first_task: just use the main task directly.
        # Email notifications are pushed via RTAC (InboxWatcherHook) automatically,
        # so agents don't need to poll their inbox on startup.
        task = f"--- MAIN TASK ---\n" + task
    instruction = role_config.get("instruction", f"You are {agent_name}.")
    instruction = instruction.replace("{agent_name}", agent_name)
    servers = list(role_config.get("servers", ["filesystem"]))
    model = role_config.get("model", "")

    # Ensure meeting_room and email are available for communication
    if "meeting_room" not in servers:
        servers.append("meeting_room")
    if "email" not in servers:
        servers.append("email")

    context_parts = [
        f"## Project Brief\n{project_brief}",
        f"\n## Shared Workspace\nPath: {workspace}",
        "Use the filesystem MCP server to read/write files.",
        f"\n{roster_ctx}",
    ]
    context = "\n\n".join(context_parts)

    roles = session.template.get("roles", {})
    team_env = _build_team_env(
        workspace, roles, role_name, my_name=agent_name,
        session_id=session.session_id, session=session,
    )

    logger.info(
        "Team %s: launching %s [%s] (background)",
        session.session_id,
        agent_name,
        role_name,
    )

    session.agents[agent_name]["status"] = "running"

    run_id = await run_isolated_agent_background(
        task=task,
        project_dir=str(project_dir),
        instruction=instruction,
        context=context,
        servers=servers,
        model=model,
        timeout_seconds=role_config.get("timeout_seconds", 0),  # 0 = no timeout for resumable agents
        role=role_name,
        agent_name=agent_name,
        team_name=team_name,
        lifecycle="resumable",
        registry=registry,
        display_manager=display_manager,
        skills=_resolve_role_skills(role_config, skills_dir),
        env_vars=team_env,
        workspace_dir=str(workspace),
        spawn_lifecycle_hooks=spawn_lifecycle_hooks,
        server_overrides=role_config.get("server_overrides"),
        session_id=session.session_id,
    )

    session.agents[agent_name]["run_id"] = run_id
    session.write_roster()
    return run_id


async def spawn_team_members_for_session(
    session_id: str,
    roles: list[str],
    registry: SpawnRegistry,
    display_manager: Any | None = None,
    project_dir: str | Path = "",
    first_task: str = "",
    spawn_lifecycle_hooks: SpawnLifecycleHooks | None = None,
) -> dict[str, dict[str, Any]]:
    """Spawn specific team members from an active team session.

    Called by the orchestrator (PM) to bring in roles on demand.

    Args:
        session_id: The team session ID.
        roles: List of role keys to spawn (e.g. ["ba", "dev", "qe"]).
        registry: SpawnRegistry for tracking.
        display_manager: TUI display manager.
        project_dir: Root project directory.

    Returns:
        Dict of role_name → {run_id, agent_name, status}.

    Raises:
        ValueError: If session not found or role invalid.
    """
    session = get_team_session(session_id)
    if not session:
        raise ValueError(f"Team session '{session_id}' not found.")

    template_roles = session.template.get("roles", {})
    pdir = Path(project_dir).resolve() if project_dir else Path.cwd()
    sdir = Path(
        os.environ.get("SPAWN_SKILLS_DIR", str(pdir / ".fast-agent" / "skills"))
    )
    paths = get_runtime_paths(str(pdir))

    results: dict[str, dict[str, Any]] = {}

    for role_name in roles:
        if role_name not in template_roles:
            results[role_name] = {
                "error": f"Role '{role_name}' not in template. Available: {list(template_roles.keys())}"
            }
            continue

        # Check if already spawned — find agent by role in session
        role_config = template_roles[role_name]
        agent_name = None
        for name, info in session.agents.items():
            if info.get("role") == role_name:
                agent_name = name
                break
        if not agent_name:
            # Agent for this role wasn't pre-registered — generate a new name
            role_display = role_config.get("role_display", role_name.upper())
            agent_name = _generate_unique_agent_name(role_display, registry)

        existing = session.agents.get(agent_name, {})
        if existing.get("status") not in ("available", None):
            results[role_name] = {
                "agent_name": agent_name,
                "status": existing.get("status"),
                "message": f"Already spawned (status: {existing.get('status')})",
            }
            continue

        roster_ctx = session.roster_context(for_role=role_name)
        # Get project_brief from session workspace
        project_brief = ""
        brief_file = session.workspace / "project_brief.txt"
        if brief_file.exists():
            project_brief = brief_file.read_text(encoding="utf-8")

        run_id = await _spawn_single_agent(
            session=session,
            role_name=role_name,
            role_config=role_config,
            workspace=session.workspace,
            roster_ctx=roster_ctx,
            project_brief=project_brief,
            registry=registry,
            project_dir=pdir,
            skills_dir=sdir,
            display_manager=display_manager,
            team_name=session.team_name,
            first_task=first_task,
            spawn_lifecycle_hooks=spawn_lifecycle_hooks,
        )

        results[role_name] = {
            "agent_name": agent_name,
            "run_id": run_id,
            "status": "running",
        }

    session.save(sessions_dir=paths["workspaces"] / "team_sessions")
    return results




# ───────────────────────────────────────────────────────────
# Session Access
# ───────────────────────────────────────────────────────────


def get_team_session(session_id: str) -> TeamSession | None:
    """Get a team session by ID.

    First checks in-memory cache, then falls back to loading from disk.
    This is critical for child processes (e.g. PM subprocess) that have
    their own empty _team_sessions dict but need to access sessions
    created by the parent process.
    """
    session = _team_sessions.get(session_id)
    if session:
        return session

    # Fall back: try loading from disk
    # Try common project dirs: SPAWN_PROJECT_DIR, cwd
    import os
    for project_dir in [
        os.environ.get("SPAWN_PROJECT_DIR", ""),
        os.getcwd(),
    ]:
        if not project_dir:
            continue
        sessions_dir = Path(project_dir) / ".runtime" / "data" / "workspaces" / "team_sessions"
        loaded = TeamSession.load(session_id, sessions_dir)
        if loaded:
            _team_sessions[session_id] = loaded  # Cache for future calls
            return loaded

    return None


def list_team_sessions() -> list[dict[str, Any]]:
    """List all team sessions (in-memory + disk-persisted).

    After server restart, in-memory cache is empty. This scans the
    disk-persisted session files so Jarvis can discover and resume
    old team sessions.
    """
    import os

    # Start with in-memory sessions
    seen_ids = set(_team_sessions.keys())
    results = [s.to_dict() for s in _team_sessions.values()]

    # Also scan disk for persisted sessions not yet in memory
    for project_dir in [
        os.environ.get("SPAWN_PROJECT_DIR", ""),
        os.getcwd(),
    ]:
        if not project_dir:
            continue
        sessions_dir = Path(project_dir) / ".runtime" / "data" / "workspaces" / "team_sessions"
        if not sessions_dir.exists():
            continue
        for session_file in sessions_dir.glob("*.json"):
            sid = session_file.stem
            if sid in seen_ids:
                continue
            try:
                loaded = TeamSession.load(sid, sessions_dir)
                if loaded:
                    _team_sessions[sid] = loaded  # Cache for future calls
                    results.append(loaded.to_dict())
                    seen_ids.add(sid)
            except Exception:
                continue

    return results
