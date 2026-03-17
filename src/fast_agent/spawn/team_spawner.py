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
    ) -> None:
        self.session_id = session_id
        self.template = template
        self.workspace = workspace
        self.parent_session_id = parent_session_id
        self.agents: dict[str, dict[str, Any]] = {}  # agent_name → {run_id, role, status, ...}
        self.sprint_status = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "template": self.template,
            "workspace": str(self.workspace),
            "parent_session_id": self.parent_session_id,
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
        lines.append("Use `post_message(to=\"Agent Name\", message=\"...\")` to send async messages to any teammate.")
        lines.append("Use `read_messages()` to check your inbox for messages from teammates.")
        lines.append("Use `check_teammate_status(agent_name=\"Agent Name\")` to check if a teammate is done.")
        lines.append("Use `create_meeting(participants=\"pm,dev\", agenda=\"...\")` for real-time discussions.")

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

    # Provide team config with agent_name -> role mapping for communication
    team_config: dict[str, Any] = {}
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
    """Resolve skill names into a temporary parent directory."""
    import shutil
    import tempfile

    skill_names = role_config.get("skills", [])
    if not skill_names:
        return []

    sdir = Path(skills_dir)
    valid_skills: list[tuple[str, Path]] = []
    for name in skill_names:
        skill_dir = sdir / name
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
            shutil.copytree(skill_dir, symlink)

    return [str(role_skills_dir)]


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

    session = TeamSession(
        session_id=session_id,
        template=template,
        workspace=workspace,
        parent_session_id=parent_session_id,
    )
    _team_sessions[session_id] = session

    roles = template.get("roles", {})
    orchestrator_role = template.get("orchestrator", "")

    # If no orchestrator specified, use the first role
    if not orchestrator_role and roles:
        orchestrator_role = next(iter(roles))

    # Pre-register all agents: orchestrator as 'pending', others as 'available'
    for role_name, role_config in roles.items():
        agent_name = role_config.get("agent_name", f"Agent - {role_name.upper()}")
        run_id = f"team_{session_id}_{role_name}_{uuid.uuid4().hex[:6]}"
        status = "pending" if role_name == orchestrator_role else "available"
        session.agents[agent_name] = {
            "run_id": run_id,
            "role": role_name,
            "agent_name": agent_name,
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
) -> str:
    """Spawn a single team agent. Returns run_id."""
    agent_name = role_config.get("agent_name", f"Agent - {role_name.upper()}")
    task = role_config.get("task", project_brief)
    instruction = role_config.get("instruction", f"You are {agent_name}.")
    instruction = instruction.replace("{agent_name}", agent_name)
    servers = list(role_config.get("servers", ["filesystem"]))
    model = role_config.get("model", "")

    # Ensure meeting_room is available for communication
    if "meeting_room" not in servers:
        servers.append("meeting_room")

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
        session_id=session.session_id,
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
        timeout_seconds=role_config.get("timeout_seconds", 600),
        role=role_name,
        agent_name=agent_name,
        team_name=team_name,
        lifecycle="resumable",
        registry=registry,
        display_manager=display_manager,
        skills=_resolve_role_skills(role_config, skills_dir),
        env_vars=team_env,
        workspace_dir=str(workspace),
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
    sdir = pdir / "skills"
    paths = get_runtime_paths(str(pdir))

    results: dict[str, dict[str, Any]] = {}

    for role_name in roles:
        if role_name not in template_roles:
            results[role_name] = {
                "error": f"Role '{role_name}' not in template. Available: {list(template_roles.keys())}"
            }
            continue

        # Check if already spawned
        role_config = template_roles[role_name]
        agent_name = role_config.get("agent_name", f"Agent - {role_name.upper()}")
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
            team_name=session.template.get("name", "team"),
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
    """List all team sessions."""
    return [s.to_dict() for s in _team_sessions.values()]
