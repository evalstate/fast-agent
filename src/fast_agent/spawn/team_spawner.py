"""Team Spawner — spawn and manage teams of agents from a template.

Provides **only primitives**: template loading, workspace creation,
agent spawning, and roster management. All workflow intelligence
lives in **skills** assigned to agents.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from fast_agent.spawn.isolated_spawner import (
    run_isolated_agent,
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
        self.agents: dict[str, dict[str, Any]] = {}  # role → {run_id, status, ...}
        self.sprint_status = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "template": self.template.get("name", "unknown"),
            "workspace": str(self.workspace),
            "parent_session_id": self.parent_session_id,
            "agents": self.agents,
            "sprint_status": self.sprint_status,
        }

    # ── Roster ───────────────────────────────────────────

    def get_roster(self) -> dict[str, dict[str, Any]]:
        """Return team roster: role → {run_id, role, status}."""
        return {
            role: {
                "run_id": info.get("run_id", ""),
                "role": role,
                "status": info.get("status", "unknown"),
            }
            for role, info in self.agents.items()
        }

    def write_roster(self) -> Path:
        """Write team_roster.json to workspace."""
        path = self.workspace / "team_roster.json"
        path.write_text(json.dumps(self.get_roster(), indent=2))
        return path

    def roster_context(self) -> str:
        """Build roster context string for agent injection."""
        lines = ["## Your Team"]
        for role, info in self.agents.items():
            run_id = info.get("run_id", "?")
            instruction = info.get("instruction", "")
            label = instruction[:60] if instruction else role
            lines.append(f"- **{role}** (run_id: {run_id}) — {label}")
        lines.append("\nUse `send_message_to_agent(to_role=..., ...)` to message any teammate.")
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
        session = cls(
            session_id=data["session_id"],
            template={"name": data.get("template", "unknown")},
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
) -> dict[str, str]:
    """Build environment variables for team agent."""
    return {
        "TEAM_WORKSPACE": str(workspace),
        "TEAM_MY_ROLE": my_role,
    }


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
    template_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
    workspace_root: Path | None = None,
    display_manager: Any | None = None,
    parent_session_id: str = "",
    mode: str = "blocking",
) -> TeamSession:
    """Spawn a team of agents from a template.

    Each agent is spawned with:
    - Their assigned skills
    - Team roster context (who's on the team)
    - Shared workspace path
    - MCP servers for communication

    Agents self-coordinate using skills + MCP tools.
    No hardcoded flow control.

    Args:
        template_name: Name of the team template.
        project_brief: Project description for agents.
        registry: SpawnRegistry for tracking agents.
        project_dir: Root directory of host application.
        template_dir: Directory with team template YAMLs.
        skills_dir: Directory with skill subdirectories.
        workspace_root: Override workspace root directory.
        display_manager: TUI display manager instance.
        parent_session_id: Optional fast-agent session ID
            for traceability.
        mode: "blocking" (wait for all agents) or
              "background" (return IDs immediately).

    Returns:
        A TeamSession with agent roster.
    """
    pdir = Path(project_dir).resolve()
    tdir = Path(template_dir) if template_dir else pdir / "team_templates"
    sdir = Path(skills_dir) if skills_dir else pdir / "skills"
    paths = get_runtime_paths(project_dir)

    template = load_team_template(template_name, tdir)
    session_id = str(uuid.uuid4())[:8]

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

    # Pre-generate run_ids and register all agents
    for role_name, role_config in roles.items():
        run_id = f"team_{session_id}_{role_name}_{uuid.uuid4().hex[:6]}"
        session.agents[role_name] = {
            "run_id": run_id,
            "role": role_name,
            "instruction": role_config.get("instruction", ""),
            "status": "pending",
        }

    # Write roster to workspace
    session.write_roster()

    # Build team roster context (shared by all agents)
    roster_ctx = session.roster_context()

    if mode == "background":
        # Return immediately, spawn in background
        session.sprint_status = "spawning"
        session.save(sessions_dir=paths["workspaces"] / "team_sessions")

        import asyncio

        async def _spawn_all_bg() -> None:
            await _spawn_all_agents(
                session=session,
                roles=roles,
                workspace=workspace,
                roster_ctx=roster_ctx,
                project_brief=project_brief,
                registry=registry,
                project_dir=pdir,
                skills_dir=sdir,
                display_manager=display_manager,
            )
            session.sprint_status = "running"
            session.save(sessions_dir=paths["workspaces"] / "team_sessions")

        asyncio.create_task(_spawn_all_bg())
        return session

    # Blocking mode: spawn all and wait
    await _spawn_all_agents(
        session=session,
        roles=roles,
        workspace=workspace,
        roster_ctx=roster_ctx,
        project_brief=project_brief,
        registry=registry,
        project_dir=pdir,
        skills_dir=sdir,
        display_manager=display_manager,
    )

    session.sprint_status = "running"
    session.save(sessions_dir=paths["workspaces"] / "team_sessions")
    return session


async def _spawn_all_agents(
    session: TeamSession,
    roles: dict[str, Any],
    workspace: Path,
    roster_ctx: str,
    project_brief: str,
    registry: SpawnRegistry,
    project_dir: str | Path,
    skills_dir: str | Path,
    display_manager: Any | None = None,
) -> None:
    """Spawn all team agents with skills and roster context."""
    for role_name, role_config in roles.items():
        task = role_config.get("task", project_brief)
        instruction = role_config.get("instruction", f"You are the {role_name}.")
        servers = list(role_config.get("servers", ["filesystem"]))
        model = role_config.get("model", "")

        # Ensure communication servers are available
        for srv in ["team_communicate"]:
            if srv not in servers:
                servers.append(srv)

        # Build context with roster + workspace
        context_parts = [
            f"## Project Brief\n{project_brief}",
            f"\n## Shared Workspace\nPath: {workspace}",
            "Use the filesystem MCP server to read/write files.",
            f"\n{roster_ctx}",
        ]
        context = "\n\n".join(context_parts)

        team_env = _build_team_env(workspace, roles, role_name)

        logger.info(
            "Team %s: spawning %s",
            session.session_id,
            role_name,
        )

        session.agents[role_name]["status"] = "running"

        result = await run_isolated_agent(
            task=task,
            project_dir=str(project_dir),
            instruction=instruction,
            context=context,
            servers=servers,
            model=model,
            timeout_seconds=600,
            role=role_name,
            lifecycle="resumable",
            registry=registry,
            env_vars=team_env,
            display_manager=display_manager,
            skills=_resolve_role_skills(role_config, skills_dir),
        )

        status = result.get("status", "error")
        session.agents[role_name].update(
            {
                "run_id": result.get("run_id", session.agents[role_name]["run_id"]),
                "status": status,
                "result": result.get("result", "")[:5000],
            }
        )

        logger.info(
            "Team %s: %s → %s",
            session.session_id,
            role_name,
            status,
        )

    # Update roster file after all agents complete
    session.write_roster()


# ───────────────────────────────────────────────────────────
# Session Access
# ───────────────────────────────────────────────────────────


def get_team_session(session_id: str) -> TeamSession | None:
    """Get a team session by ID."""
    return _team_sessions.get(session_id)


def list_team_sessions() -> list[dict[str, Any]]:
    """List all team sessions."""
    return [s.to_dict() for s in _team_sessions.values()]
