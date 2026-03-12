"""Team Spawner — spawn and orchestrate a team of agents from a template.

Loads a team template YAML, creates a shared workspace, resolves
dependencies via TaskDAG, and spawns agents in dependency order
(parallel when possible).

Workflow-specific logic (review loops, meetings, retrospectives) is
provided by ``team_orchestration`` and can be overridden via hooks.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from fast_agent.spawn.isolated_spawner import (
    run_isolated_agent,
)
from fast_agent.spawn.runtime_paths import get_runtime_paths
from fast_agent.spawn.task_dag import TaskDAG, TaskNode

if TYPE_CHECKING:
    from fast_agent.spawn.spawn_registry import SpawnRegistry
from fast_agent.spawn.workspace_manager import (
    append_changelog,
    create_workspace,
)

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

    Args:
        template_name: Name of the template (without extension).
        template_dir: Directory containing team template YAML files.

    Returns:
        Parsed template dict.

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
                    "steps": len(data.get("workflow", [])),
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
        self.step_runs: dict[str, dict[str, Any]] = {}
        self.sprint_status = "in_progress"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "template": self.template.get("name", "unknown"),
            "workspace": str(self.workspace),
            "parent_session_id": self.parent_session_id,
            "steps": self.step_runs,
            "sprint_status": self.sprint_status,
        }

    # ── Persistence ──────────────────────────────────────

    def save(self, sessions_dir: str | Path | None = None) -> Path:
        """Persist session state to JSON file.

        Default location: ``{workspace}/../team_sessions/{id}.json``
        """
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
        session.step_runs = data.get("steps", {})
        session.sprint_status = data.get("sprint_status", "unknown")
        return session


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────


def _build_team_env(
    workspace: Path,
    roles: dict[str, Any],
    my_role: str,
    depth: int = 1,
) -> dict[str, str]:
    """Build environment variables for team agent communication."""
    return {
        "TEAM_WORKSPACE": str(workspace),
        "TEAM_ROLES_CONFIG": json.dumps(roles),
        "TEAM_MY_ROLE": my_role,
        "TEAM_SPAWN_DEPTH": str(depth),
    }


def _resolve_role_skills(
    role_config: dict[str, Any],
    skills_dir: str | Path,
) -> list[str]:
    """Resolve skill names into a temporary parent directory.

    FastAgent's SkillRegistry scans subdirectories of a parent dir
    looking for SKILL.md files. This function creates a temp directory
    with symlinks to only the skills assigned to this role.
    """
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

    logger.info(
        "Prepared %d skills in %s",
        len(valid_skills),
        role_skills_dir,
    )
    return [str(role_skills_dir)]


async def _spawn_step(
    step_config: dict[str, Any],
    roles: dict[str, Any],
    workspace: Path,
    session: TeamSession,
    registry: SpawnRegistry,
    project_dir: str | Path,
    skills_dir: str | Path,
    project_brief: str = "",
    extra_context: str = "",
    display_manager: Any | None = None,
) -> dict[str, Any]:
    """Spawn a single workflow step agent and return its result."""
    step_name = step_config["step"]
    agent_role = step_config["agent"]
    role_config = roles.get(agent_role, {})

    task = step_config["task"].replace("{project_brief}", project_brief)

    context_parts = [
        f"## Shared Workspace\nPath: {workspace}",
        "You MUST read from and write to this workspace directory.",
        "Use the filesystem MCP server to access files.",
    ]

    for dep in step_config.get("depends_on", []):
        dep_info = session.step_runs.get(dep, {})
        if dep_info.get("result"):
            context_parts.append(f"\n## Output from '{dep}' step\n{dep_info['result'][:2000]}")

    if extra_context:
        context_parts.append(f"\n## Additional Context\n{extra_context}")

    context = "\n\n".join(context_parts)
    instruction = role_config.get("instruction", "")
    servers = role_config.get("servers", [])
    model = role_config.get("model", "")

    if "team_communicate" not in servers:
        servers = list(servers) + ["team_communicate"]

    append_changelog(
        workspace,
        agent_role,
        step_name,
        f"Starting: {task[:100]}...",
    )

    team_env = _build_team_env(workspace, roles, agent_role)

    logger.info(
        "Team %s: spawning %s for step '%s'",
        session.session_id,
        agent_role,
        step_name,
    )
    result = await run_isolated_agent(
        task=task,
        project_dir=project_dir,
        instruction=instruction,
        context=context,
        servers=servers,
        model=model,
        timeout_seconds=600,
        role=agent_role,
        lifecycle="persistent",
        registry=registry,
        env_vars=team_env,
        display_manager=display_manager,
        skills=_resolve_role_skills(role_config, skills_dir),
    )

    run_id = result.get("run_id", "")
    status = result.get("status", "error")
    agent_result = result.get("result", "")

    session.step_runs[step_name] = {
        "run_id": run_id,
        "agent": agent_role,
        "status": status,
        "result": agent_result[:5000],
        "duration": result.get("metadata", {}).get("duration_seconds"),
    }

    append_changelog(
        workspace,
        agent_role,
        step_name,
        f"Completed: status={status}, "
        f"duration="
        f"{result.get('metadata', {}).get('duration_seconds', '?')}s",
    )

    return result


# ───────────────────────────────────────────────────────────
# Main Orchestrator
# ───────────────────────────────────────────────────────────

# Type alias for the hook callables
MeetingHandler = Callable[..., Coroutine[Any, Any, dict[str, Any]]]
ReviewHandler = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


async def spawn_team(
    template_name: str,
    project_brief: str,
    registry: SpawnRegistry,
    project_dir: str | Path,
    template_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
    workspace_root: Path | None = None,
    display_manager: Any | None = None,
    meeting_handler: MeetingHandler | None = None,
    review_handler: ReviewHandler | None = None,
    parent_session_id: str = "",
) -> TeamSession:
    """Spawn a full team from a template.

    1. Load template -> build TaskDAG
    2. Create shared workspace
    3. Topological sort -> determine execution order
    4. Execute steps with optional review/meeting hooks
    5. Persist and return session

    Args:
        template_name: Name of the team template.
        project_brief: Project description for agents.
        registry: SpawnRegistry for tracking agents.
        project_dir: Root directory of host application.
        template_dir: Directory with team template YAMLs.
        skills_dir: Directory with skill subdirectories.
        workspace_root: Override workspace root directory.
        display_manager: TUI display manager instance.
        meeting_handler: Custom meeting handler (default:
            ``team_orchestration.run_meeting``).
        review_handler: Custom review handler (default:
            ``team_orchestration.run_review_loop``).
        parent_session_id: Optional fast-agent session ID
            for traceability.

    Returns:
        A TeamSession tracking all spawned agents.
    """
    # Lazy-import defaults from team_orchestration
    if meeting_handler is None:
        from fast_agent.spawn.team_orchestration import (
            run_meeting,
        )

        meeting_handler = run_meeting
    if review_handler is None:
        from fast_agent.spawn.team_orchestration import (
            run_review_loop,
        )

        review_handler = run_review_loop

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
    workflow = template.get("workflow", [])

    # Build DAG from workflow
    dag = TaskDAG()
    for step in workflow:
        step_name = step["step"]
        deps = step.get("depends_on", [])
        dag.add_node(TaskNode(role=step_name, depends_on=deps))

    execution_order = dag.topological_sort()
    logger.info(
        "Team %s: execution order = %s",
        session_id,
        execution_order,
    )

    completed_steps: set[str] = set()

    for step_name in execution_order:
        step_config = next(
            (s for s in workflow if s["step"] == step_name),
            None,
        )
        if not step_config:
            continue

        step_type = step_config.get("type", "")
        if step_type == "meeting":
            await meeting_handler(
                meeting_config=step_config,
                roles=roles,
                workspace=workspace,
                session=session,
                registry=registry,
                project_dir=project_dir,
                skills_dir=sdir,
                project_brief=project_brief,
                display_manager=display_manager,
            )
            completed_steps.add(step_name)
            continue

        review_target = step_config.get("review_target")
        if review_target:
            target_step = next(
                (s for s in workflow if s["step"] == review_target),
                None,
            )
            if target_step:
                await review_handler(
                    review_step=step_config,
                    target_step=target_step,
                    roles=roles,
                    workflow=workflow,
                    workspace=workspace,
                    session=session,
                    registry=registry,
                    project_dir=project_dir,
                    skills_dir=sdir,
                    project_brief=project_brief,
                    display_manager=display_manager,
                )
                completed_steps.add(step_name)
                continue

        result = await _spawn_step(
            step_config=step_config,
            roles=roles,
            workspace=workspace,
            session=session,
            registry=registry,
            project_dir=project_dir,
            skills_dir=sdir,
            project_brief=project_brief,
            display_manager=display_manager,
        )

        status = result.get("status", "error")
        if status == "completed":
            completed_steps.add(step_name)
        else:
            logger.error(
                "Team %s: step '%s' failed: %s",
                session_id,
                step_name,
                result.get("error", ""),
            )
            append_changelog(
                workspace,
                step_config.get("agent", "unknown"),
                step_name,
                f"ERROR: {result.get('error', 'unknown error')}",
            )

    session.sprint_status = "completed"

    # Persist session state
    session.save(sessions_dir=paths["workspaces"] / "team_sessions")

    return session


# ───────────────────────────────────────────────────────────
# Session Access
# ───────────────────────────────────────────────────────────


def get_team_session(session_id: str) -> TeamSession | None:
    """Get a team session by ID."""
    return _team_sessions.get(session_id)


def list_team_sessions() -> list[dict[str, Any]]:
    """List all team sessions."""
    return [
        {
            "session_id": s.session_id,
            "template": s.template.get("name", "unknown"),
            "workspace": str(s.workspace),
            "sprint_status": s.sprint_status,
            "steps_total": len(s.template.get("workflow", [])),
            "steps_completed": sum(
                1 for r in s.step_runs.values() if r.get("status") == "completed"
            ),
            "steps_errored": sum(1 for r in s.step_runs.values() if r.get("status") == "error"),
        }
        for s in _team_sessions.values()
    ]
