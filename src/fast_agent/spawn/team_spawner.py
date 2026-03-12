"""Team Spawner — spawn and orchestrate a team of agents from a template.

Loads a team template YAML, creates a shared workspace, resolves
dependencies via TaskDAG, and spawns agents in dependency order
(parallel when possible).

Supports:

- Review loops with max iterations and escalation
- User-gated sprint completion
- Multi-agent meetings with tiered escalation
- Retrospective step (triggered by user)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from fast_agent.spawn.isolated_spawner import (
    run_isolated_agent,
    run_isolated_agent_background,
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
            with open(path, encoding="utf-8") as f:
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
            with open(path, encoding="utf-8") as f:
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


class TeamSession:
    """Tracks the execution state of a team."""

    def __init__(
        self,
        session_id: str,
        template: dict[str, Any],
        workspace: Path,
    ) -> None:
        self.session_id = session_id
        self.template = template
        self.workspace = workspace
        self.step_runs: dict[str, dict[str, Any]] = {}
        self.created_at = time.time()
        self.sprint_status = "in_progress"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "template": self.template.get("name", "unknown"),
            "workspace": str(self.workspace),
            "steps": self.step_runs,
            "sprint_status": self.sprint_status,
            "created_at": self.created_at,
        }


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

    FastAgent's SkillRegistry scans subdirectories of a parent dir looking
    for SKILL.md files. This function creates a temp directory with symlinks
    to only the skills assigned to this role.
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

    logger.info("Prepared %d skills in %s", len(valid_skills), role_skills_dir)
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
        f"duration={result.get('metadata', {}).get('duration_seconds', '?')}s",
    )

    return result


def _parse_verdict(review_text: str) -> dict[str, str]:
    """Parse structured verdict from reviewer output.

    Expects: ``VERDICT: PASS`` or ``VERDICT: FAIL``.
    """
    match = re.search(r"VERDICT:\s*(PASS|FAIL)", review_text, re.IGNORECASE)
    if match:
        verdict = match.group(1).lower()
        reason_match = re.search(
            r"VERDICT:\s*(?:PASS|FAIL)\s*[-—:.]?\s*(.+)",
            review_text,
            re.IGNORECASE,
        )
        reason = reason_match.group(1).strip()[:200] if reason_match else ""
        return {"verdict": verdict, "reason": reason}

    logger.warning("No VERDICT found in review output — defaulting to pass")
    return {"verdict": "pass", "reason": "No explicit verdict from reviewer"}


async def _run_review_loop(
    review_step: dict[str, Any],
    target_step: dict[str, Any],
    roles: dict[str, Any],
    workflow: list[dict[str, Any]],
    workspace: Path,
    session: TeamSession,
    registry: SpawnRegistry,
    project_dir: str | Path,
    skills_dir: str | Path,
    project_brief: str = "",
    display_manager: Any | None = None,
) -> dict[str, Any]:
    """Run a review loop with max iterations and escalation."""
    max_iterations = review_step.get("max_iterations", 3)
    review_step_name = review_step["step"]
    target_agent = target_step["agent"]
    reviewer_agent = review_step["agent"]

    review_result: dict[str, Any] = {}

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "Team %s: review loop iteration %d/%d (%s reviewing %s)",
            session.session_id,
            iteration,
            max_iterations,
            reviewer_agent,
            target_agent,
        )

        review_context = f"Review iteration {iteration}/{max_iterations}."
        if iteration > 1:
            prev_review = session.step_runs.get(review_step_name, {}).get("result", "")
            review_context += f"\n\nPrevious review feedback:\n{prev_review[:2000]}"

        review_result = await _spawn_step(
            step_config={
                **review_step,
                "task": (
                    review_step["task"] + f"\n\nThis is review iteration "
                    f"{iteration}/{max_iterations}."
                    + (
                        "\nPrevious issues should have been fixed. Verify the fixes."
                        if iteration > 1
                        else ""
                    )
                ),
            },
            roles=roles,
            workspace=workspace,
            session=session,
            registry=registry,
            project_dir=project_dir,
            skills_dir=skills_dir,
            project_brief=project_brief,
            extra_context=review_context,
            display_manager=display_manager,
        )

        review_text = review_result.get("result", "")
        verdict = _parse_verdict(review_text)
        logger.info(
            "Team %s: verdict=%s reason=%s",
            session.session_id,
            verdict["verdict"],
            verdict.get("reason", "")[:100],
        )

        if verdict["verdict"] == "pass":
            logger.info(
                "Team %s: review PASSED at iteration %d",
                session.session_id,
                iteration,
            )
            append_changelog(
                workspace,
                reviewer_agent,
                review_step_name,
                f"✅ Review PASSED at iteration {iteration} — {verdict.get('reason', '')[:100]}",
            )
            return review_result

        # Review failed — re-run target with feedback
        if iteration < max_iterations:
            logger.info(
                "Team %s: review FAILED, re-running %s",
                session.session_id,
                target_agent,
            )
            append_changelog(
                workspace,
                reviewer_agent,
                review_step_name,
                f"❌ Review FAILED (iteration {iteration}), requesting fixes",
            )

            fix_context = (
                f"The {reviewer_agent} reviewed your work and found "
                f"issues.\n\n"
                f"## Review Feedback (iteration {iteration})\n"
                f"{review_result.get('result', '')[:3000]}\n\n"
                "Please fix the issues and update your files "
                "in the workspace."
            )

            await _spawn_step(
                step_config={
                    **target_step,
                    "task": ("Fix issues found during code review. " + target_step["task"]),
                },
                roles=roles,
                workspace=workspace,
                session=session,
                registry=registry,
                project_dir=project_dir,
                skills_dir=skills_dir,
                project_brief=project_brief,
                extra_context=fix_context,
            )

    # Exhausted iterations — escalation
    on_max_fail = review_step.get("on_max_fail", "continue")
    if on_max_fail == "escalate":
        escalate_to = review_step.get("escalate_to", [])
        escalate_task = review_step.get("escalate_task", "").replace(
            "{iterations}", str(max_iterations)
        )

        logger.warning(
            "Team %s: review exhausted %d iterations, escalating to %s",
            session.session_id,
            max_iterations,
            escalate_to,
        )
        append_changelog(
            workspace,
            "system",
            review_step_name,
            f"⚠️ Escalating to {escalate_to} after {max_iterations} failed reviews",
        )

        for escalate_role in escalate_to:
            if escalate_role in roles:
                escalate_context = (
                    "## Escalation Notice\n"
                    f"The review between {target_agent} and "
                    f"{reviewer_agent} failed after "
                    f"{max_iterations} attempts.\n\n"
                    "## Last Review Feedback\n"
                    f"{review_result.get('result', '')[:3000]}"
                )

                await _spawn_step(
                    step_config={
                        "step": f"escalation_{escalate_role}",
                        "agent": escalate_role,
                        "task": escalate_task
                        or (
                            f"Review failed after {max_iterations} "
                            f"attempts between {target_agent} and "
                            f"{reviewer_agent}. Analyze root cause "
                            "and recommend: fix, redesign, or defer."
                        ),
                        "depends_on": [],
                    },
                    roles=roles,
                    workspace=workspace,
                    session=session,
                    registry=registry,
                    project_dir=project_dir,
                    skills_dir=skills_dir,
                    project_brief=project_brief,
                    extra_context=escalate_context,
                )

    return review_result


async def run_meeting(
    meeting_config: dict[str, Any],
    roles: dict[str, Any],
    workspace: Path,
    session: TeamSession,
    registry: SpawnRegistry,
    project_dir: str | Path,
    skills_dir: str | Path,
    project_brief: str = "",
    display_manager: Any | None = None,
) -> dict[str, Any]:
    """Run a multi-agent meeting with concurrent spawning.

    1. Creates meeting directory and state files
    2. Spawns all participants as background processes
    3. Polls meeting state until ended
    4. Handles tiered escalation
    5. Spawns facilitator for MoM after meeting
    """
    step_name = meeting_config["step"]
    agenda = meeting_config.get("agenda", f"Meeting: {step_name}")
    participants = meeting_config.get("participants", [])
    max_rounds = meeting_config.get("max_rounds", 3)
    facilitator = meeting_config.get("facilitator", None)
    escalation_config = meeting_config.get("escalation", [])

    agenda = agenda.replace("{project_brief}", project_brief[:500])

    meeting_id = f"mtg_{uuid.uuid4().hex[:8]}"
    meeting_dir = workspace / "meetings" / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "meeting_id": meeting_id,
        "agenda": agenda,
        "participants": list(participants),
        "max_rounds": max_rounds,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (meeting_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    state_data: dict[str, Any] = {
        "current_turn": 0,
        "current_round": 1,
        "joined": [],
        "ended": False,
        "outcome": None,
        "started": False,
    }
    (meeting_dir / "state.json").write_text(json.dumps(state_data, indent=2))
    (meeting_dir / "transcript.json").write_text("[]")

    _audit_meeting(meeting_dir, f"Meeting created: {agenda}")

    logger.info(
        "Team %s: starting meeting '%s' (id=%s, participants=%s)",
        session.session_id,
        step_name,
        meeting_id,
        participants,
    )

    append_changelog(
        workspace,
        "system",
        step_name,
        f"Meeting started: {agenda} (participants: {', '.join(participants)})",
    )

    team_env = _build_team_env(workspace, roles, "meeting_participant")
    team_env["MEETING_ID"] = meeting_id

    context_parts = [
        f"## Shared Workspace\nPath: {workspace}",
        "Use the filesystem MCP server to access files.",
    ]
    for dep in meeting_config.get("depends_on", []):
        dep_info = session.step_runs.get(dep, {})
        if dep_info.get("result"):
            context_parts.append(f"\n## Output from '{dep}' step\n{dep_info['result'][:3000]}")
    base_context = "\n\n".join(context_parts)

    # Spawn all participants concurrently
    run_ids: dict[str, str] = {}
    for role_name in participants:
        role_config = roles.get(role_name, {})
        instruction = role_config.get("instruction", f"You are the {role_name}.")
        servers = list(role_config.get("servers", ["filesystem"]))
        model = role_config.get("model", "")

        for srv in ["meeting_room", "team_communicate"]:
            if srv not in servers:
                servers.append(srv)

        meeting_task = (
            f"You are {role_name} joining a team meeting.\n\n"
            f"## Meeting Info\n"
            f"- Meeting ID: {meeting_id}\n"
            f"- Agenda: {agenda}\n"
            f"- Your Role: {role_name}\n"
            f"- Participants: {', '.join(participants)}\n\n"
            f"## CRITICAL: Follow these steps EXACTLY\n"
            f"DO NOT call create_meeting — the meeting exists.\n\n"
            f"Step 1: Call `join_meeting(meeting_id='{meeting_id}', "
            f"role='{role_name}')`\n"
            f"Step 2: Call `wait_for_my_turn(meeting_id="
            f"'{meeting_id}', role='{role_name}')`\n"
            f"Step 3: When wait_for_my_turn returns 'your_turn':\n"
            f"   a. Call `get_transcript(meeting_id="
            f"'{meeting_id}')` to read discussion\n"
            f"   b. Read relevant workspace files if needed\n"
            f"   c. Call `speak(meeting_id='{meeting_id}', "
            f"role='{role_name}', message=...)` with your response\n"
            f"   d. OR call `skip_turn(meeting_id='{meeting_id}', "
            f"role='{role_name}')` if nothing to add\n"
            f"Step 4: After speaking, call `wait_for_my_turn` again\n"
            f"Step 5: Repeat Steps 3-4 until meeting_ended\n\n"
            f"IMPORTANT: Build on what others have said.\n"
        )

        run_id = await run_isolated_agent_background(
            task=meeting_task,
            project_dir=project_dir,
            instruction=instruction,
            context=base_context,
            servers=servers,
            model=model,
            timeout_seconds=600,
            role=role_name,
            lifecycle="oneshot",
            registry=registry,
            env_vars=team_env,
            workspace_dir=str(workspace),
            display_manager=display_manager,
            skills=_resolve_role_skills(role_config, skills_dir),
        )
        run_ids[role_name] = run_id
        logger.info(
            "Team %s: spawned %s (run_id=%s) for meeting",
            session.session_id,
            role_name,
            run_id,
        )

    # Poll meeting state
    poll_interval = 5
    max_wait = 600
    start_time = time.time()
    last_escalation_level = 0
    state_path = meeting_dir / "state.json"

    while True:
        await asyncio.sleep(poll_interval)
        elapsed = time.time() - start_time

        if state_path.exists():
            state = json.loads(state_path.read_text())
        else:
            state = {"ended": False}

        if state.get("ended"):
            outcome = state.get("outcome", "unknown")
            logger.info("Meeting %s ended: %s", meeting_id, outcome)
            break

        # Check for escalation triggers
        if escalation_config and state.get("outcome") is None:
            transcript_path = meeting_dir / "transcript.json"
            transcript = json.loads(transcript_path.read_text()) if transcript_path.exists() else []

            for msg in reversed(transcript[-5:]):
                if (
                    msg.get("type") == "speak"
                    and "VERDICT: ESCALATE" in msg.get("message", "").upper()
                ):
                    for esc_level in escalation_config:
                        level_num = esc_level.get("level", 0)
                        if level_num > last_escalation_level:
                            action = esc_level.get("action", "")
                            if action == "escalate_to_user":
                                state["ended"] = True
                                state["outcome"] = "escalate_to_user"
                                state_path.write_text(json.dumps(state, indent=2))
                                break

                            join_roles = esc_level.get("join", [])
                            extra_rounds = esc_level.get("max_rounds", 2)

                            config_path = meeting_dir / "config.json"
                            cfg = json.loads(config_path.read_text())
                            for new_role in join_roles:
                                if new_role not in cfg["participants"]:
                                    cfg["participants"].append(new_role)
                            cfg["max_rounds"] = state.get("current_round", 1) + extra_rounds
                            config_path.write_text(json.dumps(cfg, indent=2))

                            state["ended"] = False
                            state["outcome"] = None
                            state_path.write_text(json.dumps(state, indent=2))

                            for new_role in join_roles:
                                if new_role not in run_ids:
                                    rc = roles.get(new_role, {})
                                    inst = rc.get(
                                        "instruction",
                                        f"You are {new_role}.",
                                    )
                                    srvs = list(rc.get("servers", ["filesystem"]))
                                    for s in [
                                        "meeting_room",
                                        "team_communicate",
                                    ]:
                                        if s not in srvs:
                                            srvs.append(s)

                                    esc_task = (
                                        f"You are {new_role}, "
                                        "escalated into an ongoing "
                                        "meeting.\n\n"
                                        f"## Meeting Info\n"
                                        f"- Meeting ID: {meeting_id}\n"
                                        f"- Agenda: {agenda}\n"
                                        "- You were brought in for "
                                        "your expertise.\n\n"
                                        "## Instructions\n"
                                        f"1. join_meeting('{meeting_id}'"
                                        f", '{new_role}')\n"
                                        f"2. get_transcript('"
                                        f"{meeting_id}')\n"
                                        f"3. wait_for_my_turn('"
                                        f"{meeting_id}', "
                                        f"'{new_role}')\n"
                                        "4. Analyze and provide "
                                        "guidance\n"
                                    )

                                    new_run = await run_isolated_agent_background(
                                        task=esc_task,
                                        project_dir=project_dir,
                                        instruction=inst,
                                        context=base_context,
                                        servers=srvs,
                                        model=rc.get("model", ""),
                                        timeout_seconds=600,
                                        role=new_role,
                                        lifecycle="oneshot",
                                        registry=registry,
                                        env_vars=team_env,
                                        workspace_dir=str(workspace),
                                        display_manager=display_manager,
                                        skills=_resolve_role_skills(rc, skills_dir),
                                    )
                                    run_ids[new_role] = new_run

                            last_escalation_level = level_num
                            _audit_meeting(
                                meeting_dir,
                                f"Escalated to level {level_num}: added {join_roles}",
                            )
                            break
                    break

        if elapsed > max_wait:
            logger.warning(
                "Meeting %s timed out after %ds",
                meeting_id,
                max_wait,
            )
            state["ended"] = True
            state["outcome"] = "timeout"
            state_path.write_text(json.dumps(state, indent=2))
            break

    # Read final transcript
    transcript_path = meeting_dir / "transcript.json"
    transcript = json.loads(transcript_path.read_text()) if transcript_path.exists() else []
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    verdict = "unknown"
    outcome = state.get("outcome", "")
    if isinstance(outcome, str) and outcome.startswith("verdict_"):
        verdict = outcome.replace("verdict_", "")

    # Spawn facilitator for MoM
    if facilitator and facilitator in roles:
        fac_config = roles[facilitator]
        mom_task = (
            f"You are {facilitator}. A team meeting concluded.\n\n"
            f"## Meeting ID: {meeting_id}\n"
            f"## Agenda: {agenda}\n"
            f"## Outcome: {verdict}\n\n"
            "## Your Task\n"
            "1. Read the full meeting transcript from: "
            f"meetings/{meeting_id}/transcript.json\n"
            "2. Write Minutes of Meeting (MoM) to: "
            f"reviews/mom_{meeting_id}.md\n"
            "3. Include: attendees, key discussion points, "
            "decisions, action items, verdict\n"
        )

        transcript_text = "\n".join(
            f"[Round {t['round']}] {t['agent']}: {t['message']}"
            for t in transcript
            if t.get("type") == "speak"
        )

        mom_context = f"{base_context}\n\n## Meeting Transcript\n{transcript_text[:5000]}"

        fac_servers = list(fac_config.get("servers", ["filesystem"]))
        if "meeting_room" not in fac_servers:
            fac_servers.append("meeting_room")

        await run_isolated_agent(
            task=mom_task,
            project_dir=project_dir,
            instruction=fac_config.get("instruction", f"You are {facilitator}."),
            context=mom_context,
            servers=fac_servers,
            model=fac_config.get("model", ""),
            timeout_seconds=600,
            role=facilitator,
            lifecycle="oneshot",
            registry=registry,
            env_vars=team_env,
            skills=_resolve_role_skills(fac_config, skills_dir),
        )
        logger.info(
            "Facilitator %s wrote MoM for meeting %s",
            facilitator,
            meeting_id,
        )

    # Record in session
    session.step_runs[step_name] = {
        "run_id": meeting_id,
        "agent": f"meeting({','.join(participants)})",
        "status": "completed",
        "result": (
            f"Meeting {meeting_id}: verdict={verdict}, "
            f"{len(transcript)} messages, "
            f"{state.get('current_round', 1) - 1} rounds"
        ),
        "verdict": verdict,
        "meeting_id": meeting_id,
        "transcript_length": len(transcript),
    }

    append_changelog(
        workspace,
        "system",
        step_name,
        f"Meeting ended: verdict={verdict}, {len(transcript)} messages",
    )

    return {
        "status": "completed",
        "meeting_id": meeting_id,
        "verdict": verdict,
        "outcome": outcome,
        "transcript_length": len(transcript),
        "participants": list(run_ids.keys()),
    }


def _audit_meeting(meeting_dir: Path, message: str) -> None:
    """Append to meeting audit log."""
    audit_path = meeting_dir / "audit.log"
    ts = datetime.now().isoformat(timespec="seconds")
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


async def spawn_team(
    template_name: str,
    project_brief: str,
    registry: SpawnRegistry,
    project_dir: str | Path,
    template_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
    workspace_root: Path | None = None,
    display_manager: Any | None = None,
) -> TeamSession:
    """Spawn a full team from a template.

    1. Load template → build TaskDAG
    2. Create shared workspace
    3. Topological sort → determine execution order
    4. Execute steps with review loops and escalation
    5. Return session for user review

    Args:
        template_name: Name of the team template.
        project_brief: Project description for agents.
        registry: SpawnRegistry for tracking agents.
        project_dir: Root directory of host application.
        template_dir: Directory with team template YAMLs.
        skills_dir: Directory with skill subdirectories.
        workspace_root: Override workspace root directory.
        display_manager: TUI display manager instance.

    Returns:
        A TeamSession tracking all spawned agents.
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
        step_config = next((s for s in workflow if s["step"] == step_name), None)
        if not step_config:
            continue

        step_type = step_config.get("type", "")
        if step_type == "meeting":
            await run_meeting(
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
                await _run_review_loop(
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
    return session


async def run_retrospective(
    session_id: str,
    registry: SpawnRegistry,
    project_dir: str | Path,
    skills_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run retrospective for a completed team session.

    Only callable after user accepts sprint output.
    """
    session = _team_sessions.get(session_id)
    if not session:
        return {"error": f"Team session '{session_id}' not found."}

    if session.sprint_status not in ("completed", "accepted"):
        return {
            "error": (f"Sprint not completed (status: {session.sprint_status}). Cannot run retro.")
        }

    session.sprint_status = "accepted"
    template = session.template
    roles = template.get("roles", {})
    workspace = session.workspace

    pdir = Path(project_dir).resolve()
    sdir = Path(skills_dir) if skills_dir else pdir / "skills"

    retro_dir = workspace / "retrospective"
    retro_dir.mkdir(exist_ok=True)

    retro_results: dict[str, dict[str, str]] = {}

    for role_name, role_config in roles.items():
        step_results = {
            name: info for name, info in session.step_runs.items() if info.get("agent") == role_name
        }

        other_results = {
            name: {"agent": info["agent"], "status": info["status"]}
            for name, info in session.step_runs.items()
            if info.get("agent") != role_name
        }

        context_parts = [
            f"## Shared Workspace: {workspace}",
            f"\n## Your Work (as {role_name})",
        ]
        for step_name, info in step_results.items():
            context_parts.append(
                f"- Step '{step_name}': {info.get('status', '?')} — {info.get('result', '')[:500]}"
            )

        context_parts.append("\n## Team Summary")
        for step_name, info in other_results.items():
            context_parts.append(f"- {info['agent']} ({step_name}): {info['status']}")

        context = "\n".join(context_parts)

        task = (
            f"You are the {role_name} on this agile team. "
            "The sprint is complete.\n\n"
            "Write your retrospective lessons learned to file: "
            f"retrospective/{role_name}_lessons.md\n\n"
            "Cover:\n"
            "1. What went well in your role?\n"
            "2. What would you do differently next time?\n"
            "3. Suggestions for team improvement\n"
            "4. Key learnings for future sprints\n\n"
            "Be specific and reference actual work done."
        )

        instruction = role_config.get("instruction", f"You are the {role_name}.")
        servers = role_config.get("servers", ["filesystem"])

        append_changelog(
            workspace,
            role_name,
            "retrospective",
            f"Starting retrospective for {role_name}",
        )

        result = await run_isolated_agent(
            task=task,
            project_dir=project_dir,
            instruction=instruction,
            context=context,
            servers=servers,
            timeout_seconds=600,
            role=role_name,
            lifecycle="oneshot",
            registry=registry,
            skills=_resolve_role_skills(role_config, sdir),
        )

        retro_results[role_name] = {
            "status": result.get("status", "error"),
            "result": result.get("result", "")[:3000],
        }

        append_changelog(
            workspace,
            role_name,
            "retrospective",
            f"Completed: status={result.get('status', '?')}",
        )

    return {
        "session_id": session_id,
        "status": "retrospective_complete",
        "results": retro_results,
        "workspace": str(workspace),
    }


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
