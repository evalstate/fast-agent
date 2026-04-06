"""Config reader — reads MCP server configuration from fastagent.config.yaml.

Single source of truth for available servers and their commands.
All spawn modules should import from here instead of hardcoding server lists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_config(project_dir: str | Path) -> dict[str, Any]:
    """Load fastagent.config.yaml from the given project directory."""
    config_file = Path(project_dir) / "fastagent.config.yaml"
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        return yaml.safe_load(f) or {}


def get_available_servers(project_dir: str | Path) -> list[str]:
    """Get list of all available MCP server names from config.

    Reads from ``mcp.servers`` dict in fastagent.config.yaml.
    """
    config = _load_config(project_dir)
    servers = config.get("mcp", {}).get("servers", {})
    if isinstance(servers, dict):
        return list(servers.keys())
    return []


# ---------- MCP server env vars ----------

# Servers that need workspace/project env vars
_TEAM_AWARE_SERVERS = {"meeting_room", "agent_spawner", "email"}


def get_server_env(
    server_name: str,
    workspace_dir: str | None = None,
    agent_name: str | None = None,
) -> dict[str, str] | None:
    """Get extra environment variables needed by a specific MCP server.

    Returns dict of env vars, or None if no extra env is needed.

    ``meeting_room``, ``agent_spawner``, and ``email`` need:
    - SPAWN_PROJECT_DIR: to find project-level spawn_registry and team_sessions
    - TEAM_WORKSPACE: to locate workspace-specific files
    - TEAM_MESSAGES_DIR: message bus directory for email delivery
    - TEAM_ROLES_CONFIG: team member names/roles for addressing
    """
    import os

    if server_name not in _TEAM_AWARE_SERVERS:
        return None

    env: dict[str, str] = {}

    # SPAWN_PROJECT_DIR — critical for finding team sessions and spawn registry
    project_dir = os.environ.get("SPAWN_PROJECT_DIR", "")
    if project_dir:
        env["SPAWN_PROJECT_DIR"] = project_dir

    # Workspace dir for file access
    if workspace_dir:
        env["TEAM_WORKSPACE"] = workspace_dir

    # Agent identity
    if agent_name:
        env["TEAM_MY_NAME"] = agent_name

    # Session ID propagation
    session_id = os.environ.get("TEAM_SESSION_ID", "")
    if session_id:
        env["TEAM_SESSION_ID"] = session_id

    # Messages dir — needed by email server for MessageBus
    messages_dir = os.environ.get("TEAM_MESSAGES_DIR", "")
    if messages_dir:
        env["TEAM_MESSAGES_DIR"] = messages_dir

    # NOTE: TEAM_ROLES_CONFIG is NOT propagated here because it's a large
    # JSON blob that breaks YAML string concatenation in isolated_runner.
    # MCP servers inherit it from process environment instead.

    return env if env else None


def get_skills(skills_dir: str | Path, *names: str):
    """Load SkillManifest objects for specific skills by name.

    Shared helper used by both static agents (agent.py) and spawned
    team agents (isolated_runner.py).
    """
    from fast_agent.skills.registry import SkillRegistry

    manifests = []
    sdir = Path(skills_dir)
    for name in names:
        skill_md = sdir / name / "SKILL.md"
        if skill_md.exists():
            manifest, error = SkillRegistry._parse_manifest(skill_md)
            if manifest:
                manifests.append(manifest)
            elif error:
                logger.warning("Failed to parse skill %s: %s", skill_md, error)
        else:
            logger.warning("Skill '%s' not found at %s", name, sdir / name)
    return manifests


def get_default_model(project_dir: str | Path) -> str:
    """Get default model from config.

    Checks the given project_dir first, then SPAWN_PROJECT_DIR (parent config),
    then falls back to gpt-5-mini.
    """
    import os

    config = _load_config(project_dir)
    model = config.get("default_model")
    if model:
        return model

    # If running as a spawned process, check parent project config
    spawn_dir = os.environ.get("SPAWN_PROJECT_DIR")
    if spawn_dir and str(Path(spawn_dir).resolve()) != str(Path(project_dir).resolve()):
        parent_config = _load_config(spawn_dir)
        model = parent_config.get("default_model")
        if model:
            return model

    return "gpt-5-mini"
