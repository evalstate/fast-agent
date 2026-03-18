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


def get_server_commands(
    project_dir: str | Path,
    workspace_dir: str | None = None,
) -> dict[str, str]:
    """Get MCP server name → command mapping from config.

    Args:
        project_dir: Root directory containing fastagent.config.yaml,
            pyproject.toml, and server scripts.
        workspace_dir: Used for data-access paths (e.g. filesystem server's
            root). Defaults to ``project_dir``.
    """
    config = _load_config(project_dir)
    servers = config.get("mcp", {}).get("servers", {})
    workspace = workspace_dir or str(project_dir)
    project = str(Path(project_dir).resolve())

    # Detect if this is a team workspace (inside .runtime/data/workspaces/)
    is_team_workspace = workspace_dir and "/workspaces/" in workspace_dir

    commands: dict[str, str] = {}
    if not isinstance(servers, dict):
        return commands

    for name, srv_config in servers.items():
        if not isinstance(srv_config, dict):
            continue

        # Build command string from command + args
        cmd = srv_config.get("command", "")
        args = srv_config.get("args", [])
        if cmd and args:
            resolved = f"{cmd} {' '.join(str(a) for a in args)}"
        elif cmd:
            resolved = cmd
        elif srv_config.get("url"):
            # SSE/URL-based servers — store the URL
            commands[name] = srv_config["url"]
            continue
        else:
            continue

        # For team workspaces, filesystem server should serve just the
        # workspace root — not project-specific subdirs like ./data
        if is_team_workspace and name == "filesystem":
            # Extract base command (everything before the path arguments)
            # e.g. "npx -y @modelcontextprotocol/server-filesystem ./data ./ws"
            #   -> "npx -y @modelcontextprotocol/server-filesystem {workspace}"
            parts = resolved.split()
            base_parts = []
            for p in parts:
                if p.startswith(".") or p.startswith("/"):
                    break  # Stop at first path argument
                base_parts.append(p)
            resolved = " ".join(base_parts) + f" {workspace}"
        else:
            # Replace relative path "." with workspace dir
            if " ." in resolved and " .." not in resolved:
                resolved = resolved.replace(" .", f" {workspace}")

        # Replace relative "servers/" paths with absolute PROJECT paths
        if " servers/" in resolved:
            resolved = resolved.replace(" servers/", f" {project}/servers/")

        # Add --directory for uv run commands pointing to PROJECT root
        if resolved.startswith("uv run ") and "--directory" not in resolved:
            resolved = resolved.replace("uv run ", f"uv run --directory {project} ", 1)

        commands[name] = resolved

    return commands


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


def get_default_model(project_dir: str | Path) -> str:
    """Get default model from config."""
    config = _load_config(project_dir)
    return config.get("default_model", "gpt-4o-mini")
