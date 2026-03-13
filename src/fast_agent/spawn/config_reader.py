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
    """Get list of all available MCP server names from config."""
    config = _load_config(project_dir)
    targets = config.get("mcp", {}).get("targets", [])
    return [t["name"] for t in targets if "name" in t]


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
    targets = config.get("mcp", {}).get("targets", [])
    workspace = workspace_dir or str(project_dir)
    project = str(Path(project_dir).resolve())

    commands: dict[str, str] = {}
    for t in targets:
        name = t.get("name", "")
        target = t.get("target", "")
        if not (name and target):
            continue

        resolved = target

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

_WORKSPACE_SERVERS = {"meeting_room", "team_communicate"}


def get_server_env(
    server_name: str,
    workspace_dir: str | None = None,
    agent_name: str | None = None,
) -> dict[str, str] | None:
    """Get extra environment variables needed by a specific MCP server.

    Returns dict of env vars, or None if no extra env is needed.
    """
    if server_name in _WORKSPACE_SERVERS and workspace_dir:
        env: dict[str, str] = {"TEAM_WORKSPACE": workspace_dir}
        if agent_name:
            env["TEAM_MY_NAME"] = agent_name
        return env
    return None


def get_default_model(project_dir: str | Path) -> str:
    """Get default model from config."""
    config = _load_config(project_dir)
    return config.get("default_model", "gpt-4o-mini")
