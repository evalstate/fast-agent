"""Shared helpers for team MCP servers (meeting_room, email).

Provides: message bus, agent name resolution, auto-wake, team config.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_bus():  # type: ignore[no-untyped-def]
    """Get MessageBus from TEAM_MESSAGES_DIR or TEAM_WORKSPACE env var."""
    from fast_agent.spawn.message_bus import MessageBus

    # Prefer session-scoped messages dir
    messages_dir = os.environ.get("TEAM_MESSAGES_DIR", "")
    if messages_dir:
        Path(messages_dir).mkdir(parents=True, exist_ok=True)
        return MessageBus(messages_dir=messages_dir)

    workspace = os.environ.get("TEAM_WORKSPACE", "")
    if not workspace:
        return None
    cur = Path(workspace)
    while cur != cur.parent:
        if cur.name == ".runtime":
            state_dir = cur / "state" / "messages"
            state_dir.mkdir(parents=True, exist_ok=True)
            return MessageBus(messages_dir=str(state_dir))
        cur = cur.parent
    return None


def get_my_name() -> str:
    """Get current agent's name from env."""
    return os.environ.get("TEAM_MY_NAME", os.environ.get("TEAM_MY_ROLE", "agent"))


def get_team_config() -> dict:
    """Load team roles config from env."""
    try:
        return json.loads(os.environ.get("TEAM_ROLES_CONFIG", "{}"))
    except json.JSONDecodeError:
        return {}


def resolve_agent_name(name: str) -> str | None:
    """Resolve target agent name — supports both name and role key lookup."""
    team_config = get_team_config()
    for _role_key, cfg in team_config.items():
        if isinstance(cfg, dict) and cfg.get("agent_name") == name:
            return name
    if name in team_config:
        cfg = team_config[name]
        if isinstance(cfg, dict):
            return cfg.get("agent_name", name)
    return None


def parse_recipients(value: str) -> list[str]:
    """Parse comma-separated recipients. 'all' returns all team members."""
    if not value:
        return []
    if value.strip().lower() == "all":
        team_config = get_team_config()
        my_name = get_my_name()
        return [
            cfg.get("agent_name", role)
            for role, cfg in team_config.items()
            if isinstance(cfg, dict) and cfg.get("agent_name") != my_name
        ]
    return [n.strip() for n in value.split(",") if n.strip()]


def get_project_registry() -> "SpawnRegistry | None":
    """Get the PROJECT-level spawn registry (not workspace-level).

    Child processes (like PM) have TEAM_WORKSPACE pointing to their workspace
    inside .runtime/data/workspaces/. Walking up from there finds the
    workspace's own .runtime — which has an EMPTY registry.

    The correct registry is at SPAWN_PROJECT_DIR/.runtime/state/spawn_registry.json.
    """
    from fast_agent.spawn.spawn_registry import SpawnRegistry

    # Priority 1: SPAWN_PROJECT_DIR (always correct for spawned agents)
    project_dir = os.environ.get("SPAWN_PROJECT_DIR", "")
    if project_dir:
        path = Path(project_dir) / ".runtime" / "state" / "spawn_registry.json"
        if path.exists():
            return SpawnRegistry(str(path))
        logger.debug(
            "Project registry not found at %s (SPAWN_PROJECT_DIR=%s)",
            path, project_dir,
        )

    # Fallback: workspace-level registry (for non-team agents)
    workspace = os.environ.get("TEAM_WORKSPACE", "")
    if workspace:
        path = Path(workspace) / ".runtime" / "state" / "spawn_registry.json"
        if path.exists():
            return SpawnRegistry(str(path))
        logger.debug(
            "Workspace registry not found at %s (TEAM_WORKSPACE=%s)",
            path, workspace,
        )

    logger.warning(
        "No spawn registry found. SPAWN_PROJECT_DIR=%s, TEAM_WORKSPACE=%s",
        os.environ.get("SPAWN_PROJECT_DIR", "(unset)"),
        os.environ.get("TEAM_WORKSPACE", "(unset)"),
    )
    return None


def auto_wake_if_idle(agent_name: str) -> None:
    """Auto-wake an idle agent via AgentChannel socket signal.

    Priority order:
    1. AgentChannel socket signal (instant, zero-overhead — agent alive)
    2. Spawner-based resume (expensive — agent process died)
    """
    # Try AgentChannel first (agent process still alive)
    try:
        from fast_agent.spawn.agent_channel import AgentChannel

        if AgentChannel.send_signal(agent_name, "wake"):
            logger.info("📡 Woke %s via AgentChannel socket", agent_name)
            return
    except Exception:
        pass

    # Fallback: spawner-based resume (process died)
    try:
        registry = get_project_registry()
        if not registry:
            logger.warning(
                "Cannot auto-wake %s: no registry found", agent_name
            )
            return

        record = registry.find_by_name(agent_name)

        if not record:
            logger.warning(
                "Cannot auto-wake %s: not found in registry", agent_name
            )
            return

        if record.status != "idle":
            logger.warning(
                "Skip auto-wake %s: status=%s (not idle)",
                agent_name, record.status,
            )
            return

        if registry.has_running_resume(agent_name):
            logger.debug(
                "Skip auto-wake %s: already has running resume", agent_name
            )
            return

        import asyncio
        from fast_agent.spawn.isolated_spawner import _check_and_resume_on_inbox

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    _check_and_resume_on_inbox(
                        run_id=record.run_id,
                        agent_name=agent_name,
                        registry=registry,
                        display_manager=None,
                        env_vars=(
                            record.original_config.get("env_vars")
                            if record.original_config
                            else None
                        ),
                    )
                )
                logger.info("📬 Auto-waking idle agent %s via spawner (run_id=%s)", agent_name, record.run_id)
            else:
                logger.warning(
                    "Cannot auto-wake %s: event loop not running", agent_name
                )
        except RuntimeError as e:
            logger.warning(
                "Cannot auto-wake %s: event loop error: %s", agent_name, e
            )
    except Exception as e:
        logger.error("Auto-wake failed for %s: %s", agent_name, e, exc_info=True)

