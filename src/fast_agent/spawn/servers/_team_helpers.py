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


def assert_self_identity(claimed_name: str) -> tuple[str, str | None]:
    """Resolve caller identity + refuse impersonation across team servers.

    Single source of truth for the "is the caller the agent they claim to
    be" check. Used by every team-server tool that accepts a caller-
    supplied name (meeting_room: speak/skip_turn/create_meeting/
    leave_meeting; team_communicate: team_communicate/check_responses/
    reply_message). All of them previously did
    ``name = name or get_my_name()`` — i.e. trusted the caller's claim
    blindly. That let a spawn-misconfigured process pass
    ``name="<teammate>"`` and write transcript / messaging entries
    falsely attributed to the impersonated agent (production 2026-05-20:
    Taylor [PM] force-skipped 6 teammates by impersonating each one's
    turn).

    Authoritative identity is ``$TEAM_MY_NAME`` — set by the spawner at
    process-creation time, immutable for the process's lifetime. Read
    DIRECTLY (not via ``get_my_name()``) because that helper falls back
    to ``$TEAM_MY_ROLE`` and then a generic "agent" string; only
    ``TEAM_MY_NAME`` proves the process was spawned as a specific team
    member.

    Contract — no permissive escape hatch:

    1. ``claimed_name=""`` → auto-detect, fall back to ``get_my_name()``.
       No claim made, no impersonation possible.
    2. ``claimed_name`` set + ``TEAM_MY_NAME`` set + match → allow.
    3. ``claimed_name`` set + ``TEAM_MY_NAME`` set + mismatch → REFUSE.
    4. ``claimed_name`` set + ``TEAM_MY_NAME`` UNSET → REFUSE.

    Returns ``(resolved_name, error_json)`` — exactly one is non-empty.
    Callers short-circuit on ``error_json``.
    """
    caller_env = os.environ.get("TEAM_MY_NAME", "").strip()

    if not claimed_name:
        return get_my_name(), None

    if not caller_env:
        # Refusals here are the CORRECT response to an unverified claim,
        # not a server error — use warning level so they don't flood
        # error dashboards once misconfigured CLI / library callers
        # start hitting the new strict contract.
        logger.warning(
            "team server: refusing claim claimed_name=%r — TEAM_MY_NAME is "
            "unset on this process so identity cannot be verified. Set "
            "TEAM_MY_NAME before calling team tools with an explicit name.",
            claimed_name,
        )
        return "", json.dumps({
            "error": (
                "Identity unverifiable: TEAM_MY_NAME is not set on this "
                "process, so a claimed agent name cannot be verified. "
                "Either omit the name argument (auto-detect from env "
                "role) or set TEAM_MY_NAME to your authoritative identity."
            ),
            "caller_env": "",
            "claimed_agent_name": claimed_name,
        })

    if claimed_name.strip().lower() != caller_env.lower():
        logger.warning(
            "team server: refusing impersonation — caller is %r but claim is %r",
            caller_env, claimed_name,
        )
        return "", json.dumps({
            "error": (
                f"Impersonation refused: caller is {caller_env!r} but "
                f"claimed_name={claimed_name!r}. You can only act on "
                f"behalf of YOUR own identity. If a teammate is "
                f"unresponsive, use the team's escalation path — do not "
                f"put words in their mouth."
            ),
            "caller_env": caller_env,
            "claimed_agent_name": claimed_name,
        })

    return claimed_name, None


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
    """Get the PROJECT-level spawn registry via SPAWN_REGISTRY_DB (SQLite).

    Single source of truth: the SQLite database at SPAWN_REGISTRY_DB.
    This env var is propagated to all MCP server subprocesses via config_reader.
    """
    from fast_agent.spawn.spawn_registry import SpawnRegistry

    db_path = os.environ.get("SPAWN_REGISTRY_DB", "")
    if db_path:
        try:
            return SpawnRegistry(db_path)
        except Exception as e:
            logger.warning("Failed to open SQLite registry at %s: %s", db_path, e)

    logger.warning(
        "No spawn registry found. SPAWN_REGISTRY_DB=%s",
        os.environ.get("SPAWN_REGISTRY_DB", "(unset)"),
    )
    return None


def auto_wake_if_idle(agent_name: str) -> None:
    """Wake an agent — probe liveness ONCE, dispatch to the right strategy.

    Single source of truth for liveness: ``AgentChannel.is_alive`` does a
    non-blocking connect probe on the agent's channel socket. Whatever it
    returns determines the strategy:

      - alive → send a ``wake`` signal via the same channel (fast, in-process)
      - dead  → respawn the agent from its snapshot via the spawner

    No "fall back" semantics:
      - We decide UPFRONT which strategy applies based on one authoritative
        probe — not "try X, if it fails try Y" (which hides the real
        decision behind a sequence of partial failures).
      - Each strategy fails loud on its own terms; a strategy never
        silently substitutes the other.

    Why the explicit probe-then-dispatch matters:
      - ``send_signal`` returning False is ambiguous (sock missing
        vs. connect refused vs. write error). The previous code treated
        all three as "try Tier 2 spawner", which masked the rare race
        where the process is *alive but mid-shutdown* — Tier 2 then had
        to re-probe ``is_alive`` to avoid duplicate spawn.
      - With a single upfront probe, the race window collapses to one
        moment and the decision is unambiguous.
      - ``record.status`` from spawn_registry is NOT used here as a
        liveness signal — it's bridge-updated and can lag (the 1h2m
        Sasha hang regression: bridge socket died → status frozen at
        ``running`` → old code refused to wake). Channel sock is the
        truth about the process.
    """
    from fast_agent.spawn.agent_channel import AgentChannel

    try:
        alive = AgentChannel.is_alive(agent_name)
    except Exception as e:
        logger.error(
            "[AUTO-WAKE] %s: liveness probe raised %s — cannot decide wake "
            "strategy, refusing to act (would risk duplicate spawn).",
            agent_name, e, exc_info=True,
        )
        return

    if alive:
        _wake_alive_agent(agent_name)
    else:
        _respawn_dead_agent(agent_name)


def _wake_alive_agent(agent_name: str) -> None:
    """Strategy A: agent process is alive — send a wake signal via the
    AgentChannel socket. The agent's listener picks up the signal and
    triggers an LLM iteration which will check its inbox.
    """
    from fast_agent.spawn.agent_channel import AgentChannel

    try:
        delivered = AgentChannel.send_signal(agent_name, "wake")
    except Exception as e:
        logger.error(
            "[AUTO-WAKE] %s: liveness said alive, but send_signal raised %s. "
            "Agent will NOT pick up its inbox until another wake trigger. "
            "(Likely socket corruption or process state mismatch.)",
            agent_name, e, exc_info=True,
        )
        return

    if delivered:
        logger.info("📡 Woke %s via AgentChannel socket", agent_name)
    else:
        # Race: between ``is_alive`` and ``send_signal`` the process
        # exited (or the sock file was unlinked). This is a real
        # failure to wake — surface loudly so we can correlate the next
        # missed inbox to this race instead of treating it as silent.
        logger.warning(
            "[AUTO-WAKE] %s: liveness said alive but send_signal returned "
            "False — process likely exited mid-call. NOT respawning "
            "(would duplicate if process is still terminating). Caller "
            "should retry on the next event if the agent is genuinely "
            "needed.",
            agent_name,
        )


def _respawn_dead_agent(agent_name: str) -> None:
    """Strategy B: agent process is dead — respawn from snapshot.

    Requires registry access + an async event loop + the prior agent's
    saved ``original_config``. Each precondition that fails is loud.
    """
    import asyncio

    registry = get_project_registry()
    if not registry:
        logger.warning(
            "[AUTO-WAKE] Cannot respawn dead agent %s: no registry found "
            "(check SPAWN_REGISTRY_DB env var). Agent will NOT be respawned.",
            agent_name,
        )
        return

    record = registry.find_by_name(agent_name)
    if not record:
        logger.warning(
            "[AUTO-WAKE] Cannot respawn dead agent %s: not found in registry. "
            "Agent will NOT be respawned. (Did the agent ever spawn? "
            "Was the registry cleared?)",
            agent_name,
        )
        return

    if registry.has_running_resume(agent_name):
        logger.info(
            "[AUTO-WAKE] Skip respawn for %s: already has running resume "
            "in registry — another wake triggered the same path moments "
            "ago, letting that one complete.",
            agent_name,
        )
        return

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        logger.warning(
            "[AUTO-WAKE] Cannot respawn dead agent %s: event loop error: %s. "
            "Agent will NOT be respawned.",
            agent_name, e,
        )
        return

    if not loop.is_running():
        logger.warning(
            "[AUTO-WAKE] Cannot respawn dead agent %s: event loop not "
            "running. Agent will NOT be respawned. (Caller is sync-context "
            "without a live asyncio loop — must run within an async "
            "context to use the spawner.)",
            agent_name,
        )
        return

    from fast_agent.spawn.isolated_spawner import _check_and_resume_on_inbox

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
    logger.info(
        "📬 Respawning dead agent %s via spawner (prev run_id=%s)",
        agent_name, record.run_id,
    )

