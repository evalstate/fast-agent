"""Spawn Lifecycle Hooks — extension point for spawn lifecycle events.

Provides the ``SpawnLifecycleHooks`` protocol so callers can observe
and react to each phase of the isolated agent lifecycle *without*
modifying core spawner code.

All methods are optional — implement only what you need.  A concrete
no-op base class ``NoOpSpawnLifecycleHooks`` is provided for easy
subclassing.

Usage inside isolated_spawner:
    hooks = spawn_lifecycle_hooks or NoOpSpawnLifecycleHooks()
    await hooks.on_pre_spawn(run_id, agent_name, config)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SpawnLifecycleHooks(Protocol):
    """Extension point for spawn lifecycle events.

    Called by ``isolated_spawner.py`` at each lifecycle phase.
    All methods receive at minimum ``run_id`` and ``agent_name``.
    """

    # ── Phase 1: Pre-spawn ────────────────────────────────────────────
    async def on_pre_spawn(
        self, run_id: str, agent_name: str, config: dict[str, Any]
    ) -> None:
        """Called after config is built but BEFORE registry registration.

        Use cases: resource validation, rate limiting, quota checks,
        dashboard "preparing..." notification.
        """
        ...

    async def on_registered(
        self, run_id: str, agent_name: str, record: Any
    ) -> None:
        """Called AFTER the agent is registered in SpawnRegistry.

        Use cases: SQLite sync, ``agent_added`` SSE broadcast,
        team roster update.

        ``record`` is a ``SpawnRecord`` instance.
        """
        ...

    # ── Phase 2-3: Subprocess lifecycle ───────────────────────────────
    async def on_process_started(
        self, run_id: str, agent_name: str, pid: int
    ) -> None:
        """Called when the subprocess has been created (PID available).

        Use cases: PID tracking, resource monitoring.
        """
        ...

    async def on_agent_ready(self, run_id: str, agent_name: str) -> None:
        """Called when the child agent signals readiness (MCP connected).

        Use cases: "Ready" status on dashboard, meeting room availability.
        """
        ...

    # ── Phase 5-6: Completion ─────────────────────────────────────────
    async def on_completed(
        self, run_id: str, agent_name: str, result: dict[str, Any]
    ) -> None:
        """Called when the agent completes its task successfully.

        Use cases: idle transition broadcast, notification, token accounting.
        """
        ...

    async def on_error(
        self, run_id: str, agent_name: str, error: str
    ) -> None:
        """Called when the agent encounters an error.

        Use cases: error broadcast, alerting, retry decision.
        """
        ...

    async def on_idle(self, run_id: str, agent_name: str) -> None:
        """Called when a resumable agent enters keep-alive/idle mode.

        Use cases: dashboard idle transition.
        """
        ...

    # ── Phase 7: Cleanup & resume ─────────────────────────────────────
    async def on_pre_cleanup(
        self, run_id: str, agent_name: str, lifecycle: str
    ) -> None:
        """Called BEFORE the agent is removed from registry (oneshot).

        The agent record is still queryable at this point.

        Use cases: broadcast ``agent_removed`` SSE with full metadata,
        archive session history, record metrics.
        """
        ...

    async def on_after_cleanup(
        self, run_id: str, agent_name: str, lifecycle: str
    ) -> None:
        """Called AFTER the agent is removed from registry.

        The agent record is no longer in the registry.

        Use cases: final audit log, update team roster (remove from
        active list), trigger dependent workflows, release external
        resources.
        """
        ...

    async def on_auto_resume(
        self,
        run_id: str,
        agent_name: str,
        new_run_id: str,
        reason: str,
    ) -> None:
        """Called when an agent is auto-resumed (e.g. inbox messages).

        Use cases: dashboard "auto-resumed" status, chain tracking,
        rate limiting resume storms.
        """
        ...

    async def on_cancelled(self, run_id: str, agent_name: str) -> None:
        """Called when a spawn is cancelled/terminated.

        Use cases: UI cleanup, resource release, team notification.
        """
        ...


class NoOpSpawnLifecycleHooks:
    """Default no-op implementation — all hooks are silent."""

    async def on_pre_spawn(
        self, run_id: str, agent_name: str, config: dict[str, Any]
    ) -> None:
        pass

    async def on_registered(
        self, run_id: str, agent_name: str, record: Any
    ) -> None:
        pass

    async def on_process_started(
        self, run_id: str, agent_name: str, pid: int
    ) -> None:
        pass

    async def on_agent_ready(self, run_id: str, agent_name: str) -> None:
        pass

    async def on_completed(
        self, run_id: str, agent_name: str, result: dict[str, Any]
    ) -> None:
        pass

    async def on_error(
        self, run_id: str, agent_name: str, error: str
    ) -> None:
        pass

    async def on_idle(self, run_id: str, agent_name: str) -> None:
        pass

    async def on_pre_cleanup(
        self, run_id: str, agent_name: str, lifecycle: str
    ) -> None:
        pass

    async def on_after_cleanup(
        self, run_id: str, agent_name: str, lifecycle: str
    ) -> None:
        pass

    async def on_auto_resume(
        self,
        run_id: str,
        agent_name: str,
        new_run_id: str,
        reason: str,
    ) -> None:
        pass

    async def on_cancelled(self, run_id: str, agent_name: str) -> None:
        pass


async def _safe_hook(coro: Any, hook_name: str, run_id: str) -> None:
    """Call a hook coroutine with error isolation.

    Prevents hook failures from breaking the spawn lifecycle.
    """
    try:
        await coro
    except Exception:
        logger.warning(
            "SpawnLifecycleHook '%s' failed for run_id=%s",
            hook_name,
            run_id,
            exc_info=True,
        )
