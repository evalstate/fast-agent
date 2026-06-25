"""MCP server agent instance lease pool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.core.agent_instance_factory import AgentInstanceFactory
    from fast_agent.core.fastagent import AgentInstance

logger = get_logger(__name__)

InstanceScopeValue = Literal["shared", "request", "connection"]


class InstanceScope(StrEnum):
    SHARED = "shared"
    REQUEST = "request"
    CONNECTION = "connection"


@dataclass(slots=True)
class AgentInstanceLease:
    instance: AgentInstance


class AgentInstanceLeasePool(Protocol):
    async def acquire(self, ctx: object | None = None) -> AgentInstanceLease: ...

    async def release(self, lease: AgentInstanceLease, ctx: object | None = None) -> None: ...

    async def shutdown(self) -> None: ...


class ScopedAgentInstancePool:
    """Manage MCP server instances for shared, request, and connection scopes."""

    def __init__(
        self,
        *,
        primary_instance: AgentInstance,
        instance_factory: AgentInstanceFactory,
        instance_scope: InstanceScopeValue,
        get_registry_version: Callable[[], int] | None = None,
        register_missing_agents: Callable[[AgentInstance], None],
    ) -> None:
        self.primary_instance = primary_instance
        self.instance_scope = InstanceScope(instance_scope)
        self._instance_factory = instance_factory
        self._get_registry_version = get_registry_version
        self._register_missing_agents = register_missing_agents
        self._primary_registry_version = getattr(primary_instance, "registry_version", 0)
        self._shared_instance_lock = asyncio.Lock()
        self._shared_active_requests = 0
        self._shared_instance_active = True
        self._stale_instances: list[AgentInstance] = []
        self._connection_instances: dict[int, AgentInstance] = {}
        self._connection_cleanup_tasks: dict[int, Callable[[], Awaitable[None]]] = {}
        self._connection_lock = asyncio.Lock()

    async def acquire(self, ctx: object | None = None) -> AgentInstanceLease:
        if self.instance_scope is InstanceScope.SHARED:
            await self.maybe_refresh_shared_instance()
            self._shared_active_requests += 1
            return AgentInstanceLease(self.primary_instance)

        if self.instance_scope is InstanceScope.REQUEST:
            return AgentInstanceLease(await self._instance_factory.create_instance())

        if ctx is None:
            raise AssertionError("Context is required for connection-scoped instances")
        session_key = self.connection_key(ctx)
        async with self._connection_lock:
            instance = self._connection_instances.get(session_key)
            if instance is None:
                instance = await self._instance_factory.create_instance()
                self._connection_instances[session_key] = instance
                self.register_session_cleanup(ctx, session_key)
            return AgentInstanceLease(instance)

    async def release(self, lease: AgentInstanceLease, ctx: object | None = None) -> None:
        del ctx
        if self.instance_scope is InstanceScope.SHARED:
            if self._shared_active_requests > 0:
                self._shared_active_requests -= 1
            await self.dispose_stale_instances_if_idle()
            return
        if self.instance_scope is InstanceScope.REQUEST:
            await self._instance_factory.dispose_instance(lease.instance)

    @staticmethod
    def connection_key(ctx: object) -> int:
        return id(_ctx_session(ctx))

    def register_session_cleanup(self, ctx: object, session_key: int) -> None:
        async def cleanup() -> None:
            instance = self._connection_instances.pop(session_key, None)
            if instance is not None:
                await self._instance_factory.dispose_instance(instance)

        session = _ctx_session(ctx)
        exit_stack = getattr(session, "_exit_stack", None)
        if exit_stack is not None:
            exit_stack.push_async_callback(cleanup)
            return
        self._connection_cleanup_tasks[session_key] = cleanup

    async def maybe_refresh_shared_instance(self) -> None:
        if not self._get_registry_version:
            return
        latest_version = self._get_registry_version()
        if latest_version <= self._primary_registry_version:
            return

        async with self._shared_instance_lock:
            latest_version = self._get_registry_version()
            if latest_version <= self._primary_registry_version:
                return

            new_instance = await self._instance_factory.create_instance()
            old_instance = self.primary_instance
            self.primary_instance = new_instance
            self._primary_registry_version = getattr(
                new_instance, "registry_version", latest_version
            )
            self._stale_instances.append(old_instance)
            self._register_missing_agents(new_instance)

    async def dispose_stale_instances_if_idle(self) -> None:
        if self._shared_active_requests or not self._stale_instances:
            return
        stale = list(self._stale_instances)
        self._stale_instances.clear()
        for instance in stale:
            await self.dispose_instance_safely(instance, phase="shared stale instance cleanup")

    async def dispose_primary_instance(self) -> None:
        if not self._shared_instance_active:
            return
        try:
            await self.dispose_instance_safely(
                self.primary_instance,
                phase="primary instance cleanup",
            )
        finally:
            self._shared_instance_active = False

    async def dispose_all_stale_instances(self) -> None:
        if not self._stale_instances:
            return
        stale = list(self._stale_instances)
        self._stale_instances.clear()
        for instance in stale:
            await self.dispose_instance_safely(
                instance,
                phase="stale instance cleanup",
            )

    async def dispose_all_connection_instances(self) -> None:
        pending_cleanups = list(self._connection_cleanup_tasks.values())
        self._connection_cleanup_tasks.clear()
        for cleanup in pending_cleanups:
            try:
                await cleanup()
            except Exception:
                logger.exception("Connection cleanup callback failed during shutdown")

        async with self._connection_lock:
            instances = list(self._connection_instances.values())
            self._connection_instances.clear()

        for instance in instances:
            await self.dispose_instance_safely(instance, phase="connection instance cleanup")

    async def dispose_instance_safely(self, instance: AgentInstance, *, phase: str) -> None:
        try:
            await self._instance_factory.dispose_instance(instance)
        except Exception:
            logger.exception("Agent instance disposal failed during %s", phase)

    async def shutdown(self) -> None:
        await self.dispose_all_connection_instances()
        await self.dispose_primary_instance()
        await self.dispose_all_stale_instances()


def _ctx_session(ctx: object) -> Any:
    return getattr(ctx, "session")


__all__ = [
    "AgentInstanceLease",
    "AgentInstanceLeasePool",
    "InstanceScope",
    "InstanceScopeValue",
    "ScopedAgentInstancePool",
]
