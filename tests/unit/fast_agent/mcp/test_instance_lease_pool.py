from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.agent_instance_factory import CallableAgentInstanceFactory
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.server.instance_lease_pool import ScopedAgentInstancePool

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.server.instance_lease_pool import InstanceScopeValue


class _Agent:
    name = "worker"

    async def shutdown(self) -> None:
        return


class _Factory:
    def __init__(self) -> None:
        self.created: list[AgentInstance] = []
        self.disposed: list[AgentInstance] = []

    async def create(self) -> AgentInstance:
        agent = cast("AgentProtocol", _Agent())
        instance = AgentInstance(AgentApp({"worker": agent}), {"worker": agent})
        self.created.append(instance)
        return instance

    async def dispose(self, instance: AgentInstance) -> None:
        self.disposed.append(instance)
        await instance.shutdown()


class _Session:
    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()


class _Ctx:
    def __init__(self, session: _Session) -> None:
        self.session = session


def _pool(
    factory: _Factory,
    *,
    scope: "InstanceScopeValue",
    primary: AgentInstance,
    get_registry_version=None,
    registered: list[AgentInstance] | None = None,
) -> ScopedAgentInstancePool:
    if registered is None:
        registered = []
    return ScopedAgentInstancePool(
        primary_instance=primary,
        instance_factory=CallableAgentInstanceFactory(
            create=factory.create,
            dispose=factory.dispose,
        ),
        instance_scope=scope,
        get_registry_version=get_registry_version,
        register_missing_agents=registered.append,
    )


@pytest.mark.asyncio
async def test_request_scope_disposes_each_released_instance() -> None:
    factory = _Factory()
    primary = await factory.create()
    pool = _pool(factory, scope="request", primary=primary)

    first = await pool.acquire()
    second = await pool.acquire()
    await pool.release(first)
    await pool.release(second)

    assert first.instance is not second.instance
    assert factory.disposed == [first.instance, second.instance]


@pytest.mark.asyncio
async def test_connection_scope_reuses_until_session_cleanup() -> None:
    factory = _Factory()
    primary = await factory.create()
    pool = _pool(factory, scope="connection", primary=primary)
    session = _Session()
    await session._exit_stack.__aenter__()
    ctx = _Ctx(session)

    first = await pool.acquire(ctx)
    second = await pool.acquire(ctx)

    assert second.instance is first.instance
    assert factory.disposed == []

    await session._exit_stack.aclose()

    assert factory.disposed == [first.instance]


@pytest.mark.asyncio
async def test_shared_scope_refresh_defers_stale_disposal_until_release() -> None:
    factory = _Factory()
    primary = await factory.create()
    version = 1
    primary.registry_version = version
    registered: list[AgentInstance] = []

    def get_registry_version() -> int:
        return version

    pool = _pool(
        factory,
        scope="shared",
        primary=primary,
        get_registry_version=get_registry_version,
        registered=registered,
    )
    lease = await pool.acquire()
    version = 2

    await pool.maybe_refresh_shared_instance()

    assert pool.primary_instance is not primary
    assert factory.disposed == []
    assert registered == [pool.primary_instance]

    await pool.release(lease)

    assert factory.disposed == [primary]


@pytest.mark.asyncio
async def test_shutdown_disposes_connection_and_primary_instances() -> None:
    factory = _Factory()
    primary = await factory.create()
    pool = _pool(factory, scope="connection", primary=primary)
    session = _Session()
    ctx = _Ctx(session)

    lease = await pool.acquire(ctx)

    await pool.shutdown()

    assert factory.disposed == [lease.instance, primary]


@pytest.mark.asyncio
async def test_shutdown_disposes_stale_instances() -> None:
    factory = _Factory()
    primary = await factory.create()
    version = 1
    primary.registry_version = version

    def get_registry_version() -> int:
        return version

    pool = _pool(factory, scope="shared", primary=primary)
    pool._get_registry_version = get_registry_version

    version = 2
    await pool.maybe_refresh_shared_instance()
    refreshed = pool.primary_instance

    await pool.shutdown()

    assert factory.disposed == [refreshed, primary]
