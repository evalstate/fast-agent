"""Shared agent instance lifecycle boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.core.fastagent import AgentInstance


class AgentInstanceFactory(Protocol):
    """Create and dispose owned ``AgentInstance`` objects."""

    async def create_instance(self) -> AgentInstance: ...

    async def dispose_instance(self, instance: AgentInstance) -> None: ...


class ReloadableAgentInstanceFactory(AgentInstanceFactory, Protocol):
    """Agent instance factory that can reload its backing source."""

    async def reload_source(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class CallableAgentInstanceFactory:
    """Adapt existing create/dispose callbacks to ``AgentInstanceFactory``."""

    create: Callable[[], Awaitable[AgentInstance]]
    dispose: Callable[[AgentInstance], Awaitable[None]]

    async def create_instance(self) -> AgentInstance:
        return await self.create()

    async def dispose_instance(self, instance: AgentInstance) -> None:
        await self.dispose(instance)


__all__ = [
    "AgentInstanceFactory",
    "CallableAgentInstanceFactory",
    "ReloadableAgentInstanceFactory",
]
