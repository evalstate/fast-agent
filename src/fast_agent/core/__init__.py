"""
Core interfaces for fast-agent.

Public API:
- `Core`: The core application container
- `AgentApp`: Container for interacting with agents
- `FastAgent`: High-level, decorator-driven application class
- `DecoratorMixin`: Mixin providing decorator methods (@agent, @router, etc.)

Note: Agent decorators are accessed via FastAgent instances, e.g.:
    fast = FastAgent("my-app")
    @fast.agent(name="my-agent")
    async def main(): ...

Exports are resolved lazily to avoid circular imports during package init.
"""

from typing import TYPE_CHECKING


def __getattr__(name: str):
    if name == "AgentApp":
        from .agent_app import AgentApp

        return AgentApp
    elif name == "Core":
        from .core_app import Core

        return Core
    elif name == "FastAgent":
        from .fastagent import FastAgent

        return FastAgent
    elif name == "DecoratorMixin":
        from .direct_decorators import DecoratorMixin

        return DecoratorMixin
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .agent_app import AgentApp as AgentApp  # noqa: F401
    from .core_app import Core as Core  # noqa: F401
    from .direct_decorators import DecoratorMixin as DecoratorMixin  # noqa: F401
    from .fastagent import FastAgent as FastAgent  # noqa: F401


__all__ = [
    "Core",
    "AgentApp",
    "FastAgent",
    "DecoratorMixin",
]
