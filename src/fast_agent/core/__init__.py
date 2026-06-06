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

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentApp": (".agent_app", "AgentApp"),
    "Core": (".core_app", "Core"),
    "FastAgent": (".fastagent", "FastAgent"),
    "DecoratorMixin": (".direct_decorators", "DecoratorMixin"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    from importlib import import_module

    return getattr(import_module(module_name, __name__), attr_name)


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .agent_app import AgentApp as AgentApp
    from .core_app import Core as Core
    from .direct_decorators import DecoratorMixin as DecoratorMixin
    from .fastagent import FastAgent as FastAgent


__all__ = [
    "AgentApp",
    "Core",
    "DecoratorMixin",
    "FastAgent",
]
