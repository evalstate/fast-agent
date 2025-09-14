"""
Core interfaces and decorators for fast-agent.

Public API:
- `Core`: The core application container (eagerly exported)
- `FastAgent`: High-level, decorator-driven application class (lazy-loaded)
- Decorators: `agent`, `custom`, `orchestrator`, `iterative_planner`,
  `router`, `chain`, `parallel`, `evaluator_optimizer` (lazy-loaded)
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from .core_app import Core  # Eager export for external applications

__all__ = [
    "Core",
    "AgentApp",
    "FastAgent",
    # Decorators
    "agent",
    "custom",
    "orchestrator",
    "iterative_planner",
    "router",
    "chain",
    "parallel",
    "evaluator_optimizer",
]


def __getattr__(name: str):
    # Lazy imports to avoid heavy dependencies and circular imports at init time
    if name == "FastAgent":
        from .fastagent import FastAgent

        return FastAgent
    if name == "AgentApp":
        from .agent_app import AgentApp

        return AgentApp

    # Decorators from direct_decorators
    if name in {
        "agent",
        "custom",
        "orchestrator",
        "iterative_planner",
        "router",
        "chain",
        "parallel",
        "evaluator_optimizer",
    }:
        from . import direct_decorators as _dd

        return getattr(_dd, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Help static analyzers/IDEs resolve symbols and signatures without importing at runtime.
if _TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .agent_app import AgentApp as AgentApp  # noqa: F401
    from .direct_decorators import (
        agent as agent,
    )  # noqa: F401
    from .direct_decorators import (
        chain as chain,
    )
    from .direct_decorators import (
        custom as custom,
    )
    from .direct_decorators import (
        evaluator_optimizer as evaluator_optimizer,
    )
    from .direct_decorators import (
        iterative_planner as iterative_planner,
    )
    from .direct_decorators import (
        orchestrator as orchestrator,
    )
    from .direct_decorators import (
        parallel as parallel,
    )
    from .direct_decorators import (
        router as router,
    )
    from .fastagent import FastAgent as FastAgent  # noqa: F401


def __dir__():  # pragma: no cover - developer experience aid
    return sorted(__all__)
