"""
Core interfaces and decorators for fast-agent.

Public API:
- `Core`: The core application container
- `AgentApp`: Container for interacting with agents
- `FastAgent`: High-level, decorator-driven application class
- Decorators: `agent`, `custom`, `orchestrator`, `iterative_planner`,
  `router`, `chain`, `parallel`, `evaluator_optimizer`

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
    elif name in (
        "agent",
        "custom",
        "orchestrator",
        "iterative_planner",
        "router",
        "chain",
        "parallel",
        "evaluator_optimizer",
    ):
        from . import direct_decorators as _dd

        return getattr(
            _dd,
            name,
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .agent_app import AgentApp as AgentApp  # noqa: F401
    from .core_app import Core as Core  # noqa: F401
    from .fastagent import FastAgent as FastAgent  # noqa: F401
    from .direct_decorators import (  # noqa: F401
        agent as agent,
        chain as chain,
        custom as custom,
        evaluator_optimizer as evaluator_optimizer,
        iterative_planner as iterative_planner,
        orchestrator as orchestrator,
        parallel as parallel,
        router as router,
    )


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
