"""
Core interfaces and decorators for fast-agent.

Public API:
- `Core`: The core application container (lazy-loaded)
- `FastAgent`: High-level, decorator-driven application class (lazy-loaded)
- Decorators: `agent`, `custom`, `orchestrator`, `iterative_planner`,
  `router`, `chain`, `parallel`, `evaluator_optimizer` (lazy-loaded)
"""

__all__ = [
    "Core",
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
    if name == "Core":
        from .core_app import Core

        return Core
    if name == "FastAgent":
        from .fastagent import FastAgent

        return FastAgent

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
