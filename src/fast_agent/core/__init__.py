"""
Core interfaces and decorators for fast-agent.

Public API:
- `Core`: The core application container
- `AgentApp`: Container for interacting with agents
- `FastAgent`: High-level, decorator-driven application class
- Decorators: `agent`, `custom`, `orchestrator`, `iterative_planner`,
  `router`, `chain`, `parallel`, `evaluator_optimizer`
"""

# Eager exports for clear, static imports and IDE support.
# Import order matters here to avoid circular imports when loading FastAgent.
from .agent_app import AgentApp
from .core_app import Core
from .direct_decorators import (
    agent,
    chain,
    custom,
    evaluator_optimizer,
    iterative_planner,
    orchestrator,
    parallel,
    router,
)
from .fastagent import FastAgent

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
