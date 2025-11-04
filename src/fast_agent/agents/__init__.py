"""
Fast Agent - Agent implementations and workflow patterns.

This module re-exports agent classes with lazy imports to avoid circular
dependencies during package initialization while preserving a clean API:

    from fast_agent.agents import McpAgent, ToolAgent, LlmAgent
"""

from typing import TYPE_CHECKING

from fast_agent.agents.agent_types import AgentConfig


def __getattr__(name: str):
    """Lazily resolve agent classes to avoid import cycles."""
    if name == "LlmAgent":
        from .llm_agent import LlmAgent

        return LlmAgent
    elif name == "LlmDecorator":
        from .llm_decorator import LlmDecorator

        return LlmDecorator
    elif name == "ToolAgent":
        from .tool_agent import ToolAgent

        return ToolAgent
    elif name == "McpAgent":
        from .mcp_agent import McpAgent

        return McpAgent
    elif name == "ChainAgent":
        from .workflow.chain_agent import ChainAgent

        return ChainAgent
    elif name == "EvaluatorOptimizerAgent":
        from .workflow.evaluator_optimizer import EvaluatorOptimizerAgent

        return EvaluatorOptimizerAgent
    elif name == "IterativePlanner":
        from .workflow.iterative_planner import IterativePlanner

        return IterativePlanner
    elif name == "ParallelAgent":
        from .workflow.parallel_agent import ParallelAgent

        return ParallelAgent
    elif name == "RouterAgent":
        from .workflow.router_agent import RouterAgent

        return RouterAgent
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .llm_agent import LlmAgent as LlmAgent  # noqa: F401
    from .llm_decorator import LlmDecorator as LlmDecorator  # noqa: F401
    from .mcp_agent import McpAgent as McpAgent  # noqa: F401
    from .tool_agent import ToolAgent as ToolAgent  # noqa: F401
    from .workflow.chain_agent import ChainAgent as ChainAgent  # noqa: F401
    from .workflow.evaluator_optimizer import (
        EvaluatorOptimizerAgent as EvaluatorOptimizerAgent,
    )  # noqa: F401
    from .workflow.iterative_planner import IterativePlanner as IterativePlanner  # noqa: F401
    from .workflow.parallel_agent import ParallelAgent as ParallelAgent  # noqa: F401
    from .workflow.router_agent import RouterAgent as RouterAgent  # noqa: F401


__all__ = [
    # Core agents
    "LlmAgent",
    "LlmDecorator",
    "ToolAgent",
    "McpAgent",
    # Workflow agents
    "ChainAgent",
    "EvaluatorOptimizerAgent",
    "IterativePlanner",
    "ParallelAgent",
    "RouterAgent",
    # Types
    "AgentConfig",
]
