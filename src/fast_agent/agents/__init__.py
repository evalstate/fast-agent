"""
Fast Agent - Agent implementations and workflow patterns.

This module re-exports agent classes with lazy imports to avoid circular
dependencies during package initialization while preserving a clean API:

    from fast_agent.agents import McpAgent, ToolAgent, LlmAgent
"""

from typing import TYPE_CHECKING

from fast_agent.agents.agent_types import AgentConfig

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LlmAgent": (".llm_agent", "LlmAgent"),
    "LlmDecorator": (".llm_decorator", "LlmDecorator"),
    "ToolAgent": (".tool_agent", "ToolAgent"),
    "McpAgent": (".mcp_agent", "McpAgent"),
    "SmartAgent": (".smart_agent", "SmartAgent"),
    "ChainAgent": (".workflow.chain_agent", "ChainAgent"),
    "EvaluatorOptimizerAgent": (
        ".workflow.evaluator_optimizer",
        "EvaluatorOptimizerAgent",
    ),
    "IterativePlanner": (".workflow.iterative_planner", "IterativePlanner"),
    "ParallelAgent": (".workflow.parallel_agent", "ParallelAgent"),
    "RouterAgent": (".workflow.router_agent", "RouterAgent"),
}


def __getattr__(name: str):
    """Lazily resolve agent classes to avoid import cycles."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    from importlib import import_module

    return getattr(import_module(module_name, __name__), attr_name)


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .llm_agent import LlmAgent as LlmAgent
    from .llm_decorator import LlmDecorator as LlmDecorator
    from .mcp_agent import McpAgent as McpAgent
    from .smart_agent import SmartAgent as SmartAgent
    from .tool_agent import ToolAgent as ToolAgent
    from .workflow.chain_agent import ChainAgent as ChainAgent
    from .workflow.evaluator_optimizer import (
        EvaluatorOptimizerAgent as EvaluatorOptimizerAgent,
    )
    from .workflow.iterative_planner import IterativePlanner as IterativePlanner
    from .workflow.parallel_agent import ParallelAgent as ParallelAgent
    from .workflow.router_agent import RouterAgent as RouterAgent


__all__ = [
    # Types
    "AgentConfig",
    # Workflow agents
    "ChainAgent",
    "EvaluatorOptimizerAgent",
    "IterativePlanner",
    # Core agents
    "LlmAgent",
    "LlmDecorator",
    "McpAgent",
    "ParallelAgent",
    "RouterAgent",
    "SmartAgent",
    "ToolAgent",
]
