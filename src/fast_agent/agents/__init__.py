"""
Fast Agent - Agent implementations and workflow patterns.

This module exports all agent classes from the fast_agent.agents package,
providing a single import point for both core agents and workflow agents.
"""

# Core agents
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_agent import ToolAgent

# Workflow agents
from fast_agent.agents.workflow.chain_agent import ChainAgent
from fast_agent.agents.workflow.evaluator_optimizer import EvaluatorOptimizerAgent
from fast_agent.agents.workflow.iterative_planner import IterativePlanner
from fast_agent.agents.workflow.orchestrator_agent import OrchestratorAgent
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.router_agent import RouterAgent

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
    "OrchestratorAgent",
    "ParallelAgent",
    "RouterAgent",
    # Types
    "AgentConfig",
]
