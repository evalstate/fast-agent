"""
Agent implementation using the clean BaseAgent adapter.

This provides a streamlined implementation that adheres to AgentProtocol
while delegating LLM operations to an attached AugmentedLLMProtocol instance.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, TypeVar

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.interfaces import FastAgentLLMProtocol
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.context import Context

logger = get_logger(__name__)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=FastAgentLLMProtocol)


class Agent(McpAgent):
    """
    An Agent is an entity that has access to a set of MCP servers and can interact with them.
    Each agent should have a purpose defined by its instruction.

    This implementation provides a clean adapter that adheres to AgentProtocol
    while delegating LLM operations to an attached AugmentedLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,  # Can be AgentConfig or backward compatible str name
        connection_persistence: bool = True,
        context: Optional["Context"] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        # Initialize with BaseAgent constructor
        super().__init__(
            config=config,
            connection_persistence=connection_persistence,
            context=context,
            **kwargs,
        )
