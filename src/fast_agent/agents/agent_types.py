"""
Type definitions for agents and agent configurations.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Dict, List, Optional

from mcp.client.session import ElicitationFnT

# Forward imports to avoid circular dependencies
from fast_agent.types import RequestParams


class AgentType(StrEnum):
    """Enumeration of supported agent types."""

    LLM = auto()
    BASIC = auto()
    CUSTOM = auto()
    ORCHESTRATOR = auto()
    PARALLEL = auto()
    EVALUATOR_OPTIMIZER = auto()
    ROUTER = auto()
    CHAIN = auto()
    ITERATIVE_PLANNER = auto()


@dataclass
class AgentConfig:
    """Configuration for an Agent instance"""

    name: str
    instruction: str = "You are a helpful agent."
    servers: List[str] = field(default_factory=list)
    tools: Optional[Dict[str, List[str]]] = None
    resources: Optional[Dict[str, List[str]]] = None
    prompts: Optional[Dict[str, List[str]]] = None
    model: str | None = None
    use_history: bool = True
    default_request_params: RequestParams | None = None
    human_input: bool = False
    agent_type: AgentType = AgentType.BASIC
    default: bool = False
    elicitation_handler: ElicitationFnT | None = None
    api_key: str | None = None

    def __post_init__(self):
        """Ensure default_request_params exists with proper history setting"""
        if self.default_request_params is None:
            self.default_request_params = RequestParams(
                use_history=self.use_history, systemPrompt=self.instruction
            )
        else:
            # Override the request params history setting if explicitly configured
            self.default_request_params.use_history = self.use_history
