"""
Type definitions for agents and agent configurations.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, model_validator

# Forward imports to avoid circular dependencies
from mcp_agent.core.request_params import RequestParams


class AgentType(Enum):
    """Enumeration of supported agent types."""

    BASIC = "agent"
    CUSTOM = "custom"
    ORCHESTRATOR = "orchestrator"
    PARALLEL = "parallel"
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"
    ROUTER = "router"
    CHAIN = "chain"


class AgentConfig(BaseModel):
    """Configuration for an Agent instance"""

    name: str
    instruction: str = "You are a helpful agent."
    servers: List[str] = Field(default_factory=list)
    model: str | None = None
    use_history: bool = True
    default_request_params: RequestParams | None = None
    human_input: bool = False
    agent_type: AgentType = AgentType.BASIC
    default: bool = False

    @model_validator(mode="after")
    def ensure_default_request_params(self) -> "AgentConfig":
        """Ensure default_request_params exists with proper history setting"""
        if self.default_request_params is None:
            self.default_request_params = RequestParams(
                use_history=self.use_history, systemPrompt=self.instruction
            )
        else:
            # Override the request params history setting if explicitly configured
            self.default_request_params.use_history = self.use_history
        return self
