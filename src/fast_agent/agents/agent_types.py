"""
Type definitions for agents and agent configurations.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path

from mcp.client.session import ElicitationFnT

from fast_agent.skills import SkillManifest, SkillRegistry

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
    servers: list[str] = field(default_factory=list)
    tools: dict[str, list[str]] = field(default_factory=dict)  # filters for tools
    resources: dict[str, list[str]] = field(default_factory=dict)  # filters for resources
    prompts: dict[str, list[str]] = field(default_factory=dict)  # filters for prompts
    skills: SkillManifest | SkillRegistry | Path | str | None = None
    skill_manifests: list[SkillManifest] = field(default_factory=list, repr=False)
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
