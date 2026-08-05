"""
Type definitions for agents and agent configurations.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from fast_agent.command_actions import PluginCommandActionSpec
from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.hooks.lifecycle_hook_types import LifecycleHookType
from fast_agent.mcp.server_declaration import MCPServerDeclaration
from fast_agent.skills import SKILLS_DEFAULT, SkillManifest, SkillRegistry, SkillsDefault
from fast_agent.tools.function_tool_config import FunctionToolSpec

# Forward imports to avoid circular dependencies
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from mcp.client.session import ElicitationFnT
else:
    ElicitationFnT = Callable[..., Any]


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
    MAKER = auto()
    A2A = auto()


SkillConfig: TypeAlias = (
    SkillManifest
    | SkillRegistry
    | Path
    | str
    | list[SkillManifest | SkillRegistry | Path | str | None]
    | None
    | SkillsDefault
)


# Function tools can be:
# - A callable (Python function)
# - A string spec like "module.py:function_name" (for dynamic loading)
@dataclass(frozen=True)
class ScopedFunctionToolConfig:
    """A single local Python tool registration with scoped metadata."""

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None


FunctionToolConfig: TypeAlias = (
    Callable[..., Any] | str | ScopedFunctionToolConfig | FunctionToolSpec
)

FunctionToolsConfig: TypeAlias = list[FunctionToolConfig] | None
SubagentActivationSource: TypeAlias = Literal["configuration", "cli", "instruction", "runtime"]
MCPConnectSourceForm: TypeAlias = Literal["list", "mapping"]


# Tool hooks config maps hook type to function spec string
# e.g., {"after_turn_complete": "hooks.py:my_hook"}
ToolHooksConfig: TypeAlias = dict[str, str] | None
LifecycleHooksConfig: TypeAlias = dict[LifecycleHookType, str] | None
PluginCommandsConfig: TypeAlias = dict[str, PluginCommandActionSpec] | None


MCPConnectTarget: TypeAlias = MCPServerDeclaration


@dataclass
class AgentConfig:
    """Configuration for an Agent instance.

    Naming note:
    - ``tools`` filters MCP-discovered tools by server name.
    - ``function_tools`` configures local Python function tools.
    - Runtime constructors such as ``ToolAgent(..., tools=...)`` use ``tools``
      for the resolved executable function-tool objects, not these MCP filters.
    """

    name: str
    instruction: str = DEFAULT_AGENT_INSTRUCTION
    description: str | None = None
    tool_input_schema: dict[str, Any] | None = None
    servers: list[str] = field(default_factory=list)
    tools: dict[str, list[str]] = field(default_factory=dict)  # MCP tool filters by server
    resources: dict[str, list[str]] = field(default_factory=dict)  # MCP resource filters by server
    prompts: dict[str, list[str]] = field(default_factory=dict)  # MCP prompt filters by server
    skills: SkillConfig = SKILLS_DEFAULT
    skill_manifests: list[SkillManifest] = field(default_factory=list, repr=False)
    skills_resolved_for_run: bool = field(default=False, repr=False)
    model: str | None = None
    use_history: bool = True
    save_trajectory: bool = False
    default_request_params: RequestParams | None = None
    human_input: bool = False
    agent_type: AgentType = AgentType.BASIC
    default: bool = False
    tool_only: bool = False
    subagents: bool | None = None
    subagent_model: str | None = None
    harness_tools: bool = False
    subagent_activation_source: SubagentActivationSource | None = field(
        default=None,
        init=False,
    )
    subagent_child: bool = field(default=False, init=False, repr=False)
    elicitation_handler: ElicitationFnT | None = None
    api_key: str | None = None
    function_tools: FunctionToolsConfig = None  # Local Python function tools
    shell: bool = False
    cwd: Path | None = None
    tool_hooks: ToolHooksConfig = None
    lifecycle_hooks: LifecycleHooksConfig = None
    commands: PluginCommandsConfig = None
    trim_tool_history: bool = False
    mcp_connect: list[MCPConnectTarget] = field(default_factory=list)
    mcp_connect_source_form: MCPConnectSourceForm = field(default="list", repr=False)
    source_path: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure default_request_params exists with proper history setting"""
        if self.subagents is not None:
            self.subagent_activation_source = "configuration"
        if self.save_trajectory and self.use_history:
            raise AgentConfigError("save_trajectory requires use_history=False")
        if self.default_request_params is None:
            self.default_request_params = RequestParams(
                use_history=self.use_history, system_prompt=self.instruction
            )
        else:
            # Override the request params history setting if explicitly configured
            self.default_request_params.use_history = self.use_history
            # Ensure instruction takes precedence over any existing systemPrompt
            self.default_request_params.system_prompt = self.instruction
