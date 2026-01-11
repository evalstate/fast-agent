"""
Generic Fast Agent protocol interfaces and types.

These are provider- and transport-agnostic and can be safely imported
without pulling in MCP-specific code, helping to avoid circular imports.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from a2a.types import AgentCard
from mcp import Tool
from mcp.types import GetPromptResult, ListToolsResult, Prompt, PromptMessage, ReadResourceResult
from pydantic import BaseModel
from rich.text import Text

from fast_agent.llm.provider_types import Provider
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from fast_agent.acp.acp_aware_mixin import ACPCommand, ACPModeInfo
    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.agents.agent_types import AgentConfig, AgentType
    from fast_agent.agents.tool_runner import ToolRunnerHooks
    from fast_agent.context import Context
    from fast_agent.llm.model_info import ModelInfo

__all__ = [
    "FastAgentLLMProtocol",
    "StreamingAgentProtocol",
    "LlmAgentProtocol",
    "AgentProtocol",
    "ToolRunnerHookCapable",
    "ACPAwareProtocol",
    "LLMFactoryProtocol",
    "ModelFactoryFunctionProtocol",
    "ModelT",
]


ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMFactoryProtocol(Protocol):
    """Protocol for LLM factory functions that create FastAgentLLM instances."""

    def __call__(self, agent: "AgentProtocol", **kwargs: Any) -> "FastAgentLLMProtocol": ...


class ModelFactoryFunctionProtocol(Protocol):
    """Returns an LLM Model Factory for the specified model string"""

    def __call__(self, model: str | None = None) -> LLMFactoryProtocol: ...


@runtime_checkable
class FastAgentLLMProtocol(Protocol):
    """Protocol defining the interface for LLMs"""

    async def structured(
        self,
        messages: list[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]: ...

    async def generate(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended: ...

    async def apply_prompt_template(
        self, prompt_result: "GetPromptResult", prompt_name: str
    ) -> str: ...

    def get_request_params(
        self,
        request_params: RequestParams | None = None,
    ) -> RequestParams: ...

    default_request_params: RequestParams

    def add_stream_listener(self, listener: Callable[[StreamChunk], None]) -> Callable[[], None]: ...

    def add_tool_stream_listener(
        self, listener: Callable[[str, dict[str, Any] | None], None]
    ) -> Callable[[], None]: ...

    def chat_turn(self) -> int: ...

    @property
    def message_history(self) -> list[PromptMessageExtended]: ...

    def pop_last_message(self) -> PromptMessageExtended | None: ...

    @property
    def usage_accumulator(self) -> UsageAccumulator | None: ...

    @property
    def provider(self) -> Provider: ...

    @property
    def model_name(self) -> str | None: ...

    @property
    def model_info(self) -> "ModelInfo | None": ...

    def clear(self, *, clear_prompts: bool = False) -> None: ...


@runtime_checkable
class LlmAgentProtocol(Protocol):
    """Protocol defining the minimal interface for LLM agents."""

    @property
    def llm(self) -> FastAgentLLMProtocol | None: ...

    @property
    def name(self) -> str: ...

    @property
    def agent_type(self) -> "AgentType": ...

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    def clear(self, *, clear_prompts: bool = False) -> None: ...

    def pop_last_message(self) -> PromptMessageExtended | None: ...


@runtime_checkable
class AgentProtocol(LlmAgentProtocol, Protocol):
    """Standard agent interface with flexible input types."""

    async def __call__(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
    ) -> str: ...

    async def send(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
        request_params: RequestParams | None = None,
    ) -> str: ...

    async def generate(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended: ...

    async def structured(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]: ...

    @property
    def message_history(self) -> list[PromptMessageExtended]: ...

    @property
    def usage_accumulator(self) -> UsageAccumulator | None: ...

    async def apply_prompt(
        self,
        prompt: Union[str, "GetPromptResult"],
        arguments: dict[str, str] | None = None,
        as_template: bool = False,
        namespace: str | None = None,
    ) -> str: ...

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> GetPromptResult: ...

    async def list_prompts(self, namespace: str | None = None) -> Mapping[str, list[Prompt]]: ...

    async def list_resources(self, namespace: str | None = None) -> Mapping[str, list[str]]: ...

    async def list_mcp_tools(self, namespace: str | None = None) -> Mapping[str, list[Tool]]: ...

    async def list_tools(self) -> ListToolsResult: ...

    async def get_resource(
        self, resource_uri: str, namespace: str | None = None
    ) -> ReadResourceResult: ...

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageExtended],
        resource_uri: str,
        namespace: str | None = None,
    ) -> str: ...

    async def agent_card(self) -> AgentCard: ...

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended: ...

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_items: str | list[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        render_markdown: bool | None = None,
    ) -> None: ...

    async def attach_llm(
        self,
        llm_factory: LLMFactoryProtocol,
        model: str | None = None,
        request_params: RequestParams | None = None,
        **additional_kwargs,
    ) -> FastAgentLLMProtocol: ...

    @property
    def initialized(self) -> bool: ...

    instruction: str
    config: "AgentConfig"
    context: "Context | None"

    def set_instruction(self, instruction: str) -> None: ...


@runtime_checkable
class ToolRunnerHookCapable(Protocol):
    """Optional capability for agents to expose ToolRunner hooks."""

    @property
    def tool_runner_hooks(self) -> "ToolRunnerHooks | None": ...

    @tool_runner_hooks.setter
    def tool_runner_hooks(self, value: "ToolRunnerHooks | None") -> None: ...


@runtime_checkable
class StreamingAgentProtocol(AgentProtocol, Protocol):
    """Optional extension for agents that expose LLM streaming callbacks."""

    def add_stream_listener(self, listener: Callable[[StreamChunk], None]) -> Callable[[], None]: ...

    def add_tool_stream_listener(
        self, listener: Callable[[str, dict[str, Any] | None], None]
    ) -> Callable[[], None]: ...


@runtime_checkable
class ACPAwareProtocol(Protocol):
    """
    Protocol for agents that can be ACP-aware.

    This protocol defines the interface for agents that can check whether
    they're running in ACP mode and access ACP features when available.

    Agents implementing this protocol can:
    - Check if they're in ACP mode via `is_acp_mode`
    - Access ACP context via `acp` property
    - Declare slash commands via `acp_commands` property

    The ACPAwareMixin provides a concrete implementation of this protocol.
    """

    @property
    def acp(self) -> "ACPContext | None":
        """Get the ACP context if available."""
        ...

    @property
    def is_acp_mode(self) -> bool:
        """Check if the agent is running in ACP mode."""
        ...

    @property
    def acp_commands(self) -> dict[str, "ACPCommand"]:
        """
        Declare slash commands this agent exposes via ACP.

        Returns a dict mapping command names to ACPCommand instances.
        Commands are queried dynamically when the agent is the active mode.
        """
        ...

    def acp_mode_info(self) -> "ACPModeInfo | None":
        """
        Optional ACP mode metadata (name/description) for client display.
        """
        ...
