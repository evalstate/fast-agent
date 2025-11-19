"""
Generic Fast Agent protocol interfaces and types.

These are provider- and transport-agnostic and can be safely imported
without pulling in MCP-specific code, helping to avoid circular imports.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
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
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from fast_agent.agents.agent_types import AgentType
    from fast_agent.llm.model_info import ModelInfo

__all__ = [
    "FastAgentLLMProtocol",
    "StreamingAgentProtocol",
    "LlmAgentProtocol",
    "AgentProtocol",
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
        messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]: ...

    async def generate(
        self,
        messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended: ...

    async def apply_prompt_template(
        self, prompt_result: "GetPromptResult", prompt_name: str
    ) -> str: ...

    def get_request_params(
        self,
        request_params: RequestParams | None = None,
    ) -> RequestParams: ...

    def add_stream_listener(self, listener: Callable[[str], None]) -> Callable[[], None]: ...

    def add_tool_stream_listener(
        self, listener: Callable[[str, Dict[str, Any] | None], None]
    ) -> Callable[[], None]: ...

    @property
    def message_history(self) -> List[PromptMessageExtended]: ...

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
    def llm(self) -> FastAgentLLMProtocol: ...

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
    ) -> Tuple[ModelT | None, PromptMessageExtended]: ...

    @property
    def message_history(self) -> List[PromptMessageExtended]: ...

    @property
    def usage_accumulator(self) -> UsageAccumulator | None: ...

    async def apply_prompt(
        self,
        prompt: Union[str, "GetPromptResult"],
        arguments: Dict[str, str] | None = None,
        as_template: bool = False,
        namespace: str | None = None,
    ) -> str: ...

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> GetPromptResult: ...

    async def list_prompts(self, namespace: str | None = None) -> Mapping[str, List[Prompt]]: ...

    async def list_resources(self, namespace: str | None = None) -> Mapping[str, List[str]]: ...

    async def list_mcp_tools(self, namespace: str | None = None) -> Mapping[str, List[Tool]]: ...

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

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended: ...

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
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


@runtime_checkable
class StreamingAgentProtocol(AgentProtocol, Protocol):
    """Optional extension for agents that expose LLM streaming callbacks."""

    def add_stream_listener(self, listener: Callable[[str], None]) -> Callable[[], None]: ...

    def add_tool_stream_listener(
        self, listener: Callable[[str, Dict[str, Any] | None], None]
    ) -> Callable[[], None]: ...
