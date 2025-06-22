"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncContextManager,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from a2a_types.types import AgentCard
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from deprecated import deprecated
from mcp import ClientSession
from mcp.types import GetPromptResult, Prompt, PromptMessage, ReadResourceResult
from pydantic import BaseModel

from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from mcp_agent.llm.usage_tracking import UsageAccumulator


@runtime_checkable
class MCPConnectionManagerProtocol(Protocol):
    """Protocol for MCPConnectionManager functionality needed by ServerRegistry."""

    async def get_server(
        self,
        server_name: str,
        client_session_factory: Optional[
            Callable[
                [
                    MemoryObjectReceiveStream,
                    MemoryObjectSendStream,
                    Optional[timedelta],
                ],
                ClientSession,
            ]
        ] = None,
    ) -> "ServerConnection": ...

    async def disconnect_server(self, server_name: str) -> None: ...

    async def disconnect_all_servers(self) -> None: ...


@runtime_checkable
class ServerRegistryProtocol(Protocol):
    """Protocol defining the minimal interface of ServerRegistry needed by gen_client."""

    @property
    def connection_manager(self) -> MCPConnectionManagerProtocol: ...

    def initialize_server(
        self,
        server_name: str,
        client_session_factory: Optional[
            Callable[
                [
                    MemoryObjectReceiveStream,
                    MemoryObjectSendStream,
                    Optional[timedelta],
                ],
                ClientSession,
            ]
        ] = None,
        init_hook: Optional[Callable] = None,
    ) -> AsyncContextManager[ClientSession]:
        """Initialize a server and yield a client session."""
        ...


class ServerConnection(Protocol):
    """Protocol for server connection objects returned by MCPConnectionManager."""

    @property
    def session(self) -> ClientSession: ...


ModelT = TypeVar("ModelT", bound=BaseModel)


class AugmentedLLMProtocol(Protocol):
    """Protocol defining the interface for augmented LLMs"""

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Apply the prompt and return the result as a Pydantic model, or None if coercion fails"""
        ...

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Apply a list of PromptMessageMultipart messages directly to the LLM.


        Args:
            multipart_messages: List of PromptMessageMultipart objects
            request_params: Optional parameters to configure the LLM request

        Returns:
            A PromptMessageMultipart containing the Assistant response, including Tool Content
        """
        ...

    @property
    def message_history(self) -> List[PromptMessageMultipart]:
        """
        Return the LLM's message history as PromptMessageMultipart objects.

        Returns:
            List of PromptMessageMultipart objects representing the conversation history
        """
        ...

    usage_accumulator: "UsageAccumulator"


class AgentProtocol(AugmentedLLMProtocol, Protocol):
    """Protocol defining the standard agent interface"""

    name: str

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent"""
        ...

    async def __call__(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """Make the agent callable for sending messages directly."""
        ...

    async def send(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """Send a message to the agent and get a response"""
        ...

    async def apply_prompt(self, prompt_name: str, arguments: Dict[str, str] | None = None) -> str:
        """Apply an MCP prompt template by name"""
        ...

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        server_name: str | None = None,
    ) -> GetPromptResult: ...

    async def list_prompts(self, server_name: str | None = None) -> Mapping[str, List[Prompt]]: ...

    async def list_resources(self, server_name: str | None = None) -> Mapping[str, List[str]]: ...

    async def get_resource(
        self, resource_uri: str, server_name: str | None = None
    ) -> ReadResourceResult:
        """Get a resource from a specific server or search all servers"""
        ...

    @deprecated
    async def generate_str(self, message: str, request_params: RequestParams | None = None) -> str:
        """Generate a response. Deprecated: Use send(), generate() or structured()  instead"""
        ...

    @deprecated
    async def prompt(self, default_prompt: str = "") -> str:
        """Start an interactive prompt session with the agent. Deprecated. Use agent_app.interactive() instead."""
        ...

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageMultipart],
        resource_uri: str,
        server_name: str | None = None,
    ) -> str:
        """Send a message with an attached MCP resource"""
        ...

    async def agent_card(self) -> AgentCard:
        """Return an A2A Agent Card for this Agent"""
        ...

    async def initialize(self) -> None:
        """Initialize the agent and connect to MCP servers"""
        ...

    async def shutdown(self) -> None:
        """Shut down the agent and close connections"""
        ...


class ModelFactoryClassProtocol(Protocol):
    """
    Protocol defining the minimal interface of the ModelFactory class needed by sampling.
    This allows sampling.py to depend on this protocol rather than the concrete ModelFactory class.
    """

    @classmethod
    def create_factory(
        cls, model_string: str, request_params: Optional[RequestParams] = None
    ) -> Callable[..., Any]:
        """
        Creates a factory function that can be used to construct an LLM instance.

        Args:
            model_string: The model specification string
            request_params: Optional parameters to configure LLM behavior

        Returns:
            A factory function that can create an LLM instance
        """
        ...
