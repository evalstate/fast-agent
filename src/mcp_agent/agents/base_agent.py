"""
Base Agent class that implements the AgentProtocol interface.

This class provides default implementations of the standard agent methods
and delegates operations to an attached AugmentedLLMProtocol instance.
"""

import asyncio
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from a2a_types.types import AgentCapabilities, AgentCard, AgentSkill
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    GetPromptResult,
    ListToolsResult,
    PromptMessage,
    ReadResourceResult,
    TextContent,
    Tool,
)
from opentelemetry import trace
from pydantic import BaseModel

from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import PromptExitError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.human_input.types import (
    HUMAN_INPUT_SIGNAL_NAME,
    HumanInputCallback,
    HumanInputRequest,
    HumanInputResponse,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AgentProtocol, AugmentedLLMProtocol
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLMProtocol)

HUMAN_INPUT_TOOL_NAME = "__human_input__"
if TYPE_CHECKING:
    from mcp_agent.context import Context
    from mcp_agent.llm.usage_tracking import UsageAccumulator


DEFAULT_CAPABILITIES = AgentCapabilities(
    streaming=False, pushNotifications=False, stateTransitionHistory=False
)


class BaseAgent(MCPAggregator, AgentProtocol):
    """
    A base Agent class that implements the AgentProtocol interface.

    This class provides default implementations of the standard agent methods
    and delegates LLM operations to an attached AugmentedLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        functions: Optional[List[Callable]] = None,
        connection_persistence: bool = True,
        human_input_callback: Optional[HumanInputCallback] = None,
        context: Optional["Context"] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.config = config

        super().__init__(
            context=context,
            server_names=self.config.servers,
            connection_persistence=connection_persistence,
            name=self.config.name,
            **kwargs,
        )

        self._context = context
        self.tracer = trace.get_tracer(__name__)
        self.name = self.config.name
        self.instruction = self.config.instruction
        self.functions = functions or []
        self.executor = self.context.executor if context and hasattr(context, "executor") else None
        self.logger = get_logger(f"{__name__}.{self.name}")

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # Initialize the LLM to None (will be set by attach_llm)
        self._llm: Optional[AugmentedLLMProtocol] = None

        # Map function names to tools
        self._function_tool_map: Dict[str, Any] = {}

        if not self.config.human_input:
            self.human_input_callback = None
        else:
            self.human_input_callback: Optional[HumanInputCallback] = human_input_callback
            if not human_input_callback and context and hasattr(context, "human_input_handler"):
                self.human_input_callback = context.human_input_handler

    async def initialize(self) -> None:
        """
        Initialize the agent and connect to the MCP servers.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await self.__aenter__()  # This initializes the connection manager and loads the servers

    async def attach_llm(
        self,
        llm_factory: Union[Type[AugmentedLLMProtocol], Callable[..., AugmentedLLMProtocol]],
        model: Optional[str] = None,
        request_params: Optional[RequestParams] = None,
        **additional_kwargs,
    ) -> AugmentedLLMProtocol:
        """
        Create and attach an LLM instance to this agent.

        Parameters have the following precedence (highest to lowest):
        1. Explicitly passed parameters to this method
        2. Agent's default_request_params
        3. LLM's default values

        Args:
            llm_factory: A class or callable that constructs an AugmentedLLM
            model: Optional model name override
            request_params: Optional request parameters override
            **additional_kwargs: Additional parameters passed to the LLM constructor

        Returns:
            The created LLM instance
        """
        # Start with agent's default params
        effective_params = (
            self._default_request_params.model_copy() if self._default_request_params else None
        )

        # Override with explicitly passed request_params
        if request_params:
            if effective_params:
                # Update non-None values
                for k, v in request_params.model_dump(exclude_unset=True).items():
                    if v is not None:
                        setattr(effective_params, k, v)
            else:
                effective_params = request_params

        # Override model if explicitly specified
        if model and effective_params:
            effective_params.model = model

        # Create the LLM instance
        self._llm = llm_factory(
            agent=self, request_params=effective_params, context=self._context, **additional_kwargs
        )

        return self._llm

    async def shutdown(self) -> None:
        """
        Shutdown the agent and close all MCP server connections.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await super().close()

    async def __call__(
        self,
        message: Union[str, PromptMessageMultipart] | None = None,
        agent_name: Optional[str] = None,
        default_prompt: str = "",
    ) -> str:
        """
        Make the agent callable to send messages or start an interactive prompt.

        Args:
            message: Optional message to send to the agent
            agent_name: Optional name of the agent (for consistency with DirectAgentApp)
            default: Default message to use in interactive prompt mode

        Returns:
            The agent's response as a string or the result of the interactive session
        """
        if message:
            return await self.send(message)
        return await self.prompt(default_prompt=default_prompt)

    async def generate_str(self, message: str, request_params: RequestParams | None) -> str:
        result: PromptMessageMultipart = await self.generate([Prompt.user(message)], request_params)
        return result.first_text()

    async def send(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: Message content in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly

        Returns:
            The agent's response as a string
        """
        # Convert the input to a PromptMessageMultipart
        prompt = self._normalize_message_input(message)

        # Use the LLM to generate a response
        response = await self.generate([prompt], None)
        return response.all_text()

    def _normalize_message_input(
        self, message: Union[str, PromptMessage, PromptMessageMultipart]
    ) -> PromptMessageMultipart:
        """
        Convert a message of any supported type to PromptMessageMultipart.

        Args:
            message: Message in various formats (string, PromptMessage, or PromptMessageMultipart)

        Returns:
            A PromptMessageMultipart object
        """
        # Handle single message
        if isinstance(message, str):
            return Prompt.user(message)
        elif isinstance(message, PromptMessage):
            return PromptMessageMultipart(role=message.role, content=[message.content])
        elif isinstance(message, PromptMessageMultipart):
            return message
        else:
            # Try to convert to string as fallback
            return Prompt.user(str(message))

    async def prompt(self, default_prompt: str = "") -> str:
        """
        Start an interactive prompt session with the agent.

        Args:
            default_prompt: The initial prompt to send to the agent

        Returns:
            The result of the interactive session
        """
        ...

    async def request_human_input(self, request: HumanInputRequest) -> str:
        """
        Request input from a human user. Pauses the workflow until input is received.

        Args:
            request: The human input request

        Returns:
            The input provided by the human

        Raises:
            TimeoutError: If the timeout is exceeded
        """
        if not self.human_input_callback:
            raise ValueError("Human input callback not set")

        # Generate a unique ID for this request to avoid signal collisions
        request_id = f"{HUMAN_INPUT_SIGNAL_NAME}_{self.name}_{uuid.uuid4()}"
        request.request_id = request_id
        # Use metadata as a dictionary to pass agent name
        request.metadata = {"agent_name": self.name}
        self.logger.debug("Requesting human input:", data=request)

        if not self.executor:
            raise ValueError("No executor available")

        async def call_callback_and_signal() -> None:
            try:
                assert self.human_input_callback is not None
                user_input = await self.human_input_callback(request)

                self.logger.debug("Received human input:", data=user_input)
                await self.executor.signal(signal_name=request_id, payload=user_input)
            except PromptExitError as e:
                # Propagate the exit error through the signal system
                self.logger.info("User requested to exit session")
                await self.executor.signal(
                    signal_name=request_id,
                    payload={"exit_requested": True, "error": str(e)},
                )
            except Exception as e:
                await self.executor.signal(
                    request_id, payload=f"Error getting human input: {str(e)}"
                )

        asyncio.create_task(call_callback_and_signal())

        self.logger.debug("Waiting for human input signal")

        # Wait for signal (workflow is paused here)
        result = await self.executor.wait_for_signal(
            signal_name=request_id,
            request_id=request_id,
            workflow_id=request.workflow_id,
            signal_description=request.description or request.prompt,
            timeout_seconds=request.timeout_seconds,
            signal_type=HumanInputResponse,
        )

        if isinstance(result, dict) and result.get("exit_requested", False):
            raise PromptExitError(result.get("error", "User requested to exit FastAgent session"))
        self.logger.debug("Received human input signal", data=result)
        return result

    async def list_tools(self) -> ListToolsResult:
        """
        List all tools available to this agent.

        Returns:
            ListToolsResult with available tools
        """
        if not self.initialized:
            await self.initialize()

        result = await super().list_tools()

        if not self.human_input_callback:
            return result

        # Add a human_input_callback as a tool
        from mcp.server.fastmcp.tools import Tool as FastTool

        human_input_tool: FastTool = FastTool.from_function(self.request_human_input)
        result.tools.append(
            Tool(
                name=HUMAN_INPUT_TOOL_NAME,
                description=human_input_tool.description,
                inputSchema=human_input_tool.parameters,
            )
        )

        return result

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> CallToolResult:
        """
        Call a tool by name with the given arguments.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Result of the tool call
        """
        if name == HUMAN_INPUT_TOOL_NAME:
            # Call the human input tool
            return await self._call_human_input_tool(arguments)
        else:
            return await super().call_tool(name, arguments)

    async def _call_human_input_tool(
        self, arguments: Dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Handle human input request via tool calling.

        Args:
            arguments: Tool arguments

        Returns:
            Result of the human input request
        """
        # Handle human input request
        try:
            # Make sure arguments is not None
            if arguments is None:
                arguments = {}

            # Extract request data
            request_data = arguments.get("request")

            # Handle both string and dict request formats
            if isinstance(request_data, str):
                request = HumanInputRequest(prompt=request_data)
            elif isinstance(request_data, dict):
                request = HumanInputRequest(**request_data)
            else:
                # Fallback for invalid or missing request data
                request = HumanInputRequest(prompt="Please provide input:")

            result = await self.request_human_input(request=request)

            # Use response attribute if available, otherwise use the result directly
            response_text = (
                result.response if isinstance(result, HumanInputResponse) else str(result)
            )

            return CallToolResult(
                content=[TextContent(type="text", text=f"Human response: {response_text}")]
            )

        except PromptExitError:
            raise
        except asyncio.TimeoutError as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: Human input request timed out: {str(e)}",
                    )
                ],
            )
        except Exception as e:
            import traceback

            print(f"Error in _call_human_input_tool: {traceback.format_exc()}")

            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Error requesting human input: {str(e)}")],
            )

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        server_name: str | None = None,
    ) -> GetPromptResult:
        """
        Get a prompt from a server.

        Args:
            prompt_name: Name of the prompt, optionally namespaced
            arguments: Optional dictionary of arguments to pass to the prompt template
            server_name: Optional name of the server to get the prompt from

        Returns:
            GetPromptResult containing the prompt information
        """
        return await super().get_prompt(prompt_name, arguments, server_name)

    async def apply_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        agent_name: str | None = None,
        server_name: str | None = None,
    ) -> str:
        """
        Apply an MCP Server Prompt by name and return the assistant's response.
        Will search all available servers for the prompt if not namespaced and no server_name provided.

        If the last message in the prompt is from a user, this will automatically
        generate an assistant response to ensure we always end with an assistant message.

        Args:
            prompt_name: The name of the prompt to apply
            arguments: Optional dictionary of string arguments to pass to the prompt template
            agent_name: Optional agent name (ignored at this level, used by multi-agent apps)
            server_name: Optional name of the server to get the prompt from

        Returns:
            The assistant's response or error message
        """

        # Get the prompt - this will search all servers if needed
        self.logger.debug(f"Loading prompt '{prompt_name}'")
        prompt_result: GetPromptResult = await self.get_prompt(prompt_name, arguments, server_name)

        if not prompt_result or not prompt_result.messages:
            error_msg = f"Prompt '{prompt_name}' could not be found or contains no messages"
            self.logger.warning(error_msg)
            return error_msg

        # Get the display name (namespaced version)
        namespaced_name = getattr(prompt_result, "namespaced_name", prompt_name)
        self.logger.debug(f"Using prompt '{namespaced_name}'")

        # Convert prompt messages to multipart format using the safer method
        multipart_messages = PromptMessageMultipart.from_get_prompt_result(prompt_result)

        # Always call generate to ensure LLM implementations can handle prompt templates
        # This is critical for stateful LLMs like PlaybackLLM
        response = await self.generate(multipart_messages, None)
        return response.first_text()

    async def get_embedded_resources(
        self, resource_uri: str, server_name: str | None = None
    ) -> List[EmbeddedResource]:
        """
        Get a resource from an MCP server and return it as a list of embedded resources ready for use in prompts.

        Args:
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from

        Returns:
            List of EmbeddedResource objects ready to use in a PromptMessageMultipart

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        # Get the raw resource result
        result: ReadResourceResult = await self.get_resource(resource_uri, server_name)

        # Convert each resource content to an EmbeddedResource
        embedded_resources: List[EmbeddedResource] = []
        for resource_content in result.contents:
            embedded_resource = EmbeddedResource(
                type="resource", resource=resource_content, annotations=None
            )
            embedded_resources.append(embedded_resource)

        return embedded_resources

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageMultipart],
        resource_uri: str,
        server_name: str | None = None,
    ) -> str:
        """
        Create a prompt with the given content and resource, then send it to the agent.

        Args:
            prompt_content: Content in various formats:
                - String: Converted to a user message with the text
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from

        Returns:
            The agent's response as a string
        """
        # Get the embedded resources
        embedded_resources: List[EmbeddedResource] = await self.get_embedded_resources(
            resource_uri, server_name
        )

        # Create or update the prompt message
        prompt: PromptMessageMultipart
        if isinstance(prompt_content, str):
            # Create a new prompt with the text and resources
            content = [TextContent(type="text", text=prompt_content)]
            content.extend(embedded_resources)
            prompt = PromptMessageMultipart(role="user", content=content)
        elif isinstance(prompt_content, PromptMessage):
            # Convert PromptMessage to PromptMessageMultipart and add resources
            content = [prompt_content.content]
            content.extend(embedded_resources)
            prompt = PromptMessageMultipart(role=prompt_content.role, content=content)
        elif isinstance(prompt_content, PromptMessageMultipart):
            # Add resources to the existing prompt
            prompt = prompt_content
            prompt.content.extend(embedded_resources)
        else:
            raise TypeError(
                "prompt_content must be a string, PromptMessage, or PromptMessageMultipart"
            )

        response: PromptMessageMultipart = await self.generate([prompt], None)
        return response.first_text()

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Create a completion with the LLM using the provided messages.
        Delegates to the attached LLM.

        Args:
            multipart_messages: List of multipart messages to send to the LLM
            request_params: Optional parameters to configure the request

        Returns:
            The LLM's response as a PromptMessageMultipart
        """
        assert self._llm
        with self.tracer.start_as_current_span(f"Agent: '{self.name}' generate"):
            return await self._llm.generate(multipart_messages, request_params)

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Apply the prompt and return the result as a Pydantic model.
        Delegates to the attached LLM.

        Args:
            prompt: List of PromptMessageMultipart objects
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            An instance of the specified model, or None if coercion fails
        """
        assert self._llm
        with self.tracer.start_as_current_span(f"Agent: '{self.name}' structured"):
            return await self._llm.structured(multipart_messages, model, request_params)

    async def apply_prompt_messages(
        self, prompts: List[PromptMessageMultipart], request_params: RequestParams | None = None
    ) -> str:
        """
        Apply a list of prompt messages and return the result.

        Args:
            prompts: List of PromptMessageMultipart messages
            request_params: Optional request parameters

        Returns:
            The text response from the LLM
        """

        response = await self.generate(prompts, request_params)
        return response.first_text()

    @property
    def agent_type(self) -> AgentType:
        """
        Return the type of this agent.
        """
        return AgentType.BASIC

    async def agent_card(self) -> AgentCard:
        """
        Return an A2A card describing this Agent
        """

        skills: List[AgentSkill] = []
        tools: ListToolsResult = await self.list_tools()
        for tool in tools.tools:
            skills.append(await self.convert(tool))

        return AgentCard(
            name=self.name,
            description=self.instruction,
            url=f"fast-agent://agents/{self.name}/",
            version="0.1",
            capabilities=DEFAULT_CAPABILITIES,
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            provider=None,
            documentationUrl=None,
            authentication=None,
            skills=skills,
        )

    async def convert(self, tool: Tool) -> AgentSkill:
        """
        Convert a Tool to an AgentSkill.
        """

        _, tool_without_namespace = await self._parse_resource_name(tool.name, "tool")
        return AgentSkill(
            id=tool.name,
            name=tool_without_namespace,
            description=tool.description,
            tags=["tool"],
            examples=None,
            inputModes=None,  # ["text/plain"],
            # cover TextContent | ImageContent ->
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/223
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/93
            outputModes=None,  # ,["text/plain", "image/*"],
        )

    @property
    def message_history(self) -> List[PromptMessageMultipart]:
        """
        Return the agent's message history as PromptMessageMultipart objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageMultipart objects representing the conversation history
        """
        if self._llm:
            return self._llm.message_history
        return []

    @property
    def usage_accumulator(self) -> Optional["UsageAccumulator"]:
        """
        Return the usage accumulator for tracking token usage across turns.
        
        Returns:
            UsageAccumulator object if LLM is attached, None otherwise
        """
        if self._llm:
            return self._llm.usage_accumulator
        return None
