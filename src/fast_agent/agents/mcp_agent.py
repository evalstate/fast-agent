"""
Base Agent class that implements the AgentProtocol interface.

This class provides default implementations of the standard agent methods
and delegates operations to an attached AugmentedLLMProtocol instance.
"""

import asyncio
import fnmatch
import uuid
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from a2a.types import AgentCard, AgentSkill
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
from pydantic import BaseModel

from fast_agent.agents.llm_agent import DEFAULT_CAPABILITIES
from fast_agent.agents.tool_agent import ToolAgent
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
from mcp_agent.mcp.helpers.content_helpers import normalize_to_multipart_list
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol, LLMFactoryProtocol
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLMProtocol)

HUMAN_INPUT_TOOL_NAME = "__human_input__"
if TYPE_CHECKING:
    from fast_agent.context import Context
    from mcp_agent.llm.usage_tracking import UsageAccumulator


class McpAgent(ABC, ToolAgent):
    """
    A base Agent class that implements the AgentProtocol interface.

    This class provides default implementations of the standard agent methods
    and delegates LLM operations to an attached AugmentedLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        connection_persistence: bool = True,
        human_input_callback: HumanInputCallback | None = None,
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            context=context,
            **kwargs,
        )

        # Create aggregator with composition
        self._aggregator = MCPAggregator(
            server_names=self.config.servers,
            connection_persistence=connection_persistence,
            name=self.config.name,
            context=context,
            config=self.config,  # Pass the full config for access to elicitation_handler
            **kwargs,
        )

        self.instruction = self.config.instruction
        self.executor = context.executor if context else None
        self.logger = get_logger(f"{__name__}.{self._name}")

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # set with the "attach" method
        self._llm: AugmentedLLMProtocol | None = None

        if not self.config.human_input:
            self.human_input_callback = None
        else:
            self.human_input_callback: Optional[HumanInputCallback] = human_input_callback
            if not human_input_callback and context and hasattr(context, "human_input_handler"):
                self.human_input_callback = context.human_input_handler

    async def __aenter__(self):
        """Initialize the agent and its MCP aggregator."""
        await self._aggregator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the agent and its MCP aggregator."""
        await self._aggregator.__aexit__(exc_type, exc_val, exc_tb)

    async def initialize(self) -> None:
        """
        Initialize the agent and connect to the MCP servers.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await self.__aenter__()

    async def attach_llm(
        self,
        llm_factory: LLMFactoryProtocol,
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
            llm_factory: A factory function that constructs an AugmentedLLM
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
        await self._aggregator.close()

    @property
    def initialized(self) -> bool:
        """Check if both the agent and aggregator are initialized."""
        return self._initialized and self._aggregator.initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set the initialized state of both agent and aggregator."""
        self._initialized = value
        self._aggregator.initialized = value

    async def __call__(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageMultipart,
            List[Union[str, PromptMessage, PromptMessageMultipart]],
        ],
    ) -> str:
        return await self.send(message)

    async def send(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageMultipart,
            List[Union[str, PromptMessage, PromptMessageMultipart]],
        ],
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: Message content in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
                - request_params: Optional request parameters

        Returns:
            The agent's response as a string
        """
        response = await self.generate(message, request_params)
        return response.last_text()

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
        request_id = f"{HUMAN_INPUT_SIGNAL_NAME}_{self._name}_{uuid.uuid4()}"
        request.request_id = request_id
        # Use metadata as a dictionary to pass agent name
        request.metadata = {"agent_name": self._name}
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

    def _matches_pattern(self, name: str, pattern: str, server_name: str) -> bool:
        """
        Check if a name matches a pattern for a specific server.

        Args:
            name: The name to match (could be tool name, resource URI, or prompt name)
            pattern: The pattern to match against (e.g., "add", "math*", "resource://math/*")
            server_name: The server name (used for tool name prefixing)

        Returns:
            True if the name matches the pattern
        """
        # For tools, build the full pattern with server prefix: server_name-pattern
        if name.startswith(f"{server_name}-"):
            full_pattern = f"{server_name}-{pattern}"
            return fnmatch.fnmatch(name, full_pattern)

        # For resources and prompts, match directly against the pattern
        return fnmatch.fnmatch(name, pattern)

    async def list_tools(self) -> ListToolsResult:
        """
        List all tools available to this agent, filtered by configuration.

        Returns:
            ListToolsResult with available tools
        """
        # Get all tools from the aggregator
        result = await self._aggregator.list_tools()

        # Apply filtering if tools are specified in config
        if self.config.tools is not None:
            filtered_tools = []
            for tool in result.tools:
                # Extract server name from tool name, handling server names with hyphens
                server_name = None
                for configured_server in self.config.tools.keys():
                    if tool.name.startswith(f"{configured_server}-"):
                        server_name = configured_server
                        break

                # Check if this server has tool filters
                if server_name and server_name in self.config.tools:
                    # Check if tool matches any pattern for this server
                    for pattern in self.config.tools[server_name]:
                        if self._matches_pattern(tool.name, pattern, server_name):
                            filtered_tools.append(tool)
                            break
            result.tools = filtered_tools

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
            return await self._aggregator.call_tool(name, arguments)

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
        return await self._aggregator.get_prompt(prompt_name, arguments, server_name)

    async def apply_prompt(
        self,
        prompt: Union[str, GetPromptResult],
        arguments: Dict[str, str] | None = None,
        agent_name: str | None = None,
        server_name: str | None = None,
        as_template: bool = False,
    ) -> str:
        """
        Apply an MCP Server Prompt by name or GetPromptResult and return the assistant's response.
        Will search all available servers for the prompt if not namespaced and no server_name provided.

        If the last message in the prompt is from a user, this will automatically
        generate an assistant response to ensure we always end with an assistant message.

        Args:
            prompt: The name of the prompt to apply OR a GetPromptResult object
            arguments: Optional dictionary of string arguments to pass to the prompt template
            agent_name: Optional agent name (ignored at this level, used by multi-agent apps)
            server_name: Optional name of the server to get the prompt from
            as_template: If True, store as persistent template (always included in context)

        Returns:
            The assistant's response or error message
        """

        # Handle both string and GetPromptResult inputs
        if isinstance(prompt, str):
            prompt_name = prompt
            # Get the prompt - this will search all servers if needed
            self.logger.debug(f"Loading prompt '{prompt_name}'")
            prompt_result: GetPromptResult = await self.get_prompt(
                prompt_name, arguments, server_name
            )

            if not prompt_result or not prompt_result.messages:
                error_msg = f"Prompt '{prompt_name}' could not be found or contains no messages"
                self.logger.warning(error_msg)
                return error_msg

            # Get the display name (namespaced version)
            namespaced_name = getattr(prompt_result, "namespaced_name", prompt_name)
        else:
            # prompt is a GetPromptResult object
            prompt_result = prompt
            if not prompt_result or not prompt_result.messages:
                error_msg = "Provided GetPromptResult contains no messages"
                self.logger.warning(error_msg)
                return error_msg

            # Use a reasonable display name
            namespaced_name = getattr(prompt_result, "namespaced_name", "provided_prompt")

        self.logger.debug(f"Using prompt '{namespaced_name}'")

        # Convert prompt messages to multipart format using the safer method
        multipart_messages = PromptMessageMultipart.from_get_prompt_result(prompt_result)

        if as_template:
            # Use apply_prompt_template to store as persistent prompt messages
            return await self.apply_prompt_template(prompt_result, namespaced_name)
        else:
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
        result: ReadResourceResult = await self._aggregator.get_resource(resource_uri, server_name)

        # Convert each resource content to an EmbeddedResource
        embedded_resources: List[EmbeddedResource] = []
        for resource_content in result.contents:
            embedded_resource = EmbeddedResource(
                type="resource", resource=resource_content, annotations=None
            )
            embedded_resources.append(embedded_resource)

        return embedded_resources

    async def get_resource(
        self, resource_uri: str, server_name: str | None = None
    ) -> ReadResourceResult:
        """
        Get a resource from an MCP server.

        Args:
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from

        Returns:
            ReadResourceResult containing the resource data

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        # Get the raw resource result
        result: ReadResourceResult = await self._aggregator.get_resource(resource_uri, server_name)
        return result

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

    async def generate_impl(
        self,
        messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageMultipart:
        """
        Override ToolAgent's generate_impl to provide MCP tools dynamically.
        """
        if tools is None:
            # Get tools from our aggregator instead of ToolAgent's _tool_schemas
            tools_result = await self.list_tools()
            tools = tools_result.tools

        return await super().generate_impl(messages, request_params, tools=tools)

    async def _generate_impl(
        self,
        normalized_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageMultipart:
        """
        Default implementation for regular agents - delegates to attached LLM.

        Workflow agents should override this method to implement custom logic.

        Args:
            normalized_messages: Already normalized list of PromptMessageMultipart
            request_params: Optional parameters to configure the request
            tools: Optional tools to pass to the LLM

        Returns:
            The LLM's response as a PromptMessageMultipart
        """
        assert self._llm, (
            "No LLM attached to agent. Workflow agents should override _generate_impl()."
        )
        with self._tracer.start_as_current_span(f"Agent: '{self._name}' generate"):
            return await self._llm.generate(normalized_messages, request_params, tools=tools)

    async def run_tools(self, request: PromptMessageMultipart) -> PromptMessageMultipart:
        """Override ToolAgent's run_tools to use MCP tools via aggregator."""
        if not request.tool_calls:
            self.logger.warning("No tool calls found in request", data=request)
            return PromptMessageMultipart(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}

        # Cache available tool names (original, not namespaced) for display
        available_tools = [
            namespaced_tool.tool.name
            for namespaced_tool in self._aggregator._namespaced_tool_map.values()
        ]

        # Process each tool call using our aggregator
        for correlation_id, tool_request in request.tool_calls.items():
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}

            # Get the original tool name for display (not namespaced)
            namespaced_tool = self._aggregator._namespaced_tool_map.get(tool_name)
            display_tool_name = namespaced_tool.tool.name if namespaced_tool else tool_name

            self.display.show_tool_call(
                name=self._name,
                tool_args=tool_args,
                bottom_items=available_tools,
                tool_name=display_tool_name,
                highlight_items=tool_name,
                max_item_length=12,
            )

            try:
                # Use our aggregator to call the MCP tool
                result = await self.call_tool(tool_name, tool_args)
                tool_results[correlation_id] = result

                # Show tool result (like ToolAgent does)
                self.display.show_tool_result(name=self._name, result=result)

                self.logger.debug(f"MCP tool {display_tool_name} executed successfully")
            except Exception as e:
                self.logger.error(f"MCP tool {display_tool_name} failed: {e}")
                error_result = CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True,
                )
                tool_results[correlation_id] = error_result

                # Show error result too
                self.display.show_tool_result(name=self._name, result=error_result)

        return PromptMessageMultipart(role="user", tool_results=tool_results)

    async def apply_prompt_template(self, prompt_result: GetPromptResult, prompt_name: str) -> str:
        """
        Apply a prompt template as persistent context that will be included in all future conversations.
        Delegates to the attached LLM.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated
        """
        assert self._llm
        with self._tracer.start_as_current_span(f"Agent: '{self._name}' apply_prompt_template"):
            return await self._llm.apply_prompt_template(prompt_result, prompt_name)

    async def structured(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageMultipart,
            List[Union[str, PromptMessage, PromptMessageMultipart]],
        ],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Apply the prompt and return the result as a Pydantic model.
        Normalizes input messages and delegates to the attached LLM.

        Args:
            messages: Message(s) in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
                - List of any combination of the above
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            An instance of the specified model, or None if coercion fails
        """
        assert self._llm
        # Normalize all input types to a list of PromptMessageMultipart
        normalized_messages = normalize_to_multipart_list(messages)

        with self._tracer.start_as_current_span(f"Agent: '{self._name}' structured"):
            return await self._llm.structured(normalized_messages, model, request_params)

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

    async def list_prompts(self, server_name: str | None = None) -> Mapping[str, List[Prompt]]:
        """
        List all prompts available to this agent, filtered by configuration.

        Args:
            server_name: Optional server name to list prompts from

        Returns:
            Dictionary mapping server names to lists of Prompt objects
        """
        # Get all prompts from the aggregator
        result = await self._aggregator.list_prompts(server_name)

        # Apply filtering if prompts are specified in config
        if self.config.prompts is not None:
            filtered_result = {}
            for server, prompts in result.items():
                # Check if this server has prompt filters
                if server in self.config.prompts:
                    filtered_prompts = []
                    for prompt in prompts:
                        # Check if prompt matches any pattern for this server
                        for pattern in self.config.prompts[server]:
                            if self._matches_pattern(prompt.name, pattern, server):
                                filtered_prompts.append(prompt)
                                break
                    if filtered_prompts:
                        filtered_result[server] = filtered_prompts
            result = filtered_result

        return result

    async def list_resources(self, server_name: str | None = None) -> Dict[str, List[str]]:
        """
        List all resources available to this agent, filtered by configuration.

        Args:
            server_name: Optional server name to list resources from

        Returns:
            Dictionary mapping server names to lists of resource URIs
        """
        # Get all resources from the aggregator
        result = await self._aggregator.list_resources(server_name)

        # Apply filtering if resources are specified in config
        if self.config.resources is not None:
            filtered_result = {}
            for server, resources in result.items():
                # Check if this server has resource filters
                if server in self.config.resources:
                    filtered_resources = []
                    for resource in resources:
                        # Check if resource matches any pattern for this server
                        for pattern in self.config.resources[server]:
                            if self._matches_pattern(resource, pattern, server):
                                filtered_resources.append(resource)
                                break
                    if filtered_resources:
                        filtered_result[server] = filtered_resources
            result = filtered_result

        return result

    async def list_mcp_tools(self, server_name: str | None = None) -> Mapping[str, List[Tool]]:
        """
        List all tools available to this agent, grouped by server and filtered by configuration.

        Args:
            server_name: Optional server name to list tools from

        Returns:
            Dictionary mapping server names to lists of Tool objects (with original names, not namespaced)
        """
        # Get all tools from the aggregator
        result = await self._aggregator.list_mcp_tools(server_name)

        # Apply filtering if tools are specified in config
        if self.config.tools is not None:
            filtered_result = {}
            for server, tools in result.items():
                # Check if this server has tool filters
                if server in self.config.tools:
                    filtered_tools = []
                    for tool in tools:
                        # Check if tool matches any pattern for this server
                        for pattern in self.config.tools[server]:
                            if self._matches_pattern(tool.name, pattern, server):
                                filtered_tools.append(tool)
                                break
                    if filtered_tools:
                        filtered_result[server] = filtered_tools
            result = filtered_result

        # Add human input tool to a special server if human input is configured
        if self.human_input_callback:
            from mcp.server.fastmcp.tools import Tool as FastTool

            human_input_tool: FastTool = FastTool.from_function(self.request_human_input)
            special_server_name = "__human_input__"

            # If the special server doesn't exist in result, create it
            if special_server_name not in result:
                result[special_server_name] = []

            result[special_server_name].append(
                Tool(
                    name=HUMAN_INPUT_TOOL_NAME,
                    description=human_input_tool.description,
                    inputSchema=human_input_tool.parameters,
                )
            )

        return result

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
            skills=skills,
            name=self._name,
            description=self.instruction,
            url=f"fast-agent://agents/{self._name}/",
            version="0.1",
            capabilities=DEFAULT_CAPABILITIES,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            provider=None,
            documentation_url=None,
        )

    async def show_assistant_message(
        self,
        message: PromptMessageMultipart,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
    ) -> None:
        """
        Display an assistant message with MCP servers in the bottom bar.

        This override adds the list of connected MCP servers to the bottom bar
        and highlights servers that were used for tool calls in this message.
        """
        # Get the list of MCP servers (if not provided)
        if bottom_items is None:
            if self._aggregator and self._aggregator.server_names:
                server_names = self._aggregator.server_names
            else:
                server_names = []
        else:
            server_names = bottom_items

        # Extract servers from tool calls in the message for highlighting
        if highlight_items is None:
            highlight_servers = self._extract_servers_from_message(message)
        else:
            # Convert to list if needed
            if isinstance(highlight_items, str):
                highlight_servers = [highlight_items]
            else:
                highlight_servers = highlight_items

        # Call parent's implementation with server information
        await super().show_assistant_message(
            message=message,
            bottom_items=server_names,
            highlight_items=highlight_servers,
            max_item_length=max_item_length or 12,
        )

    def _extract_servers_from_message(self, message: PromptMessageMultipart) -> List[str]:
        """
        Extract server names from tool calls in the message.

        Args:
            message: The message containing potential tool calls

        Returns:
            List of server names that were called
        """
        servers = []

        # Check if message has tool calls
        if message.tool_calls:
            for tool_request in message.tool_calls.values():
                tool_name = tool_request.params.name

                # Use aggregator's mapping to find the server for this tool
                if tool_name in self._aggregator._namespaced_tool_map:
                    namespaced_tool = self._aggregator._namespaced_tool_map[tool_name]
                    if namespaced_tool.server_name not in servers:
                        servers.append(namespaced_tool.server_name)

        return servers

    async def _parse_resource_name(self, name: str, resource_type: str) -> tuple[str, str]:
        """Delegate resource name parsing to the aggregator."""
        return await self._aggregator._parse_resource_name(name, resource_type)

    async def convert(self, tool: Tool) -> AgentSkill:
        """
        Convert a Tool to an AgentSkill.
        """

        _, tool_without_namespace = await self._parse_resource_name(tool.name, "tool")
        return AgentSkill(
            id=tool.name,
            name=tool_without_namespace,
            description=tool.description or "",
            tags=["tool"],
            examples=None,
            input_modes=None,  # ["text/plain"],
            # cover TextContent | ImageContent ->
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/223
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/93
            output_modes=None,  # ,["text/plain", "image/*"],
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
