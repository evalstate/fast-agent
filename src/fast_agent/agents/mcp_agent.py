"""
Base Agent class that implements the AgentProtocol interface.

This class provides default implementations of the standard agent methods
and delegates operations to an attached FastAgentLLMProtocol instance.
"""

import asyncio
import fnmatch
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import mcp
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

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import DEFAULT_CAPABILITIES
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.constants import HUMAN_INPUT_TOOL_NAME
from fast_agent.core.exceptions import PromptExitError
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.mcp.mcp_aggregator import MCPAggregator, ServerStatus
from fast_agent.tools.elicitation import (
    get_elicitation_tool,
    run_elicitation_form,
    set_elicitation_input_callback,
)
from fast_agent.types import PromptMessageExtended, RequestParams

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)

LLM = TypeVar("LLM", bound=FastAgentLLMProtocol)

if TYPE_CHECKING:
    from rich.text import Text

    from fast_agent.context import Context
    from fast_agent.llm.usage_tracking import UsageAccumulator


class McpAgent(ABC, ToolAgent):
    """
    A base Agent class that implements the AgentProtocol interface.

    This class provides default implementations of the standard agent methods
    and delegates LLM operations to an attached FastAgentLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        connection_persistence: bool = True,
        # legacy human_input_callback removed
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
        self._llm: FastAgentLLMProtocol | None = None

        # Instantiate human input tool once if enabled in config
        self._human_input_tool: Tool | None = None
        if self.config.human_input:
            try:
                self._human_input_tool = get_elicitation_tool()
            except Exception:
                self._human_input_tool = None

        # Register the MCP UI handler as the elicitation callback so fast_agent.tools can call it
        # without importing MCP types. This avoids circular imports and ensures the callback is ready.
        try:
            from fast_agent.human_input.elicitation_handler import elicitation_input_callback
            from fast_agent.human_input.types import HumanInputRequest

            async def _mcp_elicitation_adapter(
                request_payload: dict,
                agent_name: str | None = None,
                server_name: str | None = None,
                server_info: dict | None = None,
            ) -> str:
                req = HumanInputRequest(**request_payload)
                resp = await elicitation_input_callback(
                    request=req,
                    agent_name=agent_name,
                    server_name=server_name,
                    server_info=server_info,
                )
                return resp.response if isinstance(resp.response, str) else str(resp.response)

            set_elicitation_input_callback(_mcp_elicitation_adapter)
        except Exception:
            # If UI handler import fails, leave callback unset; tool will error with a clear message
            pass

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

        # Apply template substitution to the instruction with server instructions
        await self._apply_instruction_templates()

    async def shutdown(self) -> None:
        """
        Shutdown the agent and close all MCP server connections.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await self._aggregator.close()

    async def get_server_status(self) -> Dict[str, ServerStatus]:
        """Expose server status details for UI and diagnostics consumers."""
        if not self._aggregator:
            return {}
        return await self._aggregator.collect_server_status()

    @property
    def initialized(self) -> bool:
        """Check if both the agent and aggregator are initialized."""
        return self._initialized and self._aggregator.initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set the initialized state of both agent and aggregator."""
        self._initialized = value
        self._aggregator.initialized = value

    async def _apply_instruction_templates(self) -> None:
        """
        Apply template substitution to the instruction, including server instructions.
        This is called during initialization after servers are connected.
        """
        if not self.instruction:
            return

        # Gather server instructions if the template includes {{serverInstructions}}
        if "{{serverInstructions}}" in self.instruction:
            try:
                instructions_data = await self._aggregator.get_server_instructions()
                server_instructions = self._format_server_instructions(instructions_data)
            except Exception as e:
                self.logger.warning(f"Failed to get server instructions: {e}")
                server_instructions = ""

            # Replace the template variable
            self.instruction = self.instruction.replace(
                "{{serverInstructions}}", server_instructions
            )

        # Update default request params to match
        if self._default_request_params:
            self._default_request_params.systemPrompt = self.instruction

        self.logger.debug(f"Applied instruction templates for agent {self._name}")

    def _format_server_instructions(
        self, instructions_data: Dict[str, tuple[str | None, List[str]]]
    ) -> str:
        """
        Format server instructions with XML tags and tool lists.

        Args:
            instructions_data: Dict mapping server name to (instructions, tool_names)

        Returns:
            Formatted string with server instructions
        """
        if not instructions_data:
            return ""

        formatted_parts = []
        for server_name, (instructions, tool_names) in instructions_data.items():
            # Skip servers with no instructions
            if instructions is None:
                continue

            # Format tool names with server prefix
            prefixed_tools = [f"{server_name}-{tool}" for tool in tool_names]
            tools_list = ", ".join(prefixed_tools) if prefixed_tools else "No tools available"

            formatted_parts.append(
                f'<mcp-server name="{server_name}">\n'
                f"<tools>{tools_list}</tools>\n"
                f"<instructions>\n{instructions}\n</instructions>\n"
                f"</mcp-server>"
            )

        if formatted_parts:
            return "\n\n".join(formatted_parts)
        return ""

    async def __call__(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
    ) -> str:
        return await self.send(message)

    # async def send(
    #     self,
    #     message: Union[
    #         str,
    #         PromptMessage,
    #         PromptMessageExtended,
    #         Sequence[Union[str, PromptMessage, PromptMessageExtended]],
    #     ],
    #     request_params: RequestParams | None = None,
    # ) -> str:
    #     """
    #     Send a message to the agent and get a response.

    #     Args:
    #         message: Message content in various formats:
    #             - String: Converted to a user PromptMessageExtended
    #             - PromptMessage: Converted to PromptMessageExtended
    #             - PromptMessageExtended: Used directly
    #             - request_params: Optional request parameters

    #     Returns:
    #         The agent's response as a string
    #     """
    #     response = await self.generate(message, request_params)
    #     return response.last_text() or ""

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

        # Append human input tool if enabled and available
        if self.config.human_input and getattr(self, "_human_input_tool", None):
            result.tools.append(self._human_input_tool)

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
            # Call the elicitation-backed human input tool
            return await self._call_human_input_tool(arguments)
        else:
            return await self._aggregator.call_tool(name, arguments)

    async def _call_human_input_tool(
        self, arguments: Dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Handle human input via an elicitation form.

        Expected inputs:
        - Either an object with optional 'message' and a 'schema' JSON Schema (object), or
        - The JSON Schema (object) itself as the arguments.

        Constraints:
        - No more than 7 top-level properties are allowed in the schema.
        """
        try:
            # Run via shared tool runner
            resp_text = await run_elicitation_form(arguments, agent_name=self._name)
            if resp_text == "__DECLINED__":
                return CallToolResult(
                    isError=False,
                    content=[TextContent(type="text", text="The Human declined the input request")],
                )
            if resp_text in ("__CANCELLED__", "__DISABLE_SERVER__"):
                return CallToolResult(
                    isError=False,
                    content=[
                        TextContent(type="text", text="The Human cancelled the input request")
                    ],
                )
            # Success path: return the (JSON) response as-is
            return CallToolResult(
                isError=False,
                content=[TextContent(type="text", text=resp_text)],
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
        namespace: str | None = None,
        server_name: str | None = None,
    ) -> GetPromptResult:
        """
        Get a prompt from a server.

        Args:
            prompt_name: Name of the prompt, optionally namespaced
            arguments: Optional dictionary of arguments to pass to the prompt template
            namespace: Optional namespace (server) to get the prompt from

        Returns:
            GetPromptResult containing the prompt information
        """
        target = namespace if namespace is not None else server_name
        return await self._aggregator.get_prompt(prompt_name, arguments, target)

    async def apply_prompt(
        self,
        prompt: Union[str, GetPromptResult],
        arguments: Dict[str, str] | None = None,
        as_template: bool = False,
        namespace: str | None = None,
        **_: Any,
    ) -> str:
        """
        Apply an MCP Server Prompt by name or GetPromptResult and return the assistant's response.
        Will search all available servers for the prompt if not namespaced and no server_name provided.

        If the last message in the prompt is from a user, this will automatically
        generate an assistant response to ensure we always end with an assistant message.

        Args:
            prompt: The name of the prompt to apply OR a GetPromptResult object
            arguments: Optional dictionary of string arguments to pass to the prompt template
            as_template: If True, store as persistent template (always included in context)
            namespace: Optional namespace/server to resolve the prompt from

        Returns:
            The assistant's response or error message
        """

        # Handle both string and GetPromptResult inputs
        if isinstance(prompt, str):
            prompt_name = prompt
            # Get the prompt - this will search all servers if needed
            self.logger.debug(f"Loading prompt '{prompt_name}'")
            prompt_result: GetPromptResult = await self.get_prompt(
                prompt_name, arguments, namespace
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
        multipart_messages = PromptMessageExtended.from_get_prompt_result(prompt_result)

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
            List of EmbeddedResource objects ready to use in a PromptMessageExtended

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
        self, resource_uri: str, namespace: str | None = None, server_name: str | None = None
    ) -> ReadResourceResult:
        """
        Get a resource from an MCP server.

        Args:
            resource_uri: URI of the resource to retrieve
            namespace: Optional namespace (server) to retrieve the resource from

        Returns:
            ReadResourceResult containing the resource data

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        # Get the raw resource result
        target = namespace if namespace is not None else server_name
        result: ReadResourceResult = await self._aggregator.get_resource(resource_uri, target)
        return result

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageExtended],
        resource_uri: str,
        namespace: str | None = None,
        server_name: str | None = None,
    ) -> str:
        """
        Create a prompt with the given content and resource, then send it to the agent.

        Args:
            prompt_content: Content in various formats:
                - String: Converted to a user message with the text
                - PromptMessage: Converted to PromptMessageExtended
                - PromptMessageExtended: Used directly
            resource_uri: URI of the resource to retrieve
            namespace: Optional namespace (server) to retrieve the resource from

        Returns:
            The agent's response as a string
        """
        # Get the embedded resources
        embedded_resources: List[EmbeddedResource] = await self.get_embedded_resources(
            resource_uri, namespace if namespace is not None else server_name
        )

        # Create or update the prompt message
        prompt: PromptMessageExtended
        if isinstance(prompt_content, str):
            # Create a new prompt with the text and resources
            content = [TextContent(type="text", text=prompt_content)]
            content.extend(embedded_resources)
            prompt = PromptMessageExtended(role="user", content=content)
        elif isinstance(prompt_content, PromptMessage):
            # Convert PromptMessage to PromptMessageExtended and add resources
            content = [prompt_content.content]
            content.extend(embedded_resources)
            prompt = PromptMessageExtended(role=prompt_content.role, content=content)
        elif isinstance(prompt_content, PromptMessageExtended):
            # Add resources to the existing prompt
            prompt = prompt_content
            prompt.content.extend(embedded_resources)
        else:
            raise TypeError(
                "prompt_content must be a string, PromptMessage, or PromptMessageExtended"
            )

        response: PromptMessageExtended = await self.generate([prompt], None)
        return response.first_text()

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended:
        """Override ToolAgent's run_tools to use MCP tools via aggregator."""
        if not request.tool_calls:
            self.logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}
        self._tool_loop_error = None

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

            tool_available = False
            if tool_name == HUMAN_INPUT_TOOL_NAME:
                tool_available = True
            elif namespaced_tool:
                tool_available = True
            else:
                tool_available = any(
                    candidate.tool.name == tool_name
                    for candidate in self._aggregator._namespaced_tool_map.values()
                )

            if not tool_available:
                error_message = f"Tool '{display_tool_name}' is not available"
                self.logger.error(error_message)
                self._mark_tool_loop_error(
                    correlation_id=correlation_id,
                    error_message=error_message,
                    tool_results=tool_results,
                )
                break

            # Find the index of the current tool in available_tools for highlighting
            highlight_index = None
            try:
                highlight_index = available_tools.index(display_tool_name)
            except ValueError:
                # Tool not found in list, no highlighting
                pass

            self.display.show_tool_call(
                name=self._name,
                tool_args=tool_args,
                bottom_items=available_tools,
                tool_name=display_tool_name,
                highlight_index=highlight_index,
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

        return self._finalize_tool_results(tool_results)

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

    # async def structured(
    #     self,
    #     messages: Union[
    #         str,
    #         PromptMessage,
    #         PromptMessageExtended,
    #         Sequence[Union[str, PromptMessage, PromptMessageExtended]],
    #     ],
    #     model: Type[ModelT],
    #     request_params: RequestParams | None = None,
    # ) -> Tuple[ModelT | None, PromptMessageExtended]:
    #     """
    #     Apply the prompt and return the result as a Pydantic model.
    #     Normalizes input messages and delegates to the attached LLM.

    #     Args:
    #         messages: Message(s) in various formats:
    #             - String: Converted to a user PromptMessageExtended
    #             - PromptMessage: Converted to PromptMessageExtended
    #             - PromptMessageExtended: Used directly
    #             - List of any combination of the above
    #         model: The Pydantic model class to parse the result into
    #         request_params: Optional parameters to configure the LLM request

    #     Returns:
    #         An instance of the specified model, or None if coercion fails
    #     """

    #     with self._tracer.start_as_current_span(f"Agent: '{self._name}' structured"):
    #         return await super().structured(messages, model, request_params)

    async def apply_prompt_messages(
        self, prompts: List[PromptMessageExtended], request_params: RequestParams | None = None
    ) -> str:
        """
        Apply a list of prompt messages and return the result.

        Args:
            prompts: List of PromptMessageExtended messages
            request_params: Optional request parameters

        Returns:
            The text response from the LLM
        """

        response = await self.generate(prompts, request_params)
        return response.first_text()

    async def list_prompts(
        self, namespace: str | None = None, server_name: str | None = None
    ) -> Mapping[str, List[mcp.types.Prompt]]:
        """
        List all prompts available to this agent, filtered by configuration.

        Args:
            namespace: Optional namespace (server) to list prompts from

        Returns:
            Dictionary mapping server names to lists of Prompt objects
        """
        # Get all prompts from the aggregator
        target = namespace if namespace is not None else server_name
        result = await self._aggregator.list_prompts(target)

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

    async def list_resources(
        self, namespace: str | None = None, server_name: str | None = None
    ) -> Dict[str, List[str]]:
        """
        List all resources available to this agent, filtered by configuration.

        Args:
            namespace: Optional namespace (server) to list resources from

        Returns:
            Dictionary mapping server names to lists of resource URIs
        """
        # Get all resources from the aggregator
        target = namespace if namespace is not None else server_name
        result = await self._aggregator.list_resources(target)

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

    async def list_mcp_tools(
        self, namespace: str | None = None, server_name: str | None = None
    ) -> Mapping[str, List[Tool]]:
        """
        List all tools available to this agent, grouped by server and filtered by configuration.

        Args:
            namespace: Optional namespace (server) to list tools from

        Returns:
            Dictionary mapping server names to lists of Tool objects (with original names, not namespaced)
        """
        # Get all tools from the aggregator
        target = namespace if namespace is not None else server_name
        result = await self._aggregator.list_mcp_tools(target)

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

        # Add elicitation-backed human input tool to a special server if enabled and available
        if self.config.human_input and getattr(self, "_human_input_tool", None):
            special_server_name = "__human_input__"

            # If the special server doesn't exist in result, create it
            if special_server_name not in result:
                result[special_server_name] = []

            result[special_server_name].append(self._human_input_tool)

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
        message: PromptMessageExtended,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Optional["Text"] = None,
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
            name=name,
            model=model,
            additional_message=additional_message,
        )

    def _extract_servers_from_message(self, message: PromptMessageExtended) -> List[str]:
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
    def message_history(self) -> List[PromptMessageExtended]:
        """
        Return the agent's message history as PromptMessageExtended objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageExtended objects representing the conversation history
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
