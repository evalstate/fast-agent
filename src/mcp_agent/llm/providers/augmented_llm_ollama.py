import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp
from mcp.types import TextContent, CallToolResult

from mcp_agent.core.prompt import PromptMessageMultipart
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import TurnUsage, FastAgentUsage

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"

logger = logging.getLogger(__name__)


def _extract_tool_result_text(result: CallToolResult) -> TextContent:
    """Extract text content from a CallToolResult."""
    if hasattr(result, 'content') and result.content:
        if isinstance(result.content, list) and len(result.content) > 0:
            content_item = result.content[0]
            if hasattr(content_item, 'text'):
                return content_item.text
            else:
                return content_item
        else:
            return result.content[0]
    else:
        return result.content[0]


def _convert_mcp_tools_to_ollama(mcp_tools) -> List[Dict[str, Any]]:
    """Convert MCP tools to Ollama format."""
    ollama_tools = []

    for tool in mcp_tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            }
        })

    return ollama_tools


class OllamaAugmentedLLM(AugmentedLLM):
    """Native Ollama provider with tool calling support."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.OLLAMA, **kwargs)
        self._client = None

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Ollama parameters."""
        chosen_model = kwargs.get("model", DEFAULT_OLLAMA_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        """Get Ollama base URL."""
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        if self.context.config and hasattr(self.context.config, 'ollama'):
            # Handle both dict and object access patterns
            ollama_config = self.context.config.ollama
            if isinstance(ollama_config, dict):
                base_url = ollama_config.get('base_url', base_url)
            else:
                base_url = getattr(ollama_config, 'base_url', base_url)
        return base_url

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get or create an HTTP client."""
        if self._client is None or self._client.closed:
            # Create a client with proper timeout and connector settings
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=300)
            self._client = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self._client

    async def _ensure_client_closed(self):
        """Ensure the HTTP client is properly closed."""
        if hasattr(self, '_client') and self._client and not self._client.closed:
            await self._client.close()
            self._client = None

    def __del__(self):
        """Destructor - schedule cleanup if not already done."""
        if hasattr(self, '_client') and self._client and not self._client.closed:
            # Schedule cleanup without failing if the event loop is closed
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task to close the client
                    asyncio.create_task(self._ensure_client_closed())
            except (RuntimeError, AttributeError):
                # Event loop is closed or not available, ignore
                pass

    async def _apply_prompt_provider_specific(
            self,
            multipart_messages: List[PromptMessageMultipart],
            request_params: Optional[RequestParams] = None,
            **kwargs,
    ) -> PromptMessageMultipart:
        """
        Apply prompt using Ollama's native API.
        :param **kwargs:
        """
        try:
            # Get tools from the aggregator (this should be the agent's MCPAggregator)
            tools = None
            if hasattr(self, 'aggregator') and self.aggregator:
                tools_result = await self.aggregator.list_tools()
                if tools_result and tools_result.tools:
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema
                            }
                        }
                        for tool in tools_result.tools
                    ]

            # Generate response with tools (returns Dict[str, Any])
            response_dict = await self._generate_with_tools(multipart_messages, tools, request_params)

            # Check if the response contains tool calls
            message = response_dict.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                # Handle tool calls with a proper display
                result = await self._handle_tool_calls_and_continue(
                    response_dict, multipart_messages, request_params
                )
            else:
                # Show assistant response
                response_text = message.get("content", "")
                await self.show_assistant_message(response_text)

                # Create PromptMessageMultipart with the response text
                result = PromptMessageMultipart(
                    role="assistant",
                    content=[TextContent(type="text", text=response_text)]
                )

            return result

        except Exception as e:
            logger.error(f"Error in _apply_prompt_provider_specific: {e}", exc_info=True)
            raise
        finally:
            # Always clean up the client connection after each agent execution
            await self._ensure_client_closed()

    async def generate_messages(
            self,
            multipart_messages: List[PromptMessageMultipart],
            request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Generate messages with visual feedback similar to Anthropic provider.
        This method provides the interactive conversation flow display.
        """
        try:
            # Call the main processing method
            result = await self._apply_prompt_provider_specific(multipart_messages, request_params)

            # Show usage summary if available
            if hasattr(self, 'usage_accumulator') and self.usage_accumulator:
                usage_summary = self.usage_accumulator.get_summary()
                if usage_summary:
                    # Display usage information similar to Anthropic
                    self.display.show_usage_summary(usage_summary)

            return result

        except Exception as e:
            logger.error(f"Error in generate_messages: {e}")
            raise

    async def _handle_tool_calls_and_continue(
            self,
            initial_response: Dict[str, Any],
            original_messages: List[PromptMessageMultipart],
            request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """Handle tool calls, execute them, and let the model continue with the results."""

        message = initial_response.get("message", {})
        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "")

        if not tool_calls:
            return PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text=content)]
            )

        # Execute all tool calls
        tool_results = []
        for i, tool_call in enumerate(tool_calls):
            try:
                result = await self._execute_tool_call(tool_call)

                # Extract text from CallToolResult
                tool_result_text = _extract_tool_result_text(result)
                tool_results.append({
                    "call": tool_call,
                    "result": tool_result_text
                })

            except Exception as e:
                logger.error(f"Error executing tool call: {e}", exc_info=True)
                tool_results.append({
                    "call": tool_call,
                    "result": f"Error: {str(e)}"
                })

        # Now continue the conversation with tool results
        if tool_results:
            return await self._continue_conversation_with_tool_results(
                original_messages, initial_response, tool_results, request_params
            )
        else:
            return PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="Tool calls completed.")]
            )

    async def _continue_conversation_with_tool_results(
            self,
            original_messages: List[PromptMessageMultipart],
            initial_response: Dict[str, Any],
            tool_results: List[Dict[str, Any]],
            request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """Continue the conversation after tool execution, letting the model process the results."""

        # Build conversation history including tool calls and results
        conversation_messages = []

        # Add original messages
        for msg in original_messages:
            conversation_messages.append(msg)

        # Add the assistant's response with tool calls (if any initial content)
        assistant_message = initial_response.get("message", {})
        if assistant_message.get("content"):
            assistant_content = assistant_message["content"]
            conversation_messages.append(PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text=assistant_content)]
            ))

        # Add tool results as user messages (this is how Ollama expects it)
        for tool_result in tool_results:
            tool_call = tool_result["call"]
            result_text = tool_result["result"]

            # Format the tool result as a user message
            tool_message = f"Tool call result for {tool_call.get('function', {}).get('name', 'unknown')}:\n{result_text}"
            conversation_messages.append(PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text=tool_message)]
            ))

        # Now get the model's final response
        final_response = await self._generate_with_tools(conversation_messages, None,
                                                         request_params)  # Don't pass tools again

        # Extract the final content
        final_content = final_response.get("message", {}).get("content", "")

        if not final_content:
            final_content = "No response generated."

        result = PromptMessageMultipart(
            role="assistant",
            content=[TextContent(type="text", text=final_content)]
        )

        return result

    async def _generate_with_tools(
            self,
            messages: List[PromptMessageMultipart],
            tools: Optional[List[Dict[str, Any]]] = None,
            request_params: Optional[RequestParams] = None
    ) -> Dict[str, Any]:
        """Generate a response using Ollama's native API with tool support."""
        client = await self._get_client()

        try:
            # Convert messages to Ollama format, including system prompt
            ollama_messages = self._convert_messages_to_ollama(messages, request_params)

            # Use effective request params
            effective_params = self.get_request_params(request_params)

            # Build request payload
            payload = {
                "model": effective_params.model,
                "messages": ollama_messages,
                "stream": False,
            }

            # Add tools if provided
            if tools:
                payload["tools"] = tools

            async with client.post(
                    f"{self._base_url()}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")

                response_data = await response.json()

                # Add usage tracking if the response contains usage data
                if response_data.get('done') and response_data.get('prompt_eval_count') is not None:
                    # Create a FastAgentUsage object that matches the expected schema
                    # Convert token counts to character estimates (rough approximation)
                    input_chars = response_data.get('prompt_eval_count', 0) * 4  # ~4 chars per token
                    output_chars = response_data.get('eval_count', 0) * 4

                    ollama_usage = FastAgentUsage(
                        input_chars=input_chars,
                        output_chars=output_chars,
                        model_type="ollama",
                        tool_calls=len(tools) if tools else 0,
                        delay_seconds=response_data.get('total_duration', 0) / 1_000_000_000
                        # Convert nanoseconds to seconds
                    )

                    turn_usage = TurnUsage.from_fast_agent(
                        usage=ollama_usage,
                        model=response_data.get('model', effective_params.model)
                    )
                    # self.usage_accumulator.add_turn_usage(turn_usage)
                    self.usage_accumulator.add_turn(turn_usage)

                return response_data

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}", exc_info=True)
            raise

    def _convert_messages_to_ollama(
            self,
            messages: List[PromptMessageMultipart],
            request_params: Optional[RequestParams] = None
    ) -> List[Dict[str, Any]]:
        """Convert multipart messages to Ollama format, including system prompt."""
        ollama_messages = []

        # Get effective request params to access the system prompt
        effective_params = self.get_request_params(request_params)

        # Add a system message if we have a system prompt
        if effective_params.systemPrompt:
            ollama_messages.append({
                "role": "system",
                "content": effective_params.systemPrompt
            })

        # Convert the provided messages
        for i, message in enumerate(messages):
            if len(message.content) == 1 and hasattr(message.content[0], 'text'):
                content_text = message.content[0].text
                ollama_messages.append({
                    "role": message.role,
                    "content": content_text
                })
            else:
                # Handle multipart content
                text_parts = []
                for content in message.content:
                    if hasattr(content, 'text'):
                        text_parts.append(content.text)
                    elif hasattr(content, 'resource'):
                        text_parts.append(f"[Resource: {content.resource}]")

                combined_content = " ".join(text_parts)
                ollama_messages.append({
                    "role": message.role,
                    "content": combined_content
                })

        return ollama_messages

    async def _execute_tool_call(self, tool_call: dict) -> CallToolResult:
        """Execute a single tool call and return the result as a string."""
        function_call = tool_call["function"]
        tool_name = function_call["name"]
        try:
            # Parse arguments - they might be a string or already a dict
            tool_args = function_call["arguments"]
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)

            # Get available tools for display
            available_tools = []
            if hasattr(self, 'aggregator') and self.aggregator:
                tools_result = await self.aggregator.list_tools()
                if tools_result and tools_result.tools:
                    available_tools = [
                        {
                            "name": tool.name,
                            "description": tool.description
                        }
                        for tool in tools_result.tools
                    ]

            # Show the tool call using the existing display method
            self.show_tool_call(available_tools, tool_name, tool_args)

            # Execute the tool
            if hasattr(self, 'aggregator') and self.aggregator:
                result = await self.aggregator.call_tool(tool_name, tool_args)

                # Show the tool result using the existing display method
                self.show_tool_result(result)

                # Format result as text for the conversation
                if result.isError:
                    error_text = []
                    for content_item in result.content:
                        if hasattr(content_item, "text"):
                            error_text.append(content_item.text)
                        else:
                            error_text.append(str(content_item))
                    return CallToolResult(content=[TextContent(type="text", text="\n".join(error_text))], isError=True)
                else:
                    result_text = []
                    for content_item in result.content:
                        if hasattr(content_item, "text"):
                            result_text.append(content_item.text)
                        else:
                            result_text.append(str(content_item))
                    return CallToolResult(content=[TextContent(type="text", text="\n".join(result_text))], isError=False)
            else:
                error_msg = f"No aggregator available to execute tool '{tool_name}'"
                logger.error(error_msg)
                return CallToolResult(content=[TextContent(type="text", text=error_msg)], isError=True)

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg)
            return CallToolResult(content=[TextContent(type="text", text=error_msg)], isError=True)

    async def close(self):
        """Close the HTTP client properly."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.error(f"Error closing client in close(): {e}")
            finally:
                self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
