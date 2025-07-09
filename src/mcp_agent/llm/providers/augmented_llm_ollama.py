import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional

import aiohttp
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent

from mcp_agent.core.prompt import PromptMessageMultipart
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import FastAgentUsage, TurnUsage

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"

logger = logging.getLogger(__name__)

OllamaRole = Literal["system", "user", "assistant", "tool"]


class OllamaPromptMessageMultipart(PromptMessageMultipart):
    """Extended PromptMessageMultipart that supports the 'tool' role for Ollama."""

    role: OllamaRole


def _extract_tool_result_text(result: CallToolResult) -> str:
    """Extract text content from a CallToolResult."""
    if hasattr(result, "content") and result.content:
        if isinstance(result.content, list) and len(result.content) > 0:
            content_item = result.content[0]
            if hasattr(content_item, "text"):
                return content_item.text
            else:
                return str(content_item)
        else:
            return str(result.content[0])
    else:
        return str(result.content[0])


def _convert_mcp_tools_to_ollama(mcp_tools) -> List[Dict[str, Any]]:
    """Convert MCP tools to Ollama format."""
    ollama_tools = []

    for tool in mcp_tools:
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                },
            }
        )

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
        if self.context.config and hasattr(self.context.config, "ollama"):
            # Handle both dict and object access patterns
            ollama_config = self.context.config.ollama
            if isinstance(ollama_config, dict):
                base_url = ollama_config.get("base_url", base_url)
            else:
                base_url = getattr(ollama_config, "base_url", base_url)
        return base_url

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get or create an HTTP client."""
        if self._client is None or self._client.closed:
            # Create headers - only add Authorization if we have a token
            headers = {"Content-Type": "application/json"}
            auth_header = self._get_authorization_header()
            if auth_header:
                headers["Authorization"] = auth_header

            # Create a client with proper timeout, connector settings, and headers
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=300)
            self._client = aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers=headers
            )
        return self._client

    async def _ensure_client_closed(self):
        """Ensure the HTTP client is properly closed."""
        if hasattr(self, "_client") and self._client and not self._client.closed:
            await self._client.close()
            self._client = None

    def __del__(self):
        """Destructor - schedule cleanup if not already done."""
        if hasattr(self, "_client") and self._client and not self._client.closed:
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
        is_template: bool = False,
        **kwargs,
    ) -> PromptMessageMultipart:
        """
        Apply prompt using Ollama's native API.
        """
        try:
            # Get tools from the aggregator (this should be the agent's MCPAggregator)
            tools = None
            if hasattr(self, "aggregator") and self.aggregator:
                tools_result = await self.aggregator.list_tools()
                if tools_result and tools_result.tools:
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        }
                        for tool in tools_result.tools
                    ]

            # Generate response with tools (returns Dict[str, Any])
            response_dict = await self._generate_with_tools(
                self._message_history, tools, request_params
            )

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
                    role="assistant", content=[TextContent(type="text", text=response_text)]
                )

            return result

        except Exception as e:
            logger.error(f"Error in _apply_prompt_provider_specific: {e}", exc_info=True)
            raise
        finally:
            # Always clean up the client connection after each agent execution
            await self._ensure_client_closed()

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
                role="assistant", content=[TextContent(type="text", text=content)]
            )

        # Execute all tool calls
        tool_results = []
        for i, tool_call in enumerate(tool_calls):
            try:
                result = await self._execute_tool_call(tool_call)

                # Extract text from CallToolResult
                tool_result_text = _extract_tool_result_text(result)
                tool_results.append({"call": tool_call, "result": tool_result_text})

            except Exception as e:
                logger.error(f"Error executing tool call: {e}", exc_info=True)
                tool_results.append({"call": tool_call, "result": f"Error: {str(e)}"})

        # Now continue the conversation with tool results
        if tool_results:
            return await self._continue_conversation_with_tool_results(
                original_messages, initial_response, tool_results, request_params
            )
        else:
            return PromptMessageMultipart(
                role="assistant", content=[TextContent(type="text", text="Tool calls completed.")]
            )

    async def _continue_conversation_with_tool_results(
        self,
        original_messages: List[PromptMessageMultipart],
        initial_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """Continue the conversation after tool execution, letting the model process the results."""

        # Get tools for potential follow-up calls
        tools = None
        if hasattr(self, "aggregator") and self.aggregator:
            tools_result = await self.aggregator.list_tools()
            if tools_result and tools_result.tools:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                    for tool in tools_result.tools
                ]

        # Add the assistant's response with tool calls to the main history
        assistant_message = initial_response.get("message", {})
        tool_calls = assistant_message.get("tool_calls", [])
        assistant_content = assistant_message.get("content", "")

        if tool_calls:
            assistant_msg = PromptMessageMultipart(
                role="assistant", content=[TextContent(type="text", text=assistant_content)]
            )
            self._message_history.append(assistant_msg)

        # Add tool results directly to the main history
        for tool_result in tool_results:
            result_text = tool_result["result"]

            # Use our extended model that supports the "tool" role
            tool_message = OllamaPromptMessageMultipart(
                role="tool", content=[TextContent(type="text", text=result_text)]
            )
            self._message_history.append(tool_message)

        # Now get the model's final response using the main history
        final_response = await self._generate_with_tools(
            self._message_history, tools, request_params
        )

        # Check if the final response also contains tool calls
        final_message = final_response.get("message", {})
        final_tool_calls = final_message.get("tool_calls", [])

        if final_tool_calls:
            # Handle follow-up tool calls recursively
            return await self._handle_tool_calls_and_continue(
                final_response, original_messages, request_params
            )
        else:
            # Extract the final content
            final_content = final_message.get("content", "")

            if not final_content:
                final_content = "No response generated."

            # Show the assistant's final response
            await self.show_assistant_message(final_content)

            # Create and return the final assistant response message
            # Note: This will be added to history by the calling method
            return PromptMessageMultipart(
                role="assistant", content=[TextContent(type="text", text=final_content)]
            )

    async def _generate_with_tools(
        self,
        messages: List[PromptMessageMultipart],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_params: Optional[RequestParams] = None,
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
                "stream": True,  # Enable streaming
            }

            # Add tools if provided
            if tools:
                payload["tools"] = tools

            # Log chat progress before starting the request (like OpenAI provider does)
            self._log_chat_progress(self.chat_turn(), model=effective_params.model)

            async with client.post(
                f"{self._base_url()}/api/chat",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")

                # Process streaming response
                accumulated_response = {
                    "model": effective_params.model,
                    "created_at": None,
                    "message": {"role": "assistant", "content": "", "tool_calls": []},
                    "done": False,
                    "total_duration": None,
                    "load_duration": None,
                    "prompt_eval_count": None,
                    "prompt_eval_duration": None,
                    "eval_count": None,
                    "eval_duration": None,
                }

                estimated_tokens = 0

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)

                        # Update basic response metadata
                        if chunk.get("created_at"):
                            accumulated_response["created_at"] = chunk["created_at"]
                        if chunk.get("model"):
                            accumulated_response["model"] = chunk["model"]

                        # Process message content
                        if "message" in chunk:
                            message = chunk["message"]

                            # Accumulate content
                            if "content" in message and message["content"]:
                                content = message["content"]
                                accumulated_response["message"]["content"] += content

                                # Update streaming progress
                                estimated_tokens = self._update_streaming_progress(
                                    content, effective_params.model, estimated_tokens
                                )

                            # Handle tool calls
                            if "tool_calls" in message and message["tool_calls"]:
                                accumulated_response["message"]["tool_calls"] = message[
                                    "tool_calls"
                                ]

                        # Check if done
                        if chunk.get("done", False):
                            accumulated_response["done"] = True
                            accumulated_response["done_reason"] = chunk.get("done_reason")
                            accumulated_response["total_duration"] = chunk.get("total_duration")
                            accumulated_response["load_duration"] = chunk.get("load_duration")
                            accumulated_response["prompt_eval_count"] = chunk.get(
                                "prompt_eval_count"
                            )
                            accumulated_response["prompt_eval_duration"] = chunk.get(
                                "prompt_eval_duration"
                            )
                            accumulated_response["eval_count"] = chunk.get("eval_count")
                            accumulated_response["eval_duration"] = chunk.get("eval_duration")
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON chunk: {line}")
                        continue

                # Add usage tracking if the response contains usage data
                if (
                    accumulated_response.get("done")
                    and accumulated_response.get("prompt_eval_count") is not None
                ):
                    # Create a FastAgentUsage object that matches the expected schema
                    # Convert token counts to character estimates (rough approximation)
                    input_chars = (
                        accumulated_response.get("prompt_eval_count", 0) * 4
                    )  # ~4 chars per token
                    output_chars = accumulated_response.get("eval_count", 0) * 4

                    ollama_usage = FastAgentUsage(
                        input_chars=input_chars,
                        output_chars=output_chars,
                        model_type="ollama",
                        tool_calls=len(tools) if tools else 0,
                        delay_seconds=accumulated_response.get("total_duration", 0) / 1_000_000_000,
                        # Convert nanoseconds to seconds
                    )

                    turn_usage = TurnUsage.from_fast_agent(
                        usage=ollama_usage,
                        model=accumulated_response.get("model", effective_params.model),
                    )
                    self.usage_accumulator.add_turn(turn_usage)

                # Log chat finished (like OpenAI provider does)
                self._log_chat_finished(model=effective_params.model)

                return accumulated_response

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}", exc_info=True)
            raise

    async def _execute_tool_call(self, tool_call: dict) -> CallToolResult:
        """Execute a single tool call and return the result."""
        function_call = tool_call["function"]
        tool_name = function_call["name"]
        try:
            # Parse arguments - they might be a string or already a dict
            tool_args = function_call["arguments"]
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)

            # Get available tools for display
            available_tools = []
            if hasattr(self, "aggregator") and self.aggregator:
                tools_result = await self.aggregator.list_tools()
                if tools_result and tools_result.tools:
                    available_tools = [
                        {"name": tool.name, "description": tool.description}
                        for tool in tools_result.tools
                    ]

            # Show the tool call using the existing display method
            self.show_tool_call(available_tools, tool_name, tool_args)

            # Execute the tool
            if hasattr(self, "aggregator") and self.aggregator:
                result = await self.aggregator.call_tool(tool_name, tool_args)

                # Show the tool result using the existing display method
                self.show_tool_result(result)

                return result
            else:
                error_msg = f"No aggregator available to execute tool '{tool_name}'"
                logger.error(error_msg)
                return CallToolResult(
                    content=[TextContent(type="text", text=error_msg)], isError=True
                )

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

    def _get_authorization_header(self) -> Optional[str]:
        """Get an Authorization header for Ollama API if configured."""
        # Check for auth token in environment variable first
        auth_token = os.getenv("OLLAMA_AUTH_TOKEN")

        # Then check in config
        if not auth_token and self.context.config and hasattr(self.context.config, "ollama"):
            ollama_config = self.context.config.ollama
            if isinstance(ollama_config, dict):
                auth_token = ollama_config.get("api_key", None)

        if auth_token:
            return f"Bearer {auth_token}"

        return None

    def _convert_messages_to_ollama(
        self, messages: List[PromptMessageMultipart], request_params: Optional[RequestParams] = None
    ) -> List[Dict[str, Any]]:
        """Convert multipart messages to Ollama format, including system prompt, tool messages, and multimodal content."""
        ollama_messages = []

        # Get effective request params to access the system prompt
        effective_params = self.get_request_params(request_params)

        # Add a system message if we have a system prompt
        if effective_params.systemPrompt:
            ollama_messages.append({"role": "system", "content": effective_params.systemPrompt})

        # Convert the provided messages
        for message in messages:
            if message.role == "tool":
                # Handle tool messages (text only)
                if len(message.content) == 1 and hasattr(message.content[0], "text"):
                    ollama_messages.append({"role": "tool", "content": message.content[0].text})
                else:
                    # Fallback for complex tool content
                    text_parts = self._extract_text_from_content(message.content)
                    ollama_messages.append({"role": "tool", "content": " ".join(text_parts)})
            else:
                # Handle user/assistant messages with potential multimodal content
                ollama_message = self._convert_multipart_message(message)
                ollama_messages.append(ollama_message)

        return ollama_messages

    def _convert_multipart_message(self, message: PromptMessageMultipart) -> Dict[str, Any]:
        """Convert a single multipart message to Ollama format with multimodal support."""
        if len(message.content) == 1 and hasattr(message.content[0], "text"):
            # Simple text-only message
            return {"role": message.role, "content": message.content[0].text}

        # Handle multimodal content
        text_parts = []
        images = []

        for content in message.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
            elif isinstance(content, ImageContent):
                # Convert image to base64 for Ollama
                image_data = self._convert_image_content(content)
                if image_data:
                    images.append(image_data)
            elif isinstance(content, EmbeddedResource):
                # Handle embedded resources (PDFs, etc.)
                resource_text = self._handle_embedded_resource(content)
                if resource_text:
                    text_parts.append(resource_text)
            else:
                # Handle other content types
                if hasattr(content, "text"):
                    text_parts.append(content.text)
                elif hasattr(content, "resource"):
                    text_parts.append(f"[Resource: {content.resource}]")

        # Build the Ollama message
        ollama_message = {
            "role": message.role,
            "content": " ".join(text_parts) if text_parts else "",
        }

        # Add images if present
        if images:
            ollama_message["images"] = images

        return ollama_message

    def _convert_image_content(self, image_content: ImageContent) -> Optional[str]:
        """Convert ImageContent to base64 string for Ollama."""
        try:
            if hasattr(image_content, "data") and image_content.data:
                # Image data is already base64 encoded
                return image_content.data
            elif hasattr(image_content, "url") and image_content.url:
                # Handle image URLs - would need to fetch and encode
                logger.warning(f"Image URL not directly supported in Ollama: {image_content.url}")
                return None
            else:
                logger.warning("ImageContent missing both data and url")
                return None
        except Exception as e:
            logger.error(f"Error converting image content: {e}")
            return None

    def _handle_embedded_resource(self, resource: EmbeddedResource) -> Optional[str]:
        """Handle embedded resources like PDFs."""
        try:
            if hasattr(resource, "text") and resource.text:
                return resource.text
            elif hasattr(resource, "blob") and resource.blob:
                # For PDFs and other binary content, we'd need to extract text
                logger.warning(
                    f"Binary resource content not directly supported: {resource.mimeType}"
                )
                return f"[Binary Resource: {resource.mimeType}]"
            else:
                return f"[Resource: {getattr(resource, 'uri', 'unknown')}]"
        except Exception as e:
            logger.error(f"Error handling embedded resource: {e}")
            return None

    def _extract_text_from_content(self, content_list: List[Any]) -> List[str]:
        """Extract text parts from a list of content objects."""
        text_parts = []
        for content in content_list:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
            elif hasattr(content, "text"):
                text_parts.append(content.text)
            elif isinstance(content, ImageContent):
                text_parts.append("[Image]")
            elif isinstance(content, EmbeddedResource):
                text_parts.append(f"[Resource: {getattr(content, 'uri', 'unknown')}]")
            else:
                text_parts.append(str(content))
        return text_parts
