import os
from typing import Dict, List

from huggingface_hub import AsyncInferenceClient
from mcp.types import (
    CallToolRequest,
    ContentBlock,
    TextContent,
)

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

_logger = get_logger(__name__)

DEFAULT_HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


class HuggingFaceAugmentedLLM(AugmentedLLM[dict, dict]):
    """
    HuggingFace Inference API implementation of AugmentedLLM.
    Uses the native HuggingFace InferenceClient for better compatibility.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.HUGGINGFACE, **kwargs)
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize HuggingFace-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with HuggingFace-specific settings
        chosen_model = kwargs.get("model", DEFAULT_HF_MODEL)
        base_params.model = chosen_model
        base_params.parallel_tool_calls = False  # HF doesn't support parallel tool calls yet

        return base_params

    def _get_inference_client(self) -> AsyncInferenceClient:
        """Create and return an AsyncInferenceClient instance."""
        try:
            # Use the API key from the provider key manager
            api_key = self._api_key()

            # If no API key is provided, try to get it from environment
            if not api_key:
                api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

            # Debug logging
            self.logger.debug(
                f"Creating AsyncInferenceClient with model: {self.default_request_params.model}"
            )
            self.logger.debug(f"API key present: {bool(api_key)}")
            self.logger.debug(f"API key prefix: {api_key[:10] if api_key else 'None'}...")

            # Create the client with the model and token
            # Use "auto" provider to automatically select best available provider for the model
            client = AsyncInferenceClient(
                model=self.default_request_params.model,
                token=api_key,
                provider="auto",  # Auto-select best provider for the model
            )

            self.logger.debug("Successfully created AsyncInferenceClient")
            return client
        except Exception as e:
            raise ProviderKeyError(
                "Invalid HuggingFace API token",
                "The configured HuggingFace API token was rejected.\n"
                "Please check that your API token is valid and not expired.",
            ) from e

    def _convert_multipart_to_hf_messages(
        self, multipart_messages: List[PromptMessageMultipart]
    ) -> List[dict]:
        """Convert PromptMessageMultipart to HuggingFace message format."""
        messages = []

        for msg in multipart_messages:
            # Extract text content from the message
            text_content = msg.first_text()

            # Convert role and content to HF format
            hf_message = {"role": msg.role, "content": text_content}

            messages.append(hf_message)

        return messages

    async def _process_stream(self, stream, model: str):
        """Process the streaming response from HuggingFace."""
        # Track estimated output tokens
        estimated_tokens = 0
        accumulated_content = ""
        tool_calls_map = {}  # Map tool call index to accumulated data
        final_usage = None  # Store the final usage information

        # Process the stream chunks
        async for chunk in stream:
            # Extract content from the chunk
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated_content += content
                # Use base class method for token estimation and progress emission
                estimated_tokens = self._update_streaming_progress(content, model, estimated_tokens)

            # Check for usage information in the chunk
            if hasattr(chunk, "usage") and chunk.usage:
                final_usage = chunk.usage
                self.logger.debug(
                    f"Received usage info: prompt_tokens={chunk.usage.prompt_tokens}, completion_tokens={chunk.usage.completion_tokens}, total_tokens={chunk.usage.total_tokens}"
                )

            # Handle tool calls if present - need to accumulate them properly
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                for delta_tool_call in chunk.choices[0].delta.tool_calls:
                    index = delta_tool_call.index

                    # Initialize tool call entry if not exists
                    if index not in tool_calls_map:
                        tool_calls_map[index] = {
                            "id": delta_tool_call.id,
                            "type": delta_tool_call.type or "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    # Accumulate the function name and arguments
                    if delta_tool_call.function:
                        if delta_tool_call.function.name:
                            tool_calls_map[index]["function"]["name"] += (
                                delta_tool_call.function.name
                            )
                        if delta_tool_call.function.arguments:
                            tool_calls_map[index]["function"]["arguments"] += (
                                delta_tool_call.function.arguments
                            )

                    # Update ID if provided (some chunks might have it, others might not)
                    if delta_tool_call.id:
                        tool_calls_map[index]["id"] = delta_tool_call.id

        # Convert accumulated tool calls to final format
        final_tool_calls = []
        for index in sorted(tool_calls_map.keys()):
            tool_data = tool_calls_map[index]

            # Create a properly formatted tool call object
            final_tool_call = type(
                "obj",
                (object,),
                {
                    "id": tool_data["id"] or f"call_{index}",
                    "type": tool_data["type"],
                    "function": type(
                        "obj",
                        (object,),
                        {
                            "name": tool_data["function"]["name"],
                            "arguments": tool_data["function"]["arguments"],
                        },
                    )(),
                },
            )()

            final_tool_calls.append(final_tool_call)

        # Create a final response object similar to OpenAI's format
        class FinalResponse:
            def __init__(self, content, tool_calls, usage):
                self.choices = [
                    type(
                        "obj",
                        (object,),
                        {
                            "message": type(
                                "obj",
                                (object,),
                                {
                                    "content": content,
                                    "tool_calls": tool_calls if tool_calls else None,
                                    "role": "assistant",
                                },
                            )(),
                            "finish_reason": "stop",
                        },
                    )()
                ]
                self.usage = usage

        # Use actual usage if available, otherwise estimate
        if final_usage:
            usage = type(
                "obj",
                (object,),
                {
                    "prompt_tokens": final_usage.prompt_tokens,
                    "completion_tokens": final_usage.completion_tokens,
                    "total_tokens": final_usage.total_tokens,
                },
            )()
            self.logger.debug(
                f"Using actual usage: prompt_tokens={final_usage.prompt_tokens}, completion_tokens={final_usage.completion_tokens}"
            )
        else:
            self.logger.debug(
                f"No usage data provided by HF, using estimated completion tokens: {estimated_tokens}"
            )

        return FinalResponse(accumulated_content, final_tool_calls, usage)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """Apply prompt using HuggingFace Inference API."""
        # Reset tool call counter for new turn
        self._reset_turn_tool_calls()

        last_message = multipart_messages[-1]

        # If the last message is from assistant, just return it
        if last_message.role == "assistant":
            return last_message

        # Get request parameters
        request_params = self.get_request_params(request_params)

        # Prepare messages with system prompt
        messages = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Convert and add other messages
        converted_messages = self._convert_multipart_to_hf_messages(multipart_messages)
        messages.extend(converted_messages)

        # Debug logging
        self.logger.debug(f"Final message list has {len(messages)} messages:")
        for i, msg in enumerate(messages):
            extra_fields = []
            if "tool_call_id" in msg:
                extra_fields.append(f"tool_call_id={msg['tool_call_id']}")
            if "name" in msg:
                extra_fields.append(f"name={msg['name']}")
            extra_info = f", {', '.join(extra_fields)}" if extra_fields else ""
            self.logger.debug(
                f"  Message {i}: role={msg['role']}, content_length={len(str(msg.get('content', '')))}{extra_info}"
            )

        # Prepare the client
        client = self._get_inference_client()

        # Get available tools
        response = await self.aggregator.list_tools()
        tools = None
        if response.tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": self.adjust_schema(tool.inputSchema)
                        if hasattr(self, "adjust_schema")
                        else tool.inputSchema,
                    },
                }
                for tool in response.tools
            ]

        # Run iterative completion with tool handling
        responses: List[ContentBlock] = []

        for i in range(request_params.max_iterations):
            # Log progress
            self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)

            # Make the API call with streaming
            try:
                # Debug logging for API call
                self.logger.debug("Making chat completion call with:")
                self.logger.debug(f"  Model: {self.default_request_params.model}")
                self.logger.debug(f"  Messages count: {len(messages)}")
                self.logger.debug(f"  Max tokens: {request_params.maxTokens}")
                self.logger.debug(
                    f"  Temperature: {request_params.temperature if hasattr(request_params, 'temperature') else 'None'}"
                )
                self.logger.debug(f"  Tools count: {len(tools) if tools else 0}")
                self.logger.debug(
                    f"  Stop sequences: {request_params.stopSequences if hasattr(request_params, 'stopSequences') else 'None'}"
                )

                # Prepare parameters for the API call
                call_params = {
                    "messages": messages,
                    "model": self.default_request_params.model,
                    "max_tokens": request_params.maxTokens,
                    "stream": True,
                }

                # Add optional parameters if they exist
                if (
                    hasattr(request_params, "temperature")
                    and request_params.temperature is not None
                ):
                    call_params["temperature"] = request_params.temperature

                if hasattr(request_params, "stopSequences") and request_params.stopSequences:
                    call_params["stop"] = request_params.stopSequences

                # Add tools if available, with proper tool_choice parameter
                if tools:
                    call_params["tools"] = tools
                    call_params["tool_choice"] = "auto"  # Let the model decide when to use tools

                self.logger.debug(f"Final API call parameters: {list(call_params.keys())}")

                # Try to include usage information in stream, but don't fail if not supported
                try:
                    call_params["stream_options"] = {"include_usage": True}
                    stream = await client.chat.completions.create(**call_params)
                    self.logger.debug("Stream with usage options created successfully")
                except Exception as e:
                    # If stream_options not supported, fall back to basic streaming
                    self.logger.debug(f"Stream options not supported, falling back: {e}")
                    call_params.pop("stream_options", None)
                    stream = await client.chat.completions.create(**call_params)

                self.logger.debug("Chat completion call initiated successfully")

                # Process the stream
                response = await self._process_stream(stream, self.default_request_params.model)

                # Track usage
                if hasattr(response, "usage") and response.usage:
                    try:
                        model_name = self.default_request_params.model or DEFAULT_HF_MODEL
                        turn_usage = TurnUsage.from_openai(response.usage, model_name)
                        self._finalize_turn_usage(turn_usage)
                    except Exception as e:
                        self.logger.warning(f"Failed to track usage: {e}")

                # Extract the message from response
                if not response.choices or not response.choices[0].message:
                    break

                message = response.choices[0].message

                # Add message content to responses
                if message.content:
                    responses.append(TextContent(type="text", text=message.content))
                    await self.show_assistant_message(message.content)

                # Create assistant message for conversation history
                assistant_message = {"role": "assistant", "content": message.content or ""}

                # Handle tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Add tool_calls to the assistant message
                    assistant_message["tool_calls"] = []
                    for tool_call in message.tool_calls:
                        assistant_message["tool_calls"].append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type or "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )

                # Add the assistant message to conversation
                messages.append(assistant_message)

                # Process tool calls if any
                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Process tool calls
                    for tool_call in message.tool_calls:
                        # Show the tool call
                        self.show_tool_call(
                            response.tools if hasattr(response, "tools") else [],
                            tool_call.function.name,
                            tool_call.function.arguments,
                        )

                        # Execute the tool
                        try:
                            import json

                            # Validate tool call has required fields
                            if not tool_call.function.name:
                                self.logger.error(f"Tool call missing function name: {tool_call}")
                                continue

                            # Parse arguments if they're a string
                            if isinstance(tool_call.function.arguments, str):
                                if tool_call.function.arguments.strip():
                                    try:
                                        args = json.loads(tool_call.function.arguments)
                                    except json.JSONDecodeError as e:
                                        self.logger.error(
                                            f"Failed to parse tool arguments as JSON: {tool_call.function.arguments}, error: {e}"
                                        )
                                        args = {}
                                else:
                                    args = {}
                            else:
                                args = tool_call.function.arguments or {}

                            request = CallToolRequest(
                                method="tools/call",
                                params={"name": tool_call.function.name, "arguments": args},
                            )
                            result = await self.call_tool(request, tool_call.id)

                            # Show the result
                            self.show_tool_result(result)

                            # Add tool result to messages
                            # Format tool result according to the standard
                            tool_content = ""
                            if result.content:
                                tool_content = (
                                    result.content[0].text
                                    if hasattr(result.content[0], "text")
                                    else str(result.content[0])
                                )

                            tool_message = {
                                "role": "tool",
                                "content": tool_content,
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,  # Some providers need the function name
                            }

                            self.logger.debug(
                                f"Adding tool result message: role={tool_message['role']}, content_length={len(tool_message['content'])}, tool_call_id={tool_message['tool_call_id']}, name={tool_message['name']}"
                            )
                            messages.append(tool_message)

                        except Exception as e:
                            self.logger.error(
                                f"Error executing tool {tool_call.function.name}: {e}"
                            )
                            # Add error message
                            error_message = {
                                "role": "tool",
                                "content": f"Error executing tool: {str(e)}",
                                "tool_call_id": tool_call.id,
                            }
                            messages.append(error_message)
                else:
                    # No tool calls, we're done
                    break

            except Exception as e:
                self.logger.error(f"Error during HuggingFace completion: {e}")
                self.logger.error(f"Exception type: {type(e).__name__}")
                if hasattr(e, "response"):
                    self.logger.error(
                        f"Response status: {getattr(e.response, 'status_code', 'unknown')}"
                    )
                    self.logger.error(f"Response text: {getattr(e.response, 'text', 'unknown')}")
                if hasattr(e, "request"):
                    self.logger.error(f"Request URL: {getattr(e.request, 'url', 'unknown')}")
                    self.logger.error(
                        f"Request headers: {getattr(e.request, 'headers', 'unknown')}"
                    )

                raise ProviderKeyError(
                    "HuggingFace API Error", f"Failed to complete request: {str(e)}"
                ) from e

        # Log completion
        self._log_chat_finished(model=self.default_request_params.model)

        # Return the accumulated responses
        if responses:
            return Prompt.assistant(*responses)
        else:
            return Prompt.assistant(TextContent(type="text", text=""))

    def adjust_schema(self, inputSchema: Dict) -> Dict:
        """Adjust the schema for HuggingFace compatibility."""
        # HuggingFace handles schemas properly, so we can just return as-is
        # Unlike OpenAI, we don't need to add empty properties
        return inputSchema

    async def _is_tool_stop_reason(self, finish_reason: str) -> bool:
        """Check if the finish reason indicates tool use."""
        return finish_reason in ["tool_calls", "function_call"]
