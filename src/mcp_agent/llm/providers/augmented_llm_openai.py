from typing import Dict, List

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from openai import AsyncOpenAI, AuthenticationError
from openai.lib.streaming.chat import ChatCompletionStreamState

# from openai.types.beta.chat import
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from pydantic_core import from_json
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.event_progress import ProgressAction
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    RequestParams,
)
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter, OpenAIMessage
from mcp_agent.llm.providers.sampling_converter_openai import (
    OpenAISamplingConverter,
)
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

_logger = get_logger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_REASONING_EFFORT = "medium"


class OpenAIAugmentedLLM(AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    # OpenAI-specific parameter exclusions
    OPENAI_EXCLUDE_FIELDS = {
        AugmentedLLM.PARAM_MESSAGES,
        AugmentedLLM.PARAM_MODEL,
        AugmentedLLM.PARAM_MAX_TOKENS,
        AugmentedLLM.PARAM_SYSTEM_PROMPT,
        AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS,
        AugmentedLLM.PARAM_USE_HISTORY,
        AugmentedLLM.PARAM_MAX_ITERATIONS,
        AugmentedLLM.PARAM_TEMPLATE_VARS,
    }

    def __init__(self, provider: Provider = Provider.OPENAI, *args, **kwargs) -> None:
        # Set type_converter before calling super().__init__
        if "type_converter" not in kwargs:
            kwargs["type_converter"] = OpenAISamplingConverter

        super().__init__(*args, provider=provider, **kwargs)

        # Initialize logger with name if available
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        # Set up reasoning-related attributes
        self._reasoning_effort = kwargs.get("reasoning_effort", None)
        if self.context and self.context.config and self.context.config.openai:
            if self._reasoning_effort is None and hasattr(
                self.context.config.openai, "reasoning_effort"
            ):
                self._reasoning_effort = self.context.config.openai.reasoning_effort

        # Determine if we're using a reasoning model
        # TODO -- move this to model capabilities, add o4.
        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning = chosen_model and (
            chosen_model.startswith("o3") or chosen_model.startswith("o1")
        )
        if self._reasoning:
            self.logger.info(
                f"Using reasoning model '{chosen_model}' with '{self._reasoning_effort}' reasoning effort"
            )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenAI-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with OpenAI-specific settings
        chosen_model = kwargs.get("model", DEFAULT_OPENAI_MODEL)
        base_params.model = chosen_model

        return base_params

    def _base_url(self) -> str:
        return self.context.config.openai.base_url if self.context.config.openai else None

    def _openai_client(self) -> AsyncOpenAI:
        try:
            return AsyncOpenAI(api_key=self._api_key(), base_url=self._base_url())
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

    async def _process_stream(self, stream, model: str):
        """Process the streaming response and display real-time token usage."""
        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0

        # For non-OpenAI providers (like Ollama), ChatCompletionStreamState might not work correctly
        # Fall back to manual accumulation if needed
        # TODO -- consider this and whether to subclass instead
        if self.provider in [Provider.GENERIC, Provider.OPENROUTER, Provider.GOOGLE_OAI]:
            return await self._process_stream_manual(stream, model)

        # Use ChatCompletionStreamState helper for accumulation (OpenAI only)
        state = ChatCompletionStreamState()

        # Process the stream chunks
        async for chunk in stream:
            # Handle chunk accumulation
            state.handle_chunk(chunk)

            # Count tokens in real-time from content deltas
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                # Use base class method for token estimation and progress emission
                estimated_tokens = self._update_streaming_progress(content, model, estimated_tokens)

        # Get the final completion with usage data
        final_completion = state.get_final_completion()

        # Log final usage information
        if hasattr(final_completion, "usage") and final_completion.usage:
            actual_tokens = final_completion.usage.completion_tokens
            # Emit final progress with actual token count
            token_str = str(actual_tokens).rjust(5)
            data = {
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            }
            self.logger.info("Streaming progress", data=data)

            self.logger.info(
                f"Streaming complete - Model: {model}, Input tokens: {final_completion.usage.prompt_tokens}, Output tokens: {final_completion.usage.completion_tokens}"
            )

        return final_completion

    # TODO - as per other comment this needs to go in another class. There are a number of "special" cases dealt with
    # here to deal with OpenRouter idiosyncrasies between e.g. Anthropic and Gemini models.
    async def _process_stream_manual(self, stream, model: str):
        """Manual stream processing for providers like Ollama that may not work with ChatCompletionStreamState."""
        from openai.types.chat import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function

        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0

        # Manual accumulation of response data
        accumulated_content = ""
        role = "assistant"
        tool_calls_map = {}  # Use a map to accumulate tool calls by index
        function_call = None
        finish_reason = None
        usage_data = None

        # Process the stream chunks manually
        async for chunk in stream:
            # Count tokens in real-time from content deltas
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated_content += content
                # Use base class method for token estimation and progress emission
                estimated_tokens = self._update_streaming_progress(content, model, estimated_tokens)

            # Extract other fields from the chunk
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.role:
                    role = choice.delta.role
                if choice.delta.tool_calls:
                    # Accumulate tool call deltas
                    for delta_tool_call in choice.delta.tool_calls:
                        if delta_tool_call.index is not None:
                            if delta_tool_call.index not in tool_calls_map:
                                tool_calls_map[delta_tool_call.index] = {
                                    "id": delta_tool_call.id,
                                    "type": delta_tool_call.type or "function",
                                    "function": {
                                        "name": delta_tool_call.function.name
                                        if delta_tool_call.function
                                        else None,
                                        "arguments": "",
                                    },
                                }

                            # Always update if we have new data (needed for OpenRouter Gemini)
                            if delta_tool_call.id:
                                tool_calls_map[delta_tool_call.index]["id"] = delta_tool_call.id
                            if delta_tool_call.function:
                                if delta_tool_call.function.name:
                                    tool_calls_map[delta_tool_call.index]["function"]["name"] = (
                                        delta_tool_call.function.name
                                    )
                                # Handle arguments - they might come as None, empty string, or actual content
                                if delta_tool_call.function.arguments is not None:
                                    tool_calls_map[delta_tool_call.index]["function"][
                                        "arguments"
                                    ] += delta_tool_call.function.arguments

                if choice.delta.function_call:
                    function_call = choice.delta.function_call
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            # Extract usage data if available
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage

        # Convert accumulated tool calls to proper format.
        tool_calls = None
        if tool_calls_map:
            tool_calls = []
            for idx in sorted(tool_calls_map.keys()):
                tool_call_data = tool_calls_map[idx]
                # Only add tool calls that have valid data
                if tool_call_data["id"] and tool_call_data["function"]["name"]:
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tool_call_data["id"],
                            type=tool_call_data["type"],
                            function=Function(
                                name=tool_call_data["function"]["name"],
                                arguments=tool_call_data["function"]["arguments"],
                            ),
                        )
                    )

        # Create a ChatCompletionMessage manually
        message = ChatCompletionMessage(
            content=accumulated_content,
            role=role,
            tool_calls=tool_calls if tool_calls else None,
            function_call=function_call,
            refusal=None,
            annotations=None,
            audio=None,
        )

        from types import SimpleNamespace

        final_completion = SimpleNamespace()
        final_completion.choices = [SimpleNamespace()]
        final_completion.choices[0].message = message
        final_completion.choices[0].finish_reason = finish_reason
        final_completion.usage = usage_data

        # Log final usage information
        if usage_data:
            actual_tokens = getattr(usage_data, "completion_tokens", estimated_tokens)
            token_str = str(actual_tokens).rjust(5)
            data = {
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            }
            self.logger.info("Streaming progress", data=data)

            self.logger.info(
                f"Streaming complete - Model: {model}, Input tokens: {getattr(usage_data, 'prompt_tokens', 0)}, Output tokens: {actual_tokens}"
            )

        return final_completion

    async def _openai_completion(
        self,
        message: OpenAIMessage,
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """

        request_params = self.get_request_params(request_params=request_params)

        responses: List[TextContent | ImageContent | EmbeddedResource] = []

        # TODO -- move this in to agent context management / agent group handling
        messages: List[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        messages.extend(self.history.get(include_completion_history=request_params.use_history))
        messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] | None = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description if tool.description else "",
                    "parameters": self.adjust_schema(tool.inputSchema),
                },
            )
            for tool in response.tools
        ]

        if not available_tools:
            if self.provider == Provider.DEEPSEEK:
                available_tools = None  # deepseek does not allow empty array
            else:
                available_tools = []

        # we do NOT send "stop sequences" as this causes errors with mutlimodal processing
        for i in range(request_params.max_iterations):
            arguments = self._prepare_api_request(messages, available_tools, request_params)
            self.logger.debug(f"OpenAI completion requested for: {arguments}")

            self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)

            # Use basic streaming API
            stream = await self._openai_client().chat.completions.create(**arguments)
            # Process the stream
            response = await self._process_stream(stream, self.default_request_params.model)

            # Track usage if response is valid and has usage data
            if (
                hasattr(response, "usage")
                and response.usage
                and not isinstance(response, BaseException)
            ):
                try:
                    model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL
                    turn_usage = TurnUsage.from_openai(response.usage, model_name)
                    self.usage_accumulator.add_turn(turn_usage)
                except Exception as e:
                    self.logger.warning(f"Failed to track usage: {e}")

            self.logger.debug(
                "OpenAI completion response:",
                data=response,
            )

            if isinstance(response, AuthenticationError):
                raise ProviderKeyError(
                    "Rejected OpenAI API key",
                    "The configured OpenAI API key was rejected.\n"
                    "Please check that your API key is valid and not expired.",
                ) from response
            elif isinstance(response, BaseException):
                self.logger.error(f"Error: {response}")
                break

            if not response.choices or len(response.choices) == 0:
                # No response from the model, we're done
                break

            choice = response.choices[0]
            message = choice.message
            # prep for image/audio gen models
            if message.content:
                responses.append(TextContent(type="text", text=message.content))

            # ParsedChatCompletionMessage is compatible with ChatCompletionMessage
            # since it inherits from it, so we can use it directly
            messages.append(message)

            message_text = message.content
            if choice.finish_reason in ["tool_calls", "function_call"] and message.tool_calls:
                if message_text:
                    await self.show_assistant_message(
                        message_text,
                        message.tool_calls[
                            0
                        ].function.name,  # TODO support displaying multiple tool calls
                    )
                else:
                    await self.show_assistant_message(
                        Text(
                            "the assistant requested tool calls",
                            style="dim green italic",
                        ),
                        message.tool_calls[0].function.name,
                    )

                tool_results = []
                for tool_call in message.tool_calls:
                    self.show_tool_call(
                        available_tools,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    tool_call_request = CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name=tool_call.function.name,
                            arguments={}
                            if not tool_call.function.arguments
                            or tool_call.function.arguments.strip() == ""
                            else from_json(tool_call.function.arguments, allow_partial=True),
                        ),
                    )
                    result = await self.call_tool(tool_call_request, tool_call.id)
                    self.show_oai_tool_result(str(result))

                    tool_results.append((tool_call.id, result))
                    responses.extend(result.content)
                messages.extend(OpenAIConverter.convert_function_results_to_openai(tool_results))

                self.logger.debug(
                    f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                )
            elif choice.finish_reason == "length":
                # We have reached the max tokens limit
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'length'")
                if request_params and request_params.maxTokens is not None:
                    message_text = Text(
                        f"the assistant has reached the maximum token limit ({request_params.maxTokens})",
                        style="dim green italic",
                    )
                else:
                    message_text = Text(
                        "the assistant has reached the maximum token limit",
                        style="dim green italic",
                    )

                await self.show_assistant_message(message_text)
                break
            elif choice.finish_reason == "content_filter":
                # The response was filtered by the content filter
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                )
                break
            elif choice.finish_reason == "stop":
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                if message_text:
                    await self.show_assistant_message(message_text, "")
                break

        if request_params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_completion_history=False)

            # Calculate new conversation messages (excluding prompts)
            new_messages = messages[len(prompt_messages) :]

            if system_prompt:
                new_messages = new_messages[1:]

            self.history.set(new_messages)

        self._log_chat_finished(model=self.default_request_params.model)

        return responses

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        # if the last message is a "user" inference is required
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []
        for msg in messages_to_add:
            converted.append(OpenAIConverter.convert_to_openai(msg))

        # TODO -- this looks like a defect from previous apply_prompt implementation.
        self.history.extend(converted, is_prompt=is_template)

        if "assistant" == last_message.role:
            return last_message

        # For assistant messages: Return the last message (no completion needed)
        message_param: OpenAIMessage = OpenAIConverter.convert_to_openai(last_message)
        responses: List[
            TextContent | ImageContent | EmbeddedResource
        ] = await self._openai_completion(
            message_param,
            request_params,
        )
        return Prompt.assistant(*responses)

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        return result

    def _prepare_api_request(
        self, messages, tools: List[ChatCompletionToolParam] | None, request_params: RequestParams
    ) -> dict[str, str]:
        # Create base arguments dictionary

        # overriding model via request params not supported (intentional)
        base_args = {
            "model": self.default_request_params.model,
            "messages": messages,
            "tools": tools,
            "stream": True,  # Enable basic streaming
            "stream_options": {"include_usage": True},  # Required for usage data in streaming
        }

        if self._reasoning:
            base_args.update(
                {
                    "max_completion_tokens": request_params.maxTokens,
                    "reasoning_effort": self._reasoning_effort,
                }
            )
        else:
            base_args["max_tokens"] = request_params.maxTokens
            if tools:
                base_args["parallel_tool_calls"] = request_params.parallel_tool_calls

        arguments: Dict[str, str] = self.prepare_provider_arguments(
            base_args, request_params, self.OPENAI_EXCLUDE_FIELDS.union(self.BASE_EXCLUDE_FIELDS)
        )
        return arguments

    def adjust_schema(self, inputSchema: Dict) -> Dict:
        # return inputSchema
        if self.provider not in [Provider.OPENAI, Provider.AZURE]:
            return inputSchema

        if "properties" in inputSchema:
            return inputSchema

        result = inputSchema.copy()
        result["properties"] = {}
        return result
