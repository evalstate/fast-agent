from typing import Any, Dict, List

from mcp import Tool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)
from openai import APIError, AsyncOpenAI, AuthenticationError, DefaultAioHttpClient
from openai.lib.streaming.chat import ChatCompletionStreamState

# from openai.types.beta.chat import
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from pydantic_core import from_json

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.fastagent_llm import FastAgentLLM, RequestParams
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.multipart_converter_openai import OpenAIConverter, OpenAIMessage
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import LlmStopReason, PromptMessageExtended

_logger = get_logger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_REASONING_EFFORT = "medium"


class OpenAILLM(FastAgentLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    # OpenAI-specific parameter exclusions
    OPENAI_EXCLUDE_FIELDS = {
        FastAgentLLM.PARAM_MESSAGES,
        FastAgentLLM.PARAM_MODEL,
        FastAgentLLM.PARAM_MAX_TOKENS,
        FastAgentLLM.PARAM_SYSTEM_PROMPT,
        FastAgentLLM.PARAM_PARALLEL_TOOL_CALLS,
        FastAgentLLM.PARAM_USE_HISTORY,
        FastAgentLLM.PARAM_MAX_ITERATIONS,
        FastAgentLLM.PARAM_TEMPLATE_VARS,
        FastAgentLLM.PARAM_MCP_METADATA,
        FastAgentLLM.PARAM_STOP_SEQUENCES,
    }

    def __init__(self, provider: Provider = Provider.OPENAI, *args, **kwargs) -> None:
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

        # Determine reasoning mode for the selected model
        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning_mode = ModelDatabase.get_reasoning(chosen_model)
        self._reasoning = self._reasoning_mode == "openai"
        if self._reasoning_mode:
            self.logger.info(
                f"Using reasoning model '{chosen_model}' (mode='{self._reasoning_mode}') with "
                f"'{self._reasoning_effort}' reasoning effort"
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
        """
        Create an OpenAI client instance.
        Subclasses can override this to provide different client types (e.g., AzureOpenAI).

        Note: The returned client should be used within an async context manager
        to ensure proper cleanup of aiohttp sessions.
        """
        try:
            return AsyncOpenAI(
                api_key=self._api_key(),
                base_url=self._base_url(),
                http_client=DefaultAioHttpClient(),
            )
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

    def _streams_tool_arguments(self) -> bool:
        """
        Determine whether the current provider streams tool call arguments incrementally.

        Official OpenAI and Azure OpenAI endpoints stream arguments. Most third-party
        OpenAI-compatible gateways (e.g. OpenRouter, Moonshot) deliver the full arguments
        once, so we should treat them as non-streaming to restore the legacy \"Calling Tool\"
        display experience.
        """
        if self.provider == Provider.AZURE:
            return True

        if self.provider == Provider.OPENAI:
            base_url = self._base_url()
            if not base_url:
                return True
            lowered = base_url.lower()
            return "api.openai" in lowered or "openai.azure" in lowered or "azure.com" in lowered

        return False

    def _emit_tool_notification_fallback(
        self,
        tool_calls: Any,
        notified_indices: set[int],
        *,
        streams_arguments: bool,
        model: str,
    ) -> None:
        """Emit start/stop notifications when streaming metadata was missing."""
        if not tool_calls:
            return

        for index, tool_call in enumerate(tool_calls):
            if index in notified_indices:
                continue

            tool_name = None
            tool_use_id = None

            try:
                tool_use_id = getattr(tool_call, "id", None)
                function = getattr(tool_call, "function", None)
                if function:
                    tool_name = getattr(function, "name", None)
            except Exception:
                tool_use_id = None
                tool_name = None

            if not tool_name:
                tool_name = "tool"
            if not tool_use_id:
                tool_use_id = f"tool-{index}"

            payload = {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "index": index,
                "streams_arguments": streams_arguments,
            }

            self._notify_tool_stream_listeners("start", payload)
            self.logger.info(
                "Model emitted fallback tool notification",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": self.name,
                    "model": model,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_event": "start",
                    "streams_arguments": streams_arguments,
                    "fallback": True,
                },
            )
            self._notify_tool_stream_listeners("stop", payload)
            self.logger.info(
                "Model emitted fallback tool notification",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": self.name,
                    "model": model,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_event": "stop",
                    "streams_arguments": streams_arguments,
                    "fallback": True,
                },
            )

    async def _process_stream(self, stream, model: str):
        """Process the streaming response and display real-time token usage."""
        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0

        # For providers/models that emit non-OpenAI deltas, fall back to manual accumulation
        stream_mode = ModelDatabase.get_stream_mode(model)
        provider_requires_manual = self.provider in [
            Provider.GENERIC,
            Provider.OPENROUTER,
            Provider.GOOGLE_OAI,
        ]
        if stream_mode == "manual" or provider_requires_manual:
            return await self._process_stream_manual(stream, model)

        # Use ChatCompletionStreamState helper for accumulation (OpenAI only)
        state = ChatCompletionStreamState()

        # Track tool call state for stream events
        tool_call_started: dict[int, dict[str, Any]] = {}
        streams_arguments = self._streams_tool_arguments()
        notified_tool_indices: set[int] = set()

        # Process the stream chunks
        async for chunk in stream:
            # Handle chunk accumulation
            state.handle_chunk(chunk)
            # Process streaming events for tool calls
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                # Handle tool call streaming
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        index = tool_call.index

                        # Fire "start" event on first chunk for this tool call
                        if index is None:
                            continue

                        existing_info = tool_call_started.get(index)
                        tool_use_id = tool_call.id or (
                            existing_info.get("tool_use_id") if existing_info else None
                        )
                        function_name = (
                            tool_call.function.name
                            if tool_call.function and tool_call.function.name
                            else (existing_info.get("tool_name") if existing_info else None)
                        )

                        if existing_info is None and tool_use_id and function_name:
                            tool_call_started[index] = {
                                "tool_name": function_name,
                                "tool_use_id": tool_use_id,
                                "streams_arguments": streams_arguments,
                            }
                            self._notify_tool_stream_listeners(
                                "start",
                                {
                                    "tool_name": function_name,
                                    "tool_use_id": tool_use_id,
                                    "index": index,
                                    "streams_arguments": streams_arguments,
                                },
                            )
                            self.logger.info(
                                "Model started streaming tool call",
                                data={
                                    "progress_action": ProgressAction.CALLING_TOOL,
                                    "agent_name": self.name,
                                    "model": model,
                                    "tool_name": function_name,
                                    "tool_use_id": tool_use_id,
                                    "tool_event": "start",
                                    "streams_arguments": streams_arguments,
                                },
                            )
                            notified_tool_indices.add(index)
                        elif existing_info:
                            if tool_use_id:
                                existing_info["tool_use_id"] = tool_use_id
                            if function_name:
                                existing_info["tool_name"] = function_name

                        # Fire "delta" event for argument chunks
                        if tool_call.function and tool_call.function.arguments:
                            info = tool_call_started.setdefault(
                                index,
                                {
                                    "tool_name": function_name,
                                    "tool_use_id": tool_use_id,
                                    "streams_arguments": streams_arguments,
                                },
                            )
                            self._notify_tool_stream_listeners(
                                "delta",
                                {
                                    "tool_name": info.get("tool_name"),
                                    "tool_use_id": info.get("tool_use_id"),
                                    "index": index,
                                    "chunk": tool_call.function.arguments,
                                    "streams_arguments": info.get("streams_arguments", False),
                                },
                            )

                # Handle text content streaming
                if delta.content:
                    content = delta.content
                    # Use base class method for token estimation and progress emission
                    estimated_tokens = self._update_streaming_progress(
                        content, model, estimated_tokens
                    )
                    self._notify_tool_stream_listeners(
                        "text",
                        {
                            "chunk": content,
                            "streams_arguments": streams_arguments,
                        },
                    )

                # Fire "stop" event when tool calls complete
                if choice.finish_reason == "tool_calls":
                    for index, info in list(tool_call_started.items()):
                        self._notify_tool_stream_listeners(
                            "stop",
                            {
                                "tool_name": info.get("tool_name"),
                                "tool_use_id": info.get("tool_use_id"),
                                "index": index,
                                "streams_arguments": info.get("streams_arguments", False),
                            },
                        )
                        self.logger.info(
                            "Model finished streaming tool call",
                            data={
                                "progress_action": ProgressAction.CALLING_TOOL,
                                "agent_name": self.name,
                                "model": model,
                                "tool_name": info.get("tool_name"),
                                "tool_use_id": info.get("tool_use_id"),
                                "tool_event": "stop",
                                "streams_arguments": info.get("streams_arguments", False),
                            },
                        )
                        notified_tool_indices.add(index)
                    tool_call_started.clear()

        # Check if we hit the length limit to avoid LengthFinishReasonError
        current_snapshot = state.current_completion_snapshot
        if current_snapshot.choices and current_snapshot.choices[0].finish_reason == "length":
            # Return the current snapshot directly to avoid exception
            final_completion = current_snapshot
        else:
            # Get the final completion with usage data (may include structured output parsing)
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

        final_message = None
        if hasattr(final_completion, "choices") and final_completion.choices:
            final_message = getattr(final_completion.choices[0], "message", None)
        tool_calls = getattr(final_message, "tool_calls", None) if final_message else None
        self._emit_tool_notification_fallback(
            tool_calls,
            notified_tool_indices,
            streams_arguments=streams_arguments,
            model=model,
        )

        return final_completion

    def _normalize_role(self, role: str | None) -> str:
        """Ensure the role string matches MCP expectations."""
        default_role = "assistant"
        if not role:
            return default_role

        lowered = role.lower()
        allowed_roles = {"assistant", "user", "system", "tool"}
        if lowered in allowed_roles:
            return lowered

        for candidate in allowed_roles:
            if len(lowered) % len(candidate) == 0:
                repetitions = len(lowered) // len(candidate)
                if candidate * repetitions == lowered:
                    self.logger.warning(
                        "Collapsing repeated role value from provider",
                        data={
                            "original_role": role,
                            "normalized_role": candidate,
                        },
                    )
                    return candidate

        self.logger.warning(
            "Model emitted unsupported role; defaulting to assistant",
            data={"original_role": role},
        )
        return default_role

    # TODO - as per other comment this needs to go in another class. There are a number of "special" cases dealt with
    # here to deal with OpenRouter idiosyncrasies between e.g. Anthropic and Gemini models.
    async def _process_stream_manual(self, stream, model: str):
        """Manual stream processing for providers like Ollama that may not work with ChatCompletionStreamState."""

        from openai.types.chat import ChatCompletionMessageToolCall

        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0

        # Manual accumulation of response data
        accumulated_content = ""
        role = "assistant"
        tool_calls_map = {}  # Use a map to accumulate tool calls by index
        function_call = None
        finish_reason = None
        usage_data = None

        # Track tool call state for stream events
        tool_call_started: dict[int, dict[str, Any]] = {}
        streams_arguments = self._streams_tool_arguments()
        notified_tool_indices: set[int] = set()

        # Process the stream chunks manually
        async for chunk in stream:
            # Process streaming events for tool calls
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                # Handle tool call streaming
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.index is not None:
                            index = tool_call.index

                            existing_info = tool_call_started.get(index)
                            tool_use_id = tool_call.id or (
                                existing_info.get("tool_use_id") if existing_info else None
                            )
                            function_name = (
                                tool_call.function.name
                                if tool_call.function and tool_call.function.name
                                else (existing_info.get("tool_name") if existing_info else None)
                            )

                            # Fire "start" event on first chunk for this tool call
                            if index not in tool_call_started and tool_use_id and function_name:
                                tool_call_started[index] = {
                                    "tool_name": function_name,
                                    "tool_use_id": tool_use_id,
                                    "streams_arguments": streams_arguments,
                                }
                                self._notify_tool_stream_listeners(
                                    "start",
                                    {
                                        "tool_name": function_name,
                                        "tool_use_id": tool_use_id,
                                        "index": index,
                                        "streams_arguments": streams_arguments,
                                    },
                                )
                                self.logger.info(
                                    "Model started streaming tool call",
                                    data={
                                        "progress_action": ProgressAction.CALLING_TOOL,
                                        "agent_name": self.name,
                                        "model": model,
                                        "tool_name": function_name,
                                        "tool_use_id": tool_use_id,
                                        "tool_event": "start",
                                        "streams_arguments": streams_arguments,
                                    },
                                )
                                notified_tool_indices.add(index)
                            elif existing_info:
                                if tool_use_id:
                                    existing_info["tool_use_id"] = tool_use_id
                                if function_name:
                                    existing_info["tool_name"] = function_name

                            # Fire "delta" event for argument chunks
                            if tool_call.function and tool_call.function.arguments:
                                info = tool_call_started.setdefault(
                                    index,
                                    {
                                        "tool_name": function_name,
                                        "tool_use_id": tool_use_id,
                                        "streams_arguments": streams_arguments,
                                    },
                                )
                                self._notify_tool_stream_listeners(
                                    "delta",
                                    {
                                        "tool_name": info.get("tool_name"),
                                        "tool_use_id": info.get("tool_use_id"),
                                        "index": index,
                                        "chunk": tool_call.function.arguments,
                                        "streams_arguments": info.get("streams_arguments", False),
                                    },
                                )

                # Handle text content streaming
                if delta.content:
                    content = delta.content
                    accumulated_content += content
                    # Use base class method for token estimation and progress emission
                    estimated_tokens = self._update_streaming_progress(
                        content, model, estimated_tokens
                    )
                    self._notify_tool_stream_listeners(
                        "text",
                        {
                            "chunk": content,
                            "streams_arguments": streams_arguments,
                        },
                    )

                # Fire "stop" event when tool calls complete
                if choice.finish_reason == "tool_calls":
                    for index, info in list(tool_call_started.items()):
                        self._notify_tool_stream_listeners(
                            "stop",
                            {
                                "tool_name": info.get("tool_name"),
                                "tool_use_id": info.get("tool_use_id"),
                                "index": index,
                                "streams_arguments": info.get("streams_arguments", False),
                            },
                        )
                        self.logger.info(
                            "Model finished streaming tool call",
                            data={
                                "progress_action": ProgressAction.CALLING_TOOL,
                                "agent_name": self.name,
                                "model": model,
                                "tool_name": info.get("tool_name"),
                                "tool_use_id": info.get("tool_use_id"),
                                "tool_event": "stop",
                                "streams_arguments": info.get("streams_arguments", False),
                            },
                        )
                        notified_tool_indices.add(index)
                    tool_call_started.clear()

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
                            function={
                                "name": tool_call_data["function"]["name"],
                                "arguments": tool_call_data["function"]["arguments"],
                            },
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

        final_message = final_completion.choices[0].message if final_completion.choices else None
        tool_calls = getattr(final_message, "tool_calls", None) if final_message else None
        self._emit_tool_notification_fallback(
            tool_calls,
            notified_tool_indices,
            streams_arguments=streams_arguments,
            model=model,
        )

        return final_completion

    async def _openai_completion(
        self,
        message: List[OpenAIMessage] | None,
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """

        request_params = self.get_request_params(request_params=request_params)

        response_content_blocks: List[ContentBlock] = []
        model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL

        # TODO -- move this in to agent context management / agent group handling
        messages: List[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        messages.extend(self.history.get(include_completion_history=request_params.use_history))
        if message is not None:
            messages.extend(message)

        available_tools: List[ChatCompletionToolParam] | None = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description if tool.description else "",
                    "parameters": self.adjust_schema(tool.inputSchema),
                },
            }
            for tool in tools or []
        ]

        if not available_tools:
            if self.provider in [Provider.DEEPSEEK, Provider.ALIYUN]:
                available_tools = None  # deepseek/aliyun does not allow empty array
            else:
                available_tools = []

        # we do NOT send "stop sequences" as this causes errors with mutlimodal processing
        arguments: dict[str, Any] = self._prepare_api_request(
            messages, available_tools, request_params
        )
        if not self._reasoning and request_params.stopSequences:
            arguments["stop"] = request_params.stopSequences

        self.logger.debug(f"OpenAI completion requested for: {arguments}")

        self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)
        model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL

        # Use basic streaming API with context manager to properly close aiohttp session
        try:
            async with self._openai_client() as client:
                stream = await client.chat.completions.create(**arguments)
                # Process the stream
                response = await self._process_stream(stream, model_name)
        except APIError as error:
            self.logger.error("APIError during OpenAI completion", exc_info=error)
            return self._stream_failure_response(error, model_name)
        # Track usage if response is valid and has usage data
        if (
            hasattr(response, "usage")
            and response.usage
            and not isinstance(response, BaseException)
        ):
            try:
                turn_usage = TurnUsage.from_openai(response.usage, model_name)
                self._finalize_turn_usage(turn_usage)
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

        choice = response.choices[0]
        message = choice.message
        normalized_role = self._normalize_role(getattr(message, "role", None))
        # prep for image/audio gen models
        if message.content:
            response_content_blocks.append(TextContent(type="text", text=message.content))

        # ParsedChatCompletionMessage is compatible with ChatCompletionMessage
        # since it inherits from it, so we can use it directly
        # Convert to dict and remove None values
        message_dict = message.model_dump()
        message_dict = {k: v for k, v in message_dict.items() if v is not None}
        if normalized_role:
            try:
                message.role = normalized_role
            except Exception:
                pass

        if model_name in (
            "deepseek-r1-distill-llama-70b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ):
            message_dict.pop("reasoning", None)
            message_dict.pop("channel", None)

        message_dict["role"] = normalized_role or message_dict.get("role", "assistant")

        messages.append(message_dict)
        stop_reason = LlmStopReason.END_TURN
        requested_tool_calls: Dict[str, CallToolRequest] | None = None
        if await self._is_tool_stop_reason(choice.finish_reason) and message.tool_calls:
            requested_tool_calls = {}
            stop_reason = LlmStopReason.TOOL_USE
            for tool_call in message.tool_calls:
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
                requested_tool_calls[tool_call.id] = tool_call_request
        elif choice.finish_reason == "length":
            stop_reason = LlmStopReason.MAX_TOKENS
            # We have reached the max tokens limit
            self.logger.debug(" Stopping because finish_reason is 'length'")
        elif choice.finish_reason == "content_filter":
            stop_reason = LlmStopReason.SAFETY
            self.logger.debug(" Stopping because finish_reason is 'content_filter'")

        if request_params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_completion_history=False)

            # Calculate new conversation messages (excluding prompts)
            new_messages = messages[len(prompt_messages) :]

            if system_prompt:
                new_messages = new_messages[1:]

            self.history.set(new_messages)

        self._log_chat_finished(model=self.default_request_params.model)

        return Prompt.assistant(
            *response_content_blocks, stop_reason=stop_reason, tool_calls=requested_tool_calls
        )

    def _stream_failure_response(self, error: APIError, model_name: str) -> PromptMessageExtended:
        """Convert streaming API errors into a graceful assistant reply."""

        provider_label = (
            self.provider.value if isinstance(self.provider, Provider) else str(self.provider)
        )
        detail = getattr(error, "message", None) or str(error)
        detail = detail.strip() if isinstance(detail, str) else ""

        parts: list[str] = [f"{provider_label} request failed"]
        if model_name:
            parts.append(f"for model '{model_name}'")
        code = getattr(error, "code", None)
        if code:
            parts.append(f"(code: {code})")
        status = getattr(error, "status_code", None)
        if status:
            parts.append(f"(status={status})")

        message = " ".join(parts)
        if detail:
            message = f"{message}: {detail}"

        user_summary = " ".join(message.split()) if message else ""
        if user_summary and len(user_summary) > 280:
            user_summary = user_summary[:277].rstrip() + "..."

        if user_summary:
            assistant_text = f"I hit an internal error while calling the model: {user_summary}"
            if not assistant_text.endswith((".", "!", "?")):
                assistant_text += "."
            assistant_text += " See fast-agent-error for additional details."
        else:
            assistant_text = (
                "I hit an internal error while calling the model; see fast-agent-error for details."
            )

        assistant_block = text_content(assistant_text)
        error_block = text_content(message)

        return PromptMessageExtended(
            role="assistant",
            content=[assistant_block],
            channels={FAST_AGENT_ERROR_CHANNEL: [error_block]},
            stop_reason=LlmStopReason.ERROR,
        )

    async def _is_tool_stop_reason(self, finish_reason: str) -> bool:
        return True

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageExtended"],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        # Determine effective params to respect use_history for this turn
        req_params = self.get_request_params(request_params)

        last_message = multipart_messages[-1]

        # Prepare prior messages (everything before the last user message), or all if last is assistant
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )

        converted_prior: List[ChatCompletionMessageParam] = []
        for msg in messages_to_add:
            # convert_to_openai now returns a list of messages
            converted_prior.extend(OpenAIConverter.convert_to_openai(msg))

        # If the last message is from the assistant, no inference required
        if last_message.role == "assistant":
            return last_message

        # Convert the last user message
        converted_last = OpenAIConverter.convert_to_openai(last_message)
        if not converted_last:
            # Fallback for empty conversion
            converted_last = [{"role": "user", "content": ""}]

        # History-aware vs stateless turn construction
        if req_params.use_history:
            # Persist prior context to provider memory; send only the last message for this turn
            self.history.extend(converted_prior, is_prompt=is_template)
            turn_messages = converted_last
        else:
            # Do NOT persist; inline the full turn context to the provider call
            turn_messages = converted_prior + converted_last

        return await self._openai_completion(turn_messages, req_params, tools)

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
