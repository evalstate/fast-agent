import json
from typing import Any, List, Tuple, Type, Union, cast

from anthropic import APIError, AsyncAnthropic, AuthenticationError
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolUseBlock,
    ToolUseBlockParam,
    Usage,
)
from mcp import Tool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    TextContent,
)

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import ModelT
from fast_agent.llm.fastagent_llm import (
    FastAgentLLM,
    RequestParams,
)
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import (
    AnthropicConverter,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-0"
STRUCTURED_OUTPUT_TOOL_NAME = "return_structured_output"

# Type alias for system field - can be string or list of text blocks with cache control
SystemParam = Union[str, List[TextBlockParam]]

logger = get_logger(__name__)


class AnthropicLLM(FastAgentLLM[MessageParam, Message]):
    # Anthropic-specific parameter exclusions
    ANTHROPIC_EXCLUDE_FIELDS = {
        FastAgentLLM.PARAM_MESSAGES,
        FastAgentLLM.PARAM_MODEL,
        FastAgentLLM.PARAM_SYSTEM_PROMPT,
        FastAgentLLM.PARAM_STOP_SEQUENCES,
        FastAgentLLM.PARAM_MAX_TOKENS,
        FastAgentLLM.PARAM_METADATA,
        FastAgentLLM.PARAM_USE_HISTORY,
        FastAgentLLM.PARAM_MAX_ITERATIONS,
        FastAgentLLM.PARAM_PARALLEL_TOOL_CALLS,
        FastAgentLLM.PARAM_TEMPLATE_VARS,
        FastAgentLLM.PARAM_MCP_METADATA,
    }

    def __init__(self, *args, **kwargs) -> None:
        # Initialize logger - keep it simple without name reference
        super().__init__(*args, provider=Provider.ANTHROPIC, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Anthropic-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with Anthropic-specific settings
        chosen_model = kwargs.get("model", DEFAULT_ANTHROPIC_MODEL)
        base_params.model = chosen_model

        return base_params

    def _base_url(self) -> str | None:
        assert self.context.config
        return self.context.config.anthropic.base_url if self.context.config.anthropic else None

    def _get_cache_mode(self) -> str:
        """Get the cache mode configuration."""
        cache_mode = "auto"  # Default to auto
        if self.context.config and self.context.config.anthropic:
            cache_mode = self.context.config.anthropic.cache_mode
        return cache_mode

    async def _prepare_tools(
        self, structured_model: Type[ModelT] | None = None, tools: List[Tool] | None = None
    ) -> List[ToolParam]:
        """Prepare tools based on whether we're in structured output mode."""
        if structured_model:
            return [
                ToolParam(
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    description="Return the response in the required JSON format",
                    input_schema=structured_model.model_json_schema(),
                )
            ]
        else:
            # Regular mode - use tools from aggregator
            return [
                ToolParam(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )
                for tool in tools or []
            ]

    def _apply_system_cache(self, base_args: dict, cache_mode: str) -> None:
        """Apply cache control to system prompt if cache mode allows it."""
        system_content: SystemParam | None = base_args.get("system")

        if cache_mode != "off" and system_content:
            # Convert string to list format with cache control
            if isinstance(system_content, str):
                base_args["system"] = [
                    TextBlockParam(
                        type="text", text=system_content, cache_control={"type": "ephemeral"}
                    )
                ]
                logger.debug(
                    "Applied cache_control to system prompt (caches tools+system in one block)"
                )
            # If it's already a list (shouldn't happen in current flow but type-safe)
            elif isinstance(system_content, list):
                logger.debug("System prompt already in list format")
            else:
                logger.debug(f"Unexpected system prompt type: {type(system_content)}")

    async def _apply_conversation_cache(self, messages: List[MessageParam], cache_mode: str) -> int:
        """Apply conversation caching if in auto mode. Returns number of cache blocks applied."""
        applied_count = 0
        if cache_mode == "auto" and self.history.should_apply_conversation_cache():
            cache_updates = self.history.get_conversation_cache_updates()

            # Remove cache control from old positions
            if cache_updates["remove"]:
                self.history.remove_cache_control_from_messages(messages, cache_updates["remove"])
                logger.debug(
                    f"Removed conversation cache_control from positions {cache_updates['remove']}"
                )

            # Add cache control to new positions
            if cache_updates["add"]:
                applied_count = self.history.add_cache_control_to_messages(
                    messages, cache_updates["add"]
                )
                if applied_count > 0:
                    self.history.apply_conversation_cache_updates(cache_updates)
                    logger.debug(
                        f"Applied conversation cache_control to positions {cache_updates['add']} ({applied_count} blocks)"
                    )
                else:
                    logger.debug(
                        f"Failed to apply conversation cache_control to positions {cache_updates['add']}"
                    )

        return applied_count

    def _is_structured_output_request(self, tool_uses: List[Any]) -> bool:
        """
        Check if the tool uses contain a structured output request.

        Args:
            tool_uses: List of tool use blocks from the response

        Returns:
            True if any tool is the structured output tool
        """
        return any(tool.name == STRUCTURED_OUTPUT_TOOL_NAME for tool in tool_uses)

    def _build_tool_calls_dict(self, tool_uses: List[ToolUseBlock]) -> dict[str, CallToolRequest]:
        """
        Convert Anthropic tool use blocks into our CallToolRequest.

        Args:
            tool_uses: List of tool use blocks from Anthropic response

        Returns:
            Dictionary mapping tool_use_id to CallToolRequest objects
        """
        tool_calls = {}
        for tool_use in tool_uses:
            tool_call = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name=tool_use.name,
                    arguments=cast("dict[str, Any] | None", tool_use.input),
                ),
            )
            tool_calls[tool_use.id] = tool_call
        return tool_calls

    async def _handle_structured_output_response(
        self,
        tool_use_block: ToolUseBlock,
        structured_model: Type[ModelT],
        messages: List[MessageParam],
    ) -> Tuple[LlmStopReason, List[ContentBlock]]:
        """
        Handle a structured output tool response from Anthropic.

        This handles the special case where Anthropic's model was forced to use
        a 'return_structured_output' tool via tool_choice. The tool input contains
        the JSON data we want, so we extract it and format it for display.

        Even though we don't call an external tool, we must create a CallToolResult
        to satisfy Anthropic's API requirement that every tool_use has a corresponding
        tool_result in the next message.

        Args:
            tool_use_block: The tool use block containing structured output
            structured_model: The model class for structured output
            messages: The message list to append tool results to

        Returns:
            Tuple of (stop_reason, response_content_blocks)
        """
        tool_args = tool_use_block.input
        tool_use_id = tool_use_block.id

        # Create the content for responses
        structured_content = TextContent(type="text", text=json.dumps(tool_args))

        tool_result = CallToolResult(isError=False, content=[structured_content])
        messages.append(
            AnthropicConverter.create_tool_results_message([(tool_use_id, tool_result)])
        )

        logger.debug("Structured output received, treating as END_TURN")

        return LlmStopReason.END_TURN, [structured_content]

    async def _process_stream(self, stream: AsyncMessageStream, model: str) -> Message:
        """Process the streaming response and display real-time token usage."""
        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0
        tool_streams: dict[int, dict[str, Any]] = {}

        try:
            # Process the raw event stream to get token counts
            async for event in stream:
                if (
                    event.type == "content_block_start"
                    and hasattr(event, "content_block")
                    and getattr(event.content_block, "type", None) == "tool_use"
                ):
                    content_block = event.content_block
                    tool_streams[event.index] = {
                        "name": content_block.name,
                        "id": content_block.id,
                        "buffer": [],
                    }
                    self._notify_tool_stream_listeners(
                        "start",
                        {
                            "tool_name": content_block.name,
                            "tool_use_id": content_block.id,
                            "index": event.index,
                            "streams_arguments": False,  # Anthropic doesn't stream arguments
                        },
                    )
                    self.logger.info(
                        "Model started streaming tool input",
                        data={
                            "progress_action": ProgressAction.CALLING_TOOL,
                            "agent_name": self.name,
                            "model": model,
                            "tool_name": content_block.name,
                            "tool_use_id": content_block.id,
                            "tool_event": "start",
                        },
                    )
                    continue

                if (
                    event.type == "content_block_delta"
                    and hasattr(event, "delta")
                    and event.delta.type == "input_json_delta"
                ):
                    info = tool_streams.get(event.index)
                    if info is not None:
                        chunk = event.delta.partial_json or ""
                        info["buffer"].append(chunk)
                        preview = chunk if len(chunk) <= 80 else chunk[:77] + "..."
                        self._notify_tool_stream_listeners(
                            "delta",
                            {
                                "tool_name": info.get("name"),
                                "tool_use_id": info.get("id"),
                                "index": event.index,
                                "chunk": chunk,
                                "streams_arguments": False,
                            },
                        )
                        self.logger.debug(
                            "Streaming tool input delta",
                            data={
                                "tool_name": info.get("name"),
                                "tool_use_id": info.get("id"),
                                "chunk": preview,
                            },
                        )
                    continue

                if (
                    event.type == "content_block_stop"
                    and event.index in tool_streams
                ):
                    info = tool_streams.pop(event.index)
                    preview_raw = "".join(info.get("buffer", []))
                    if preview_raw:
                        preview = (
                            preview_raw if len(preview_raw) <= 120 else preview_raw[:117] + "..."
                        )
                        self.logger.debug(
                            "Completed tool input stream",
                            data={
                                "tool_name": info.get("name"),
                                "tool_use_id": info.get("id"),
                                "input_preview": preview,
                            },
                        )
                    self._notify_tool_stream_listeners(
                        "stop",
                        {
                            "tool_name": info.get("name"),
                            "tool_use_id": info.get("id"),
                            "index": event.index,
                            "streams_arguments": False,
                        },
                    )
                    self.logger.info(
                        "Model finished streaming tool input",
                        data={
                            "progress_action": ProgressAction.CALLING_TOOL,
                            "agent_name": self.name,
                            "model": model,
                            "tool_name": info.get("name"),
                            "tool_use_id": info.get("id"),
                            "tool_event": "stop",
                        },
                    )
                    continue

                # Count tokens in real-time from content_block_delta events
                if (
                    event.type == "content_block_delta"
                    and hasattr(event, "delta")
                    and event.delta.type == "text_delta"
                ):
                    # Use base class method for token estimation and progress emission
                    estimated_tokens = self._update_streaming_progress(
                        event.delta.text, model, estimated_tokens
                    )
                    self._notify_tool_stream_listeners(
                        "text",
                        {
                            "chunk": event.delta.text,
                            "index": event.index,
                            "streams_arguments": False,
                        },
                    )

                # Also check for final message_delta events with actual usage info
                elif (
                    event.type == "message_delta"
                    and hasattr(event, "usage")
                    and event.usage.output_tokens
                ):
                    actual_tokens = event.usage.output_tokens
                    # Emit final progress with actual token count
                    token_str = str(actual_tokens).rjust(5)
                    data = {
                        "progress_action": ProgressAction.STREAMING,
                        "model": model,
                        "agent_name": self.name,
                        "chat_turn": self.chat_turn(),
                        "details": token_str.strip(),
                    }
                    logger.info("Streaming progress", data=data)

            # Get the final message with complete usage data
            message = await stream.get_final_message()

            # Log final usage information
            if hasattr(message, "usage") and message.usage:
                logger.info(
                    f"Streaming complete - Model: {model}, Input tokens: {message.usage.input_tokens}, Output tokens: {message.usage.output_tokens}"
                )

            return message
        except APIError as error:
            logger.error("Streaming APIError during Anthropic completion", exc_info=error)
            raise  # Re-raise to be handled by _anthropic_completion
        except Exception as error:
            logger.error("Unexpected error during Anthropic stream processing", exc_info=error)
            # Convert to APIError for consistent handling
            raise APIError(f"Stream processing error: {str(error)}") from error

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

    async def _anthropic_completion(
        self,
        message_param,
        request_params: RequestParams | None = None,
        structured_model: Type[ModelT] | None = None,
        tools: List[Tool] | None = None,
        pre_messages: List[MessageParam] | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using an LLM and available tools.
        Override this method to use a different LLM.
        """

        api_key = self._api_key()
        base_url = self._base_url()
        if base_url and base_url.endswith("/v1"):
            base_url = base_url.rstrip("/v1")

        try:
            anthropic = AsyncAnthropic(api_key=api_key, base_url=base_url)
            messages: List[MessageParam] = list(pre_messages) if pre_messages else []
            params = self.get_request_params(request_params)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from e

        # Always include prompt messages, but only include conversation history if enabled
        messages.extend(self.history.get(include_completion_history=params.use_history))
        messages.append(message_param)  # message_param is the current user turn

        # Get cache mode configuration
        cache_mode = self._get_cache_mode()
        logger.debug(f"Anthropic cache_mode: {cache_mode}")

        available_tools = await self._prepare_tools(structured_model, tools)

        response_content_blocks: List[ContentBlock] = []
        tool_calls: dict[str, CallToolRequest] | None = None
        model = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL

        # Create base arguments dictionary
        base_args = {
            "model": model,
            "messages": messages,
            "stop_sequences": params.stopSequences,
            "tools": available_tools,
        }

        if self.instruction or params.systemPrompt:
            base_args["system"] = self.instruction or params.systemPrompt

        if structured_model:
            base_args["tool_choice"] = {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME}

        if params.maxTokens is not None:
            base_args["max_tokens"] = params.maxTokens

        self._log_chat_progress(self.chat_turn(), model=model)
        # Use the base class method to prepare all arguments with Anthropic-specific exclusions
        # Do this BEFORE applying cache control so metadata doesn't override cached fields
        arguments = self.prepare_provider_arguments(
            base_args, params, self.ANTHROPIC_EXCLUDE_FIELDS
        )

        # Apply cache control to system prompt AFTER merging arguments
        self._apply_system_cache(arguments, cache_mode)

        # Apply conversation caching
        applied_count = await self._apply_conversation_cache(messages, cache_mode)

        # Verify we don't exceed Anthropic's 4 cache block limit
        if applied_count > 0:
            total_cache_blocks = applied_count
            if cache_mode != "off" and arguments["system"]:
                total_cache_blocks += 1  # tools+system cache block
            if total_cache_blocks > 4:
                logger.warning(
                    f"Total cache blocks ({total_cache_blocks}) exceeds Anthropic limit of 4"
                )

        logger.debug(f"{arguments}")
        # Use streaming API with helper
        try:
            async with anthropic.messages.stream(**arguments) as stream:
                # Process the stream
                response = await self._process_stream(stream, model)
        except APIError as error:
            logger.error("Streaming APIError during Anthropic completion", exc_info=error)
            return self._stream_failure_response(error, model)

        # Track usage if response is valid and has usage data
        if (
            hasattr(response, "usage")
            and response.usage
            and not isinstance(response, BaseException)
        ):
            try:
                turn_usage = TurnUsage.from_anthropic(
                    response.usage, model or DEFAULT_ANTHROPIC_MODEL
                )
                self._finalize_turn_usage(turn_usage)
            except Exception as e:
                logger.warning(f"Failed to track usage: {e}")

        if isinstance(response, AuthenticationError):
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from response
        elif isinstance(response, BaseException):
            # This path shouldn't be reached anymore since we handle APIError above,
            # but keeping for backward compatibility
            logger.error(f"Unexpected error type: {type(response).__name__}", exc_info=response)
            return self._stream_failure_response(
                APIError(f"Unexpected error: {str(response)}"), model
            )

        logger.debug(
            f"{model} response:",
            data=response,
        )

        response_as_message = self.convert_message_to_message_param(response)
        messages.append(response_as_message)
        if response.content and response.content[0].type == "text":
            response_content_blocks.append(TextContent(type="text", text=response.content[0].text))

        stop_reason: LlmStopReason = LlmStopReason.END_TURN

        match response.stop_reason:
            case "stop_sequence":
                stop_reason = LlmStopReason.STOP_SEQUENCE
            case "max_tokens":
                stop_reason = LlmStopReason.MAX_TOKENS
            case "refusal":
                stop_reason = LlmStopReason.SAFETY
            case "pause":
                stop_reason = LlmStopReason.PAUSE
            case "tool_use":
                stop_reason = LlmStopReason.TOOL_USE
                tool_uses: list[ToolUseBlock] = [
                    c for c in response.content if c.type == "tool_use"
                ]
                if structured_model and self._is_structured_output_request(tool_uses):
                    stop_reason, structured_blocks = await self._handle_structured_output_response(
                        tool_uses[0], structured_model, messages
                    )
                    response_content_blocks.extend(structured_blocks)
                else:
                    tool_calls = self._build_tool_calls_dict(tool_uses)

        # Only save the new conversation messages to history if use_history is true
        # Keep the prompt messages separate
        if params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_completion_history=False)
            new_messages = messages[len(prompt_messages) :]
            self.history.set(new_messages)

        self._log_chat_finished(model=model)

        return Prompt.assistant(
            *response_content_blocks, stop_reason=stop_reason, tool_calls=tool_calls
        )

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageExtended"],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        # Effective params for this turn
        params = self.get_request_params(request_params)

        # Check the last message role
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted: List[MessageParam] = []

        # Get cache mode configuration
        cache_mode = self._get_cache_mode()

        for msg in messages_to_add:
            anthropic_msg = AnthropicConverter.convert_to_anthropic(msg)

            # Apply caching to template messages if cache_mode is "prompt" or "auto"
            if is_template and cache_mode in ["prompt", "auto"] and anthropic_msg.get("content"):
                content_list = anthropic_msg["content"]
                if isinstance(content_list, list) and content_list:
                    # Apply cache control to the last content block
                    last_block = content_list[-1]
                    if isinstance(last_block, dict):
                        last_block["cache_control"] = {"type": "ephemeral"}
                        logger.debug(
                            f"Applied cache_control to template message with role {anthropic_msg.get('role')}"
                        )

            converted.append(anthropic_msg)

        # Persist prior only when history is enabled; otherwise inline for this call
        pre_messages: List[MessageParam] | None = None
        if params.use_history:
            self.history.extend(converted, is_prompt=is_template)
        else:
            pre_messages = converted

        if last_message.role == "user":
            logger.debug("Last message in prompt is from user, generating assistant response")
            message_param = AnthropicConverter.convert_to_anthropic(last_message)
            return await self._anthropic_completion(
                message_param, request_params, tools=tools, pre_messages=pre_messages
            )
        else:
            # For assistant messages: Return the last message content as text
            logger.debug("Last message in prompt is from assistant, returning it directly")
            return last_message

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:  # noqa: F821
        request_params = self.get_request_params(request_params)

        # Check the last message role
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []

        for msg in messages_to_add:
            anthropic_msg = AnthropicConverter.convert_to_anthropic(msg)
            converted.append(anthropic_msg)

        self.history.extend(converted, is_prompt=False)

        if last_message.role == "user":
            logger.debug("Last message in prompt is from user, generating structured response")
            message_param = AnthropicConverter.convert_to_anthropic(last_message)

            # Call _anthropic_completion with the structured model
            result: PromptMessageExtended = await self._anthropic_completion(
                message_param, request_params, structured_model=model
            )

            for content in result.content:
                if content.type == "text":
                    try:
                        data = json.loads(content.text)
                        parsed_model = model(**data)
                        return parsed_model, result
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse structured output: {e}")
                        return None, result

            # If no valid response found
            return None, Prompt.assistant()
        else:
            # For assistant messages: Return the last message content
            logger.debug("Last message in prompt is from assistant, returning it directly")
            return None, last_message

    @classmethod
    def convert_message_to_message_param(cls, message: Message, **kwargs) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []

        for content_block in message.content:
            if content_block.type == "text":
                content.append(TextBlock(type="text", text=content_block.text))
            elif content_block.type == "tool_use":
                content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        name=content_block.name,
                        input=content_block.input,
                        id=content_block.id,
                    )
                )

        return MessageParam(role="assistant", content=content, **kwargs)

    def _show_usage(self, raw_usage: Usage, turn_usage: TurnUsage) -> None:
        """This is a debug routine, leaving in for convenience"""
        # Print raw usage for debugging
        print(f"\n=== USAGE DEBUG ({turn_usage.model}) ===")
        print(f"Raw usage: {raw_usage}")
        print(
            f"Turn usage: input={turn_usage.input_tokens}, output={turn_usage.output_tokens}, current_context={turn_usage.current_context_tokens}"
        )
        print(
            f"Cache: read={turn_usage.cache_usage.cache_read_tokens}, write={turn_usage.cache_usage.cache_write_tokens}"
        )
        print(f"Effective input: {turn_usage.effective_input_tokens}")
        print(
            f"Accumulator: total_turns={self.usage_accumulator.turn_count}, cumulative_billing={self.usage_accumulator.cumulative_billing_tokens}, current_context={self.usage_accumulator.current_context_tokens}"
        )
        if self.usage_accumulator.context_usage_percentage:
            print(
                f"Context usage: {self.usage_accumulator.context_usage_percentage:.1f}% of {self.usage_accumulator.context_window_size}"
            )
        if self.usage_accumulator.cache_hit_rate:
            print(f"Cache hit rate: {self.usage_accumulator.cache_hit_rate:.1f}%")
        print("===========================\n")
