import asyncio
import json
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Tuple,
    Type,
    cast,
)

from mcp.types import TextContent

if TYPE_CHECKING:
    from mcp import ListToolsResult

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
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
)
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError

# endregion
# region Internal Imports
## Internal Imports -- Core
from mcp_agent.core.prompt import Prompt

## Internal Imports -- Progress
from mcp_agent.event_progress import ProgressAction
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    RequestParams,
)
from mcp_agent.llm.context_truncation_and_summarization import ContextTruncation

## Internal Imports -- LLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.multipart_converter_anthropic import AnthropicConverter
from mcp_agent.llm.providers.sampling_converter_anthropic import AnthropicSamplingConverter
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.logging.logger import get_logger

## Internal Imports -- MCP
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# endregion


## Internal Imports -- MCP

# endregion


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-0"


class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    # Anthropic-specific parameter exclusions
    ANTHROPIC_EXCLUDE_FIELDS = {
        AugmentedLLM.PARAM_MESSAGES,
        AugmentedLLM.PARAM_MODEL,
        AugmentedLLM.PARAM_SYSTEM_PROMPT,
        AugmentedLLM.PARAM_STOP_SEQUENCES,
        AugmentedLLM.PARAM_MAX_TOKENS,
        AugmentedLLM.PARAM_METADATA,
        AugmentedLLM.PARAM_USE_HISTORY,
        AugmentedLLM.PARAM_MAX_ITERATIONS,
        AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS,
        AugmentedLLM.PARAM_TEMPLATE_VARS,
        AugmentedLLM.PARAM_CONTEXT_TRUNCATION_MODE,
        AugmentedLLM.PARAM_CONTEXT_TRUNCATION_LENGTH_LIMIT,
        AugmentedLLM.PARAM_REQUEST_DELAY_SECONDS,
    }

    def __init__(self, *args, **kwargs) -> None:
        # Initialize logger - keep it simple without name reference
        self.logger = get_logger(__name__)

        super().__init__(
            *args, provider=Provider.ANTHROPIC, type_converter=AnthropicSamplingConverter, **kwargs
        )

        self.client = self._initialize_client()  # Initialize the client once and reuse it

    def _initialize_client(self) -> AsyncAnthropic:
        """Initializes and returns the Anthropic API client."""
        try:
            api_key = self._api_key()
            base_url = self._base_url()
            if base_url and base_url.endswith("/v1"):
                base_url = base_url.rstrip("/v1")
            return AsyncAnthropic(api_key=api_key, base_url=base_url)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from e

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Anthropic-specific default parameters"""
        base_params = super()._initialize_default_params(
            kwargs
        )  # Get base defaults from parent (includes ModelDatabase lookup)
        base_params.model = kwargs.get(
            "model", DEFAULT_ANTHROPIC_MODEL
        )  # Override with Anthropic-specific settings

        return base_params

    def _base_url(self) -> Optional[str]:
        assert self.context.config
        return self.context.config.anthropic.base_url if self.context.config.anthropic else None

    def _get_cache_mode(self) -> str:
        """Get the cache mode configuration. Default 'auto'"""
        if self.context.config and self.context.config.anthropic:
            return self.context.config.anthropic.cache_mode
        return "auto"  # Default

    async def _prepare_tools(
        self, structured_model: Optional[Type[ModelT]] = None
    ) -> List[ToolParam]:
        """Prepare tools for the API call, handling structured output mode."""
        if structured_model:
            return [
                ToolParam(
                    name="return_structured_output",
                    description="Return the response in the required JSON format",
                    input_schema=structured_model.model_json_schema(),
                )
            ]

        tool_list: ListToolsResult = await self.aggregator.list_tools()
        return [
            ToolParam(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in tool_list.tools
        ]

    def _apply_system_cache(self, system_prompt: Any, cache_mode: str) -> Any:
        """
        Apply cache control to system prompt if cache mode allows it.
        Apply conversation caching. Returns number of cache blocks applied.
        """
        if cache_mode != "off" and isinstance(system_prompt, str) and system_prompt:
            self.logger.debug(
                "Applied cache_control to system prompt (caches tools+system in one block)"
            )
            return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

        if not isinstance(system_prompt, str):
            self.logger.debug(f"System prompt is not a string: {type(system_prompt)}")
        return system_prompt

    async def _apply_conversation_cache(self, messages: List[MessageParam], cache_mode: str) -> int:
        """Apply conversation caching if in auto mode. Returns number of cache blocks applied."""

        if cache_mode != "auto" or not self.history.should_apply_conversation_cache():
            return 0

        cache_updates = self.history.get_conversation_cache_updates()

        if cache_updates["remove"]:
            self.history.remove_cache_control_from_messages(messages, cache_updates["remove"])
            self.logger.debug(
                f"Removed conversation cache_control from positions {cache_updates['remove']}"
            )

        if cache_updates["add"]:
            applied_count = self.history.add_cache_control_to_messages(
                messages, cache_updates["add"]
            )
            if applied_count > 0:
                self.history.apply_conversation_cache_updates(cache_updates)
                self.logger.debug(
                    f"Applied conversation cache_control to positions {cache_updates['add']} ({applied_count} blocks)"
                )
                return applied_count
            else:
                self.logger.debug(
                    f"Failed to apply conversation cache_control to positions {cache_updates['add']}"
                )

        return 0

    def _check_cache_limit(
        self, conversation_cache_count: int, system_prompt: Any, cache_mode: str
    ):
        """Warns if the number of cache blocks exceeds Anthropic's limit."""
        system_cache_count = 1 if cache_mode != "off" and system_prompt else 0
        total_cache_blocks = conversation_cache_count + system_cache_count
        if total_cache_blocks > 4:
            self.logger.warning(
                f"Total cache blocks ({total_cache_blocks}) exceeds Anthropic limit of 4"
            )

    async def _process_structured_output(
        self,
        content_block: ToolUseBlock,
    ) -> Tuple[str, CallToolResult, TextContent]:
        """
        Process a structured output tool call from Anthropic.
        (For the special 'return_structured_output' tool call.)

        This handles the special case where Anthropic's model was forced to use
        a 'return_structured_output' tool via tool_choice. The tool input contains
        the JSON data we want, so we extract it and format it for display.

        Even though we don't call an external tool, we must create a CallToolResult
        to satisfy Anthropic's API requirement that every tool_use has a corresponding
        tool_result in the next message.

        Returns:
            Tuple of (tool_use_id, tool_result, content_block) for the structured data
        """
        tool_args = content_block.input
        tool_use_id = content_block.id

        # Show the formatted JSON response to the user
        json_response = json.dumps(tool_args, indent=2)
        await self.show_assistant_message(json_response)

        structured_content = TextContent(
            type="text", text=json.dumps(tool_args)
        )  # Create the content for responses

        tool_result = CallToolResult(
            isError=False, content=[structured_content]
        )  # Create a CallToolResult to satisfy Anthropic's API requirements. This represents the "result" of our structured output "tool"

        return tool_use_id, tool_result, structured_content

    async def _process_regular_tool_call(
        self,
        content_block: ToolUseBlock,
        available_tools: List[ToolParam],
        is_first_tool: bool,
        message_text: str | Text,
    ) -> Tuple[str, CallToolResult]:
        """
        Process a regular MCP tool call via the MCP aggregator.
        """
        if is_first_tool:
            await self.show_assistant_message(message_text, content_block.name)

        self.show_tool_call(
            available_tools=available_tools,
            tool_name=content_block.name,
            tool_args=content_block.input,
        )
        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name=content_block.name,
                arguments=content_block.input,
            ),
        )
        result = await self.call_tool(request=tool_call_request, tool_call_id=content_block.id)
        self.show_tool_result(result)
        return content_block.id, result

    async def _process_tool_calls(
        self,
        tool_uses: List[ToolUseBlock],
        available_tools: List[ToolParam],
        message_text: str | Text,
        structured_model: Optional[Type[ModelT]] = None,
    ) -> Tuple[List[Tuple[str, CallToolResult]], List[ContentBlock]]:
        """
        Process tool calls, handling both structured output and regular MCP tools.

        For structured output mode:
        - Extracts JSON data from the forced 'return_structured_output' tool
        - Does NOT create fake CallToolResults
        - Returns the JSON content directly

        For regular tools:
        - Calls actual MCP tools via the aggregator
        - Returns real CallToolResults
        """
        tool_results_for_api = []
        final_content_responses = []

        for tool_idx, content_block in enumerate(tool_uses):
            is_first_tool = tool_idx == 0

            if content_block.name == "return_structured_output" and structured_model:
                # Structured output: extract JSON, don't call external tools
                (
                    tool_use_id,
                    tool_result,
                    structured_content,
                ) = await self._process_structured_output(content_block=content_block)

                final_content_responses.append(structured_content)

                tool_results_for_api.append(
                    (tool_use_id, tool_result)
                )  # Add to tool_results to satisfy Anthropic's API requirement for tool_result messages
            else:
                # Regular tool: call external MCP tool
                tool_use_id, tool_result = await self._process_regular_tool_call(
                    content_block=content_block,
                    available_tools=available_tools,
                    is_first_tool=is_first_tool,
                    message_text=message_text,
                )

                final_content_responses.extend(tool_result.content)
                tool_results_for_api.append((tool_use_id, tool_result))

        return tool_results_for_api, final_content_responses

    def _prepare_request_payload(
        self,
        messages: List[MessageParam],
        params: RequestParams,
        tools: List[ToolParam],
        system_prompt: Any,
        structured_model: Optional[Type[ModelT]],
    ) -> dict:
        """Assembles the final dictionary of arguments for the Anthropic API call."""
        base_args = {
            "model": params.model,
            "messages": messages,
            "system": system_prompt,
            "stop_sequences": params.stopSequences,
            "tools": tools,
        }
        if structured_model:
            base_args["tool_choice"] = {"type": "tool", "name": "return_structured_output"}
        if params.maxTokens is not None:
            base_args["max_tokens"] = params.maxTokens

        # Use the base class method to merge remaining sampling parameters
        return self.prepare_provider_arguments(base_args, params, self.ANTHROPIC_EXCLUDE_FIELDS)

    async def execute_simple_api_call(self, message_string, max_tokens=2_000) -> str:
        model = self.default_request_params.model
        arguments = {
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message_string}],
            "model": model,
        }

        response = await self._execute_streaming_call(arguments=arguments, model=model)
        response_string = response.content[0].text

        return response_string

    async def _execute_streaming_call(self, arguments: dict, model: str) -> Message:
        """Executes the API call, processes the stream for real-time feedback, and returns the final message."""
        estimated_tokens = 0
        try:
            async with self.client.messages.stream(**arguments) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        estimated_tokens = self._update_streaming_progress(
                            event.delta.text, model, estimated_tokens
                        )
                    elif event.type == "message_delta" and hasattr(event, "usage"):
                        self._log_final_streaming_progress(event.usage.output_tokens, model)

                message = await stream.get_final_message()
                if hasattr(message, "usage") and message.usage:
                    self.logger.info(
                        f"Streaming complete - Model: {model}, Input tokens: {message.usage.input_tokens}, Output tokens: {message.usage.output_tokens}"
                    )
                return message
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Anthropic API key was rejected during a call.",
                "Please check your API key.",
            ) from e
        except APIError as e:
            self.logger.error(f"Anthropic API Error: {e}", exc_info=True)

            return Message(  # Create a synthetic error message to avoid crashing the agent
                id="error",
                model="error",
                role="assistant",
                type="message",
                content=[TextBlock(type="text", text=f"Error during generation: {e}")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0),
            )

    async def _process_response_actions(
        self,
        response: Message,
        messages: List[MessageParam],
        available_tools: List[ToolParam],
        params: RequestParams,
        structured_model: Optional[Type[ModelT]],
    ) -> Tuple[str, List[ContentBlock], Optional[MessageParam]]:
        """
        Processes the final API message, handles actions based on stop_reason, and returns the outcome.
        Returns a tuple of (action, content_responses, next_message_to_append).
        """
        response_as_message_param = self.convert_message_to_message_param(response)

        text_content = "".join(
            [
                block.text
                for block in response.content
                if hasattr(block, "type") and block.type == "text"
            ]
        )

        if response.stop_reason == "tool_use":
            tool_uses = [c for c in response.content if isinstance(c, ToolUseBlock)]
            if not tool_uses:
                return self.ACTIONS.STOP, [], response_as_message_param

            message_text = text_content or Text(
                "the assistant requested tool calls", style="dim green italic"
            )
            tool_results_for_api, tool_content = await self._process_tool_calls(
                tool_uses, available_tools, message_text, structured_model
            )

            # For structured output, we stop after getting the tool call result.
            if structured_model:
                return self.ACTIONS.STOP, tool_content, response_as_message_param

            # For regular tools, we create a tool_results message and continue the loop.
            tool_results_message = AnthropicConverter.create_tool_results_message(
                tool_results_for_api
            )
            return "CONTINUE_WITH_TOOLS", tool_content, tool_results_message

        # Handle all terminal states
        if response.stop_reason in ["end_turn", "stop_sequence"]:
            await self.show_assistant_message(text_content)
        elif response.stop_reason == "max_tokens":
            limit = f"({params.maxTokens})" if params.maxTokens else ""
            await self.show_assistant_message(
                Text(
                    f"the assistant has reached the maximum token limit {limit}",
                    style="dim green italic",
                )
            )

        final_responses = [TextContent(type="text", text=text_content)] if text_content else []
        return self.ACTIONS.STOP, final_responses, response_as_message_param

    async def _process_stream(self, stream: AsyncMessageStream, model: str) -> Message:
        """Process the streaming response and display real-time token usage."""
        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0

        # Process the raw event stream to get token counts
        async for event in stream:
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
                self.logger.info("Streaming progress", data=data)

        # Get the final message with complete usage data
        message = await stream.get_final_message()

        # Log final usage information
        if hasattr(message, "usage") and message.usage:
            self.logger.info(
                f"Streaming complete - Model: {model}, Input tokens: {message.usage.input_tokens}, Output tokens: {message.usage.output_tokens}"
            )

        return message

    async def _anthropic_completion(
        self,
        message_param: MessageParam,
        request_params: Optional[RequestParams] = None,
        structured_model: Optional[Type[ModelT]] = None,
    ) -> list[ContentBlock]:
        """
        Orchestrates the process of sending a prompt to Anthropic and handling the response.
        Process a query using an LLM and available tools.
        """
        # Initialization of History Incl. the New Message
        params = self.get_request_params(request_params)
        messages: List[MessageParam] = self.history.get(
            include_completion_history=params.use_history
        )
        messages.append(message_param)
        all_content_responses: List[ContentBlock] = []

        # System Prompt
        system_prompt = self.instruction or params.systemPrompt

        # Caching
        cache_mode = self._get_cache_mode()
        self.logger.debug(f"Anthropic cache_mode: {cache_mode}")

        available_tools = await self._prepare_tools(structured_model)

        for i in range(params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=params.model)

            if hasattr(params, "context_truncation_mode") and params.context_truncation_mode:
                # 1. Convert from Anthropic's format to your internal MCP format
                multipart_messages = AnthropicConverter.convert_from_anthropic_list_to_multipart(
                    messages
                )

                if (
                    hasattr(params, "context_truncation_length_limit")
                    and params.context_truncation_length_limit
                ):
                    token_limit_for_truncation = params.context_truncation_length_limit
                else:
                    token_limit_for_truncation = params.maxTokens

                # 2. Call the truncation manager to truncate the history if needed
                truncated_multipart: List[
                    PromptMessageMultipart
                ] = await ContextTruncation.truncate_if_required(
                    messages=multipart_messages,
                    truncation_mode=params.context_truncation_mode,
                    limit=token_limit_for_truncation,
                    model_name=params.model,
                    system_prompt=system_prompt,
                    provider=self,
                )

                # 3. If truncation occurred, convert back and update the messages list
                old_token_length = self.get_token_count(
                    messages=multipart_messages, system_prompt=None
                )
                new_token_length = self.get_token_count(
                    messages=truncated_multipart, system_prompt=None
                )

                if new_token_length < old_token_length:
                    self.logger.info(
                        f"History truncated from {old_token_length} to {new_token_length} tokens."
                    )
                    messages = AnthropicConverter.convert_from_multipart_to_anthropic_list(
                        truncated_multipart
                    )

            # 4. Apply Caching
            final_system_prompt = self._apply_system_cache(
                system_prompt=system_prompt, cache_mode=cache_mode
            )
            conversation_cache_count = await self._apply_conversation_cache(
                messages=messages, cache_mode=cache_mode
            )
            self._check_cache_limit(
                conversation_cache_count=conversation_cache_count,
                system_prompt=final_system_prompt,
                cache_mode=cache_mode,
            )

            # 5. Build Payload and Execute API Call
            arguments = self._prepare_request_payload(
                messages=messages,
                params=params,
                tools=available_tools,
                system_prompt=final_system_prompt,
                structured_model=structured_model,
            )

            self.logger.debug(f"Prepared arguments for Anthropic API: {str(arguments)[:50]}")
            self.logger.debug(f"params: {params}")

            if params.request_delay_seconds > 0.0:
                self.logger.info(f"Sleeping for {self.PARAM_REQUEST_DELAY_SECONDS} seconds.")
                await asyncio.sleep(self.PARAM_REQUEST_DELAY_SECONDS)
            response = await self._execute_streaming_call(
                arguments=arguments,
                model=params.model,
            )
            assistant_message = self.convert_message_to_message_param(response)
            messages.append(assistant_message)

            # 4. Track Usage
            if hasattr(response, "usage") and response.usage:
                turn_usage = TurnUsage.from_anthropic(usage=response.usage, model=params.model)
                self._finalize_turn_usage(turn_usage=turn_usage)

            # 5. Process Response and Determine Next Action
            action, new_content, tool_results_message = await self._process_response_actions(
                response=response,
                messages=messages,
                available_tools=available_tools,
                params=params,
                structured_model=structured_model,
            )
            if new_content:
                all_content_responses.extend(new_content)

            if tool_results_message:
                messages.append(tool_results_message)

            if action == self.ACTIONS.STOP:
                self.logger.debug(
                    f"Iteration {i}: Stopping because action is {response.stop_reason}"
                )
                break
        else:
            self.logger.warning(
                f"Exceeded max iterations ({params.max_iterations}) without stopping."
            )

        # 6. Finalize History
        # Apply cache control to system prompt
        if params.use_history:
            prompt_len = len(self.history.get(include_completion_history=False))
            self.history.set(messages[prompt_len:])

        self._log_chat_finished(model=params.model)
        return all_content_responses

    async def generate_messages(
        self,
        message_param: MessageParam,
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        self._reset_turn_tool_calls()  # Reset tool call counter for new turn

        res = await self._anthropic_completion(
            message_param=message_param,
            request_params=request_params,
        )
        return Prompt.assistant(*res)

    def _prepare_and_set_history(
        self, multipart_messages: List[PromptMessageMultipart], is_template: bool
    ) -> None:
        """Converts messages and adds them to history, applying prompt caching if applicable."""
        cache_mode = self._get_cache_mode()
        converted = []
        for msg in multipart_messages:
            anthropic_msg = AnthropicConverter.convert_to_anthropic(msg)
            # Apply caching to template messages
            if (
                is_template
                and cache_mode in ["prompt", "auto"]
                and isinstance(anthropic_msg.get("content"), list)
            ):
                content_list = cast("list", anthropic_msg["content"])
                if content_list and isinstance(content_list[-1], dict):
                    content_list[-1]["cache_control"] = {"type": "ephemeral"}
                    self.logger.debug(
                        f"Applied cache_control to template message with role {anthropic_msg.get('role')}"
                    )
            converted.append(anthropic_msg)
        self.history.extend(converted, is_prompt=is_template)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """Applies a prompt, handling history and generating a response if the last message is from the user."""
        last_message = multipart_messages[-1]  # Check the last message role
        messages_to_add_to_history = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        self._prepare_and_set_history(messages_to_add_to_history, is_template)

        if last_message.role == "user":
            self.logger.debug("Last message in prompt is from user, generating assistant response")
            message_param = AnthropicConverter.convert_to_anthropic(last_message)
            return await self.generate_messages(message_param, request_params)

        self.logger.debug("Last message in prompt is from assistant, returning it directly.")
        return last_message

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[Optional[ModelT], PromptMessageMultipart]:
        """Applies a prompt and generates a structured (JSON) response."""
        last_message = multipart_messages[-1]  # Check the last message role

        messages_to_add_to_history = (  # Add all previous messages to history (or all messages if last is from assistant)
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )

        self._prepare_and_set_history(messages_to_add_to_history, is_template=False)

        if last_message.role == "user":
            self.logger.debug("Last message in prompt is from user, generating structured response")
            message_param = AnthropicConverter.convert_to_anthropic(last_message)
            response_content = await self._anthropic_completion(
                message_param=message_param, request_params=request_params, structured_model=model
            )

            for content in response_content:  # Extract the structured data from the response
                if content.type == "text":
                    try:
                        data = json.loads(content.text)  # Parse the JSON response from the tool
                        parsed_model = model(**data)
                        return parsed_model, Prompt.assistant(content)

                    except (json.JSONDecodeError, ValueError) as e:
                        self.logger.error(f"Failed to parse structured output: {e}")
                        return None, Prompt.assistant(content)

            return None, Prompt.assistant()  # If no valid response found

        # For assistant messages: Return the last message content
        self.logger.debug("Last message in prompt is from assistant, returning it directly")
        return None, last_message

    def _update_streaming_progress(
        self,
        text_chunk: str,
        model: str,
        estimated_tokens: int,
    ) -> int:
        """
        This calls a method on the parent class AugmentedLLM.
        """
        return super()._update_streaming_progress(text_chunk, model, estimated_tokens)

    def _show_usage(self, raw_usage: Usage, turn_usage: TurnUsage) -> None:
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

    @classmethod
    def convert_message_to_message_param(
        cls,
        message: Message,
        **kwargs,
    ) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []

        for content_block in message.content:
            if content_block.type == "text":
                content.append(TextBlockParam(type="text", text=content_block.text))
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

