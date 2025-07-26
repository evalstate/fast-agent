# region External Imports
import json
from warnings import deprecated
from typing import Dict, List, Optional, Tuple, Type
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    TextContent,
)
from openai import APIError, AsyncOpenAI, AuthenticationError
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from openai.types.responses import (
    ResponseFunctionToolCall
)
from pydantic_core import from_json
from rich.text import Text
# endregion

# region Internal Imports
## Internal Imports -- Core
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
## Internal Imports -- MCP
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
## Internal Imports -- Other Stuff
from mcp_agent.event_progress import ProgressAction
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    RequestParams,
)
from mcp_agent.logging.logger import get_logger
## Internal Imports -- LLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.multipart_converter_openai import (
    OpenAIConverter, 
    OpenAIMessage,
)
from mcp_agent.llm.context_truncation_and_summarization import (
    ContextTruncation
)
from mcp_agent.llm.providers.sampling_converter_openai import (
    OpenAISamplingConverter,
)
from mcp_agent.llm.usage_tracking import TurnUsage
# endregion


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


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
        if "type_converter" not in kwargs:  # Set type_converter before calling super().__init__()
            kwargs["type_converter"] = OpenAISamplingConverter

        super().__init__(*args, provider=provider, **kwargs)
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)
        self.client = self._initialize_client()

        # Set up reasoning-related attributes
        self._reasoning_effort = kwargs.get("reasoning_effort", None)
        if self.context and self.context.config and self.context.config.openai:
            if self._reasoning_effort is None and hasattr(
                self.context.config.openai, "reasoning_effort"
            ):
                self._reasoning_effort = self.context.config.openai.reasoning_effort

        # Determine if we're using a reasoning model -- TODO -- move this to model capabilities.
        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning = chosen_model and (
            chosen_model.startswith("o1")
            or chosen_model.startswith("o2")
            or chosen_model.startswith("o3")
            or chosen_model.startswith("o4")
            or chosen_model.startswith("o5")
        )
        if self._reasoning:
            self.logger.info(
                f"Using reasoning model '{chosen_model}' with '{self._reasoning_effort}' reasoning effort"
            )
        else:
            self.logger.info(
                f"Using non-reasoning model '{chosen_model}' without reasoning effort self._reasoning_effort: '{self._reasoning_effort}'"
            )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenAI-specific default parameters. Get base defaults from parent (includes ModelDatabase lookup)"""
        base_params = super()._initialize_default_params(kwargs)
        base_params.model = kwargs.get("model", DEFAULT_OPENAI_MODEL)

        return base_params

    def _base_url(self) -> str:
        return self.context.config.openai.base_url if self.context.config.openai else None

    async def _prepare_tools(
        self, structured_model: Optional[Type[ModelT]] = None
    ) -> List[ChatCompletionToolParam]:
        """Prepare tools for the API call."""
        # Note: OpenAI does not currently support a forced structured output mode
        # in the same way Anthropic does. This parameter is kept for interface consistency.
        if structured_model:
            self.logger.warning(
                "OpenAI provider does not have a dedicated structured output mode like Anthropic; "
                "standard tool-use will be used."
            )

        tool_list = await self.aggregator.list_tools()
        # OpenAI requires 'properties' to be an object, even if empty.
        available_tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                    if "properties" in tool.inputSchema
                    else {**tool.inputSchema, "properties": {}},
                },
            )
            for tool in tool_list.tools
        ]
        return available_tools if available_tools else None

    def _initialize_client(self) -> AsyncOpenAI:
        try:
            return AsyncOpenAI(api_key=self._api_key(), base_url=self._base_url())

        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

    #####################################
    ############## API Call Handling ####
    #####################################

    def _prepare_request_payload(
        self,
        messages: List[ChatCompletionMessageParam],
        params: RequestParams,
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> dict:
        """Assembles the final dictionary of arguments for the OpenAI API call."""
        base_args = {
            "model": params.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Add tools if they are available
        if tools:
            base_args["tools"] = tools
            base_args["parallel_tool_calls"] = params.parallel_tool_calls

        # Add max_tokens if specified
        if params.maxTokens is not None:
            base_args["max_tokens"] = params.maxTokens

        # Use the base class method to merge remaining sampling parameters
        return self.prepare_provider_arguments(
            base_args, params, self.OPENAI_EXCLUDE_FIELDS.union(self.BASE_EXCLUDE_FIELDS)
        )

    async def _execute_streaming_call(
        self, arguments: dict, model: str
    ) -> ChatCompletionMessage:
        """Executes the API call, processes the stream, and returns the final message."""
        estimated_tokens = 0
        state = ChatCompletionStreamState()

        try:
            stream = await self.client.chat.completions.create(**arguments)
            async for chunk in stream:
                state.handle_chunk(chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    estimated_tokens = self._update_streaming_progress(
                        chunk.choices[0].delta.content, model, estimated_tokens
                    )

            final_completion = state.get_final_completion()

            if hasattr(final_completion, "usage") and final_completion.usage:
                self.logger.info(
                    f"Streaming complete - Model: {model}, "
                    f"Input tokens: {final_completion.usage.prompt_tokens}, "
                    f"Output tokens: {final_completion.usage.completion_tokens}"
                )
                self._log_final_streaming_progress(
                    final_completion.usage.completion_tokens, model
                )
                turn_usage = TurnUsage.from_openai(final_completion.usage, model)
                self._finalize_turn_usage(turn_usage)

            if not final_completion.choices:
                raise APIError("No response choices received from OpenAI.", request=None)

            return final_completion.choices[0].message

        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key rejected during a call.", "Please check your API key."
            ) from e
        except APIError as e:
            self.logger.error(f"OpenAI API Error: {e}", exc_info=True)
            return ChatCompletionMessage(
                role="assistant", content=f"Error during generation: {e}"
            )

    async def _process_tool_calls(
        self, tool_calls: List[ChatCompletionMessageToolCall], available_tools: List[ChatCompletionToolParam]
    ) -> Tuple[List[ContentBlock], ChatCompletionMessageParam]:
        """Processes tool calls by executing them and preparing the result message."""
        final_content_responses = []
        tool_results_for_api = []

        for tool_call in tool_calls:
            self.show_tool_call(
                available_tools, tool_call.function.name, tool_call.function.arguments
            )

            arguments = {} if not tool_call.function.arguments or tool_call.function.arguments.strip() == "" else from_json(tool_call.function.arguments, allow_partial=True)
            function_name = tool_call.function.name
            
            tool_call_request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name=function_name,
                    arguments=arguments,
                ),
            )
            result = await self.call_tool(tool_call_request, tool_call.id)
            self.show_tool_result(result)

            final_content_responses.extend(result.content)
            tool_results_for_api.append((tool_call.id, result))

        tool_results_message = OpenAIConverter.convert_function_results_to_openai(
            results=tool_results_for_api,
            arguments=arguments,
            function_name=function_name,
            concatenate_text_blocks=False,

        )
        return final_content_responses, tool_results_message

    async def _process_response_actions(
        self,
        message: ChatCompletionMessage,
        finish_reason: str,
        available_tools: List[ChatCompletionToolParam],
        params: RequestParams,
    ) -> Tuple[str, List[ContentBlock], Optional[List[ChatCompletionMessageParam]]]:
        """Processes the API message, handles actions based on finish_reason, and returns the outcome."""
        messages_to_append = [message]
        text_content = message.content or ""

        if finish_reason == "tool_calls" and message.tool_calls:
            message_text = text_content or Text(
                "the assistant requested tool calls", style="dim green italic"
            )
            await self.show_assistant_message(message_text, message.tool_calls[0].function.name)

            tool_calls: List[ResponseFunctionToolCall] = message.tool_calls

            tool_content, tool_results_message = await self._process_tool_calls(
                tool_calls, available_tools
            )

            # Why extend? Not append?
            messages_to_append.extend(tool_results_message)
            return "CONTINUE_WITH_TOOLS", tool_content, messages_to_append

        # Handle all terminal states
        if finish_reason == "stop":
            await self.show_assistant_message(text_content)
        elif finish_reason == "length":
            limit = f"({params.maxTokens})" if params.maxTokens else ""
            await self.show_assistant_message(
                Text(
                    f"the assistant has reached the maximum token limit {limit}",
                    style="dim green italic",
                )
            )
        elif finish_reason == "content_filter":
            await self.show_assistant_message(
                Text("the response was filtered due to content policies.", style="dim red")
            )

        final_responses = [TextContent(type="text", text=text_content)] if text_content else []
        return self.ACTIONS.STOP, final_responses, messages_to_append

    async def _openai_completion(
        self,
        message: OpenAIMessage,
        request_params: Optional[RequestParams] = None,
        structured_model: Optional[Type[ModelT]] = None,

    ) -> List[ContentBlock]:
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        """
        request_params = self.get_request_params(request_params)
        messages: List[ChatCompletionMessageParam] = self.history.get(include_completion_history=request_params.use_history)
        messages.append(message)
        all_content_responses: List[ContentBlock] = []

        system_prompt = self.instruction or request_params.systemPrompt

        messages.extend(self.history.get(include_completion_history=request_params.use_history))
        messages.append(message)  # New message

        available_tools = await self._prepare_tools(structured_model)
        
        if not available_tools:
            if self.provider == Provider.DEEPSEEK:
                available_tools = None  # deepseek does not allow empty array
            else:
                available_tools = []

        # we do NOT send "stop sequences" as this causes errors with mutlimodal processing
        for i in range(request_params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=request_params.model)

            if hasattr(request_params, "context_truncation_mode") and request_params.context_truncation_mode:
                # 1. Convert from Anthropic's format to your internal MCP format                
                
                for message in messages:
                    message["role"]
                
                multipart_messages = OpenAIConverter.convert_from_openai_list_to_multipart(messages)

                if hasattr(request_params, "context_truncation_length_limit") and request_params.context_truncation_length_limit:
                    token_limit_for_truncation = request_params.context_truncation_length_limit
                else:
                    token_limit_for_truncation = request_params.maxTokens

                # 2. Call the truncation manager to truncate the history if needed
                truncated_multipart: List[PromptMessageMultipart] = await ContextTruncation.truncate_if_required(
                    messages=multipart_messages,
                    truncation_mode=request_params.context_truncation_mode,
                    limit=token_limit_for_truncation,
                    model_name=request_params.model,
                    system_prompt=system_prompt,
                    provider=self,
                )

                # 3. If truncation occurred, convert back and update the messages list
                old_token_length = ContextTruncation._estimate_tokens(messages=multipart_messages, model=request_params.model, system_prompt="")
                new_token_length = ContextTruncation._estimate_tokens(messages=truncated_multipart, model=request_params.model, system_prompt="")

                if new_token_length < old_token_length:
                    self.logger.info(
                        f"History truncated from {old_token_length} to {new_token_length} tokens."
                    )
                    messages = OpenAIConverter.convert_from_openai_list_to_multipart(truncated_multipart)

            arguments = self._prepare_api_request(
                messages=messages, 
                tools=available_tools, 
                request_params=request_params
            )
            response_message = await self._execute_streaming_call(arguments, request_params.model)
            
            self.logger.debug(f"OpenAI completion requested for: {arguments}")
            self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)

            # The finish_reason is on the choice object, not the message itself.
            # We get this from the stateful stream processor's final result.
            # This is a simplification; in a real scenario, we'd get this from `_execute_streaming_call`.
            # For this refactor, we assume the reason is on the message for simplicity.
            # A more robust implementation would return a tuple (message, finish_reason) from execute_call.
            finish_reason = getattr(response_message, "finish_reason", "stop")
            if response_message.tool_calls:
                finish_reason = "tool_calls"


            #
            #
            #          -.-
            #  
            #
            action, new_content, messages_to_append = await self._process_response_actions(
                message=response_message, finish_reason=finish_reason, available_tools=available_tools, params=request_params
            )

            if new_content:
                all_content_responses.extend(new_content)
            if messages_to_append:
                messages.extend(messages_to_append)

            # Track usage if response is valid and has usage data
            if (
                hasattr(response_message, "usage")
                and response_message.usage
                and not isinstance(response_message, BaseException)
            ):
                try:
                    model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL
                    turn_usage = TurnUsage.from_openai(response_message.usage, model_name)
                    self._finalize_turn_usage(turn_usage)
                except Exception as e:
                    self.logger.warning(f"Failed to track usage: {e}")

            self.logger.debug(
                "OpenAI completion response_message:",
                data=response_message,
            )

            if action == self.ACTIONS.STOP:
                self.logger.debug(f"Iteration {i}: Stopping because action is '{finish_reason}'")
                break

        else:
            self.logger.warning(
                f"Exceeded max iterations ({request_params.max_iterations}) without stopping."
            )

        if request_params.use_history:
            prompt_messages = self.history.get(include_completion_history=False)
            system_offset = 1 if system_prompt else 0
            new_history_start_index = len(prompt_messages) + system_offset
            self.history.set(messages[new_history_start_index:])

        self._log_chat_finished(model=request_params.model)
        return all_content_responses

    async def _is_tool_stop_reason(self, finish_reason: str) -> bool:
        return True

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """Applies a prompt, handling history and generating a response."""
        self._reset_turn_tool_calls()
        
        # Add all previous messages to history (or all messages if last is from assistant)
        # if the last message is a "user", inference is required
        last_message = multipart_messages[-1]

        if last_message.role == "user":
            messages_to_add = multipart_messages[:-1]
        else:
            messages_to_add = multipart_messages
       
        #######################################
        #### Convert and Add to History #######
        #######################################
        openai_messages = []                 ##
        for message in messages_to_add:      ##
            openai_messages.append(OpenAIConverter.convert_to_openai(message)
        )                                    ##
        self.history.extend(openai_messages, is_prompt=is_template)
        #######################################

        if "assistant" == last_message.role:
            return last_message  # For assistant messages: Return the last message (no completion needed)

        #######################################
        #### Call OpenAI API ##################
        #######################################
        message_param: OpenAIMessage = OpenAIConverter.convert_to_openai(last_message)
        responses: List[ContentBlock] = await self._openai_completion(
            message=message_param,           ##
            request_params=request_params,   ##
        )                                    ##
        return Prompt.assistant(*responses)  ##
        #######################################

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


