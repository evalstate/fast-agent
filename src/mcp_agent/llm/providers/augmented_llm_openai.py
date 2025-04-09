import os
from typing import List, Tuple, Type

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from openai import AuthenticationError, OpenAI

# from openai.types.beta.chat import
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from pydantic_core import from_json
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    ModelT,
    RequestParams,
)
from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter
from mcp_agent.llm.providers.sampling_converter_openai import (
    OpenAISamplingConverter,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

_logger = get_logger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_REASONING_EFFORT = "medium"


class OpenAIAugmentedLLM(AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    def __init__(self, provider_name: str = "OpenAI", *args, **kwargs) -> None:
        # Set type_converter before calling super().__init__
        if "type_converter" not in kwargs:
            kwargs["type_converter"] = OpenAISamplingConverter

        super().__init__(*args, **kwargs)

        self.provider = provider_name
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
        chosen_model = kwargs.get("model", DEFAULT_OPENAI_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _api_key(self) -> str:
        config = self.context.config
        api_key = None

        if hasattr(config, "openai") and config.openai:
            api_key = config.openai.api_key
            if api_key == "<your-api-key-here>":
                api_key = None

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "OpenAI API key not configured",
                "The OpenAI API key is required but not set.\n"
                "Add it to your configuration file under openai.api_key\n"
                "Or set the OPENAI_API_KEY environment variable",
            )
        return api_key

    def _base_url(self) -> str:
        return self.context.config.openai.base_url if self.context.config.openai else None

    async def generate_internal(
        self,
        message,
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """

        try:
            openai_client = OpenAI(api_key=self._api_key(), base_url=self._base_url())
            messages: List[ChatCompletionMessageParam] = []
            params = self.get_request_params(request_params)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

        system_prompt = self.instruction or params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        # Always include prompt messages, but only include conversation history
        # if use_history is True
        messages.extend(self.history.get(include_history=params.use_history))

        if isinstance(message, str):
            messages.append(ChatCompletionUserMessageParam(role="user", content=message))
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] | None = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                    # TODO: saqadri - determine if we should specify "strict" to True by default
                },
            )
            for tool in response.tools
        ]
        if not available_tools:
            available_tools = None  # deepseek does not allow empty array

        responses: List[TextContent | ImageContent | EmbeddedResource] = []
        model = self.default_request_params.model

        # we do NOT send stop sequences as this causes errors with mutlimodal processing
        for i in range(params.max_iterations):
            arguments = {
                "model": model or "gpt-4o",
                "messages": messages,
                "tools": available_tools,
            }
            if self._reasoning:
                arguments = {
                    **arguments,
                    "max_completion_tokens": params.maxTokens,
                    "reasoning_effort": self._reasoning_effort,
                }
            else:
                arguments = {**arguments, "max_tokens": params.maxTokens}
                if available_tools:
                    arguments["parallel_tool_calls"] = params.parallel_tool_calls

            if params.metadata:
                arguments = {**arguments, **params.metadata}

            self.logger.debug(f"{arguments}")
            self._log_chat_progress(self.chat_turn(), model=model)

            executor_result = await self.executor.execute(
                openai_client.chat.completions.create, **arguments
            )

            response = executor_result[0]

            self.logger.debug(
                "OpenAI ChatCompletion response:",
                data=response,
            )

            if isinstance(response, AuthenticationError):
                raise ProviderKeyError(
                    "Invalid OpenAI API key",
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

            converted_message = self.convert_message_to_message_param(message, name=self.name)
            messages.append(converted_message)
            message_text = converted_message.content
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
                            arguments=from_json(tool_call.function.arguments, allow_partial=True),
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
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            elif choice.finish_reason == "content_filter":
                # The response was filtered by the content filter
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            elif choice.finish_reason == "stop":
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                if message_text:
                    await self.show_assistant_message(message_text, "")
                break

        # Only save the new conversation messages to history if use_history is true
        # Keep the prompt messages separate
        if params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_history=False)

            # Calculate new conversation messages (excluding prompts)
            new_messages = messages[len(prompt_messages) :]

            # Update conversation history
            self.history.set(new_messages)

        self._log_chat_finished(model=model)

        return responses

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        # TODO -- this is very similar to Anthropic (just the converter class changes).
        # TODO -- potential refactor to base class, standardize Converter interface
        # Check the last message role
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []
        for msg in messages_to_add:
            converted.append(OpenAIConverter.convert_to_openai(msg))
        self.history.extend(converted, is_prompt=True)

        if last_message.role == "user":
            # For user messages: Generate response to the last one
            self.logger.debug("Last message in prompt is from user, generating assistant response")
            message_param = OpenAIConverter.convert_to_openai(last_message)
            responses: List[
                TextContent | ImageContent | EmbeddedResource
            ] = await self.generate_internal(
                message_param,
                request_params,
            )
            return Prompt.assistant(*responses)
        else:
            # For assistant messages: Return the last message content as text
            self.logger.debug("Last message in prompt is from assistant, returning it directly")
            return last_message

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Apply the prompt and return the result as a Pydantic model.

        Uses OpenAI's beta parse feature when compatible, falling back to standard
        JSON parsing if the beta feature fails or is unavailable.

        Args:
            prompt: List of messages to process
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed response as a Pydantic model, or None if parsing fails
        """

        if not "OpenAI" == self.provider:
            return await super().structured(prompt, model, request_params)

        logger = get_logger(__name__)

        # First try to use OpenAI's beta.chat.completions.parse feature
        try:
            # Convert the multipart messages to OpenAI format
            messages = []
            for msg in prompt:
                messages.append(OpenAIConverter.convert_to_openai(msg))

            # Add system prompt if available and not already present
            if self.instruction and not any(m.get("role") == "system" for m in messages):
                system_msg = ChatCompletionSystemMessageParam(
                    role="system", content=self.instruction
                )
                messages.insert(0, system_msg)
            model_name = self.default_request_params.model
            self.show_user_message(prompt[-1].first_text(), model_name, self.chat_turn())
            # Use the beta parse feature
            try:
                openai_client = OpenAI(api_key=self._api_key(), base_url=self._base_url())
                model_name = self.default_request_params.model

                logger.debug(
                    f"Using OpenAI beta parse with model {model_name} for structured output"
                )
                response = await self.executor.execute(
                    openai_client.beta.chat.completions.parse,
                    model=model_name,
                    messages=messages,
                    response_format=model,
                )

                if response and isinstance(response[0], BaseException):
                    raise response[0]
                parsed_result = response[0].choices[0].message
                await self.show_assistant_message(parsed_result.content)
                logger.debug("Successfully used OpenAI beta parse feature for structured output")
                return parsed_result.parsed, Prompt.assistant(parsed_result.content)

            except (ImportError, AttributeError, NotImplementedError) as e:
                # Beta feature not available, log and continue to fallback
                logger.debug(f"OpenAI beta parse feature not available: {str(e)}")
                # Continue to fallback

        except Exception as e:
            logger.debug(f"OpenAI beta parse failed: {str(e)}, falling back to standard method")
            # Continue to standard method as fallback

        # Fallback to standard method (inheriting from base class)
        return await super().structured(prompt, model, request_params)

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        return result
