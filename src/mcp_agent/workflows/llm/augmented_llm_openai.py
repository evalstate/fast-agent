import json
import os
from typing import List, Type, TYPE_CHECKING

from pydantic_core import from_json

from mcp_agent.workflows.llm.providers.multipart_converter_openai import OpenAIConverter
from mcp_agent.workflows.llm.providers.sampling_converter_openai import (
    OpenAISamplingConverter,
)

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from openai import OpenAI, AuthenticationError

# from openai.types.beta.chat import
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessage,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    CallToolResult,
)

from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    ModelT,
    RequestParams,
)
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.logging.logger import get_logger
from rich.text import Text

_logger = get_logger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_REASONING_EFFORT = "medium"


class OpenAIAugmentedLLM(
    AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]
):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    def __init__(self, *args, **kwargs):
        # Set type_converter before calling super().__init__
        if "type_converter" not in kwargs:
            kwargs["type_converter"] = OpenAISamplingConverter

        super().__init__(*args, **kwargs)

        self.provider = "OpenAI"
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
        chosen_model = (
            self.default_request_params.model if self.default_request_params else None
        )
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

        # Get default model from config if available
        if self.context and self.context.config and self.context.config.openai:
            if hasattr(self.context.config.openai, "default_model"):
                chosen_model = self.context.config.openai.default_model

        return RequestParams(
            model=chosen_model,
            modelPreferences=self.model_preferences,
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
        return (
            self.context.config.openai.base_url if self.context.config.openai else None
        )

    async def generate(
        self,
        message,
        request_params: RequestParams | None = None,
        response_model: Type[ModelT] | None = None,
    ) -> List[ChatCompletionMessage]:
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
            messages.append(
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            )

        # Always include prompt messages, but only include conversation history
        # if use_history is True
        messages.extend(self.history.get(include_history=params.use_history))

        if isinstance(message, str):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=message)
            )
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] = [
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
            available_tools = []

        responses: List[ChatCompletionMessage] = []
        model = await self.select_model(params)
        chat_turn = len(messages) // 2
        if self._reasoning:
            self.show_user_message(
                str(message), f"{model} ({self._reasoning_effort})", chat_turn
            )
        else:
            self.show_user_message(str(message), model, chat_turn)

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
            self._log_chat_progress(chat_turn, model=model)

            if response_model is None:
                executor_result = await self.executor.execute(
                    openai_client.chat.completions.create, **arguments
                )
            else:
                executor_result = await self.executor.execute(
                    openai_client.beta.chat.completions.parse,
                    **arguments,
                    response_format=response_model,
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

            # TODO: saqadri - handle multiple choices for more complex interactions.
            # Keeping it simple for now because multiple choices will also complicate memory management
            choice = response.choices[0]
            message = choice.message
            responses.append(message)

            converted_message = self.convert_message_to_message_param(
                message, name=self.name
            )
            messages.append(converted_message)
            message_text = converted_message.content
            if (
                choice.finish_reason in ["tool_calls", "function_call"]
                and message.tool_calls
            ):
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
                            arguments=from_json(
                                tool_call.function.arguments, allow_partial=True
                            ),
                        ),
                    )
                    result = await self.call_tool(tool_call_request, tool_call.id)
                    self.show_oai_tool_result(str(result))

                    tool_results.append((tool_call.id, result))

                messages.extend(
                    OpenAIConverter.convert_function_results_to_openai(tool_results)
                )

                self.logger.debug(
                    f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                )
            elif choice.finish_reason == "length":
                # We have reached the max tokens limit
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'length'"
                )
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
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'stop'"
                )
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

    async def generate_str(
        self,
        message,
        request_params: RequestParams | None = None,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.

        Special commands:
        - "***SAVE_HISTORY <filename.md>" - Saves the conversation history to the specified file
          in MCP prompt format with user/assistant delimiters.
        """
        # Check if this is a special command to save history
        if isinstance(message, str) and message.startswith("***SAVE_HISTORY "):
            return await self._save_history_to_file(message)

        responses = await self.generate(
            message=message,
            request_params=request_params,
        )

        final_text: List[str] = []

        for response in responses:
            content = response.content
            if not content:
                continue

            if isinstance(content, str):
                final_text.append(content)
                continue

        return "\n".join(final_text)

    async def _apply_prompt_template_provider_specific(
        self, multipart_messages: List["PromptMessageMultipart"]
    ) -> str:
        """
        OpenAI-specific implementation of apply_prompt_template that handles
        multimodal content natively.

        Args:
            multipart_messages: List of PromptMessageMultipart objects parsed from the prompt template

        Returns:
            String representation of the assistant's response if generated,
            or the last assistant message in the prompt
        """
        from openai.types.chat import ChatCompletionSystemMessageParam

        # Check the last message role
        last_message = multipart_messages[-1]

        if last_message.role == "user":
            # For user messages: Add all previous messages to history, then generate response to the last one
            self.logger.debug(
                "Last message in prompt is from user, generating assistant response"
            )

            # Add all but the last message to history
            if len(multipart_messages) > 1:
                previous_messages = multipart_messages[:-1]
                converted = []

                # Convert all previous messages to OpenAI format
                for msg in previous_messages:
                    converted.append(OpenAIConverter.convert_to_openai(msg))

                # Add system prompt at the beginning if it exists
                if self.instruction or (
                    self.default_request_params
                    and self.default_request_params.systemPrompt
                ):
                    system_prompt = (
                        self.instruction or self.default_request_params.systemPrompt
                    )
                    if system_prompt and system_prompt not in [
                        m.get("content") for m in converted if m.get("role") == "system"
                    ]:
                        converted.insert(
                            0,
                            ChatCompletionSystemMessageParam(
                                role="system", content=system_prompt
                            ),
                        )

                self.history.extend(converted, is_prompt=True)

            # Convert the last message to OpenAI format and generate a response
            message_param = OpenAIConverter.convert_to_openai(last_message)
            return await self.generate_str(message_param)
        else:
            # For assistant messages: Add all messages to history and return the last one
            self.logger.debug(
                "Last message in prompt is from assistant, returning it directly"
            )

            # Convert and add all messages to history
            converted = []

            # Convert all messages to OpenAI format
            for msg in multipart_messages:
                converted.append(OpenAIConverter.convert_to_openai(msg))

            # Add system prompt at the beginning if it exists
            if self.instruction or (
                self.default_request_params and self.default_request_params.systemPrompt
            ):
                system_prompt = (
                    self.instruction or self.default_request_params.systemPrompt
                )
                if system_prompt and system_prompt not in [
                    m.get("content") for m in converted if m.get("role") == "system"
                ]:
                    converted.insert(
                        0,
                        ChatCompletionSystemMessageParam(
                            role="system", content=system_prompt
                        ),
                    )

            self.history.extend(converted, is_prompt=True)

            # Process the last message content for display
            assistant_text_parts = []
            has_non_text_content = False

            for content in last_message.content:
                if content.type == "text":
                    assistant_text_parts.append(content.text)
                elif content.type == "resource" and hasattr(content.resource, "text"):
                    # Add resource text with metadata
                    mime_type = getattr(content.resource, "mimeType", "text/plain")
                    uri = getattr(content.resource, "uri", "")
                    if uri:
                        assistant_text_parts.append(
                            f"[Resource: {uri}, Type: {mime_type}]\n{content.resource.text}"
                        )
                    else:
                        assistant_text_parts.append(
                            f"[Resource Type: {mime_type}]\n{content.resource.text}"
                        )
                elif content.type == "image":
                    # Note the presence of images
                    mime_type = getattr(content, "mimeType", "image/unknown")
                    assistant_text_parts.append(f"[Image: {mime_type}]")
                    has_non_text_content = True
                else:
                    # Other content types
                    assistant_text_parts.append(f"[Content of type: {content.type}]")
                    has_non_text_content = True

            # Join all parts with double newlines for better readability
            result = (
                "\n\n".join(assistant_text_parts)
                if assistant_text_parts
                else str(last_message.content)
            )

            # Add a note if non-text content was present
            if has_non_text_content:
                result += "\n\n[Note: This message contained non-text content that may not be fully represented in text format]"

            return result

    async def _save_history_to_file(self, command: str) -> str:
        """
        Save the conversation history to a file in MCP prompt format.

        Args:
            command: The command string, expected format: "***SAVE_HISTORY <filename.md>"

        Returns:
            Success or error message
        """
        try:
            # Extract the filename from the command
            parts = command.split(" ", 1)
            if len(parts) != 2 or not parts[1].strip():
                return "Error: Invalid format. Expected '***SAVE_HISTORY <filename.md>'"

            filename = parts[1].strip()

            # Get all messages from history
            messages = self.history.get(include_history=True)

            # Import required utilities
            from mcp_agent.workflows.llm.openai_utils import (
                openai_message_param_to_prompt_message_multipart,
            )
            from mcp_agent.mcp.prompt_serialization import (
                multipart_messages_to_delimited_format,
            )

            # Convert message params to PromptMessageMultipart objects
            multipart_messages = []
            for msg in messages:
                # Skip system messages - PromptMessageMultipart only supports user and assistant roles
                if isinstance(msg, dict) and msg.get("role") == "system":
                    continue

                # Convert the message to a multipart message
                multipart_messages.append(
                    openai_message_param_to_prompt_message_multipart(msg)
                )

            # Convert to delimited format
            delimited_content = multipart_messages_to_delimited_format(
                multipart_messages,
                user_delimiter="---USER",
                assistant_delimiter="---ASSISTANT",
            )

            # Write to file
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n\n".join(delimited_content))

            self.logger.info(f"Saved conversation history to {filename}")
            return f"Done. Saved conversation history to {filename}"

        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
            return f"Error saving history: {str(e)}"

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        responses = await self.generate(
            message=message,
            request_params=request_params,
            response_model=response_model,
        )
        return responses[0].parsed

    async def generate_prompt(
        self, prompt: "PromptMessageMultipart", request_params: RequestParams | None
    ) -> str:
        converted_prompt = OpenAIConverter.convert_to_openai(prompt)
        return await self.generate_str(converted_prompt, request_params)

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        return result

    def message_param_str(self, message: ChatCompletionMessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:  # content is a list
                final_text: List[str] = []
                for part in content:
                    text_part = part.get("text")
                    if text_part:
                        final_text.append(str(text_part))
                    else:
                        final_text.append(str(part))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: ChatCompletionMessage) -> str:
        """Convert an output message to a string representation."""
        content = message.content
        if content:
            return content

        return str(message)
