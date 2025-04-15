import os
from typing import List

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from openai import AuthenticationError, AzureOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from pydantic_core import from_json
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.augmented_llm import (
    RequestParams,
)
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter
from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_REASONING_EFFORT = "medium"


class AzureOpenAIAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, provider_name: str = "OpenAI", *args, **kwargs) -> None:
        super().__init__(provider_name=provider_name, *args, **kwargs)

    def _api_version(self) -> str:
        config = self.context.config
        api_version = None

        if hasattr(config, "openai") and config.openai:
            api_version = config.openai.api_version

        if api_version is None:
            api_version = os.getenv("AZURE_OPENAI_VERSION")

        if api_version is None:
            url = self._base_url()
            if "api-version" in url:
                api_version = url.split("api-version=")[-1].split("&")[0]

        if not api_version:
            raise ProviderKeyError(
                "Azure OpenAI API version not configured",
                "The OpenAI API Version is required but not set.\n"
                "Add it to your configuration file under azure_openai.api_version\n"
                "Or set the AZURE_OPENAI_VERSION environment variable",
            )
        return api_version

    def _api_key(self) -> str:
        config = self.context.config
        api_key = None

        if hasattr(config, "azure_openai") and config.azure_openai:
            api_key = config.azure_openai.api_key
            if api_key == "<your-api-key-here>":
                api_key = None

        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "Azure OpenAI API key not configured",
                "The Azure OpenAI API key is required but not set.\n"
                "Add it to your configuration file under azure_openai.api_key\n"
                "Or set the Azure_OPENAI_API_KEY environment variable",
            )
        return api_key

    def _base_url(self) -> str:
        return (
            self.context.config.azure_openai.base_url if self.context.config.azure_openai else None
        )

    async def generate_internal(
        self,
        message,
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using an LLM and available tools.
        This implementation uses AzureOpenAI
        Override this method to use a different LLM.
        """

        try:
            openai_client = AzureOpenAI(
                api_key=self._api_key(), base_url=self._base_url(), api_version=self._api_version()
            )
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
