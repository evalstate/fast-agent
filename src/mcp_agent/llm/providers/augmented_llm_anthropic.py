import os
from typing import TYPE_CHECKING, List

from mcp.types import EmbeddedResource, ImageContent, TextContent

from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.providers.multipart_converter_anthropic import (
    AnthropicConverter,
)
from mcp_agent.llm.providers.sampling_converter_anthropic import (
    AnthropicSamplingConverter,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from mcp import ListToolsResult


from anthropic import Anthropic, AuthenticationError
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolUseBlockParam,
    Usage,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
)
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    RequestParams,
)
from mcp_agent.logging.logger import get_logger

DEFAULT_ANTHROPIC_MODEL = "claude-3-7-sonnet-latest"


class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.provider = "Anthropic"
        # Initialize logger - keep it simple without name reference
        self.logger = get_logger(__name__)

        # Now call super().__init__
        super().__init__(*args, type_converter=AnthropicSamplingConverter, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Anthropic-specific default parameters"""
        return RequestParams(
            model=kwargs.get("model", DEFAULT_ANTHROPIC_MODEL),
            maxTokens=4096,  # default haiku3
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str | None:
        assert self.context.config
        return self.context.config.anthropic.base_url if self.context.config.anthropic else None

    async def generate_internal(
        self,
        message_param,
        request_params: RequestParams | None = None,
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using an LLM and available tools.
        Override this method to use a different LLM.
        """

        api_key = self._api_key(self.context.config)
        base_url = self._base_url()
        if base_url and base_url.endswith("/v1"):
            base_url = base_url.rstrip("/v1")

        try:
            anthropic = Anthropic(api_key=api_key, base_url=base_url)
            messages: List[MessageParam] = []
            params = self.get_request_params(request_params)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from e

        # Always include prompt messages, but only include conversation history
        # if use_history is True
        messages.extend(self.history.get(include_history=params.use_history))

        messages.append(message_param)

        tool_list: ListToolsResult = await self.aggregator.list_tools()
        available_tools: List[ToolParam] = [
            ToolParam(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in tool_list.tools
        ]

        responses: List[TextContent | ImageContent | EmbeddedResource] = []

        model = self.default_request_params.model

        for i in range(params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=model)
            arguments = {
                "model": model,
                "messages": messages,
                "system": self.instruction or params.systemPrompt,
                "stop_sequences": params.stopSequences,
                "tools": available_tools,
            }

            if params.maxTokens is not None:
                arguments["max_tokens"] = params.maxTokens

            if params.metadata:
                arguments = {**arguments, **params.metadata}

            self.logger.debug(f"{arguments}")

            executor_result = await self.executor.execute(anthropic.messages.create, **arguments)

            response = executor_result[0]

            if isinstance(response, AuthenticationError):
                raise ProviderKeyError(
                    "Invalid Anthropic API key",
                    "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
                ) from response
            elif isinstance(response, BaseException):
                error_details = str(response)
                self.logger.error(f"Error: {error_details}", data=executor_result)

                # Try to extract more useful information for API errors
                if hasattr(response, "status_code") and hasattr(response, "response"):
                    try:
                        error_json = response.response.json()
                        error_details = f"Error code: {response.status_code} - {error_json}"
                    except:  # noqa: E722
                        error_details = f"Error code: {response.status_code} - {str(response)}"

                # Convert other errors to text response
                error_message = f"Error during generation: {error_details}"
                response = Message(
                    id="error",  # Required field
                    model="error",  # Required field
                    role="assistant",
                    type="message",
                    content=[TextBlock(type="text", text=error_message)],
                    stop_reason="end_turn",  # Must be one of the allowed values
                    usage=Usage(input_tokens=0, output_tokens=0),  # Required field
                )

            self.logger.debug(
                f"{model} response:",
                data=response,
            )

            response_as_message = self.convert_message_to_message_param(response)
            messages.append(response_as_message)
            if response.content[0].type == "text":
                responses.append(TextContent(type="text", text=response.content[0].text))

            if response.stop_reason == "end_turn":
                message_text = ""
                for block in response_as_message["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        message_text += block.get("text", "")
                    elif hasattr(block, "type") and block.type == "text":
                        message_text += block.text

                await self.show_assistant_message(message_text)

                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'end_turn'")
                break
            elif response.stop_reason == "stop_sequence":
                # We have reached a stop sequence
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'"
                )
                break
            elif response.stop_reason == "max_tokens":
                # We have reached the max tokens limit

                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'max_tokens'")
                if params.maxTokens is not None:
                    message_text = Text(
                        f"the assistant has reached the maximum token limit ({params.maxTokens})",
                        style="dim green italic",
                    )
                else:
                    message_text = Text(
                        "the assistant has reached the maximum token limit",
                        style="dim green italic",
                    )

                await self.show_assistant_message(message_text)

                break
            else:
                message_text = ""
                for block in response_as_message["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        message_text += block.get("text", "")
                    elif hasattr(block, "type") and block.type == "text":
                        message_text += block.text

                # response.stop_reason == "tool_use":
                # First, collect all tool uses in this turn
                tool_uses = [c for c in response.content if c.type == "tool_use"]

                if tool_uses:
                    if message_text == "":
                        message_text = Text(
                            "the assistant requested tool calls",
                            style="dim green italic",
                        )

                    # Process all tool calls and collect results
                    tool_results = []
                    for i, content in enumerate(tool_uses):
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id

                        if i == 0:  # Only show message for first tool use
                            await self.show_assistant_message(message_text, tool_name)

                        self.show_tool_call(available_tools, tool_name, tool_args)
                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
                        )
                        # TODO -- support MCP isError etc.
                        result = await self.call_tool(
                            request=tool_call_request, tool_call_id=tool_use_id
                        )
                        self.show_tool_result(result)

                        # Add each result to our collection
                        tool_results.append((tool_use_id, result))
                        responses.extend(result.content)

                    messages.append(AnthropicConverter.create_tool_results_message(tool_results))

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

    def _api_key(self, config):
        api_key = None

        if hasattr(config, "anthropic") and config.anthropic:
            api_key = config.anthropic.api_key
            if api_key == "<your-api-key-here>":
                api_key = None

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "Anthropic API key not configured",
                "The Anthropic API key is required but not set.\n"
                "Add it to your configuration file under anthropic.api_key "
                "or set the ANTHROPIC_API_KEY environment variable.",
            )

        return api_key

    async def generate_messages(
        self,
        message_param,
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.

        """
        res = await self.generate_internal(
            message_param=message_param,
            request_params=request_params,
        )
        return Prompt.assistant(*res)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        # Check the last message role
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []
        for msg in messages_to_add:
            converted.append(AnthropicConverter.convert_to_anthropic(msg))

        self.history.extend(converted, is_prompt=True)

        if last_message.role == "user":
            self.logger.debug("Last message in prompt is from user, generating assistant response")
            message_param = AnthropicConverter.convert_to_anthropic(last_message)
            return await self.generate_messages(message_param, request_params)
        else:
            # For assistant messages: Return the last message content as text
            self.logger.debug("Last message in prompt is from assistant, returning it directly")
            return last_message

    @classmethod
    def convert_message_to_message_param(cls, message: Message, **kwargs) -> MessageParam:
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
