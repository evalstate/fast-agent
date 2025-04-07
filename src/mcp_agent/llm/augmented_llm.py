import json
import re
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from mcp.types import (
    CallToolRequest,
    CallToolResult,
    GetPromptResult,
    PromptMessage,
    TextContent,
)
from pydantic_core import from_json
from rich.text import Text

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.core.exceptions import PromptExitError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.event_progress import ProgressAction
from mcp_agent.llm.memory import Memory, SimpleMemory
from mcp_agent.llm.sampling_format_converter import (
    BasicFormatConverter,
    ProviderFormatConverter,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import (
    AugmentedLLMProtocol,
    ModelT,
)
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompt_render import render_multipart_message
from mcp_agent.mcp.prompt_serialization import multipart_messages_to_delimited_format
from mcp_agent.ui.console_display import ConsoleDisplay

# Define type variables locally
MessageParamT = TypeVar("MessageParamT")
MessageT = TypeVar("MessageT")

# Forward reference for type annotations
if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.context import Context


# TODO -- move this to a constant
HUMAN_INPUT_TOOL_NAME = "__human_input__"


class AugmentedLLM(ContextDependent, AugmentedLLMProtocol, Generic[MessageParamT, MessageT]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    provider: str | None = None

    def __init__(
        self,
        agent: Optional["Agent"] = None,
        server_names: List[str] | None = None,
        instruction: str | None = None,
        name: str | None = None,
        request_params: RequestParams | None = None,
        type_converter: Type[
            ProviderFormatConverter[MessageParamT, MessageT]
        ] = BasicFormatConverter,
        context: Optional["Context"] = None,
        model: Optional[str] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize the LLM with a list of server names and an instruction.
        If a name is provided, it will be used to identify the LLM.
        If an agent is provided, all other properties are optional

        Args:
            agent: Optional Agent that owns this LLM
            server_names: List of MCP server names to connect to
            instruction: System prompt for the LLM
            name: Optional name identifier for the LLM
            request_params: RequestParams to configure LLM behavior
            type_converter: Provider-specific format converter class
            context: Application context
            model: Optional model name override
            **kwargs: Additional provider-specific parameters
        """
        # Extract request_params before super() call
        self._init_request_params = request_params
        super().__init__(context=context, **kwargs)
        self.logger = get_logger(__name__)
        self.executor = self.context.executor
        self.aggregator = agent if agent is not None else MCPAggregator(server_names or [])
        self.name = agent.name if agent else name
        self.instruction = agent.instruction if agent else instruction

        # memory contains provider specific API types.
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()

        self._message_history: List[PromptMessageMultipart] = []

        # Initialize the display component
        self.display = ConsoleDisplay(config=self.context.config)

        # Initialize default parameters
        self.default_request_params = self._initialize_default_params(kwargs)

        # Apply model override if provided
        if model:
            self.default_request_params.model = model

        # Merge with provided params if any
        if self._init_request_params:
            self.default_request_params = self._merge_request_params(
                self.default_request_params, self._init_request_params
            )

        self.type_converter = type_converter
        self.verb = kwargs.get("verb")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize default parameters for the LLM.
        Should be overridden by provider implementations to set provider-specific defaults."""
        return RequestParams(
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Apply the prompt and return the result as a Pydantic model, or None if coercion fails"""
        logger = get_logger(__name__) # Ensure logger is available
        try:
            result: PromptMessageMultipart = await self.generate(prompt, request_params)
            raw_text = result.first_text()

            # Attempt to find and extract the first JSON object or array
            json_match = re.search(r'(\{.*\}|\[.*\])', raw_text, re.DOTALL)
            extracted_json_str = None
            if json_match:
                potential_json = json_match.group(0)
                # Basic validation: check if it starts and ends with matching braces/brackets
                if (potential_json.startswith('{') and potential_json.endswith('}')) or \
                   (potential_json.startswith('[') and potential_json.endswith(']')):
                   extracted_json_str = potential_json

            if not extracted_json_str:
                 # Fallback: Try finding the start of a JSON object/array
                 start_index = -1
                 first_char = ''
                 brace_index = raw_text.find('{')
                 bracket_index = raw_text.find('[')

                 if brace_index != -1 and (bracket_index == -1 or brace_index < bracket_index):
                     start_index = brace_index
                     first_char = '{'
                 elif bracket_index != -1:
                     start_index = bracket_index
                     first_char = '['

                 if start_index != -1:
                    extracted_json_str = raw_text[start_index:]
                    # More robust extraction (optional, simple find might suffice with pydantic_core)
                    # try:
                    #     # Use json.JSONDecoder for potentially more robust partial parsing if needed
                    #     decoder = json.JSONDecoder()
                    #     parsed_obj, end_index = decoder.raw_decode(extracted_json_str)
                    #     extracted_json_str = extracted_json_str[:end_index] # Keep only the valid part
                    # except json.JSONDecodeError:
                    #      extracted_json_str = None # Failed to extract robustly


            if extracted_json_str:
                try:
                    # Use pydantic_core's from_json for validation AND parsing
                    json_data = from_json(extracted_json_str, allow_partial=True)
                    validated_model = model.model_validate(json_data)
                    # Return the validated model and the *original* raw assistant response for history
                    return cast("ModelT", validated_model), result
                except Exception as validation_error:
                    logger.warning(f"JSON found but failed Pydantic validation: {validation_error}. Raw JSON tried: '{extracted_json_str}'")
                    # Fall through to the main error handling
            else:
                 logger.warning(f"No valid JSON object or array found in the response. Raw text: '{raw_text}'")
                 # Fall through

            # If extraction or validation failed up to this point
            raise ValueError("Failed to extract or validate JSON from the response.")

        except Exception as e:
            # Log the original error if it's different from our ValueError
            if not isinstance(e, ValueError) or "Failed to extract" not in str(e) :
                 logger.error(f"Failed to parse structured response: {str(e)}. Raw text was: '{result.first_text() if 'result' in locals() else 'N/A'}'")
            # Return None and an assistant message indicating failure (using the original raw response)
            failure_message = result if 'result' in locals() else Prompt.assistant(f"Failed to process structured response: {str(e)}")
            # Ensure the failure message has helpful text content if the original was complex
            if not failure_message.content or not isinstance(failure_message.content[0], TextContent):
                 failure_message = Prompt.assistant(f"Failed to process structured response: {str(e)}")

            return None, failure_message

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Create a completion with the LLM using the provided messages.
        """
        if multipart_messages[-1].first_text().startswith("***SAVE_HISTORY"):
            parts: list[str] = multipart_messages[-1].first_text().split(" ", 1)
            filename: str = (
                parts[1].strip() if len(parts) > 1 else f"{self.name or 'assistant'}_prompts.txt"
            )
            await self._save_history(filename)
            self.show_user_message(
                f"History saved to {filename}", model=self.default_request_params.model, chat_turn=0
            )
            return Prompt.assistant(f"History saved to {filename}")

        self._message_history.extend(multipart_messages)

        if multipart_messages[-1].role == "user":
            self.show_user_message(
                render_multipart_message(multipart_messages[-1]),
                model=self.default_request_params.model,
                chat_turn=self.chat_turn(),
            )

        assistant_response: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )

        self._message_history.append(assistant_response)
        return assistant_response

    def chat_turn(self) -> int:
        """Return the current chat turn number"""
        return 1 + sum(1 for message in self._message_history if message.role == "assistant")

    def _merge_request_params(
        self, default_params: RequestParams, provided_params: RequestParams
    ) -> RequestParams:
        """Merge default and provided request parameters"""

        merged = default_params.model_dump()
        merged.update(provided_params.model_dump(exclude_unset=True))
        final_params = RequestParams(**merged)

        return final_params

    def get_request_params(
        self,
        request_params: RequestParams | None = None,
        default: RequestParams | None = None,
    ) -> RequestParams:
        """
        Get request parameters with merged-in defaults and overrides.
        Args:
            request_params: The request parameters to use as overrides.
            default: The default request parameters to use as the base.
                If unspecified, self.default_request_params will be used.
        """
        # Start with the defaults
        default_request_params = default or self.default_request_params

        if not default_request_params:
            default_request_params = self._initialize_default_params({})

        # If user provides overrides, merge them with defaults
        if request_params:
            return self._merge_request_params(default_request_params, request_params)

        return default_request_params

    @classmethod
    def convert_message_to_message_param(
        cls, message: MessageT, **kwargs: dict[str, Any]
    ) -> MessageParamT:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        # Many LLM implementations will allow the same type for input and output messages
        return cast("MessageParamT", message)

    def show_tool_result(self, result: CallToolResult) -> None:
        """Display a tool result in a formatted panel."""
        self.display.show_tool_result(result)

    def show_oai_tool_result(self, result: str) -> None:
        """Display a tool result in a formatted panel."""
        self.display.show_oai_tool_result(result)

    def show_tool_call(self, available_tools, tool_name, tool_args) -> None:
        """Display a tool call in a formatted panel."""
        self.display.show_tool_call(available_tools, tool_name, tool_args)

    async def show_assistant_message(
        self,
        message_text: str | Text | None,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
    ) -> None:
        if message_text is None:
            message_text = Text("No content to display", style="dim green italic")
        """Display an assistant message in a formatted panel."""
        await self.display.show_assistant_message(
            message_text,
            aggregator=self.aggregator,
            highlight_namespaced_tool=highlight_namespaced_tool,
            title=title,
            name=self.name,
        )

    def show_user_message(self, message, model: str | None, chat_turn: int) -> None:
        """Display a user message in a formatted panel."""
        self.display.show_user_message(message, model, chat_turn, name=self.name)

    async def pre_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        """Called before a tool is executed. Return False to prevent execution."""
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        """Called after a tool execution. Can modify the result before it's returned."""
        return result

    async def call_tool(
        self,
        request: CallToolRequest,
        tool_call_id: str | None = None,
    ) -> CallToolResult:
        """Call a tool with the given parameters and optional ID"""

        try:
            preprocess = await self.pre_tool_call(
                tool_call_id=tool_call_id,
                request=request,
            )

            if isinstance(preprocess, bool):
                if not preprocess:
                    return CallToolResult(
                        isError=True,
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error: Tool '{request.params.name}' was not allowed to run.",
                            )
                        ],
                    )
            else:
                request = preprocess

            tool_name = request.params.name
            tool_args = request.params.arguments
            result = await self.aggregator.call_tool(tool_name, tool_args)

            postprocess = await self.post_tool_call(
                tool_call_id=tool_call_id, request=request, result=result
            )

            if isinstance(postprocess, CallToolResult):
                result = postprocess

            return result
        except PromptExitError:
            raise
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error executing tool '{request.params.name}': {str(e)}",
                    )
                ],
            )

    def _log_chat_progress(
        self, chat_turn: Optional[int] = None, model: Optional[str] = None
    ) -> None:
        """Log a chat progress event"""
        # Determine action type based on verb
        if hasattr(self, "verb") and self.verb:
            # Use verb directly regardless of type
            act = self.verb
        else:
            act = ProgressAction.CHATTING

        data = {
            "progress_action": act,
            "model": model,
            "agent_name": self.name,
            "chat_turn": chat_turn if chat_turn is not None else None,
        }
        self.logger.debug("Chat in progress", data=data)

    def _log_chat_finished(self, model: Optional[str] = None) -> None:
        """Log a chat finished event"""
        data = {
            "progress_action": ProgressAction.READY,
            "model": model,
            "agent_name": self.name,
        }
        self.logger.debug("Chat finished", data=data)

    def _convert_prompt_messages(self, prompt_messages: List[PromptMessage]) -> List[MessageParamT]:
        """
        Convert prompt messages to this LLM's specific message format.
        To be implemented by concrete LLM classes.
        """
        raise NotImplementedError("Must be implemented by subclass")

    async def show_prompt_loaded(
        self,
        prompt_name: str,
        description: Optional[str] = None,
        message_count: int = 0,
        arguments: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Display information about a loaded prompt template.

        Args:
            prompt_name: The name of the prompt
            description: Optional description of the prompt
            message_count: Number of messages in the prompt
            arguments: Optional dictionary of arguments passed to the prompt
        """
        await self.display.show_prompt_loaded(
            prompt_name=prompt_name,
            description=description,
            message_count=message_count,
            agent_name=self.name,
            aggregator=self.aggregator,
            arguments=arguments,
        )

    async def apply_prompt_template(self, prompt_result: GetPromptResult, prompt_name: str) -> str:
        """
        Apply a prompt template by adding it to the conversation history.
        If the last message in the prompt is from a user, automatically
        generate an assistant response.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated,
            or the last assistant message in the prompt
        """
        from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

        # Check if we have any messages
        if not prompt_result.messages:
            return "Prompt contains no messages"

        # Extract arguments if they were stored in the result
        arguments = getattr(prompt_result, "arguments", None)

        # Display information about the loaded prompt
        await self.show_prompt_loaded(
            prompt_name=prompt_name,
            description=prompt_result.description,
            message_count=len(prompt_result.messages),
            arguments=arguments,
        )

        # Convert to PromptMessageMultipart objects
        multipart_messages = PromptMessageMultipart.parse_get_prompt_result(prompt_result)

        # Delegate to the provider-specific implementation
        result = await self._apply_prompt_provider_specific(multipart_messages, None)
        return result.first_text()

    async def _save_history(self, filename: str) -> None:
        """
        Save the Message History to a file in a simple delimeted format.
        """
        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(
            self._message_history,
        )

        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n\n".join(delimited_content))

    @abstractmethod
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Provider-specific implementation of apply_prompt_template.
        This default implementation handles basic text content for any LLM type.
        Provider-specific subclasses should override this method to handle
        multimodal content appropriately.

        Args:
            multipart_messages: List of PromptMessageMultipart objects parsed from the prompt template

        Returns:
            String representation of the assistant's response if generated,
            or the last assistant message in the prompt
        """

    @property
    def message_history(self) -> List[PromptMessageMultipart]:
        """
        Return the agent's message history as PromptMessageMultipart objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageMultipart objects representing the conversation history
        """
        return self._message_history
