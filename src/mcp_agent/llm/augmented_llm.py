from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
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
from openai import NotGiven
from openai.lib._parsing import type_to_response_format_param as _type_to_response_format
from pydantic_core import from_json
from rich.text import Text

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.core.exceptions import PromptExitError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.event_progress import ProgressAction
from mcp_agent.llm.memory import Memory, SimpleMemory
from mcp_agent.llm.model_database import ModelDatabase
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.sampling_format_converter import (
    BasicFormatConverter,
    ProviderFormatConverter,
)
from mcp_agent.llm.usage_tracking import UsageAccumulator
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import get_text
from mcp_agent.mcp.interfaces import (
    AugmentedLLMProtocol,
    ModelT,
)
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompt_render import render_multipart_message
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


def deep_merge(dict1: Dict[Any, Any], dict2: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively merges `dict2` into `dict1` in place.

    If a key exists in both dictionaries and their values are dictionaries,
    the function merges them recursively. Otherwise, the value from `dict2`
    overwrites or is added to `dict1`.

    Args:
        dict1 (Dict): The dictionary to be updated.
        dict2 (Dict): The dictionary to merge into `dict1`.

    Returns:
        Dict: The updated `dict1`.
    """
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            deep_merge(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1


class AugmentedLLM(ContextDependent, AugmentedLLMProtocol, Generic[MessageParamT, MessageT]):
    # Common parameter names used across providers
    PARAM_MESSAGES = "messages"
    PARAM_MODEL = "model"
    PARAM_MAX_TOKENS = "maxTokens"
    PARAM_SYSTEM_PROMPT = "systemPrompt"
    PARAM_STOP_SEQUENCES = "stopSequences"
    PARAM_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    PARAM_METADATA = "metadata"
    PARAM_USE_HISTORY = "use_history"
    PARAM_MAX_ITERATIONS = "max_iterations"
    PARAM_TEMPLATE_VARS = "template_vars"
    # Base set of fields that should always be excluded
    BASE_EXCLUDE_FIELDS = {PARAM_METADATA}

    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    provider: Provider | None = None

    def __init__(
        self,
        provider: Provider,
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
        self.provider = provider
        # memory contains provider specific API types.
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()

        self._message_history: List[PromptMessageMultipart] = []

        # Initialize the display component
        self.display = ConsoleDisplay(config=self.context.config)

        # Initialize default parameters, passing model info
        model_kwargs = kwargs.copy()
        if model:
            model_kwargs["model"] = model
        self.default_request_params = self._initialize_default_params(model_kwargs)

        # Merge with provided params if any
        if self._init_request_params:
            self.default_request_params = self._merge_request_params(
                self.default_request_params, self._init_request_params
            )

        self.type_converter = type_converter
        self.verb = kwargs.get("verb")

        # Initialize usage tracking
        self.usage_accumulator = UsageAccumulator()

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize default parameters for the LLM.
        Should be overridden by provider implementations to set provider-specific defaults."""
        # Get model-aware default max tokens
        model = kwargs.get("model")
        max_tokens = ModelDatabase.get_default_max_tokens(model)

        return RequestParams(
            model=model,
            maxTokens=max_tokens,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=20,
            use_history=True,
        )

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Create a completion with the LLM using the provided messages.
        """
        # note - check changes here are mirrored in structured(). i've thought hard about
        # a strategy to reduce duplication etc, but aiming for simple but imperfect for the moment

        # We never expect this for structured() calls - this is for interactive use - developers
        # can do this programatically
        # TODO -- create a "fast-agent" control role rather than magic strings

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

        self._precall(multipart_messages)

        assistant_response: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )

        # add generic error and termination reason handling/rollback
        self._message_history.append(assistant_response)
        return assistant_response

    @abstractmethod
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
        is_template: bool = False,
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

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Return a structured response from the LLM using the provided messages."""

        self._precall(multipart_messages)
        result, assistant_response = await self._apply_prompt_provider_specific_structured(
            multipart_messages, model, request_params
        )

        self._message_history.append(assistant_response)
        return result, assistant_response

    @staticmethod
    def model_to_response_format(
        model: Type[Any],
    ) -> Any:
        """
        Convert a pydantic model to the appropriate response format schema.
        This allows for reuse in multiple provider implementations.

        Args:
            model: The pydantic model class to convert to a schema

        Returns:
            Provider-agnostic schema representation or NotGiven if conversion fails
        """
        return _type_to_response_format(model)

    @staticmethod
    def model_to_schema_str(
        model: Type[Any],
    ) -> str:
        """
        Convert a pydantic model to a schema string representation.
        This provides a simpler interface for provider implementations
        that need a string representation.

        Args:
            model: The pydantic model class to convert to a schema

        Returns:
            Schema as a string, or empty string if conversion fails
        """
        import json

        try:
            schema = model.model_json_schema()
            return json.dumps(schema)
        except Exception:
            return ""

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Base class attempts to parse JSON - subclasses can use provider specific functionality"""

        request_params = self.get_request_params(request_params)

        if not request_params.response_format:
            schema = self.model_to_response_format(model)
            if schema is not NotGiven:
                request_params.response_format = schema

        result: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )
        return self._structured_from_multipart(result, model)

    def _structured_from_multipart(
        self, message: PromptMessageMultipart, model: Type[ModelT]
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Parse the content of a PromptMessage and return the structured model and message itself"""
        try:
            text = get_text(message.content[-1]) or ""
            json_data = from_json(text, allow_partial=True)
            validated_model = model.model_validate(json_data)
            return cast("ModelT", validated_model), message
        except ValueError as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to parse structured response: {str(e)}")
            return None, message

    def _precall(self, multipart_messages: List[PromptMessageMultipart]) -> None:
        """Pre-call hook to modify the message before sending it to the provider."""
        self._message_history.extend(multipart_messages)
        if multipart_messages[-1].role == "user":
            self.show_user_message(
                render_multipart_message(multipart_messages[-1]),
                model=self.default_request_params.model,
                chat_turn=self.chat_turn(),
            )

    def chat_turn(self) -> int:
        """Return the current chat turn number"""
        return 1 + sum(1 for message in self._message_history if message.role == "assistant")

    def prepare_provider_arguments(
        self,
        base_args: dict,
        request_params: RequestParams,
        exclude_fields: set | None = None,
    ) -> dict:
        """
        Prepare arguments for provider API calls by merging request parameters.

        Args:
            base_args: Base arguments dictionary with provider-specific required parameters
            params: The RequestParams object containing all parameters
            exclude_fields: Set of field names to exclude from params. If None, uses BASE_EXCLUDE_FIELDS.

        Returns:
            Complete arguments dictionary with all applicable parameters
        """
        # Start with base arguments
        arguments = base_args.copy()

        # Use provided exclude_fields or fall back to base exclusions
        exclude_fields = exclude_fields or self.BASE_EXCLUDE_FIELDS.copy()

        # Add all fields from params that aren't explicitly excluded
        params_dict = request_params.model_dump(exclude=exclude_fields)
        for key, value in params_dict.items():
            if value is not None and key not in arguments:
                arguments[key] = value

        # Finally, add any metadata fields as a last layer of overrides
        if request_params.metadata:
            arguments.update(request_params.metadata)

        return arguments

    def _merge_request_params(
        self, default_params: RequestParams, provided_params: RequestParams
    ) -> RequestParams:
        """Merge default and provided request parameters"""

        merged = deep_merge(
            default_params.model_dump(),
            provided_params.model_dump(exclude_unset=True),
        )
        final_params = RequestParams(**merged)

        return final_params

    def get_request_params(
        self,
        request_params: RequestParams | None = None,
    ) -> RequestParams:
        """
        Get request parameters with merged-in defaults and overrides.
        Args:
            request_params: The request parameters to use as overrides.
            default: The default request parameters to use as the base.
                If unspecified, self.default_request_params will be used.
        """

        # If user provides overrides, merge them with defaults
        if request_params:
            return self._merge_request_params(self.default_request_params, request_params)

        return self.default_request_params.model_copy()

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
        result = await self._apply_prompt_provider_specific(
            multipart_messages, None, is_template=True
        )
        return result.first_text()

    async def _save_history(self, filename: str) -> None:
        """
        Save the Message History to a file in a format determined by the file extension.

        Uses JSON format for .json files (MCP SDK compatible format) and
        delimited text format for other extensions.
        """
        from mcp_agent.mcp.prompt_serialization import save_messages_to_file

        # Save messages using the unified save function that auto-detects format
        save_messages_to_file(self._message_history, filename)

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

    def _api_key(self):
        from mcp_agent.llm.provider_key_manager import ProviderKeyManager

        assert self.provider
        return ProviderKeyManager.get_api_key(self.provider.value, self.context.config)

    def get_usage_summary(self) -> dict:
        """
        Get a summary of usage statistics for this LLM instance.

        Returns:
            Dictionary containing usage statistics including tokens, cache metrics,
            and context window utilization.
        """
        return self.usage_accumulator.get_summary()
