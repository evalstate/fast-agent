import asyncio
import inspect
import json
import os
import time
from abc import abstractmethod
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Generic,
    Type,
    TypeVar,
    Union,
    cast,
)

from mcp import Tool
from mcp.types import (
    GetPromptResult,
    PromptMessage,
    TextContent,
)
from openai import NotGiven
from openai.lib._parsing import type_to_response_format_param as _type_to_response_format
from pydantic_core import from_json
from rich import print as rich_print

from fast_agent.constants import (
    CONTROL_MESSAGE_SAVE_HISTORY,
    DEFAULT_MAX_ITERATIONS,
    FAST_AGENT_TIMING,
)
from fast_agent.context_dependent import ContextDependent
from fast_agent.core.exceptions import AgentConfigError, ProviderKeyError, ServerConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import (
    FastAgentLLMProtocol,
    ModelT,
)
from fast_agent.llm.memory import Memory, SimpleMemory
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.usage_tracking import TurnUsage, UsageAccumulator
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types import PromptMessageExtended, RequestParams

# Define type variables locally
MessageParamT = TypeVar("MessageParamT")
MessageT = TypeVar("MessageT")

# Forward reference for type annotations
if TYPE_CHECKING:
    from fast_agent.context import Context


# Context variable for storing MCP metadata
_mcp_metadata_var: ContextVar[dict[str, Any] | None] = ContextVar("mcp_metadata", default=None)


def deep_merge(dict1: dict[Any, Any], dict2: dict[Any, Any]) -> dict[Any, Any]:
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


class FastAgentLLM(ContextDependent, FastAgentLLMProtocol, Generic[MessageParamT, MessageT]):
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
    PARAM_MCP_METADATA = "mcp_metadata"

    # Base set of fields that should always be excluded
    BASE_EXCLUDE_FIELDS = {PARAM_METADATA}

    """
    Implementation of the Llm Protocol - intended be subclassed for Provider
    or behaviour specific reasons. Contains convenience and template methods.
    """

    def __init__(
        self,
        provider: Provider,
        instruction: str | None = None,
        name: str | None = None,
        request_params: RequestParams | None = None,
        context: Union["Context", None] = None,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """

        Args:
            provider: LLM API Provider
            instruction: System prompt for the LLM
            name: Name for the LLM (usually attached Agent name)
            request_params: RequestParams to configure LLM behaviour
            context: Application context
            model: Optional model name override
            **kwargs: Additional provider-specific parameters
        """
        # Extract request_params before super() call
        self._init_request_params = request_params
        super().__init__(context=context, **kwargs)
        self.logger = get_logger(__name__)
        self.executor = self.context.executor
        self.name: str = name or "fast-agent"
        self.instruction = instruction
        self._provider = provider
        # memory contains provider specific API types.
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()

        # Initialize the display component
        from fast_agent.ui.console_display import ConsoleDisplay

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

        # Cache effective model name for type-safe access
        self._model_name: str | None = self.default_request_params.model

        self.verb = kwargs.get("verb")

        self._init_api_key = api_key

        # Initialize usage tracking
        self._usage_accumulator = UsageAccumulator()
        self._stream_listeners: set[Callable[[StreamChunk], None]] = set()
        self._tool_stream_listeners: set[Callable[[str, dict[str, Any] | None], None]] = set()
        self.retry_count = self._resolve_retry_count()
        self.retry_backoff_seconds: float = 10.0

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
            max_iterations=DEFAULT_MAX_ITERATIONS,
            use_history=True,
        )



    async def _execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        on_final_error: Callable[[Exception], Awaitable[Any] | Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Executes a function with robust retry logic for transient API errors.
        """
        retries = max(0, int(self.retry_count))
        
        def _is_fatal_error(e: Exception) -> bool:
            if isinstance(e, (KeyboardInterrupt, AgentConfigError, ServerConfigError)):
                return True
            if isinstance(e, ProviderKeyError):
                msg = str(e).lower()
                # Retry on Rate Limits (429, Quota, Overloaded)
                keywords = ["429", "503", "quota", "exhausted", "overloaded", "unavailable", "timeout"]
                if any(k in msg for k in keywords):
                    return False 
                return True
            return False

        last_error = None
        
        for attempt in range(retries + 1):
            try:
                # Await the async function
                return await func(*args, **kwargs)
            except Exception as e:
                if _is_fatal_error(e):
                    raise e
                
                last_error = e
                if attempt < retries:
                    wait_time = self.retry_backoff_seconds * (attempt + 1)
                    
                    # Try to import progress_display safely
                    try:
                        from fast_agent.ui.progress_display import progress_display
                        with progress_display.paused():
                            rich_print(f"\n[yellow]⚠ Provider Error: {str(e)[:300]}...[/yellow]")
                            rich_print(f"[dim]⟳ Retrying in {wait_time}s... (Attempt {attempt+1}/{retries})[/dim]")
                    except ImportError:
                        print(f"⚠ Provider Error: {str(e)[:300]}...")
                        print(f"⟳ Retrying in {wait_time}s... (Attempt {attempt+1}/{retries})")

                    await asyncio.sleep(wait_time)
        
        if last_error:
            handler = on_final_error or getattr(self, "_handle_retry_failure", None)
            if handler:
                handled = handler(last_error)
                if inspect.isawaitable(handled):
                    handled = await handled
                if handled is not None:
                    return handled

            raise last_error

        # This line satisfies Pylance that we never implicitly return None
        raise RuntimeError("Retry loop finished without success or exception")
        
    def _handle_retry_failure(self, error: Exception) -> Any | None:
        """
        Optional hook for providers to convert an exhausted retry into a user-facing response.

        Return a non-None value to short-circuit raising the final exception.
        """
        return None

    def _resolve_retry_count(self) -> int:
        """Resolve retries from config first, then env, defaulting to 0."""
        config_retries = None
        try:
            config_retries = getattr(self.context.config, "llm_retries", None)
        except Exception:
            config_retries = None

        if config_retries is not None:
            try:
                return int(config_retries)
            except (TypeError, ValueError):
                pass

        env_retries = os.getenv("FAST_AGENT_RETRIES")
        if env_retries is not None:
            try:
                return int(env_retries)
            except (TypeError, ValueError):
                pass

        return 0
        
        
    async def generate(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Generate a completion using normalized message lists.

        This is the primary LLM interface that works directly with
        list[PromptMessageExtended] for efficient internal usage.

        Args:
            messages: List of PromptMessageExtended objects
            request_params: Optional parameters to configure the LLM request
            tools: Optional list of tools available to the LLM

        Returns:
            A PromptMessageExtended containing the Assistant response

        Raises:
            asyncio.CancelledError: If the operation is cancelled via task.cancel()
        """
        # TODO -- create a "fast-agent" control role rather than magic strings

        if messages[-1].first_text().startswith(CONTROL_MESSAGE_SAVE_HISTORY):
            parts: list[str] = messages[-1].first_text().split(" ", 1)
            if len(parts) > 1:
                filename: str = parts[1].strip()
            else:
                from datetime import datetime

                timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")
                filename = f"{timestamp}-conversation.json"
            await self._save_history(filename, messages)
            return Prompt.assistant(f"History saved to {filename}")

        # Store MCP metadata in context variable
        final_request_params = self.get_request_params(request_params)
        if final_request_params.mcp_metadata:
            _mcp_metadata_var.set(final_request_params.mcp_metadata)

        # The caller supplies the full conversation to send
        full_history = messages

        # Track timing for this generation
        start_time = time.perf_counter()
        assistant_response: PromptMessageExtended = await self._execute_with_retry(
            self._apply_prompt_provider_specific,
            full_history, 
            request_params, 
            tools
        )
        end_time = time.perf_counter()
        duration_ms = round((end_time - start_time) * 1000, 2)

        # Add timing data to channels only if not already present
        # (preserves original timing when loading saved history)
        channels = dict(assistant_response.channels or {})
        if FAST_AGENT_TIMING not in channels:
            timing_data = {
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
            }
            channels[FAST_AGENT_TIMING] = [TextContent(type="text", text=json.dumps(timing_data))]
            assistant_response.channels = channels

        self.usage_accumulator.count_tools(len(assistant_response.tool_calls or {}))

        return assistant_response

    @abstractmethod
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list["PromptMessageExtended"],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """
        Provider-specific implementation of apply_prompt_template.
        This default implementation handles basic text content for any LLM type.
        Provider-specific subclasses should override this method to handle
        multimodal content appropriately.

        Args:
            multipart_messages: List of PromptMessageExtended objects parsed from the prompt template
            request_params: Optional parameters to configure the LLM request
            tools: Optional list of tools available to the LLM
            is_template: Whether this is a template application

        Returns:
            String representation of the assistant's response if generated,
            or the last assistant message in the prompt
        """

    async def structured(
        self,
        messages: list[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """
        Generate a structured response using normalized message lists.

        This is the primary LLM interface for structured output that works directly with
        list[PromptMessageExtended] for efficient internal usage.

        Args:
            messages: List of PromptMessageExtended objects
            model: The Pydantic model class to parse the response into
            request_params: Optional parameters to configure the LLM request

        Returns:
            Tuple of (parsed model instance or None, assistant response message)
        """

        # Store MCP metadata in context variable
        final_request_params = self.get_request_params(request_params)

        # TODO -- this doesn't need to go here anymore.
        if final_request_params.mcp_metadata:
            _mcp_metadata_var.set(final_request_params.mcp_metadata)

        full_history = messages

        # Track timing for this structured generation
        start_time = time.perf_counter()
        result_or_response = await self._execute_with_retry(
            self._apply_prompt_provider_specific_structured,
            full_history, 
            model, 
            request_params,
            on_final_error=self._handle_retry_failure,
        )
        if isinstance(result_or_response, PromptMessageExtended):
            result, assistant_response = self._structured_from_multipart(result_or_response, model)
        else:
            result, assistant_response = result_or_response
        end_time = time.perf_counter()
        duration_ms = round((end_time - start_time) * 1000, 2)

        # Add timing data to channels only if not already present
        # (preserves original timing when loading saved history)
        channels = dict(assistant_response.channels or {})
        if FAST_AGENT_TIMING not in channels:
            timing_data = {
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
            }
            channels[FAST_AGENT_TIMING] = [TextContent(type="text", text=json.dumps(timing_data))]
            assistant_response.channels = channels

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
        multipart_messages: list[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """Base class attempts to parse JSON - subclasses can use provider specific functionality"""

        request_params = self.get_request_params(request_params)

        if not request_params.response_format:
            schema = self.model_to_response_format(model)
            if schema is not NotGiven:
                request_params.response_format = schema

        result: PromptMessageExtended = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )
        return self._structured_from_multipart(result, model)

    def _structured_from_multipart(
        self, message: PromptMessageExtended, model: Type[ModelT]
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """Parse the content of a PromptMessage and return the structured model and message itself"""
        try:
            text = get_text(message.content[-1]) or ""
            text = self._prepare_structured_text(text)
            json_data = from_json(text, allow_partial=True)
            validated_model = model.model_validate(json_data)
            return cast("ModelT", validated_model), message
        except ValueError as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to parse structured response: {str(e)}")
            return None, message

    def _prepare_structured_text(self, text: str) -> str:
        """Hook for subclasses to adjust structured output text before parsing."""
        return text

    def record_templates(self, templates: list[PromptMessageExtended]) -> None:
        """Hook for providers that need template visibility (e.g., caching)."""
        return

    def _precall(self, multipart_messages: list[PromptMessageExtended]) -> None:
        """Pre-call hook to modify the message before sending it to the provider."""
        # No-op placeholder; history is managed by the agent

    def chat_turn(self) -> int:
        """Return the current chat turn number"""
        return 1 + len(self._usage_accumulator.turns)

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

        # Combine base exclusions with provider-specific exclusions
        final_exclude_fields = self.BASE_EXCLUDE_FIELDS.copy()
        if exclude_fields:
            final_exclude_fields.update(exclude_fields)

        # Add all fields from params that aren't explicitly excluded
        # Ensure model_dump only includes set fields if that's the desired behavior,
        # or adjust exclude_unset=True/False as needed.
        # Default Pydantic v2 model_dump is exclude_unset=False
        params_dict = request_params.model_dump(exclude=final_exclude_fields)

        for key, value in params_dict.items():
            # Only add if not None and not already in base_args (base_args take precedence)
            # or if None is a valid value for the provider, this logic might need adjustment.
            if value is not None and key not in arguments:
                arguments[key] = value
            elif value is not None and key in arguments and arguments[key] is None:
                # Allow overriding a None in base_args with a set value from params
                arguments[key] = value

        # Finally, add any metadata fields as a last layer of overrides
        # This ensures metadata can override anything previously set if keys conflict.
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

    def _finalize_turn_usage(self, turn_usage: "TurnUsage") -> None:
        """Set tool call count on TurnUsage and add to accumulator."""
        self._usage_accumulator.add_turn(turn_usage)

    def _log_chat_progress(self, chat_turn: int | None = None, model: str | None = None) -> None:
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

    def _update_streaming_progress(self, content: str, model: str, estimated_tokens: int) -> int:
        """Update streaming progress with token estimation and formatting.

        Args:
            content: The text content from the streaming event
            model: The model name
            estimated_tokens: Current token count to update

        Returns:
            Updated estimated token count
        """
        # Rough estimate: 1 token per 4 characters (OpenAI's typical ratio)
        text_length = len(content)
        additional_tokens = max(1, text_length // 4)
        new_total = estimated_tokens + additional_tokens

        # Format token count for display
        token_str = str(new_total).rjust(5)

        # Emit progress event
        data = {
            "progress_action": ProgressAction.STREAMING,
            "model": model,
            "agent_name": self.name,
            "chat_turn": self.chat_turn(),
            "details": token_str.strip(),  # Token count goes in details for STREAMING action
        }
        self.logger.info("Streaming progress", data=data)

        return new_total

    def add_stream_listener(self, listener: Callable[[StreamChunk], None]) -> Callable[[], None]:
        """
        Register a callback invoked with streaming text chunks.

        Args:
            listener: Callable receiving a StreamChunk emitted by the provider.

        Returns:
            A function that removes the listener when called.
        """
        self._stream_listeners.add(listener)

        def remove() -> None:
            self._stream_listeners.discard(listener)

        return remove

    def _notify_stream_listeners(self, chunk: StreamChunk) -> None:
        """Notify registered listeners with a streaming chunk."""
        if not chunk.text:
            return
        for listener in list(self._stream_listeners):
            try:
                listener(chunk)
            except Exception:
                self.logger.exception("Stream listener raised an exception")

    def add_tool_stream_listener(
        self, listener: Callable[[str, dict[str, Any] | None], None]
    ) -> Callable[[], None]:
        """Register a callback invoked with tool streaming events.

        Args:
            listener: Callable receiving event_type (str) and optional info dict.

        Returns:
            A function that removes the listener when called.
        """

        self._tool_stream_listeners.add(listener)

        def remove() -> None:
            self._tool_stream_listeners.discard(listener)

        return remove

    def _notify_tool_stream_listeners(
        self, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        """Notify listeners about tool streaming lifecycle events."""

        data = payload or {}
        for listener in list(self._tool_stream_listeners):
            try:
                listener(event_type, data)
            except Exception:
                self.logger.exception("Tool stream listener raised an exception")

    def _log_chat_finished(self, model: str | None = None) -> None:
        """Log a chat finished event"""
        data = {
            "progress_action": ProgressAction.READY,
            "model": model,
            "agent_name": self.name,
        }
        self.logger.debug("Chat finished", data=data)

    def _convert_prompt_messages(self, prompt_messages: list[PromptMessage]) -> list[MessageParamT]:
        """
        Convert prompt messages to this LLM's specific message format.
        To be implemented by concrete LLM classes.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def _convert_to_provider_format(
        self, messages: list[PromptMessageExtended]
    ) -> list[MessageParamT]:
        """
        Convert provided messages to provider-specific format.
        Called fresh on EVERY API call - no caching.

        Args:
            messages: List of PromptMessageExtended

        Returns:
            List of provider-specific message objects
        """
        return self._convert_extended_messages_to_provider(messages)

    @abstractmethod
    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[MessageParamT]:
        """
        Convert PromptMessageExtended list to provider-specific format.
        Must be implemented by each provider.

        Args:
            messages: List of PromptMessageExtended objects

        Returns:
            List of provider-specific message parameter objects
        """
        raise NotImplementedError("Must be implemented by subclass")

    async def show_prompt_loaded(
        self,
        prompt_name: str,
        description: str | None = None,
        message_count: int = 0,
        arguments: dict[str, str] | None = None,
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
        from fast_agent.types import PromptMessageExtended

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

        # Convert to PromptMessageExtended objects and delegate
        multipart_messages = PromptMessageExtended.parse_get_prompt_result(prompt_result)
        result = await self._apply_prompt_provider_specific(
            multipart_messages, None, is_template=True
        )
        return result.first_text()

    async def _save_history(self, filename: str, messages: list[PromptMessageExtended]) -> None:
        """
        Save the Message History to a file in a format determined by the file extension.

        Uses JSON format for .json files (MCP SDK compatible format) and
        delimited text format for other extensions.
        """
        from fast_agent.mcp.prompt_serialization import save_messages

        # Drop control messages like ***SAVE_HISTORY before persisting
        filtered = [
            msg.model_copy(deep=True)
            for msg in messages
            if not msg.first_text().startswith(CONTROL_MESSAGE_SAVE_HISTORY)
        ]

        # Save messages using the unified save function that auto-detects format
        save_messages(filtered, filename)

    @property
    def message_history(self) -> list[PromptMessageExtended]:
        """
        Return the agent's message history as PromptMessageExtended objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageExtended objects representing the conversation history
        """
        return []

    def pop_last_message(self) -> PromptMessageExtended | None:
        """Remove and return the most recent message from the conversation history."""
        return None

    def clear(self, *, clear_prompts: bool = False) -> None:
        """Reset stored message history while optionally retaining prompt templates."""

        self.history.clear(clear_prompts=clear_prompts)

    def _api_key(self):
        if self._init_api_key:
            return self._init_api_key

        from fast_agent.llm.provider_key_manager import ProviderKeyManager

        assert self.provider
        return ProviderKeyManager.get_api_key(self.provider.value, self.context.config)

    @property
    def usage_accumulator(self):
        return self._usage_accumulator

    def get_usage_summary(self) -> dict:
        """
        Get a summary of usage statistics for this LLM instance.

        Returns:
            Dictionary containing usage statistics including tokens, cache metrics,
            and context window utilization.
        """
        return self._usage_accumulator.get_summary()

    @property
    def provider(self) -> Provider:
        """
        Return the LLM provider type.

        Returns:
            The Provider enum value representing the LLM provider
        """
        return self._provider

    @property
    def model_name(self) -> str | None:
        """Return the effective model name, if set."""
        return self._model_name

    @property
    def model_info(self):
        """Return resolved model information with capabilities.

        Uses a lightweight resolver backed by the ModelDatabase and provides
        text/document/vision flags, context window, etc.
        """
        from fast_agent.llm.model_info import ModelInfo

        if not self._model_name:
            return None
        return ModelInfo.from_name(self._model_name, self._provider)
