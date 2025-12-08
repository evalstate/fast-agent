import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp import Tool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)
from openai import APIError, AsyncOpenAI, AuthenticationError, DefaultAioHttpClient
from openai.lib.streaming.chat import ChatCompletionStreamState

# from openai.types.beta.chat import
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from pydantic_core import from_json

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL, REASONING
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.fastagent_llm import FastAgentLLM, RequestParams
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.multipart_converter_openai import OpenAIConverter, OpenAIMessage
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.mcp.helpers.content_helpers import get_text, text_content
from fast_agent.types import LlmStopReason, PromptMessageExtended

_logger = get_logger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "low"

# Stream capture mode - when enabled, saves all streaming chunks to files for debugging
# Set FAST_AGENT_LLM_TRACE=1 (or any non-empty value) to enable
STREAM_CAPTURE_ENABLED = bool(os.environ.get("FAST_AGENT_LLM_TRACE"))
STREAM_CAPTURE_DIR = Path("stream-debug")


def _stream_capture_filename(turn: int) -> Path | None:
    """Generate filename for stream capture. Returns None if capture is disabled."""
    if not STREAM_CAPTURE_ENABLED:
        return None
    STREAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STREAM_CAPTURE_DIR / f"{timestamp}_turn{turn}"


def _save_stream_request(filename_base: Path | None, arguments: dict[str, Any]) -> None:
    """Save the request arguments to a _request.json file."""
    if not filename_base:
        return
    try:
        request_file = filename_base.with_name(f"{filename_base.name}_request.json")
        with open(request_file, "w") as f:
            json.dump(arguments, f, indent=2, default=str)
    except Exception as e:
        _logger.debug(f"Failed to save stream request: {e}")


def _save_stream_chunk(filename_base: Path | None, chunk: Any) -> None:
    """Save a streaming chunk to file when capture mode is enabled."""
    if not filename_base:
        return
    try:
        chunk_file = filename_base.with_name(f"{filename_base.name}.jsonl")
        try:
            payload: Any = chunk.model_dump()
        except Exception:
            payload = str(chunk)

        with open(chunk_file, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        _logger.debug(f"Failed to save stream chunk: {e}")


class OpenAILLM(FastAgentLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    # Config section name override (falls back to provider value)
    config_section: str | None = None
    # OpenAI-specific parameter exclusions
    OPENAI_EXCLUDE_FIELDS = {
        FastAgentLLM.PARAM_MESSAGES,
        FastAgentLLM.PARAM_MODEL,
        FastAgentLLM.PARAM_MAX_TOKENS,
        FastAgentLLM.PARAM_SYSTEM_PROMPT,
        FastAgentLLM.PARAM_PARALLEL_TOOL_CALLS,
        FastAgentLLM.PARAM_USE_HISTORY,
        FastAgentLLM.PARAM_MAX_ITERATIONS,
        FastAgentLLM.PARAM_TEMPLATE_VARS,
        FastAgentLLM.PARAM_MCP_METADATA,
        FastAgentLLM.PARAM_STOP_SEQUENCES,
    }

    def __init__(self, provider: Provider = Provider.OPENAI, *args, **kwargs) -> None:
        super().__init__(*args, provider=provider, **kwargs)

        # Initialize logger with name if available
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        # Set up reasoning-related attributes
        self._reasoning_effort = kwargs.get("reasoning_effort", None)
        if self.context and self.context.config and self.context.config.openai:
            if self._reasoning_effort is None and hasattr(
                self.context.config.openai, "reasoning_effort"
            ):
                self._reasoning_effort = self.context.config.openai.reasoning_effort

        # Determine reasoning mode for the selected model
        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning_mode = ModelDatabase.get_reasoning(chosen_model)
        self._reasoning = self._reasoning_mode == "openai"
        if self._reasoning_mode:
            self.logger.info(
                f"Using reasoning model '{chosen_model}' (mode='{self._reasoning_mode}') with "
                f"'{self._reasoning_effort}' reasoning effort"
            )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenAI-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with OpenAI-specific settings
        chosen_model = kwargs.get("model", DEFAULT_OPENAI_MODEL)
        base_params.model = chosen_model

        return base_params

    def _base_url(self) -> str:
        return self.context.config.openai.base_url if self.context.config.openai else None

    def _default_headers(self) -> dict[str, str] | None:
        """
        Get custom headers from configuration.
        Subclasses can override this to provide provider-specific headers.
        """
        provider_config = self._get_provider_config()
        return getattr(provider_config, "default_headers", None) if provider_config else None

    def _get_provider_config(self):
        """Return the config section for this provider, if available."""
        context_config = getattr(self.context, "config", None)
        if not context_config:
            return None
        section_name = self.config_section or getattr(self.provider, "value", None)
        if not section_name:
            return None
        return getattr(context_config, section_name, None)

    def _openai_client(self) -> AsyncOpenAI:
        """
        Create an OpenAI client instance.
        Subclasses can override this to provide different client types (e.g., AzureOpenAI).

        Note: The returned client should be used within an async context manager
        to ensure proper cleanup of aiohttp sessions.
        """
        try:
            kwargs: dict[str, Any] = {
                "api_key": self._api_key(),
                "base_url": self._base_url(),
                "http_client": DefaultAioHttpClient(),
            }

            # Add custom headers if configured
            default_headers = self._default_headers()
            if default_headers:
                kwargs["default_headers"] = default_headers

            return AsyncOpenAI(**kwargs)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

    def _streams_tool_arguments(self) -> bool:
        """
        Determine whether the current provider streams tool call arguments incrementally.

        Official OpenAI and Azure OpenAI endpoints stream arguments. Most third-party
        OpenAI-compatible gateways (e.g. OpenRouter, Moonshot) deliver the full arguments
        once, so we should treat them as non-streaming to restore the legacy \"Calling Tool\"
        display experience.
        """
        if self.provider in (Provider.AZURE, Provider.HUGGINGFACE):
            return True

        if self.provider == Provider.OPENAI:
            base_url = self._base_url()
            if not base_url:
                return True
            lowered = base_url.lower()
            return "api.openai" in lowered or "openai.azure" in lowered or "azure.com" in lowered

        return False

    def _emit_tool_notification_fallback(
        self,
        tool_calls: Any,
        notified_indices: set[int],
        *,
        streams_arguments: bool,
        model: str,
    ) -> None:
        """Emit start/stop notifications when streaming metadata was missing."""
        if not tool_calls:
            return

        for index, tool_call in enumerate(tool_calls):
            if index in notified_indices:
                continue

            tool_name = None
            tool_use_id = None

            try:
                tool_use_id = getattr(tool_call, "id", None)
                function = getattr(tool_call, "function", None)
                if function:
                    tool_name = getattr(function, "name", None)
            except Exception:
                tool_use_id = None
                tool_name = None

            if not tool_name:
                tool_name = "tool"
            if not tool_use_id:
                tool_use_id = f"tool-{index}"

            payload = {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "index": index,
                "streams_arguments": streams_arguments,
            }

            self._notify_tool_stream_listeners("start", payload)
            self.logger.info(
                "Model emitted fallback tool notification",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": self.name,
                    "model": model,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_event": "start",
                    "streams_arguments": streams_arguments,
                    "fallback": True,
                },
            )
            self._notify_tool_stream_listeners("stop", payload)
            self.logger.info(
                "Model emitted fallback tool notification",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": self.name,
                    "model": model,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_event": "stop",
                    "streams_arguments": streams_arguments,
                    "fallback": True,
                },
            )

    def _handle_reasoning_delta(
        self,
        *,
        reasoning_mode: str | None,
        reasoning_text: str,
        reasoning_active: bool,
        reasoning_segments: list[str],
    ) -> bool:
        """Stream reasoning text and track whether a thinking block is open."""
        if not reasoning_text:
            return reasoning_active

        if reasoning_mode == "tags":
            if not reasoning_active:
                reasoning_active = True
            self._notify_stream_listeners(StreamChunk(text=reasoning_text, is_reasoning=True))
            reasoning_segments.append(reasoning_text)
            return reasoning_active

        if reasoning_mode in {"stream", "reasoning_content", "gpt_oss"}:
            # Emit reasoning as-is
            self._notify_stream_listeners(StreamChunk(text=reasoning_text, is_reasoning=True))
            reasoning_segments.append(reasoning_text)
            return reasoning_active

        return reasoning_active

    def _handle_tool_delta(
        self,
        *,
        delta_tool_calls: Any,
        tool_call_started: dict[int, dict[str, Any]],
        streams_arguments: bool,
        model: str,
        notified_tool_indices: set[int],
    ) -> None:
        """Emit tool call start/delta events and keep state in sync."""
        for tool_call in delta_tool_calls:
            index = tool_call.index
            if index is None:
                continue

            existing_info = tool_call_started.get(index)
            tool_use_id = tool_call.id or (
                existing_info.get("tool_use_id") if existing_info else None
            )
            function_name = (
                tool_call.function.name
                if tool_call.function and tool_call.function.name
                else (existing_info.get("tool_name") if existing_info else None)
            )

            if existing_info is None and tool_use_id and function_name:
                tool_call_started[index] = {
                    "tool_name": function_name,
                    "tool_use_id": tool_use_id,
                    "streams_arguments": streams_arguments,
                }
                self._notify_tool_stream_listeners(
                    "start",
                    {
                        "tool_name": function_name,
                        "tool_use_id": tool_use_id,
                        "index": index,
                        "streams_arguments": streams_arguments,
                    },
                )
                self.logger.info(
                    "Model started streaming tool call",
                    data={
                        "progress_action": ProgressAction.CALLING_TOOL,
                        "agent_name": self.name,
                        "model": model,
                        "tool_name": function_name,
                        "tool_use_id": tool_use_id,
                        "tool_event": "start",
                        "streams_arguments": streams_arguments,
                    },
                )
                notified_tool_indices.add(index)
            elif existing_info:
                if tool_use_id:
                    existing_info["tool_use_id"] = tool_use_id
                if function_name:
                    existing_info["tool_name"] = function_name

            if tool_call.function and tool_call.function.arguments:
                info = tool_call_started.setdefault(
                    index,
                    {
                        "tool_name": function_name,
                        "tool_use_id": tool_use_id,
                        "streams_arguments": streams_arguments,
                    },
                )
                self._notify_tool_stream_listeners(
                    "delta",
                    {
                        "tool_name": info.get("tool_name"),
                        "tool_use_id": info.get("tool_use_id"),
                        "index": index,
                        "chunk": tool_call.function.arguments,
                        "streams_arguments": info.get("streams_arguments", False),
                    },
                )

    def _finalize_tool_calls_on_stop(
        self,
        *,
        tool_call_started: dict[int, dict[str, Any]],
        streams_arguments: bool,
        model: str,
        notified_tool_indices: set[int],
    ) -> None:
        """Emit stop events for any in-flight tool calls and clear state."""
        for index, info in list(tool_call_started.items()):
            self._notify_tool_stream_listeners(
                "stop",
                {
                    "tool_name": info.get("tool_name"),
                    "tool_use_id": info.get("tool_use_id"),
                    "index": index,
                    "streams_arguments": info.get("streams_arguments", False),
                },
            )
            self.logger.info(
                "Model finished streaming tool call",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": self.name,
                    "model": model,
                    "tool_name": info.get("tool_name"),
                    "tool_use_id": info.get("tool_use_id"),
                    "tool_event": "stop",
                    "streams_arguments": info.get("streams_arguments", False),
                },
            )
            notified_tool_indices.add(index)
        tool_call_started.clear()

    def _emit_text_delta(
        self,
        *,
        content: str,
        model: str,
        estimated_tokens: int,
        streams_arguments: bool,
        reasoning_active: bool,
    ) -> tuple[int, bool]:
        """Emit text deltas and close any active reasoning block."""
        if reasoning_active:
            reasoning_active = False

        self._notify_stream_listeners(StreamChunk(text=content, is_reasoning=False))
        estimated_tokens = self._update_streaming_progress(content, model, estimated_tokens)
        self._notify_tool_stream_listeners(
            "text",
            {
                "chunk": content,
                "streams_arguments": streams_arguments,
            },
        )

        return estimated_tokens, reasoning_active

    def _close_reasoning_if_active(self, reasoning_active: bool) -> bool:
        """Return reasoning state; kept for symmetry."""
        return False if reasoning_active else reasoning_active

    async def _process_stream(
        self,
        stream,
        model: str,
        capture_filename: Path | None = None,
    ) -> tuple[Any, list[str]]:
        """Process the streaming response and display real-time token usage."""
        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0
        reasoning_active = False
        reasoning_segments: list[str] = []
        reasoning_mode = ModelDatabase.get_reasoning(model)

        # For providers/models that emit non-OpenAI deltas, fall back to manual accumulation
        stream_mode = ModelDatabase.get_stream_mode(model)
        provider_requires_manual = self.provider in [
            Provider.GENERIC,
            Provider.OPENROUTER,
            Provider.GOOGLE_OAI,
        ]
        if stream_mode == "manual" or provider_requires_manual:
            return await self._process_stream_manual(stream, model, capture_filename)

        # Use ChatCompletionStreamState helper for accumulation (OpenAI only)
        state = ChatCompletionStreamState()

        # Track tool call state for stream events
        tool_call_started: dict[int, dict[str, Any]] = {}
        streams_arguments = self._streams_tool_arguments()
        notified_tool_indices: set[int] = set()

        # Process the stream chunks
        # Cancellation is handled via asyncio.Task.cancel() which raises CancelledError
        async for chunk in stream:
            # Save chunk if stream capture is enabled
            _save_stream_chunk(capture_filename, chunk)
            # Handle chunk accumulation
            state.handle_chunk(chunk)
            # Process streaming events for tool calls
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                reasoning_text = self._extract_reasoning_text(
                    reasoning=getattr(delta, "reasoning", None),
                    reasoning_content=getattr(delta, "reasoning_content", None),
                )
                reasoning_active = self._handle_reasoning_delta(
                    reasoning_mode=reasoning_mode,
                    reasoning_text=reasoning_text,
                    reasoning_active=reasoning_active,
                    reasoning_segments=reasoning_segments,
                )

                # Handle tool call streaming
                if delta.tool_calls:
                    self._handle_tool_delta(
                        delta_tool_calls=delta.tool_calls,
                        tool_call_started=tool_call_started,
                        streams_arguments=streams_arguments,
                        model=model,
                        notified_tool_indices=notified_tool_indices,
                    )

                # Handle text content streaming
                if delta.content:
                    estimated_tokens, reasoning_active = self._emit_text_delta(
                        content=delta.content,
                        model=model,
                        estimated_tokens=estimated_tokens,
                        streams_arguments=streams_arguments,
                        reasoning_active=reasoning_active,
                    )

                # Fire "stop" event when tool calls complete
                if choice.finish_reason == "tool_calls":
                    self._finalize_tool_calls_on_stop(
                        tool_call_started=tool_call_started,
                        streams_arguments=streams_arguments,
                        model=model,
                        notified_tool_indices=notified_tool_indices,
                    )

        # Check if we hit the length limit to avoid LengthFinishReasonError
        current_snapshot = state.current_completion_snapshot
        if current_snapshot.choices and current_snapshot.choices[0].finish_reason == "length":
            # Return the current snapshot directly to avoid exception
            final_completion = current_snapshot
        else:
            # Get the final completion with usage data (may include structured output parsing)
            final_completion = state.get_final_completion()

        reasoning_active = self._close_reasoning_if_active(reasoning_active)

        # Log final usage information
        if hasattr(final_completion, "usage") and final_completion.usage:
            actual_tokens = final_completion.usage.completion_tokens
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

            self.logger.info(
                f"Streaming complete - Model: {model}, Input tokens: {final_completion.usage.prompt_tokens}, Output tokens: {final_completion.usage.completion_tokens}"
            )

        final_message = None
        if hasattr(final_completion, "choices") and final_completion.choices:
            final_message = getattr(final_completion.choices[0], "message", None)
        tool_calls = getattr(final_message, "tool_calls", None) if final_message else None
        self._emit_tool_notification_fallback(
            tool_calls,
            notified_tool_indices,
            streams_arguments=streams_arguments,
            model=model,
        )

        return final_completion, reasoning_segments

    def _normalize_role(self, role: str | None) -> str:
        """Ensure the role string matches MCP expectations."""
        default_role = "assistant"
        if not role:
            return default_role

        lowered = role.lower()
        allowed_roles = {"assistant", "user", "system", "tool"}
        if lowered in allowed_roles:
            return lowered

        for candidate in allowed_roles:
            if len(lowered) % len(candidate) == 0:
                repetitions = len(lowered) // len(candidate)
                if candidate * repetitions == lowered:
                    self.logger.info(
                        "Collapsing repeated role value from provider",
                        data={
                            "original_role": role,
                            "normalized_role": candidate,
                        },
                    )
                    return candidate

        self.logger.warning(
            "Model emitted unsupported role; defaulting to assistant",
            data={"original_role": role},
        )
        return default_role

    # TODO - as per other comment this needs to go in another class. There are a number of "special" cases dealt with
    # here to deal with OpenRouter idiosyncrasies between e.g. Anthropic and Gemini models.
    async def _process_stream_manual(
        self,
        stream,
        model: str,
        capture_filename: Path | None = None,
    ) -> tuple[Any, list[str]]:
        """Manual stream processing for providers like Ollama that may not work with ChatCompletionStreamState."""

        from openai.types.chat import ChatCompletionMessageToolCall

        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0
        reasoning_active = False
        reasoning_segments: list[str] = []
        reasoning_mode = ModelDatabase.get_reasoning(model)

        # Manual accumulation of response data
        accumulated_content = ""
        role = "assistant"
        tool_calls_map = {}  # Use a map to accumulate tool calls by index
        function_call = None
        finish_reason = None
        usage_data = None

        # Track tool call state for stream events
        tool_call_started: dict[int, dict[str, Any]] = {}
        streams_arguments = self._streams_tool_arguments()
        notified_tool_indices: set[int] = set()

        # Process the stream chunks manually
        # Cancellation is handled via asyncio.Task.cancel() which raises CancelledError
        async for chunk in stream:
            # Save chunk if stream capture is enabled
            _save_stream_chunk(capture_filename, chunk)
            # Process streaming events for tool calls
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                reasoning_text = self._extract_reasoning_text(
                    reasoning=getattr(delta, "reasoning", None),
                    reasoning_content=getattr(delta, "reasoning_content", None),
                )
                reasoning_active = self._handle_reasoning_delta(
                    reasoning_mode=reasoning_mode,
                    reasoning_text=reasoning_text,
                    reasoning_active=reasoning_active,
                    reasoning_segments=reasoning_segments,
                )

                # Handle tool call streaming
                if delta.tool_calls:
                    self._handle_tool_delta(
                        delta_tool_calls=delta.tool_calls,
                        tool_call_started=tool_call_started,
                        streams_arguments=streams_arguments,
                        model=model,
                        notified_tool_indices=notified_tool_indices,
                    )

                # Handle text content streaming
                if delta.content:
                    estimated_tokens, reasoning_active = self._emit_text_delta(
                        content=delta.content,
                        model=model,
                        estimated_tokens=estimated_tokens,
                        streams_arguments=streams_arguments,
                        reasoning_active=reasoning_active,
                    )
                    accumulated_content += delta.content

                # Fire "stop" event when tool calls complete
                if choice.finish_reason == "tool_calls":
                    self._finalize_tool_calls_on_stop(
                        tool_call_started=tool_call_started,
                        streams_arguments=streams_arguments,
                        model=model,
                        notified_tool_indices=notified_tool_indices,
                    )

            # Extract other fields from the chunk
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.role:
                    role = choice.delta.role
                if choice.delta.tool_calls:
                    # Accumulate tool call deltas
                    for delta_tool_call in choice.delta.tool_calls:
                        if delta_tool_call.index is not None:
                            if delta_tool_call.index not in tool_calls_map:
                                tool_calls_map[delta_tool_call.index] = {
                                    "id": delta_tool_call.id,
                                    "type": delta_tool_call.type or "function",
                                    "function": {
                                        "name": delta_tool_call.function.name
                                        if delta_tool_call.function
                                        else None,
                                        "arguments": "",
                                    },
                                }

                            # Always update if we have new data (needed for OpenRouter Gemini)
                            if delta_tool_call.id:
                                tool_calls_map[delta_tool_call.index]["id"] = delta_tool_call.id
                            if delta_tool_call.function:
                                if delta_tool_call.function.name:
                                    tool_calls_map[delta_tool_call.index]["function"]["name"] = (
                                        delta_tool_call.function.name
                                    )
                                # Handle arguments - they might come as None, empty string, or actual content
                                if delta_tool_call.function.arguments is not None:
                                    tool_calls_map[delta_tool_call.index]["function"][
                                        "arguments"
                                    ] += delta_tool_call.function.arguments

                if choice.delta.function_call:
                    function_call = choice.delta.function_call
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            # Extract usage data if available
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage

        # Convert accumulated tool calls to proper format.
        tool_calls = None
        if tool_calls_map:
            tool_calls = []
            for idx in sorted(tool_calls_map.keys()):
                tool_call_data = tool_calls_map[idx]
                # Only add tool calls that have valid data
                if tool_call_data["id"] and tool_call_data["function"]["name"]:
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tool_call_data["id"],
                            type=tool_call_data["type"],
                            function={
                                "name": tool_call_data["function"]["name"],
                                "arguments": tool_call_data["function"]["arguments"],
                            },
                        )
                    )

        # Create a ChatCompletionMessage manually
        message = ChatCompletionMessage(
            content=accumulated_content,
            role=role,
            tool_calls=tool_calls if tool_calls else None,
            function_call=function_call,
            refusal=None,
            annotations=None,
            audio=None,
        )

        reasoning_active = False

        from types import SimpleNamespace

        final_completion = SimpleNamespace()
        final_completion.choices = [SimpleNamespace()]
        final_completion.choices[0].message = message
        final_completion.choices[0].finish_reason = finish_reason
        final_completion.usage = usage_data

        # Log final usage information
        if usage_data:
            actual_tokens = getattr(usage_data, "completion_tokens", estimated_tokens)
            token_str = str(actual_tokens).rjust(5)
            data = {
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            }
            self.logger.info("Streaming progress", data=data)

            self.logger.info(
                f"Streaming complete - Model: {model}, Input tokens: {getattr(usage_data, 'prompt_tokens', 0)}, Output tokens: {actual_tokens}"
            )

        final_message = final_completion.choices[0].message if final_completion.choices else None
        tool_calls = getattr(final_message, "tool_calls", None) if final_message else None
        self._emit_tool_notification_fallback(
            tool_calls,
            notified_tool_indices,
            streams_arguments=streams_arguments,
            model=model,
        )

        return final_completion, reasoning_segments

    async def _openai_completion(
        self,
        message: list[OpenAIMessage] | None,
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """

        request_params = self.get_request_params(request_params=request_params)

        response_content_blocks: list[ContentBlock] = []
        model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL

        # TODO -- move this in to agent context management / agent group handling
        messages: list[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        # The caller supplies the full history; convert it directly
        if message:
            messages.extend(message)

        available_tools: list[ChatCompletionToolParam] | None = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description if tool.description else "",
                    "parameters": self.adjust_schema(tool.inputSchema),
                },
            }
            for tool in tools or []
        ]

        if not available_tools:
            if self.provider in [Provider.DEEPSEEK, Provider.ALIYUN]:
                available_tools = None  # deepseek/aliyun does not allow empty array
            else:
                available_tools = []

        # we do NOT send "stop sequences" as this causes errors with mutlimodal processing
        arguments: dict[str, Any] = self._prepare_api_request(
            messages, available_tools, request_params
        )
        if not self._reasoning and request_params.stopSequences:
            arguments["stop"] = request_params.stopSequences

        self.logger.debug(f"OpenAI completion requested for: {arguments}")

        self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)
        model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL

        # Generate stream capture filename once (before streaming starts)
        capture_filename = _stream_capture_filename(self.chat_turn())
        _save_stream_request(capture_filename, arguments)

        # Use basic streaming API with context manager to properly close aiohttp session
        try:
            async with self._openai_client() as client:
                stream = await client.chat.completions.create(**arguments)
                # Process the stream
                response, streamed_reasoning = await self._process_stream(
                    stream, model_name, capture_filename
                )
        except asyncio.CancelledError as e:
            reason = str(e) if e.args else "cancelled"
            self.logger.info(f"OpenAI completion cancelled: {reason}")
            # Return a response indicating cancellation
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )
        except APIError as error:
            self.logger.error("APIError during OpenAI completion", exc_info=error)
            raise error
        except Exception:
            streamed_reasoning = []
            raise
        # Track usage if response is valid and has usage data
        if (
            hasattr(response, "usage")
            and response.usage
            and not isinstance(response, BaseException)
        ):
            try:
                turn_usage = TurnUsage.from_openai(response.usage, model_name)
                self._finalize_turn_usage(turn_usage)
            except Exception as e:
                self.logger.warning(f"Failed to track usage: {e}")

        self.logger.debug(
            "OpenAI completion response:",
            data=response,
        )

        if isinstance(response, AuthenticationError):
            raise ProviderKeyError(
                "Rejected OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from response
        elif isinstance(response, BaseException):
            self.logger.error(f"Error: {response}")

        choice = response.choices[0]
        message = choice.message
        normalized_role = self._normalize_role(getattr(message, "role", None))
        # prep for image/audio gen models
        if message.content:
            response_content_blocks.append(TextContent(type="text", text=message.content))

        # ParsedChatCompletionMessage is compatible with ChatCompletionMessage
        # since it inherits from it, so we can use it directly
        # Convert to dict and remove None values
        message_dict = message.model_dump()
        message_dict = {k: v for k, v in message_dict.items() if v is not None}
        if normalized_role:
            try:
                message.role = normalized_role
            except Exception:
                pass

        if model_name in (
            "deepseek-r1-distill-llama-70b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ):
            message_dict.pop("reasoning", None)
            message_dict.pop("channel", None)

        message_dict["role"] = normalized_role or message_dict.get("role", "assistant")

        messages.append(message_dict)
        stop_reason = LlmStopReason.END_TURN
        requested_tool_calls: dict[str, CallToolRequest] | None = None
        if await self._is_tool_stop_reason(choice.finish_reason) and message.tool_calls:
            requested_tool_calls = {}
            stop_reason = LlmStopReason.TOOL_USE
            for tool_call in message.tool_calls:
                tool_call_request = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name=tool_call.function.name,
                        arguments={}
                        if not tool_call.function.arguments
                        or tool_call.function.arguments.strip() == ""
                        else from_json(tool_call.function.arguments, allow_partial=True),
                    ),
                )
                requested_tool_calls[tool_call.id] = tool_call_request
        elif choice.finish_reason == "length":
            stop_reason = LlmStopReason.MAX_TOKENS
            # We have reached the max tokens limit
            self.logger.debug(" Stopping because finish_reason is 'length'")
        elif choice.finish_reason == "content_filter":
            stop_reason = LlmStopReason.SAFETY
            self.logger.debug(" Stopping because finish_reason is 'content_filter'")

        # Update diagnostic snapshot (never read again)
        # This provides a snapshot of what was sent to the provider for debugging
        self.history.set(messages)

        self._log_chat_finished(model=self.default_request_params.model)

        reasoning_blocks: list[ContentBlock] | None = None
        if streamed_reasoning:
            reasoning_blocks = [TextContent(type="text", text="".join(streamed_reasoning))]

        return PromptMessageExtended(
            role="assistant",
            content=response_content_blocks,
            tool_calls=requested_tool_calls,
            channels={REASONING: reasoning_blocks} if reasoning_blocks else None,
            stop_reason=stop_reason,
        )

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

    def _handle_retry_failure(self, error: Exception) -> PromptMessageExtended | None:
        """Return the legacy error-channel response when retries are exhausted."""
        if isinstance(error, APIError):
            model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL
            return self._stream_failure_response(error, model_name)
        return None

    async def _is_tool_stop_reason(self, finish_reason: str) -> bool:
        return True

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list["PromptMessageExtended"],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """
        Provider-specific prompt application.
        Templates are handled by the agent; messages already include them.
        """
        # Determine effective params
        req_params = self.get_request_params(request_params)

        last_message = multipart_messages[-1]

        # If the last message is from the assistant, no inference required
        if last_message.role == "assistant":
            return last_message

        # Convert the supplied history/messages directly
        converted_messages = self._convert_to_provider_format(multipart_messages)
        if not converted_messages:
            converted_messages = [{"role": "user", "content": ""}]

        return await self._openai_completion(converted_messages, req_params, tools)

    def _prepare_api_request(
        self, messages, tools: list[ChatCompletionToolParam] | None, request_params: RequestParams
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

        arguments: dict[str, str] = self.prepare_provider_arguments(
            base_args, request_params, self.OPENAI_EXCLUDE_FIELDS.union(self.BASE_EXCLUDE_FIELDS)
        )
        return arguments

    @staticmethod
    def _extract_reasoning_text(reasoning: Any = None, reasoning_content: Any | None = None) -> str:
        """Extract text from provider-specific reasoning payloads.

        Priority: explicit `reasoning` field (string/object/list) > `reasoning_content` list.
        """

        def _coerce_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return str(value.get("text") or value)
            text_attr = None
            try:
                text_attr = getattr(value, "text", None)
            except Exception:
                text_attr = None
            if text_attr:
                return str(text_attr)
            return str(value)

        if reasoning is not None:
            if isinstance(reasoning, (list, tuple)):
                combined = "".join(_coerce_text(item) for item in reasoning)
            else:
                combined = _coerce_text(reasoning)
            if combined.strip():
                return combined

        if reasoning_content:
            parts: list[str] = []
            for item in reasoning_content:
                text = _coerce_text(item)
                if text:
                    parts.append(text)
            combined = "".join(parts)
            if combined.strip():
                return combined

        return ""

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert PromptMessageExtended list to OpenAI ChatCompletionMessageParam format.
        This is called fresh on every API call from _convert_to_provider_format().

        Args:
            messages: List of PromptMessageExtended objects

        Returns:
            List of OpenAI ChatCompletionMessageParam objects
        """
        converted: list[ChatCompletionMessageParam] = []
        reasoning_mode = ModelDatabase.get_reasoning(self.default_request_params.model)

        for msg in messages:
            # convert_to_openai returns a list of messages
            openai_msgs = OpenAIConverter.convert_to_openai(msg)

            if reasoning_mode == "reasoning_content" and msg.channels:
                reasoning_blocks = msg.channels.get(REASONING) if msg.channels else None
                if reasoning_blocks:
                    reasoning_texts = [get_text(block) for block in reasoning_blocks]
                    reasoning_texts = [txt for txt in reasoning_texts if txt]
                    if reasoning_texts:
                        reasoning_content = "\n\n".join(reasoning_texts)
                        for oai_msg in openai_msgs:
                            oai_msg["reasoning_content"] = reasoning_content

            # gpt-oss: per docs, reasoning should be dropped on subsequent sampling
            # UNLESS tool calling is involved. For tool calls, prefix the assistant
            # message content with the reasoning text.
            if reasoning_mode == "gpt_oss" and msg.channels and msg.tool_calls:
                reasoning_blocks = msg.channels.get(REASONING) if msg.channels else None
                if reasoning_blocks:
                    reasoning_texts = [get_text(block) for block in reasoning_blocks]
                    reasoning_texts = [txt for txt in reasoning_texts if txt]
                    if reasoning_texts:
                        reasoning_text = "\n\n".join(reasoning_texts)
                        for oai_msg in openai_msgs:
                            existing_content = oai_msg.get("content", "") or ""
                            oai_msg["content"] = reasoning_text + existing_content

            converted.extend(openai_msgs)

        return converted

    def adjust_schema(self, inputSchema: dict) -> dict:
        # return inputSchema
        if self.provider not in [Provider.OPENAI, Provider.AZURE]:
            return inputSchema

        if "properties" in inputSchema:
            return inputSchema

        result = inputSchema.copy()
        result["properties"] = {}
        return result
