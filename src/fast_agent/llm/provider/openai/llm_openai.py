import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import httpx
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
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic_core import from_json

from fast_agent.constants import FAST_AGENT_SAFETY_DETAILS, REASONING
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.fastagent_llm import FastAgentLLM, RequestParams
from fast_agent.llm.provider.error_utils import build_stream_failure_response
from fast_agent.llm.provider.openai._stream_capture import (
    save_stream_chunk as _save_stream_chunk,
)
from fast_agent.llm.provider.openai._stream_capture import (
    save_stream_request as _save_stream_request,
)
from fast_agent.llm.provider.openai._stream_capture import (
    stream_capture_filename as _stream_capture_filename,
)
from fast_agent.llm.provider.openai.multipart_converter_openai import OpenAIConverter
from fast_agent.llm.provider.openai.responses_files import ResponsesFileMixin
from fast_agent.llm.provider.openai.schema_sanitizer import (
    sanitize_tool_input_schema,
    should_strip_tool_schema_defaults,
)
from fast_agent.llm.provider.openai.streaming_utils import with_stream_idle_timeout
from fast_agent.llm.provider.openai.structured_output import OpenAIStructuredOutputMixin
from fast_agent.llm.provider.openai.tool_notifications import OpenAIToolNotificationMixin
from fast_agent.llm.provider.reasoning_config import reasoning_setting_from_config
from fast_agent.llm.provider.streaming_timeouts import await_stream_start
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import format_reasoning_setting, parse_reasoning_setting
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.tool_call_errors import format_incomplete_tool_call_error
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mime_utils import guess_mime_type
from fast_agent.types import LlmStopReason, PromptMessageExtended
from fast_agent.utils.reasoning_chunk_join import (
    ReasoningTextAccumulator,
    normalize_reasoning_delta,
)
from fast_agent.utils.text import strip_casefold, strip_to_none

_logger = get_logger(__name__)


class EmptyStreamError(RuntimeError):
    """Raised when a streaming response yields no chunks."""


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "low"
OPENAI_FINISH_REASON_MAP: dict[str, LlmStopReason] = {
    "length": LlmStopReason.MAX_TOKENS,
    "content_filter": LlmStopReason.SAFETY,
}


@runtime_checkable
class _OpenAIUsageHolder(Protocol):
    usage: Any


@runtime_checkable
class _OpenAIChoicesHolder(Protocol):
    choices: Any


def _openai_usage(value: object) -> Any | None:
    if not isinstance(value, _OpenAIUsageHolder):
        return None
    return value.usage


def _openai_choices(value: object) -> Any | None:
    if not isinstance(value, _OpenAIChoicesHolder):
        return None
    return value.choices


def _coerce_reasoning_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text") or value)
    try:
        text_attr = value.text
    except Exception:
        text_attr = None
    if text_attr:
        return str(text_attr)
    return str(value)


def _combine_reasoning_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "".join(_coerce_reasoning_text(item) for item in value)
    return _coerce_reasoning_text(value)


@dataclass(slots=True)
class _ManualOpenAIStreamState:
    estimated_tokens: int = 0
    reasoning_active: bool = False
    reasoning_segments: ReasoningTextAccumulator = field(
        default_factory=lambda: ReasoningTextAccumulator(normalizer=normalize_reasoning_delta)
    )
    accumulated_content: str = ""
    cumulative_content: str = ""
    role: str = "assistant"
    tool_calls_map: dict[int, dict[str, Any]] = field(default_factory=dict)
    function_call: Any | None = None
    finish_reason: Any | None = None
    usage_data: Any | None = None
    tool_call_started: dict[int, dict[str, Any]] = field(default_factory=dict)
    notified_tool_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _OpenAICompletionRequest:
    params: RequestParams
    model_name: str
    messages: list[ChatCompletionMessageParam]
    arguments: dict[str, Any]


@dataclass(slots=True)
class _OpenAICompletionResponse:
    response: Any
    streamed_reasoning: list[str]


@dataclass(slots=True)
class _OpenAIStopResult:
    stop_reason: LlmStopReason
    requested_tool_calls: dict[str, CallToolRequest] | None = None


class OpenAILLM(
    OpenAIToolNotificationMixin,
    ResponsesFileMixin,
    OpenAIStructuredOutputMixin,
    FastAgentLLM[ChatCompletionMessageParam, ChatCompletionMessage],
):
    # Config section name override (falls back to provider value)
    config_section: str | None = None
    # OpenAI-specific parameter exclusions
    OPENAI_EXCLUDE_FIELDS: ClassVar[set[str]] = {
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

    def __init__(self, provider: Provider = Provider.OPENAI, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)

        # Initialize logger with name if available
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)
        self._file_id_cache: dict[str, str] = {}

        # Set up reasoning-related attributes
        raw_setting = kwargs.get("reasoning_effort")
        if self.context and self.context.config and self.context.config.openai:
            config = self.context.config.openai
            if raw_setting is None:
                raw_setting, warn_deprecated_reasoning_effort = reasoning_setting_from_config(
                    config
                )
                if warn_deprecated_reasoning_effort:
                    self.logger.warning(
                        "OpenAI config 'reasoning_effort' is deprecated; use 'reasoning'."
                    )

        setting = parse_reasoning_setting(raw_setting)
        if setting is not None:
            try:
                self.set_reasoning_effort(setting)
            except ValueError as exc:
                self.logger.warning(f"Invalid reasoning setting: {exc}")

        # Determine reasoning mode for the selected model
        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning_mode = self._get_model_reasoning(chosen_model)
        self._reasoning = self._reasoning_mode == "openai"
        if self._reasoning_mode:
            self.logger.info(
                f"Using reasoning model '{chosen_model}' (mode='{self._reasoning_mode}') with "
                f"'{format_reasoning_setting(self.reasoning_effort)}' reasoning effort"
            )

    async def _download_remote_file(
        self,
        file_url: str,
    ) -> tuple[bytes | None, str | None]:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                response = await client.get(file_url)
                response.raise_for_status()
        except Exception as exc:
            self.logger.warning(
                "Failed to download remote attachment for OpenAI chat completions",
                data={"url": file_url, "error": str(exc)},
            )
            return None, None

        content_type = strip_to_none(response.headers.get("content-type", "").split(";", 1)[0])
        return response.content, content_type

    @staticmethod
    def _chat_file_filename(file_obj: dict[str, Any]) -> str | None:
        filename = file_obj.get("filename")
        if isinstance(filename, str) and filename:
            return filename
        return None

    async def _chat_file_bytes_from_url(
        self,
        file_url: str,
        filename: str | None,
    ) -> tuple[bytes | None, str | None, str | None]:
        if file_url.startswith("data:"):
            data_bytes, mime_type = self._decode_file_data(file_url)
            return data_bytes, filename, mime_type
        if file_url.startswith("file://"):
            local_path = Path(file_url[len("file://") :])
            if local_path.exists():
                resolved_filename = filename or local_path.name
                return (
                    local_path.read_bytes(),
                    resolved_filename,
                    guess_mime_type(local_path.name),
                )
            return None, filename, None
        if file_url.startswith(("http://", "https://")):
            data_bytes, mime_type = await self._download_remote_file(file_url)
            return data_bytes, filename, mime_type
        return None, filename, None

    async def _normalize_chat_file_part(
        self,
        client: AsyncOpenAI,
        part: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if part.get("type") != "file":
            return part, False

        file_obj = part.get("file")
        if not isinstance(file_obj, dict):
            return part, False

        file_url = file_obj.get("file_url")
        if not isinstance(file_url, str) or not file_url:
            return part, False

        filename = self._chat_file_filename(file_obj)
        data_bytes, filename, mime_type = await self._chat_file_bytes_from_url(file_url, filename)
        if data_bytes is None:
            return part, False

        resolved_mime_type = mime_type or guess_mime_type(filename or file_url)
        file_id = await self._upload_file_bytes(client, data_bytes, filename, resolved_mime_type)
        return {"type": "file", "file": {"file_id": file_id}}, True

    async def _normalize_chat_content_parts(
        self,
        client: AsyncOpenAI,
        content: list[Any],
    ) -> tuple[list[Any], bool]:
        updated_content: list[Any] = []
        changed = False
        for part in content:
            if not isinstance(part, dict):
                updated_content.append(part)
                continue

            updated_part, part_changed = await self._normalize_chat_file_part(client, part)
            updated_content.append(updated_part)
            changed = changed or part_changed
        return updated_content, changed

    async def _normalize_chat_message_files(
        self,
        client: AsyncOpenAI,
        message: ChatCompletionMessageParam,
    ) -> ChatCompletionMessageParam:
        content = message.get("content")
        if not isinstance(content, list):
            return message

        updated_content, changed = await self._normalize_chat_content_parts(client, content)
        if not changed:
            return message

        return cast(
            "ChatCompletionMessageParam",
            {**message, "content": updated_content},
        )

    async def _normalize_chat_completion_files(
        self,
        client: AsyncOpenAI,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        return [await self._normalize_chat_message_files(client, message) for message in messages]

    def _resolve_reasoning_effort(self) -> str | None:
        setting = self.reasoning_effort
        if setting is None:
            return DEFAULT_REASONING_EFFORT
        if setting.kind == "effort":
            return str(setting.value)
        if setting.kind == "toggle":
            return None if setting.value is False else DEFAULT_REASONING_EFFORT
        if setting.kind == "budget":
            self.logger.warning("Ignoring budget reasoning setting for OpenAI models.")
            return DEFAULT_REASONING_EFFORT
        return DEFAULT_REASONING_EFFORT

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenAI-specific default parameters"""
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_OPENAI_MODEL)

    def _provider_base_url(self) -> str | None:
        if self.context.config and self.context.config.openai:
            return self.context.config.openai.base_url
        return None

    def _provider_default_headers(self) -> dict[str, str] | None:
        """
        Get custom headers from configuration.
        Subclasses can override this to provide provider-specific headers.
        """
        provider_config = self._get_provider_config()
        return getattr(provider_config, "default_headers", None) if provider_config else None

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

    def _emit_tool_notification_fallback(
        self,
        tool_calls: Any,
        notified_indices: set[int],
        *,
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

            self._emit_fallback_tool_notification_event(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                index=index,
                model=model,
            )

    def _handle_reasoning_delta(
        self,
        *,
        reasoning_mode: str | None,
        reasoning_text: str,
        reasoning_active: bool,
        reasoning_segments: ReasoningTextAccumulator,
    ) -> bool:
        """Stream reasoning text and track whether a thinking block is open."""
        if not self._should_emit_reasoning_stream(reasoning_mode):
            return reasoning_active

        if not reasoning_text:
            return reasoning_active

        normalized_text = reasoning_segments.append(reasoning_text)
        if not normalized_text:
            return reasoning_active

        if reasoning_mode == "tags":
            if not reasoning_active:
                reasoning_active = True
            self._notify_stream_listeners(StreamChunk(text=normalized_text, is_reasoning=True))
            return reasoning_active

        if reasoning_mode in {"stream", "reasoning_content", "gpt_oss"}:
            # Emit reasoning as-is
            self._notify_stream_listeners(StreamChunk(text=normalized_text, is_reasoning=True))
            return reasoning_active

        return reasoning_active

    def _should_emit_reasoning_stream(self, reasoning_mode: str | None) -> bool:
        """Allow subclasses to suppress streamed reasoning display."""
        return True

    def _handle_tool_delta(
        self,
        *,
        delta_tool_calls: Any,
        tool_call_started: dict[int, dict[str, Any]],
        model: str,
        notified_tool_indices: set[int],
    ) -> None:
        """Emit tool call start/delta events and keep state in sync."""
        for tool_call in delta_tool_calls:
            index = tool_call.index
            if index is None:
                continue

            existing_info = tool_call_started.get(index)

            # Get current chunk values
            chunk_id = tool_call.id
            chunk_name = (
                tool_call.function.name if tool_call.function and tool_call.function.name else None
            )

            # Accumulate values: prefer new, fall back to existing
            tool_use_id = chunk_id or (existing_info.get("tool_use_id") if existing_info else None)
            function_name = chunk_name or (
                existing_info.get("tool_name") if existing_info else None
            )

            # Always create/update tracking entry when we have any new info
            # This ensures we accumulate metadata across chunks
            if chunk_id or chunk_name:
                if existing_info is None:
                    tool_call_started[index] = {
                        "tool_name": function_name,
                        "tool_use_id": tool_use_id,
                        "notified": False,
                    }
                    existing_info = tool_call_started[index]
                else:
                    if tool_use_id:
                        existing_info["tool_use_id"] = tool_use_id
                    if function_name:
                        existing_info["tool_name"] = function_name

            # Fire "start" notification once we have BOTH values
            if (
                existing_info
                and not existing_info.get("notified")
                and existing_info.get("tool_use_id")
                and existing_info.get("tool_name")
            ):
                self._notify_tool_stream_listeners(
                    "start",
                    {
                        "tool_name": existing_info["tool_name"],
                        "tool_use_id": existing_info["tool_use_id"],
                        "index": index,
                    },
                )
                self.logger.info(
                    "Model started streaming tool call",
                    data={
                        "progress_action": ProgressAction.CALLING_TOOL,
                        "agent_name": self.name,
                        "model": model,
                        "tool_name": existing_info["tool_name"],
                        "tool_use_id": existing_info["tool_use_id"],
                        "tool_event": "start",
                    },
                )
                existing_info["notified"] = True
                notified_tool_indices.add(index)

            if tool_call.function and tool_call.function.arguments:
                info = tool_call_started.setdefault(
                    index,
                    {
                        "tool_name": function_name,
                        "tool_use_id": tool_use_id,
                        "notified": False,
                    },
                )
                self._notify_tool_stream_listeners(
                    "delta",
                    {
                        "tool_name": info.get("tool_name"),
                        "tool_use_id": info.get("tool_use_id"),
                        "index": index,
                        "chunk": tool_call.function.arguments,
                    },
                )

    def _process_stream_chunk_common(
        self,
        chunk: Any,
        *,
        reasoning_mode: Any,
        reasoning_active: bool,
        reasoning_segments: ReasoningTextAccumulator,
        tool_call_started: dict[int, dict[str, Any]],
        model: str,
        notified_tool_indices: set[int],
        cumulative_content: str,
        estimated_tokens: int,
    ) -> tuple[str, int, bool, str | None]:
        """Process common streaming chunk logic shared by multiple stream processing methods.

        Returns:
            Tuple of (cumulative_content, estimated_tokens, reasoning_active, incremental_content)
        """
        incremental: str | None = None
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
                    model=model,
                    notified_tool_indices=notified_tool_indices,
                )

            # Handle text content streaming
            cumulative_content, estimated_tokens, reasoning_active, incremental = (
                self._apply_content_delta(
                    delta_content=delta.content,
                    cumulative_content=cumulative_content,
                    model=model,
                    estimated_tokens=estimated_tokens,
                    reasoning_active=reasoning_active,
                )
            )

            # Fire "stop" event when tool calls complete
            if choice.finish_reason == "tool_calls":
                self._finalize_tool_calls_on_stop(
                    tool_call_started=tool_call_started,
                    model=model,
                    notified_tool_indices=notified_tool_indices,
                )

        return cumulative_content, estimated_tokens, reasoning_active, incremental

    def _finalize_tool_calls_on_stop(
        self,
        *,
        tool_call_started: dict[int, dict[str, Any]],
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
                    "tool_terminal": True,
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
            },
        )

        return estimated_tokens, reasoning_active

    def _close_reasoning_if_active(self, reasoning_active: bool) -> bool:
        """Return reasoning state; kept for symmetry."""
        return False if reasoning_active else reasoning_active

    @staticmethod
    def _extract_incremental_delta(delta: str, cumulative: str) -> tuple[str, str]:
        """Return the incremental portion of a possibly cumulative stream delta."""
        if not delta:
            return "", cumulative
        if cumulative and delta.startswith(cumulative):
            return delta[len(cumulative) :], delta
        return delta, cumulative + delta

    def _apply_content_delta(
        self,
        *,
        delta_content: str | None,
        cumulative_content: str,
        model: str,
        estimated_tokens: int,
        reasoning_active: bool,
    ) -> tuple[str, int, bool, str]:
        """Apply a content delta, returning updated state and any incremental text."""
        if not delta_content:
            return cumulative_content, estimated_tokens, reasoning_active, ""

        incremental, cumulative_content = self._extract_incremental_delta(
            delta_content, cumulative_content
        )
        if incremental:
            estimated_tokens, reasoning_active = self._emit_text_delta(
                content=incremental,
                model=model,
                estimated_tokens=estimated_tokens,
                reasoning_active=reasoning_active,
            )

        return cumulative_content, estimated_tokens, reasoning_active, incremental

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
        reasoning_segments = ReasoningTextAccumulator(normalizer=normalize_reasoning_delta)
        reasoning_mode = self._get_model_reasoning(model)

        # For providers/models that emit non-OpenAI deltas, fall back to manual accumulation
        stream_mode = self._get_model_stream_mode(model)
        provider_requires_manual = self.provider in [
            Provider.GENERIC,
            Provider.OPENROUTER,
            Provider.GOOGLE_OAI,
        ]
        if stream_mode == "manual" or provider_requires_manual:
            return await self._process_stream_manual(stream, model, capture_filename)

        # Use ChatCompletionStreamState helper for accumulation (OpenAI only)
        state = ChatCompletionStreamState()
        cumulative_content = ""
        chunk_count = 0

        # Track tool call state for stream events
        tool_call_started: dict[int, dict[str, Any]] = {}
        notified_tool_indices: set[int] = set()

        # Process the stream chunks
        # Cancellation is handled via asyncio.Task.cancel() which raises CancelledError
        async for chunk in stream:
            chunk_count += 1
            # Save chunk if stream capture is enabled
            _save_stream_chunk(capture_filename, chunk)
            self._raise_stream_chunk_error(chunk)
            # Handle chunk accumulation
            state.handle_chunk(chunk)
            # Process streaming events for tool calls
            cumulative_content, estimated_tokens, reasoning_active, _ = (
                self._process_stream_chunk_common(
                    chunk,
                    reasoning_mode=reasoning_mode,
                    reasoning_active=reasoning_active,
                    reasoning_segments=reasoning_segments,
                    tool_call_started=tool_call_started,
                    model=model,
                    notified_tool_indices=notified_tool_indices,
                    cumulative_content=cumulative_content,
                    estimated_tokens=estimated_tokens,
                )
            )

        if tool_call_started:
            incomplete_tools = [
                f"{info.get('tool_name', 'unknown')}:{info.get('tool_use_id', 'unknown')}"
                for info in tool_call_started.values()
            ]
            self.logger.error(
                "Tool call streaming incomplete - started but never finished",
                data={
                    "incomplete_tools": incomplete_tools,
                    "tool_count": len(tool_call_started),
                },
            )
            raise RuntimeError(format_incomplete_tool_call_error(incomplete_tools))

        if chunk_count == 0:
            raise EmptyStreamError("OpenAI streaming response yielded no chunks")

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
        usage = _openai_usage(final_completion)
        if usage:
            actual_tokens = usage.completion_tokens
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
                f"Streaming complete - Model: {model}, Input tokens: {usage.prompt_tokens}, Output tokens: {usage.completion_tokens}"
            )

        final_message = None
        choices = _openai_choices(final_completion)
        if choices:
            final_message = getattr(choices[0], "message", None)
        tool_calls = getattr(final_message, "tool_calls", None) if final_message else None
        self._emit_tool_notification_fallback(
            tool_calls,
            notified_tool_indices,
            model=model,
        )

        return final_completion, reasoning_segments.parts()

    def _raise_stream_chunk_error(self, chunk: Any) -> None:
        if getattr(chunk, "choices", None):
            return
        error_message = getattr(chunk, "error_message", None)
        if error_message:
            raise RuntimeError(f"Provider stream error: {error_message}")

    def _normalize_role(self, role: str | None) -> str:
        """Ensure the role string matches MCP expectations."""
        default_role = "assistant"
        if not role:
            return default_role

        lowered = strip_casefold(role)
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

    def _record_manual_tool_call_delta(
        self,
        delta_tool_call: Any,
        state: _ManualOpenAIStreamState,
    ) -> None:
        if delta_tool_call.index is None:
            return

        tool_call = state.tool_calls_map.setdefault(
            delta_tool_call.index,
            {
                "id": delta_tool_call.id,
                "type": delta_tool_call.type or "function",
                "function": {
                    "name": (delta_tool_call.function.name if delta_tool_call.function else None),
                    "arguments": "",
                },
            },
        )

        if delta_tool_call.id:
            tool_call["id"] = delta_tool_call.id
        if not delta_tool_call.function:
            return
        if delta_tool_call.function.name:
            tool_call["function"]["name"] = delta_tool_call.function.name
        if delta_tool_call.function.arguments is not None:
            tool_call["function"]["arguments"] += delta_tool_call.function.arguments

    def _record_manual_stream_choice(
        self,
        chunk: Any,
        state: _ManualOpenAIStreamState,
    ) -> None:
        if not chunk.choices:
            return

        choice = chunk.choices[0]
        if choice.delta.role:
            state.role = choice.delta.role
        if choice.delta.tool_calls:
            for delta_tool_call in choice.delta.tool_calls:
                self._record_manual_tool_call_delta(delta_tool_call, state)
        if choice.delta.function_call:
            state.function_call = choice.delta.function_call
        if choice.finish_reason:
            state.finish_reason = choice.finish_reason

    def _record_manual_stream_chunk(
        self,
        chunk: Any,
        *,
        model: str,
        reasoning_mode: str | None,
        state: _ManualOpenAIStreamState,
    ) -> None:
        (
            state.cumulative_content,
            state.estimated_tokens,
            state.reasoning_active,
            incremental,
        ) = self._process_stream_chunk_common(
            chunk,
            reasoning_mode=reasoning_mode,
            reasoning_active=state.reasoning_active,
            reasoning_segments=state.reasoning_segments,
            tool_call_started=state.tool_call_started,
            model=model,
            notified_tool_indices=state.notified_tool_indices,
            cumulative_content=state.cumulative_content,
            estimated_tokens=state.estimated_tokens,
        )
        if incremental:
            state.accumulated_content += incremental

        self._record_manual_stream_choice(chunk, state)
        usage = _openai_usage(chunk)
        if usage:
            state.usage_data = usage

    @staticmethod
    def _manual_stream_tool_calls(
        state: _ManualOpenAIStreamState,
    ) -> list[ChatCompletionMessageToolCall] | None:
        tool_calls = []
        for idx in sorted(state.tool_calls_map):
            tool_call_data = state.tool_calls_map[idx]
            if not tool_call_data["id"] or not tool_call_data["function"]["name"]:
                continue
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call_data["id"],
                    type=tool_call_data["type"],
                    function=Function(
                        name=tool_call_data["function"]["name"],
                        arguments=tool_call_data["function"]["arguments"],
                    ),
                )
            )
        return tool_calls or None

    def _raise_for_incomplete_manual_tools(
        self,
        state: _ManualOpenAIStreamState,
    ) -> None:
        if not state.tool_call_started:
            return

        incomplete_tools = [
            f"{info.get('tool_name', 'unknown')}:{info.get('tool_use_id', 'unknown')}"
            for info in state.tool_call_started.values()
        ]
        self.logger.error(
            "Tool call streaming incomplete - started but never finished",
            data={
                "incomplete_tools": incomplete_tools,
                "tool_count": len(state.tool_call_started),
            },
        )
        raise RuntimeError(format_incomplete_tool_call_error(incomplete_tools))

    def _manual_stream_completion(self, state: _ManualOpenAIStreamState) -> Any:
        message = ChatCompletionMessage(
            content=state.accumulated_content,
            role=cast("Any", state.role),
            tool_calls=cast("Any", self._manual_stream_tool_calls(state)),
            function_call=state.function_call,
            refusal=None,
            annotations=None,
            audio=None,
        )

        final_completion = SimpleNamespace()
        final_completion.choices = [SimpleNamespace()]
        final_completion.choices[0].message = message
        final_completion.choices[0].finish_reason = state.finish_reason
        final_completion.usage = state.usage_data
        return final_completion

    def _log_manual_stream_usage(
        self,
        state: _ManualOpenAIStreamState,
        model: str,
    ) -> None:
        if not state.usage_data:
            return

        actual_tokens = getattr(state.usage_data, "completion_tokens", state.estimated_tokens)
        token_str = str(actual_tokens).rjust(5)
        self.logger.info(
            "Streaming progress",
            data={
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            },
        )
        self.logger.info(
            f"Streaming complete - Model: {model}, Input tokens: {getattr(state.usage_data, 'prompt_tokens', 0)}, Output tokens: {actual_tokens}"
        )

    # TODO - as per other comment this needs to go in another class. There are a number of "special" cases dealt with
    # here to deal with OpenRouter idiosyncrasies between e.g. Anthropic and Gemini models.
    async def _process_stream_manual(
        self,
        stream,
        model: str,
        capture_filename: Path | None = None,
    ) -> tuple[Any, list[str]]:
        """Manual stream processing for providers like Ollama that may not work with ChatCompletionStreamState."""
        state = _ManualOpenAIStreamState()
        reasoning_mode = self._get_model_reasoning(model)

        async for chunk in stream:
            _save_stream_chunk(capture_filename, chunk)
            self._record_manual_stream_chunk(
                chunk,
                model=model,
                reasoning_mode=reasoning_mode,
                state=state,
            )

        self._raise_for_incomplete_manual_tools(state)
        final_completion = self._manual_stream_completion(state)
        self._log_manual_stream_usage(state, model)

        final_message = final_completion.choices[0].message if final_completion.choices else None
        tool_calls = getattr(final_message, "tool_calls", None) if final_message else None
        self._emit_tool_notification_fallback(
            tool_calls,
            state.notified_tool_indices,
            model=model,
        )

        return final_completion, state.reasoning_segments.parts()

    def _openai_completion_request(
        self,
        message: list[ChatCompletionMessageParam] | None,
        request_params: RequestParams | None,
        tools: list[Tool] | None,
    ) -> _OpenAICompletionRequest:
        params = self.get_request_params(request_params=request_params)
        model_name = params.model or self.default_request_params.model or DEFAULT_OPENAI_MODEL
        messages = self._openai_completion_messages(message, params)
        available_tools = self._openai_completion_tools(tools, model_name)
        arguments = self._prepare_api_request(messages, available_tools, params)
        if not self._reasoning and params.stopSequences:
            arguments["stop"] = params.stopSequences
        return _OpenAICompletionRequest(
            params=params,
            model_name=model_name,
            messages=messages,
            arguments=arguments,
        )

    def _openai_completion_messages(
        self,
        message: list[ChatCompletionMessageParam] | None,
        request_params: RequestParams,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
        if message:
            messages.extend(cast("list[ChatCompletionMessageParam]", message))
        return messages

    def _openai_completion_tools(
        self,
        tools: list[Tool] | None,
        model_name: str,
    ) -> list[ChatCompletionToolParam] | None:
        available_tools = cast(
            "list[ChatCompletionToolParam]",
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description if tool.description else "",
                        "parameters": self.adjust_schema(tool.inputSchema, model_name=model_name),
                    },
                }
                for tool in tools or []
            ],
        )
        if available_tools:
            return available_tools
        if self.provider in [Provider.DEEPSEEK, Provider.ALIYUN]:
            return None
        return []

    async def _create_openai_streaming_response(
        self,
        client: AsyncOpenAI,
        request: _OpenAICompletionRequest,
        capture_filename: Path | None,
    ) -> _OpenAICompletionResponse:
        timeout = request.params.streaming_timeout
        stream = await await_stream_start(
            client.chat.completions.create(**request.arguments),
            timeout_seconds=timeout,
            timeout_message=f"OpenAI stream did not start within {timeout} seconds.",
        )
        timed_stream = with_stream_idle_timeout(
            stream,
            idle_timeout_seconds=timeout,
            timeout_message=f"Streaming was idle for more than {timeout} seconds.",
        )
        try:
            response, streamed_reasoning = await self._process_stream(
                timed_stream, request.model_name, capture_filename
            )
        except EmptyStreamError as exc:
            self.logger.error(
                "OpenAI stream returned no chunks; retrying without streaming",
                data={
                    "model": request.model_name,
                    "error": str(exc),
                },
            )
            response = await client.chat.completions.create(
                **self._prepare_non_streaming_request(request.arguments)
            )
            streamed_reasoning = []
        except TimeoutError:
            if timeout is None:
                raise
            self.logger.error(
                "Streaming idle timeout while waiting for OpenAI completion",
                data={
                    "model": request.model_name,
                    "timeout_seconds": timeout,
                },
            )
            raise
        return _OpenAICompletionResponse(response, streamed_reasoning)

    async def _run_openai_completion_request(
        self,
        request: _OpenAICompletionRequest,
    ) -> _OpenAICompletionResponse:
        async with self._openai_client() as client:
            arguments = dict(request.arguments)
            messages_arg = arguments.get("messages")
            if isinstance(messages_arg, list):
                arguments["messages"] = await self._normalize_chat_completion_files(
                    client, messages_arg
                )
            normalized_request = _OpenAICompletionRequest(
                params=request.params,
                model_name=request.model_name,
                messages=request.messages,
                arguments=arguments,
            )
            self.logger.debug(f"OpenAI completion requested for: {arguments}")
            self._log_chat_progress(self.chat_turn(), model=request.model_name)
            capture_filename = _stream_capture_filename(self.chat_turn())
            _save_stream_request(capture_filename, arguments)
            return await self._create_openai_streaming_response(
                client, normalized_request, capture_filename
            )

    def _track_openai_response_usage(self, response: Any, model_name: str) -> None:
        if isinstance(response, BaseException):
            return
        usage = _openai_usage(response)
        if not usage:
            return
        try:
            turn_usage = TurnUsage.from_openai(usage, model_name)
            self._finalize_turn_usage(turn_usage)
        except Exception as e:
            self.logger.warning(f"Failed to track usage: {e}")

    def _raise_openai_response_error(self, response: Any) -> None:
        if isinstance(response, AuthenticationError):
            raise ProviderKeyError(
                "Rejected OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from response
        if isinstance(response, BaseException):
            self.logger.error(f"Error: {response}")

    def _append_openai_assistant_history(
        self,
        messages: list[ChatCompletionMessageParam],
        message: ChatCompletionMessage,
        model_name: str,
    ) -> list[ContentBlock]:
        normalized_role = self._normalize_role(getattr(message, "role", None))
        response_content_blocks: list[ContentBlock] = []
        if message.content:
            response_content_blocks.append(TextContent(type="text", text=message.content))

        message_dict = message.model_dump()
        message_dict = {key: value for key, value in message_dict.items() if value is not None}
        with suppress(Exception):
            cast("Any", message).role = normalized_role
        if model_name in (
            "deepseek-r1-distill-llama-70b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ):
            message_dict.pop("reasoning", None)
            message_dict.pop("channel", None)
        message_dict["role"] = normalized_role or message_dict.get("role", "assistant")
        messages.append(cast("ChatCompletionMessageParam", message_dict))
        return response_content_blocks

    async def _openai_stop_result(self, choice: Any, message: Any) -> _OpenAIStopResult:
        if await self._is_tool_stop_reason(choice.finish_reason) and message.tool_calls:
            return _OpenAIStopResult(
                stop_reason=LlmStopReason.TOOL_USE,
                requested_tool_calls=self._openai_requested_tool_calls(message.tool_calls),
            )
        mapped_stop_reason = OPENAI_FINISH_REASON_MAP.get(choice.finish_reason)
        if mapped_stop_reason is not None:
            self.logger.debug(f" Stopping because finish_reason is {choice.finish_reason!r}")
            return _OpenAIStopResult(stop_reason=mapped_stop_reason)
        return _OpenAIStopResult(stop_reason=LlmStopReason.END_TURN)

    @staticmethod
    def _openai_requested_tool_calls(tool_calls: Any) -> dict[str, CallToolRequest]:
        requested_tool_calls: dict[str, CallToolRequest] = {}
        for tool_call in tool_calls:
            arguments = tool_call.function.arguments
            tool_arguments = (
                {}
                if not arguments or arguments.strip() == ""
                else from_json(arguments, allow_partial=True)
            )
            requested_tool_calls[tool_call.id] = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name=tool_call.function.name,
                    arguments=tool_arguments,
                ),
            )
        return requested_tool_calls

    @staticmethod
    def _openai_reasoning_blocks(streamed_reasoning: list[str]) -> list[ContentBlock] | None:
        if not streamed_reasoning:
            return None
        return [TextContent(type="text", text="".join(streamed_reasoning))]

    @staticmethod
    def _openai_response_channels(
        *,
        reasoning_blocks: list[ContentBlock] | None,
        choice: Any,
        stop_result: _OpenAIStopResult,
    ) -> dict[str, list[ContentBlock]] | None:
        channels: dict[str, list[ContentBlock]] = {}
        if reasoning_blocks:
            channels[REASONING] = reasoning_blocks
        if stop_result.stop_reason == LlmStopReason.SAFETY:
            channels[FAST_AGENT_SAFETY_DETAILS] = [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "provider": "openai",
                            "reason": str(getattr(choice, "finish_reason", "content_filter")),
                        }
                    ),
                )
            ]
        return channels or None

    async def _finalize_openai_completion(
        self,
        request: _OpenAICompletionRequest,
        completion: _OpenAICompletionResponse,
    ) -> PromptMessageExtended:
        response = completion.response
        self._track_openai_response_usage(response, request.model_name)
        self.logger.debug("OpenAI completion response:", data=response)
        self._raise_openai_response_error(response)

        choice = response.choices[0]
        message = choice.message
        response_content_blocks = self._append_openai_assistant_history(
            request.messages, message, request.model_name
        )
        stop_result = await self._openai_stop_result(choice, message)
        self.history.set(request.messages)
        self._log_chat_finished(model=request.model_name)
        reasoning_blocks = self._openai_reasoning_blocks(completion.streamed_reasoning)
        return PromptMessageExtended(
            role="assistant",
            content=response_content_blocks,
            tool_calls=stop_result.requested_tool_calls,
            channels=self._openai_response_channels(
                reasoning_blocks=reasoning_blocks,
                choice=choice,
                stop_result=stop_result,
            ),
            stop_reason=stop_result.stop_reason,
        )

    async def _openai_completion(
        self,
        message: list[ChatCompletionMessageParam] | None,
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        request = self._openai_completion_request(message, request_params, tools)
        try:
            completion = await self._run_openai_completion_request(request)
        except asyncio.CancelledError as e:
            reason = str(e) if e.args else "cancelled"
            self.logger.info(f"OpenAI completion cancelled: {reason}")
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )
        except APIError as error:
            self.logger.error("APIError during OpenAI completion", exc_info=error)
            raise
        except Exception:
            raise
        return await self._finalize_openai_completion(request, completion)

    def _handle_retry_failure(self, error: Exception) -> PromptMessageExtended | None:
        """Return the legacy error-channel response when retries are exhausted."""
        if isinstance(error, APIError):
            model_name = self.default_request_params.model or DEFAULT_OPENAI_MODEL
            return build_stream_failure_response(self.provider, error, model_name)
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
        converted_messages: list[ChatCompletionMessageParam] = self._convert_to_provider_format(
            multipart_messages
        )
        if not converted_messages:
            converted_messages = [ChatCompletionUserMessageParam(role="user", content="")]

        return await self._openai_completion(converted_messages, req_params, tools)

    def _prepare_api_request(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        # Create base arguments dictionary

        base_args: dict[str, Any] = {
            "model": request_params.model
            or self.default_request_params.model
            or DEFAULT_OPENAI_MODEL,
            "messages": messages,
            "tools": tools,
            "stream": True,  # Enable basic streaming
            "stream_options": {"include_usage": True},  # Required for usage data in streaming
        }

        if self._reasoning:
            effort = self._resolve_reasoning_effort()
            if request_params.maxTokens is not None:
                base_args["max_completion_tokens"] = request_params.maxTokens
            if effort:
                base_args["reasoning_effort"] = effort
        else:
            if request_params.maxTokens is not None:
                base_args["max_tokens"] = request_params.maxTokens
            if tools:
                base_args["parallel_tool_calls"] = request_params.parallel_tool_calls

        arguments: dict[str, str] = self.prepare_provider_arguments(
            base_args, request_params, self.OPENAI_EXCLUDE_FIELDS.union(self.BASE_EXCLUDE_FIELDS)
        )
        return arguments

    @staticmethod
    def _prepare_non_streaming_request(arguments: dict[str, Any]) -> dict[str, Any]:
        non_stream_args = dict(arguments)
        non_stream_args["stream"] = False
        non_stream_args.pop("stream_options", None)
        return non_stream_args

    @staticmethod
    def _extract_reasoning_text(reasoning: Any = None, reasoning_content: Any | None = None) -> str:
        """Extract text from provider-specific reasoning payloads.

        Priority: explicit `reasoning` field (string/object/list) > `reasoning_content` list.
        """
        combined = _combine_reasoning_text(reasoning)
        if combined.strip():
            return combined
        if reasoning_content:
            combined = "".join(
                text
                for text in (_coerce_reasoning_text(item) for item in reasoning_content)
                if text
            )
            if combined.strip():
                return combined

        return ""

    @staticmethod
    def _reasoning_channel_text(msg: PromptMessageExtended) -> str:
        if not msg.channels:
            return ""

        reasoning_blocks = msg.channels.get(REASONING)
        if not reasoning_blocks:
            return ""

        reasoning_texts = [text for block in reasoning_blocks if (text := get_text(block))]
        return "\n\n".join(reasoning_texts)

    def _apply_reasoning_replay(
        self,
        openai_msgs: list[ChatCompletionMessageParam],
        msg: PromptMessageExtended,
        reasoning_mode: str | None,
    ) -> None:
        reasoning_text = self._reasoning_channel_text(msg)
        if not reasoning_text:
            return

        if reasoning_mode == "reasoning_content":
            for oai_msg in openai_msgs:
                # reasoning_content is an OpenAI extension not in the TypedDict
                cast("dict[str, Any]", oai_msg)["reasoning_content"] = reasoning_text
            return

        # gpt-oss: per docs, reasoning should be dropped on subsequent sampling
        # UNLESS tool calling is involved. For tool calls, prefix the assistant
        # message content with the reasoning text.
        if reasoning_mode != "gpt_oss" or not msg.tool_calls:
            return

        for oai_msg in openai_msgs:
            oai_dict = cast("dict[str, Any]", oai_msg)
            existing_content = oai_dict.get("content", "") or ""
            if isinstance(existing_content, str):
                oai_dict["content"] = reasoning_text + existing_content

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
        model = self.default_request_params.model
        reasoning_mode = self._get_model_reasoning(model)

        for msg in messages:
            # convert_to_openai returns a list of messages
            openai_msgs = OpenAIConverter.convert_to_openai(msg)
            self._apply_reasoning_replay(openai_msgs, msg, reasoning_mode)
            converted.extend(openai_msgs)

        return converted

    def adjust_schema(self, inputSchema: dict, model_name: str | None = None) -> dict:
        effective_model = model_name or self.default_request_params.model
        result = (
            sanitize_tool_input_schema(inputSchema)
            if should_strip_tool_schema_defaults(effective_model)
            else inputSchema
        )

        if self.provider not in [Provider.OPENAI, Provider.AZURE]:
            return result

        if "properties" in result:
            return result

        result = result.copy()
        result["properties"] = {}
        return result
