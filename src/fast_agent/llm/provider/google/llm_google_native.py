import json
import logging
import secrets
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, cast

from google import genai
from google.genai import (
    errors,
    types,
)
from mcp import Tool as McpTool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)

from fast_agent.constants import DEFAULT_MAX_ITERATIONS, REASONING
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.prompt import Prompt
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.google._stream_capture import (
    save_stream_chunk,
    save_stream_request,
    stream_capture_filename,
)
from fast_agent.llm.provider.google.google_converter import GoogleConverter, GoogleToolResult
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import (
    format_reasoning_setting,
    parse_reasoning_setting,
)
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.tool_call_errors import format_incomplete_tool_call_error
from fast_agent.llm.tool_tracking import ToolCallTracker
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.text import strip_casefold

# Suppress noisy internal warnings and AFC logs from the Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.ERROR)

# Define default model and potentially other Google-specific defaults
DEFAULT_GOOGLE_MODEL = "gemini3"
_GOOGLE_VERTEX_PARTNER_MODEL_PREFIXES = ("claude",)
_GOOGLE_FINISH_REASON_MAP: dict[str, LlmStopReason] = {
    "STOP": LlmStopReason.END_TURN,
    "MAX_TOKENS": LlmStopReason.MAX_TOKENS,
    "LENGTH": LlmStopReason.MAX_TOKENS,
    "PROHIBITED_CONTENT": LlmStopReason.SAFETY,
    "SAFETY": LlmStopReason.SAFETY,
    "RECITATION": LlmStopReason.SAFETY,
    "BLOCKLIST": LlmStopReason.SAFETY,
    "SPII": LlmStopReason.SAFETY,
    "IMAGE_SAFETY": LlmStopReason.SAFETY,
    "IMAGE_PROHIBITED_CONTENT": LlmStopReason.SAFETY,
    "IMAGE_RECITATION": LlmStopReason.SAFETY,
    "MALFORMED_FUNCTION_CALL": LlmStopReason.ERROR,
    "UNEXPECTED_TOOL_CALL": LlmStopReason.ERROR,
    "TOO_MANY_TOOL_CALLS": LlmStopReason.ERROR,
    "NO_IMAGE": LlmStopReason.ERROR,
    "IMAGE_OTHER": LlmStopReason.ERROR,
}


def _google_thinking_effort_config(effort: str) -> tuple[int | None, str | None]:
    if effort == "none":
        return 0, None
    if effort == "auto":
        return -1, None

    level_map: dict[str, str] = {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }
    level = level_map.get(effort)
    if level is not None:
        return None, level
    return -1, None


# Define Google-specific parameter exclusions if necessary
GOOGLE_EXCLUDE_FIELDS = {
    # Add fields that should not be passed directly from RequestParams to google.genai config
    FastAgentLLM.PARAM_MESSAGES,  # Handled by contents
    FastAgentLLM.PARAM_MODEL,  # Handled during client/call setup
    FastAgentLLM.PARAM_SYSTEM_PROMPT,  # Handled by system_instruction in config
    FastAgentLLM.PARAM_USE_HISTORY,  # Handled by FastAgentLLM base / this class's logic
    FastAgentLLM.PARAM_MAX_ITERATIONS,  # Handled by this class's loop
    FastAgentLLM.PARAM_MCP_METADATA,
}.union(FastAgentLLM.BASE_EXCLUDE_FIELDS)


@dataclass(slots=True)
class _GoogleTextTimelineEntry:
    text: str


@dataclass(slots=True)
class _GoogleReasoningTimelineEntry:
    text: str
    thought_signature: bytes | None = None


@dataclass(slots=True)
class _GoogleToolTimelineEntry:
    tool_use_id: str


@dataclass(slots=True)
class _GoogleSignatureTimelineEntry:
    thought_signature: bytes


@dataclass(slots=True)
class _GoogleToolBuffer:
    tool_use_id: str
    name: str
    buffer: str = ""
    provider_call_id: str | None = None
    thought_signature: bytes | None = None


@dataclass(slots=True)
class _GoogleStreamState:
    model: str
    timeline: list["GoogleTimelineEntry"] = field(default_factory=list)
    tracker: ToolCallTracker = field(default_factory=ToolCallTracker)
    tool_buffers: dict[str, _GoogleToolBuffer] = field(default_factory=dict)
    active_tool_index: int | None = None
    tool_counter: int = 0
    estimated_tokens: int = 0
    usage_metadata: types.GenerateContentResponseUsageMetadata | None = None
    last_chunk: types.GenerateContentResponse | None = None


GoogleTimelineEntry = (
    _GoogleTextTimelineEntry
    | _GoogleReasoningTimelineEntry
    | _GoogleToolTimelineEntry
    | _GoogleSignatureTimelineEntry
)


class GoogleNativeLLM(FastAgentLLM[types.Content, types.Content]):
    """
    Google LLM provider using the native google.genai library.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        web_search_override = kwargs.pop("web_search", None)
        self._web_search_override: bool | None = (
            bool(web_search_override) if isinstance(web_search_override, bool) else None
        )
        super().__init__(provider=Provider.GOOGLE, **kwargs)
        # Initialize the converter
        self._converter = GoogleConverter()
        self._init_reasoning(kwargs)

    @property
    def web_search_supported(self) -> bool:
        """Whether provider-side web search is supported by this model/provider."""
        if self._resolved_model_spec is None:
            return False
        params = self._resolved_model_spec.model_params
        return bool(params and getattr(params, "google_search_supported", False))

    @property
    def web_search_enabled(self) -> bool:
        """Whether provider-side web search is enabled for this LLM instance."""
        if not self.web_search_supported:
            return False
        return self._web_search_override if self._web_search_override is not None else False

    def set_web_search_enabled(self, value: bool | None) -> None:
        if value is None:
            self._web_search_override = None
            return
        if not self.web_search_supported:
            raise ValueError("Current model does not support web search configuration.")
        self._web_search_override = value

    def _init_reasoning(self, kwargs: dict) -> None:
        """Wire up reasoning/thinking from kwargs or config."""
        raw_setting = kwargs.get("reasoning_effort")
        model_name = self.default_request_params.model or DEFAULT_GOOGLE_MODEL

        if raw_setting is None:
            google_cfg = getattr(getattr(self.context, "config", None), "google", None)
            if google_cfg:
                raw_setting = (
                    google_cfg.get("reasoning")
                    if isinstance(google_cfg, Mapping)
                    else getattr(google_cfg, "reasoning", None)
                )

        reasoning_mode = self._get_model_reasoning(model_name)
        spec = self._get_model_reasoning_effort_spec(model_name)

        if raw_setting is not None and reasoning_mode != "google_thinking":
            self.logger.warning(
                "Reasoning setting ignored for model without Google thinking support."
            )
            raw_setting = None

        if raw_setting is None and reasoning_mode == "google_thinking" and spec and spec.default:
            raw_setting = spec.default

        setting = parse_reasoning_setting(raw_setting)
        if setting is not None:
            try:
                self.set_reasoning_effort(setting)
            except ValueError as exc:
                self.logger.warning(f"Invalid reasoning setting: {exc}")
                if spec and spec.default:
                    self.set_reasoning_effort(spec.default)
                else:
                    self.set_reasoning_effort(None)
        else:
            self.set_reasoning_effort(None)

        if reasoning_mode == "google_thinking":
            self.logger.info(
                f"Google reasoning resolved: {format_reasoning_setting(self.reasoning_effort)}"
            )

    def _resolve_thinking_config(self) -> tuple[int | None, str | None]:
        """Resolve thinking config from reasoning_effort setting.

        Returns:
            (thinking_budget, thinking_level) tuple where:
            - thinking_budget: None if not configured, 0 to disable, -1 for auto,
              or a positive token count for explicit budgets.
            - thinking_level: SDK ThinkingLevel name (MINIMAL/LOW/MEDIUM/HIGH)
              when an effort level is selected, None otherwise.
        """
        setting = self.reasoning_effort
        if setting is None:
            return (None, None)

        thinking_budget: int | None
        thinking_level: str | None
        match setting.kind:
            case "toggle":
                thinking_budget, thinking_level = (-1 if setting.value else 0), None
            case "budget" if isinstance(setting.value, int):
                thinking_budget, thinking_level = max(0, setting.value), None
            case "effort":
                thinking_budget, thinking_level = _google_thinking_effort_config(
                    strip_casefold(str(setting.value))
                )
            case _:
                thinking_budget, thinking_level = None, None
        return thinking_budget, thinking_level

    def _vertex_cfg(self) -> tuple[bool, str | None, str | None]:
        """(enabled, project_id, location) for Vertex config; supports dict/mapping or object."""
        google_cfg = getattr(getattr(self.context, "config", None), "google", None)
        vertex = (
            (google_cfg or {}).get("vertex_ai")
            if isinstance(google_cfg, Mapping)
            else getattr(google_cfg, "vertex_ai", None)
        )
        if not vertex:
            return (False, None, None)
        if isinstance(vertex, Mapping):
            return (bool(vertex.get("enabled")), vertex.get("project_id"), vertex.get("location"))
        return (
            bool(getattr(vertex, "enabled", False)),
            getattr(vertex, "project_id", None),
            getattr(vertex, "location", None),
        )

    def _resolve_model_name(self, model: str) -> str:
        """Resolve model name; for Vertex, expand first-party short ids.

        * If the caller passes a full publisher resource name, it is respected as-is.
        * If Vertex is not enabled, the short id is returned unchanged (Developer API path).
        * If Vertex is enabled, short first-party Google model ids are expanded under
          `publishers/google`.
        * Known partner model ids such as Anthropic Claude are left untouched so Vertex can
          resolve them using the provider-native short model name from the docs.
        """
        # Fully-qualified publisher / model resource: do not rewrite.
        if model.startswith(("projects/", "publishers/")) or "/publishers/" in model:
            return model

        enabled, project_id, location = self._vertex_cfg()
        # Developer API path: return the short model id unchanged.
        if not (enabled and project_id and location):
            return model

        normalized = strip_casefold(model)
        if normalized.startswith(_GOOGLE_VERTEX_PARTNER_MODEL_PREFIXES):
            return model

        return f"projects/{project_id}/locations/{location}/publishers/google/models/{model}"

    def _initialize_google_client(self) -> genai.Client:
        """
        Initializes the google.genai client.

        Reads Google API key or Vertex AI configuration from context config.
        """
        try:
            # Prefer Vertex AI (ADC/IAM) if enabled. This path must NOT require an API key.
            vertex_enabled, project_id, location = self._vertex_cfg()
            if vertex_enabled:
                return genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                    # http_options=types.HttpOptions(api_version='v1')
                )

            # Otherwise, default to Gemini Developer API (API key required).
            api_key = self._api_key()
            if not api_key:
                raise ProviderKeyError(
                    "Google API key not found.",
                    "Please configure your Google API key.",
                )

            return genai.Client(
                api_key=api_key,
                # http_options=types.HttpOptions(api_version='v1')
            )
        except Exception as e:
            # Catch potential initialization errors and raise ProviderKeyError
            raise ProviderKeyError("Failed to initialize Google GenAI client.", str(e)) from e

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Google-specific default parameters."""
        chosen_model = (
            self._resolve_default_model_name(kwargs.get("model"), DEFAULT_GOOGLE_MODEL)
            or DEFAULT_GOOGLE_MODEL
        )
        # Gemini models have different max output token limits; for example,
        # gemini-2.0-flash only supports up to 8192 output tokens.
        resolved_model = self._resolved_model_spec
        if (
            resolved_model is not None
            and chosen_model == resolved_model.wire_model_name
            and resolved_model.max_output_tokens is not None
        ):
            max_tokens = resolved_model.max_output_tokens
        else:
            max_tokens = ModelDatabase.get_max_output_tokens(chosen_model) or 65536

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,  # System instruction will be mapped in _google_completion
            parallel_tool_calls=True,  # Assume parallel tool calls are supported by default with native API
            max_iterations=DEFAULT_MAX_ITERATIONS,
            use_history=True,
            # Pick a safe default per model (e.g. gemini-2.0-flash is limited to 8192).
            maxTokens=max_tokens,
            # Include other relevant default parameters
        )

    async def _stream_generate_content(
        self,
        *,
        model: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        client: genai.Client,
    ) -> types.GenerateContentResponse | None:
        """Stream Gemini responses and return the final aggregated completion."""
        capture_base = stream_capture_filename(self.chat_turn())
        save_stream_request(
            capture_base,
            {
                "model": model,
                "contents": contents,
                "config": config,
            },
        )
        try:
            response_stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=cast("types.ContentListUnion", contents),
                config=config,
            )
        except AttributeError:
            # Older SDKs might not expose streaming; fall back to non-streaming.
            return None
        except errors.APIError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "Google streaming failed during setup; falling back to non-streaming",
                exc_info=exc,
            )
            return None

        return await self._consume_google_stream(
            response_stream,
            model=model,
            capture_base=capture_base,
        )

    @staticmethod
    def _append_google_text_timeline(
        timeline: list[GoogleTimelineEntry],
        text: str,
    ) -> None:
        if not text:
            return
        if timeline and isinstance(timeline[-1], _GoogleTextTimelineEntry):
            timeline[-1].text += text
            return
        timeline.append(_GoogleTextTimelineEntry(text=text))

    @staticmethod
    def _append_google_reasoning_timeline(
        timeline: list[GoogleTimelineEntry],
        text: str,
        thought_signature: bytes | None = None,
    ) -> None:
        if not text:
            return
        if (
            timeline
            and isinstance(timeline[-1], _GoogleReasoningTimelineEntry)
            and timeline[-1].thought_signature == thought_signature
        ):
            timeline[-1].text += text
            return
        timeline.append(
            _GoogleReasoningTimelineEntry(
                text=text,
                thought_signature=thought_signature,
            )
        )

    @staticmethod
    def _serialize_google_tool_args(args: object) -> str:
        try:
            return json.dumps(args, separators=(",", ":"))
        except Exception:
            return str(args)

    def _start_google_tool_stream(
        self,
        *,
        tracker: ToolCallTracker,
        tool_buffers: dict[str, _GoogleToolBuffer],
        timeline: list[GoogleTimelineEntry],
        tool_index: int,
        tool_name: str,
        provider_call_id: str | None = None,
        thought_signature: bytes | None = None,
    ) -> _GoogleToolBuffer:
        tool_use_id = provider_call_id or f"tool_{self.chat_turn()}_{tool_index}"
        state = tracker.register(
            tool_use_id=tool_use_id,
            name=tool_name,
            index=tool_index,
        )
        buffer = _GoogleToolBuffer(
            tool_use_id=state.tool_use_id,
            name=state.name,
            provider_call_id=tool_use_id,
            thought_signature=thought_signature,
        )
        tool_buffers[state.tool_use_id] = buffer
        self._notify_tool_stream_listeners(
            "start",
            {
                "tool_name": state.name,
                "tool_use_id": state.tool_use_id,
                "index": tool_index,
            },
        )
        state.start_notified = True
        timeline.append(_GoogleToolTimelineEntry(tool_use_id=state.tool_use_id))
        return buffer

    def _close_google_tool_stream(
        self,
        *,
        tracker: ToolCallTracker,
        tool_index: int,
    ) -> None:
        state = tracker.close(index=tool_index)
        if state is None:
            return
        self._notify_tool_stream_listeners(
            "stop",
            {
                "tool_name": state.name,
                "tool_use_id": state.tool_use_id,
                "index": tool_index,
            },
        )

    def _build_google_final_response(
        self,
        *,
        last_chunk: types.GenerateContentResponse | None,
        usage_metadata: types.GenerateContentResponseUsageMetadata | None,
        timeline: list[GoogleTimelineEntry],
        tool_buffers: dict[str, _GoogleToolBuffer],
    ) -> types.GenerateContentResponse | None:
        if not timeline and last_chunk is None:
            return None

        final_parts: list[types.Part] = []
        for entry in timeline:
            part = self._google_timeline_part(entry, tool_buffers)
            if part is None:
                continue
            final_parts.append(part)

        final_content = types.Content(role="model", parts=final_parts)
        final_response = self._google_response_with_content(last_chunk, final_content)

        if usage_metadata:
            final_response.usage_metadata = usage_metadata

        return final_response

    def _google_timeline_part(
        self,
        entry: GoogleTimelineEntry,
        tool_buffers: dict[str, _GoogleToolBuffer],
    ) -> types.Part | None:
        if isinstance(entry, _GoogleTextTimelineEntry):
            return types.Part.from_text(text=entry.text)
        if isinstance(entry, _GoogleReasoningTimelineEntry):
            return types.Part(
                text=entry.text,
                thought=True,
                thought_signature=entry.thought_signature,
            )
        if isinstance(entry, _GoogleSignatureTimelineEntry):
            return types.Part(text="", thought_signature=entry.thought_signature)

        tool_buffer = tool_buffers.get(entry.tool_use_id)
        if tool_buffer is None:
            return None
        try:
            args_obj = json.loads(tool_buffer.buffer) if tool_buffer.buffer else {}
        except json.JSONDecodeError:
            args_obj = {"__raw": tool_buffer.buffer}
        return types.Part(
            function_call=types.FunctionCall(
                id=tool_buffer.provider_call_id or tool_buffer.tool_use_id,
                name=str(tool_buffer.name or "tool"),
                args=args_obj,
            ),
            thought_signature=tool_buffer.thought_signature,
        )

    @staticmethod
    def _google_response_with_content(
        last_chunk: types.GenerateContentResponse | None,
        final_content: types.Content,
    ) -> types.GenerateContentResponse:
        if last_chunk is not None:
            final_response = last_chunk.model_copy(deep=True)
            candidates = final_response.candidates or []
            if candidates:
                final_candidate = candidates[0]
                final_candidate.content = final_content
            else:
                final_response.candidates = [types.Candidate(content=final_content)]
        else:
            final_response = types.GenerateContentResponse(
                candidates=[types.Candidate(content=final_content)]
            )

        return final_response

    def _record_google_stream_chunk(
        self,
        state: _GoogleStreamState,
        chunk: types.GenerateContentResponse,
        *,
        capture_base,
    ) -> types.Candidate | None:
        save_stream_chunk(capture_base, chunk)
        state.last_chunk = chunk
        if getattr(chunk, "usage_metadata", None):
            state.usage_metadata = chunk.usage_metadata

        candidates = chunk.candidates
        if not candidates:
            return None
        return candidates[0]

    def _handle_google_stream_part(
        self,
        state: _GoogleStreamState,
        part: types.Part,
    ) -> None:
        self._handle_google_text_part(state, part)
        self._handle_google_function_call_part(state, part)
        self._handle_google_signature_part(state, part)

    def _handle_google_text_part(
        self,
        state: _GoogleStreamState,
        part: types.Part,
    ) -> None:
        text = getattr(part, "text", None) or ""
        if not text:
            return

        if getattr(part, "thought", False):
            self._notify_stream_listeners(StreamChunk(text=text, is_reasoning=True))
            self._append_google_reasoning_timeline(
                state.timeline,
                text,
                part.thought_signature,
            )
            return

        self._append_google_text_timeline(state.timeline, text)
        state.estimated_tokens = self._emit_stream_text_delta(
            text=text,
            model=state.model,
            estimated_tokens=state.estimated_tokens,
        )

    def _handle_google_function_call_part(
        self,
        state: _GoogleStreamState,
        part: types.Part,
    ) -> None:
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            return

        if state.active_tool_index is None:
            state.active_tool_index = state.tool_counter
            state.tool_counter += 1
            self._start_google_tool_stream(
                tracker=state.tracker,
                tool_buffers=state.tool_buffers,
                timeline=state.timeline,
                tool_index=state.active_tool_index,
                tool_name=getattr(function_call, "name", None) or "tool",
                provider_call_id=getattr(function_call, "id", None),
                thought_signature=part.thought_signature,
            )

        self._update_google_tool_buffer(state, function_call, part)

    def _update_google_tool_buffer(
        self,
        state: _GoogleStreamState,
        function_call: Any,
        part: types.Part,
    ) -> None:
        active_tool_index = state.active_tool_index
        if active_tool_index is None:
            return

        tool_state = state.tracker.resolve_open(index=active_tool_index)
        if tool_state is None:
            return

        thought_signature = part.thought_signature
        tool_buffer = state.tool_buffers.get(tool_state.tool_use_id)
        if tool_buffer is None:
            tool_buffer = _GoogleToolBuffer(
                tool_use_id=tool_state.tool_use_id,
                name=tool_state.name,
                thought_signature=thought_signature,
            )
            state.tool_buffers[tool_state.tool_use_id] = tool_buffer
        if thought_signature is not None:
            tool_buffer.thought_signature = thought_signature

        serialized_args = self._serialize_google_tool_args(
            getattr(function_call, "args", None) or {}
        )
        delta = self._google_tool_buffer_delta(tool_buffer.buffer, serialized_args)
        tool_buffer.buffer = serialized_args
        if delta:
            self._notify_google_tool_delta(tool_buffer, active_tool_index, delta)

    @staticmethod
    def _google_tool_buffer_delta(previous: str, current: str) -> str:
        if current.startswith(previous):
            return current.removeprefix(previous)
        return current

    def _notify_google_tool_delta(
        self,
        tool_buffer: _GoogleToolBuffer,
        tool_index: int,
        delta: str,
    ) -> None:
        self._notify_tool_stream_listeners(
            "delta",
            {
                "tool_name": tool_buffer.name,
                "tool_use_id": tool_buffer.tool_use_id,
                "index": tool_index,
                "chunk": delta,
            },
        )

    @staticmethod
    def _handle_google_signature_part(
        state: _GoogleStreamState,
        part: types.Part,
    ) -> None:
        thought_signature = part.thought_signature
        if (
            thought_signature is not None
            and not getattr(part, "function_call", None)
            and not getattr(part, "text", None)
        ):
            state.timeline.append(
                _GoogleSignatureTimelineEntry(
                    thought_signature=thought_signature,
                )
            )

    def _close_google_tool_if_finished(
        self,
        state: _GoogleStreamState,
        candidate: types.Candidate,
    ) -> None:
        finish_reason = getattr(candidate, "finish_reason", None)
        if not finish_reason or state.active_tool_index is None:
            return

        finish_value = str(finish_reason).split(".")[-1].upper()
        if finish_value not in {"FUNCTION_CALL", "STOP"}:
            return

        self._close_google_tool_stream(
            tracker=state.tracker,
            tool_index=state.active_tool_index,
        )
        state.active_tool_index = None

    def _close_google_active_tool(self, state: _GoogleStreamState) -> None:
        if state.active_tool_index is None:
            return
        self._close_google_tool_stream(
            tracker=state.tracker,
            tool_index=state.active_tool_index,
        )
        state.active_tool_index = None

    @staticmethod
    def _raise_if_google_tools_incomplete(tracker: ToolCallTracker) -> None:
        incomplete_tools = tracker.incomplete()
        if incomplete_tools:
            raise RuntimeError(
                format_incomplete_tool_call_error(
                    [f"{tool.name}:{tool.tool_use_id}" for tool in incomplete_tools]
                )
            )

    def _google_structured_response_options(
        self,
        request_params: RequestParams,
        response_mime_type: str | None,
        response_schema: object | None,
    ) -> tuple[str | None, object | None]:
        if request_params.structured_schema and response_schema is None:
            return (
                response_mime_type or "application/json",
                self._converter._clean_schema_for_google(request_params.structured_schema),
            )
        return response_mime_type, response_schema

    def _google_suppress_tools(
        self,
        request_params: RequestParams,
        tools: list[McpTool] | None,
        suppress_tools: bool | None,
    ) -> bool:
        if suppress_tools is not None:
            return suppress_tools
        return (
            self._has_structured_intent(request_params)
            and bool(tools)
            and self._resolve_structured_tool_policy(request_params) == "no_tools"
        )

    def _google_available_tools(
        self,
        tools: list[McpTool] | None,
        *,
        suppress_tools: bool,
    ) -> types.ToolListUnion:
        available_tools: types.ToolListUnion = []
        if tools and not suppress_tools:
            available_tools.extend(self._converter.convert_to_google_tools(tools))
        if self.web_search_enabled:
            available_tools.append(types.Tool(google_search=types.GoogleSearch()))
        return available_tools

    def _google_generate_content_config(
        self,
        request_params: RequestParams,
        *,
        tools: list[McpTool] | None,
        available_tools: types.ToolListUnion,
        response_mime_type: str | None,
        response_schema: object | None,
        suppress_tools: bool,
    ) -> types.GenerateContentConfig:
        thinking_budget, thinking_level = self._resolve_thinking_config()
        generate_content_config = self._converter.convert_request_params_to_google_config(
            request_params,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )
        if response_mime_type:
            generate_content_config.response_mime_type = response_mime_type
        if response_schema is not None:
            generate_content_config.response_schema = response_schema
        if available_tools:
            generate_content_config.tools = available_tools
            if tools and not suppress_tools:
                generate_content_config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.AUTO,
                    ),
                    include_server_side_tool_invocations=bool(self.web_search_enabled),
                )
        return generate_content_config

    async def _call_google_generate_content(
        self,
        *,
        client: genai.Client,
        model_name: str,
        conversation_history: list[types.Content],
        generate_content_config: types.GenerateContentConfig,
        response_mime_type: str | None,
        response_schema: object | None,
    ) -> types.GenerateContentResponse:
        async with client.aio:
            api_response = None
            streaming_supported = response_schema is None and response_mime_type is None
            if streaming_supported:
                api_response = await self._stream_generate_content(
                    model=model_name,
                    contents=conversation_history,
                    config=generate_content_config,
                    client=client,
                )
            if api_response is None:
                api_response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=cast("types.ContentListUnion", conversation_history),
                    config=generate_content_config,
                )
            return api_response

    def _track_google_usage(
        self,
        api_response: types.GenerateContentResponse,
        model_name: str,
    ) -> None:
        usage_metadata = getattr(api_response, "usage_metadata", None)
        if not usage_metadata:
            return
        try:
            turn_usage = TurnUsage.from_google(usage_metadata, model_name)
            self._finalize_turn_usage(turn_usage)
        except Exception as e:
            self.logger.warning(f"Failed to track usage: {e}")

    def _google_model_response_parts(
        self,
        candidate: types.Candidate,
        api_response: types.GenerateContentResponse,
        *,
        response_mime_type: str | None,
        response_schema: object | None,
    ) -> list[ContentBlock | CallToolRequestParams]:
        candidate_content = candidate.content
        if candidate_content is None:
            model_response_content_parts: list[ContentBlock | CallToolRequestParams] = []
        else:
            model_response_content_parts = self._converter.convert_from_google_content(
                candidate_content
            )

        model_response_content_parts = self._apply_google_grounding_citations(
            model_response_content_parts,
            getattr(candidate, "grounding_metadata", None),
        )
        if not model_response_content_parts and (response_schema or response_mime_type):
            structured_text = self._extract_structured_response_text(api_response)
            if structured_text:
                model_response_content_parts.append(TextContent(type="text", text=structured_text))
        return model_response_content_parts

    def _apply_google_grounding_citations(
        self,
        parts: list[ContentBlock | CallToolRequestParams],
        grounding_metadata: Any,
    ) -> list[ContentBlock | CallToolRequestParams]:
        if not grounding_metadata:
            return parts

        text_parts = [part for part in parts if isinstance(part, TextContent)]
        if not text_parts:
            return parts

        cited_text = self._apply_citations(
            "".join(part.text for part in text_parts), grounding_metadata
        )
        new_parts: list[ContentBlock | CallToolRequestParams] = []
        replaced = False
        for part in parts:
            if isinstance(part, TextContent):
                if not replaced:
                    new_parts.append(TextContent(type="text", text=cited_text))
                    replaced = True
                continue
            new_parts.append(part)
        return new_parts

    @staticmethod
    def _provider_google_tool_calls(
        candidate_content: types.Content | None,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        provider_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
        if candidate_content is None or candidate_content.parts is None:
            return provider_tool_calls

        for content_part in candidate_content.parts:
            function_call = content_part.function_call
            if function_call is None:
                continue
            tool_name = function_call.name or "unknown_function"
            tool_args = function_call.args or {}
            tool_call_id = function_call.id or secrets.token_hex(3)[:5]
            if function_call.id is None:
                function_call.id = tool_call_id
            provider_tool_calls.append((tool_call_id, tool_name, dict(tool_args)))
        return provider_tool_calls

    @staticmethod
    def _google_response_text_and_tool_params(
        parts: list[ContentBlock | CallToolRequestParams],
        provider_tool_calls: list[tuple[str, str, dict[str, Any]]],
    ) -> tuple[list[ContentBlock], list[CallToolRequestParams]]:
        responses: list[ContentBlock] = []
        tool_calls_to_execute: list[CallToolRequestParams] = []
        for part in parts:
            if isinstance(part, TextContent):
                responses.append(part)
            elif isinstance(part, CallToolRequestParams) and not provider_tool_calls:
                tool_calls_to_execute.append(part)

        if provider_tool_calls:
            tool_calls_to_execute = [
                CallToolRequestParams(name=name, arguments=args)
                for _, name, args in provider_tool_calls
            ]
        return responses, tool_calls_to_execute

    def _google_tool_call_requests(
        self,
        tool_calls_to_execute: list[CallToolRequestParams],
        provider_tool_calls: list[tuple[str, str, dict[str, Any]]],
    ) -> dict[str, CallToolRequest] | None:
        if not tool_calls_to_execute:
            return None

        tool_calls: dict[str, CallToolRequest] = {}
        for index, tool_call_params in enumerate(tool_calls_to_execute):
            tool_call_request = CallToolRequest(method="tools/call", params=tool_call_params)
            tool_call_id = (
                provider_tool_calls[index][0] if provider_tool_calls else secrets.token_hex(3)[:5]
            )
            tool_calls[tool_call_id] = tool_call_request
        self.logger.debug("Tool call results processed.")
        return tool_calls

    @staticmethod
    def _google_assistant_with_reasoning(
        responses: list[ContentBlock],
        *,
        stop_reason: LlmStopReason,
        tool_calls: dict[str, CallToolRequest] | None,
        candidate_content: types.Content | None,
    ) -> PromptMessageExtended:
        assistant = Prompt.assistant(*responses, stop_reason=stop_reason, tool_calls=tool_calls)
        reasoning_blocks = GoogleNativeLLM._extract_reasoning_blocks(candidate_content)
        if reasoning_blocks:
            channels = dict(assistant.channels or {})
            channels[REASONING] = reasoning_blocks
            assistant.channels = channels
        return assistant

    async def _consume_google_stream(
        self,
        response_stream,
        *,
        model: str,
        capture_base=None,
    ) -> types.GenerateContentResponse | None:
        """Consume the async streaming iterator and aggregate the final response."""
        state = _GoogleStreamState(model=model)

        try:
            # Cancellation is handled via asyncio.Task.cancel() which raises CancelledError
            async for chunk in response_stream:
                candidate = self._record_google_stream_chunk(
                    state,
                    chunk,
                    capture_base=capture_base,
                )
                if candidate is None:
                    continue
                content = getattr(candidate, "content", None)
                if content is None or not getattr(content, "parts", None):
                    continue

                for part in content.parts:
                    self._handle_google_stream_part(state, part)

                self._close_google_tool_if_finished(state, candidate)
        finally:
            stream_close = getattr(response_stream, "aclose", None)
            if callable(stream_close):
                with suppress(Exception):
                    await stream_close()

        self._close_google_active_tool(state)
        self._raise_if_google_tools_incomplete(state.tracker)

        return self._build_google_final_response(
            last_chunk=state.last_chunk,
            usage_metadata=state.usage_metadata,
            timeline=state.timeline,
            tool_buffers=state.tool_buffers,
        )

    async def _google_completion(
        self,
        message: list[types.Content] | None,
        request_params: RequestParams | None = None,
        tools: list[McpTool] | None = None,
        *,
        response_mime_type: str | None = None,
        response_schema: object | None = None,
        suppress_tools: bool | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using Google's generate_content API and available tools.
        """
        request_params = self.get_request_params(request_params=request_params)
        response_mime_type, response_schema = self._google_structured_response_options(
            request_params,
            response_mime_type,
            response_schema,
        )

        # Caller supplies the full set of messages to send (history + turn)
        conversation_history: list[types.Content] = list(message or [])

        self.logger.debug(f"Google completion requested with messages: {conversation_history}")
        self._log_chat_progress(self.chat_turn(), model=request_params.model)

        suppress_tools = self._google_suppress_tools(request_params, tools, suppress_tools)
        available_tools = self._google_available_tools(tools, suppress_tools=suppress_tools)
        generate_content_config = self._google_generate_content_config(
            request_params,
            tools=tools,
            available_tools=available_tools,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            suppress_tools=suppress_tools,
        )

        client = self._initialize_google_client()
        model_name = self._resolve_model_name(request_params.model or DEFAULT_GOOGLE_MODEL)
        try:
            api_response = await self._call_google_generate_content(
                client=client,
                model_name=model_name,
                conversation_history=conversation_history,
                generate_content_config=generate_content_config,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            )
            self.logger.debug("Google generate_content response:", data=api_response)
            self._track_google_usage(api_response, model_name)
        except errors.APIError as e:
            # Handle specific Google API errors
            self.logger.error(f"Google API Error: {e.code} - {e.message}")
            raise ProviderKeyError(f"Google API Error: {e.code}", e.message or "") from e
        except Exception as e:
            self.logger.error(f"Error during Google generate_content call: {e}")
            raise

        if not api_response.candidates:
            self.logger.debug("No candidates returned.")
            return Prompt.assistant(stop_reason=LlmStopReason.END_TURN)

        candidate = api_response.candidates[0]
        candidate_content = candidate.content
        model_response_content_parts = self._google_model_response_parts(
            candidate,
            api_response,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )
        provider_tool_calls = self._provider_google_tool_calls(candidate_content)
        if candidate_content is not None:
            conversation_history.append(candidate_content)

        responses, tool_calls_to_execute = self._google_response_text_and_tool_params(
            model_response_content_parts,
            provider_tool_calls,
        )
        tool_calls = self._google_tool_call_requests(tool_calls_to_execute, provider_tool_calls)
        if tool_calls_to_execute:
            stop_reason = LlmStopReason.TOOL_USE
        else:
            stop_reason = self._map_finish_reason(getattr(candidate, "finish_reason", None))

        self.history.set(conversation_history)

        self._log_chat_finished(model=model_name)  # Use resolved model name
        return self._google_assistant_with_reasoning(
            responses,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            candidate_content=candidate_content,
        )

    #        return responses  # Return the accumulated responses (fast-agent content types)

    @staticmethod
    def _extract_reasoning_blocks(content: types.Content | None) -> list[TextContent]:
        if content is None or content.parts is None:
            return []

        reasoning_segments = [
            part.text for part in content.parts if getattr(part, "thought", False) and part.text
        ]

        reasoning_text = "".join(reasoning_segments).strip()
        if not reasoning_text:
            return []
        return [TextContent(type="text", text=reasoning_text)]

    @staticmethod
    def _extract_structured_response_text(
        api_response: types.GenerateContentResponse,
    ) -> str | None:
        try:
            text = api_response.text
        except Exception:
            text = None
        if text:
            return text

        try:
            parsed = api_response.parsed
        except Exception:
            parsed = None
        if parsed is None:
            return None
        if isinstance(parsed, str):
            return parsed
        try:
            return json.dumps(parsed)
        except Exception:
            return str(parsed)

    def _prepare_structured_request(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams,
        tools: list[McpTool] | None = None,
    ) -> tuple[list[PromptMessageExtended], RequestParams]:
        if not self._should_defer_structured_schema_for_tools(messages, request_params, tools):
            return messages, request_params
        return messages, request_params.model_copy(update={"structured_schema": None})

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[McpTool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """
        Provider-specific prompt application.
        Templates are handled by the agent; messages already include them.
        """
        request_params = self.get_request_params(request_params=request_params)

        # Determine the last message
        last_message = multipart_messages[-1]

        if last_message.role == "assistant":
            # No generation required; the provided assistant message is the output
            return last_message

        conversation_history = self._google_conversation_history(multipart_messages, request_params)

        return await self._google_completion(
            conversation_history,
            request_params=request_params,
            tools=tools,
            suppress_tools=self._should_suppress_tools_for_structured_final(
                multipart_messages, request_params, tools
            ),
        )

    def _google_conversation_history(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams,
    ) -> list[types.Content]:
        conversation_history: list[types.Content] = []
        provider_history = self.history.get()
        if request_params.use_history and provider_history and len(multipart_messages) > 1:
            conversation_history.extend(provider_history)
            conversation_history.extend(self._new_messages_since_last_assistant(multipart_messages))
        elif request_params.use_history and len(multipart_messages) > 1:
            conversation_history.extend(self._convert_to_provider_format(multipart_messages[:-1]))
        conversation_history.extend(self._google_turn_messages(multipart_messages))
        return conversation_history

    def _new_messages_since_last_assistant(
        self,
        multipart_messages: list[PromptMessageExtended],
    ) -> list[types.Content]:
        last_assistant_index = -1
        for idx, message in enumerate(multipart_messages):
            if message.role == "assistant":
                last_assistant_index = idx

        new_messages = multipart_messages[last_assistant_index + 1 : -1]
        if not new_messages:
            return []
        return self._convert_to_provider_format(new_messages)

    def _google_turn_messages(
        self,
        multipart_messages: list[PromptMessageExtended],
    ) -> list[types.Content]:
        last_message = multipart_messages[-1]
        turn_messages = self._google_tool_result_messages(multipart_messages, last_message)
        if last_message.content:
            turn_messages.extend(self._converter.convert_to_google_content([last_message]))
        if not turn_messages:
            turn_messages.append(types.Content(role="user", parts=[types.Part.from_text(text="")]))
        return turn_messages

    def _google_tool_result_messages(
        self,
        multipart_messages: list[PromptMessageExtended],
        last_message: PromptMessageExtended,
    ) -> list[types.Content]:
        if not last_message.tool_results:
            return []

        id_to_name = self._last_assistant_tool_names(multipart_messages)
        tool_results_pairs: list[GoogleToolResult] = [
            (id_to_name.get(call_id, "tool"), call_id, result)
            for call_id, result in last_message.tool_results.items()
        ]
        if not tool_results_pairs:
            return []
        return self._converter.convert_function_results_to_google(tool_results_pairs)

    @staticmethod
    def _last_assistant_tool_names(
        multipart_messages: list[PromptMessageExtended],
    ) -> dict[str, str]:
        for message in reversed(multipart_messages):
            if message.role != "assistant" or not message.tool_calls:
                continue

            id_to_name: dict[str, str] = {}
            for call_id, call in message.tool_calls.items():
                with suppress(Exception):
                    id_to_name[call_id] = call.params.name
            return id_to_name
        return {}

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[types.Content]:
        """
        Convert PromptMessageExtended list to Google types.Content format.
        This is called fresh on every API call from _convert_to_provider_format().

        Args:
            messages: List of PromptMessageExtended objects

        Returns:
            List of Google types.Content objects
        """
        # Build mapping of tool call ID to tool name from all assistant messages in the history
        id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                for call_id, call in msg.tool_calls.items():
                    with suppress(Exception):
                        id_to_name[call_id] = call.params.name

        converted: list[types.Content] = []
        for msg in messages:
            if msg.tool_results:
                tool_results_pairs: list[GoogleToolResult] = []
                for call_id, result in msg.tool_results.items():
                    tool_name = id_to_name.get(call_id, "tool")
                    tool_results_pairs.append((tool_name, call_id, result))

                if tool_results_pairs:
                    converted.extend(
                        self._converter.convert_function_results_to_google(tool_results_pairs)
                    )
                # If there is also direct content in this message, convert and append it
                if msg.content:
                    converted.extend(self._converter.convert_to_google_content([msg]))
            else:
                converted.extend(self._converter.convert_to_google_content([msg]))

        return converted

    def _map_finish_reason(self, finish_reason: object) -> LlmStopReason:
        """Map Google finish reasons to LlmStopReason robustly."""
        # Normalize to string if it's an enum-like object
        reason = None
        try:
            reason = str(finish_reason) if finish_reason is not None else None
        except Exception:
            reason = None

        if not reason:
            return LlmStopReason.END_TURN

        # Extract last token after any dots or enum prefixes
        key = reason.split(".")[-1].upper()

        # Some SDKs include OTHER, LANGUAGE, GROUNDING, UNSPECIFIED, etc.
        return _GOOGLE_FINISH_REASON_MAP.get(key, LlmStopReason.ERROR)

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages,
        model,
        request_params=None,
    ):
        """
        Provider-specific structured output implementation.
        Note: Message history is managed by base class and converted via
        _convert_to_provider_format() on each call.
        """
        import json

        # Determine the last message
        last_message = multipart_messages[-1] if multipart_messages else None

        # If the last message is an assistant message, attempt to parse its JSON and return
        if last_message and last_message.role == "assistant":
            assistant_text = last_message.last_text()
            if assistant_text:
                try:
                    json_data = json.loads(assistant_text)
                    validated_model = model.model_validate(json_data)
                    return validated_model, last_message
                except (json.JSONDecodeError, Exception) as e:
                    self.logger.warning(
                        f"Failed to parse assistant message as structured response: {e}"
                    )
                    return None, last_message

        # Prepare request params
        request_params = self.get_request_params(request_params)

        # Google genai accepts Pydantic models directly for response_schema and
        # applies its own schema processing. Use that model route instead of
        # eagerly converting to a dict so Pydantic and raw-schema inputs remain
        # distinct and match downstream SDK behavior.
        response_schema = model

        # Convert the last user message to provider-native content for the current turn
        turn_messages: list[types.Content] = []
        if last_message:
            turn_messages = self._converter.convert_to_google_content([last_message])

        # Delegate to unified completion with structured options enabled (no tools)
        assistant_msg = await self._google_completion(
            turn_messages,
            request_params=request_params,
            tools=None,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        # Parse using shared helper for consistency
        parsed, _ = self._structured_from_multipart(assistant_msg, model)
        return parsed, assistant_msg

    async def _apply_prompt_provider_specific_structured_schema(
        self,
        multipart_messages: list[PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        last_message = multipart_messages[-1] if multipart_messages else None

        if last_message and last_message.role == "assistant":
            return self._structured_schema_from_multipart(last_message, schema)

        request_params = self.get_request_params(request_params)
        response_schema = self._converter._clean_schema_for_google(schema)

        turn_messages: list[types.Content] = []
        if last_message:
            turn_messages = self._converter.convert_to_google_content([last_message])

        assistant_msg = await self._google_completion(
            turn_messages,
            request_params=request_params,
            tools=None,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        return self._structured_schema_from_multipart(assistant_msg, schema)

    def _apply_citations(self, text: str, grounding_metadata: Any) -> str:
        """Apply citations and footnotes using grounding metadata."""
        supports = getattr(grounding_metadata, "grounding_supports", None)
        chunks = getattr(grounding_metadata, "grounding_chunks", None)
        if not supports or not chunks:
            return text

        try:
            for support in self._citation_supports_by_descending_end(supports):
                end_index = self._citation_end_index(support)
                if not end_index:
                    continue

                citation_string = self._citation_string(support, chunks)
                if citation_string:
                    text = text[:end_index] + citation_string + text[end_index:]
        except Exception as e:
            self.logger.warning(
                f"Failed to process Google Search grounding metadata citations: {e}"
            )

        return text

    def _citation_supports_by_descending_end(self, supports: Any) -> list[Any]:
        return sorted(supports, key=self._citation_end_index, reverse=True)

    @staticmethod
    def _citation_end_index(support_item: Any) -> int:
        segment = getattr(support_item, "segment", None)
        if segment is None:
            return 0
        value = getattr(segment, "end_index", None)
        if value is None:
            value = getattr(segment, "endIndex", None)
        return int(value) if value is not None else 0

    def _citation_string(self, support: Any, chunks: Any) -> str | None:
        citation_links = [
            link
            for index in self._citation_chunk_indices(support)
            if (link := self._citation_link(index, chunks)) is not None
        ]
        if not citation_links:
            return None
        return " " + ", ".join(citation_links)

    @staticmethod
    def _citation_chunk_indices(support: Any) -> list[int]:
        indices = getattr(support, "grounding_chunk_indices", None)
        if not indices:
            indices = getattr(support, "groundingChunkIndices", None)
        return list(indices or [])

    @staticmethod
    def _citation_link(index: int, chunks: Any) -> str | None:
        if index >= len(chunks):
            return None
        chunk = chunks[index]
        web = getattr(chunk, "web", None)
        uri = getattr(web, "uri", None) if web else None
        if not uri:
            return None
        return f"[{index + 1}]({uri})"
