import asyncio
import base64
import hashlib
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, cast, runtime_checkable

from anthropic import (
    APIError,
    AsyncAnthropic,
    AuthenticationError,
    transform_schema,
)
from anthropic.lib.streaming import BetaAsyncMessageStream
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaInputJSONDelta,
    BetaMCPToolResultBlock,
    BetaMCPToolUseBlock,
    BetaMessage,
    BetaMessageParam,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRedactedThinkingBlock,
    BetaServerToolUseBlock,
    BetaSignatureDelta,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaTextDelta,
    BetaThinkingBlock,
    BetaThinkingDelta,
    BetaToolParam,
    BetaToolUseBlock,
    BetaToolUseBlockParam,
)
from mcp import Tool
from mcp.types import (
    BlobResourceContents,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    TextContent,
)
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel

from fast_agent.constants import (
    ANTHROPIC_ASSISTANT_RAW_CONTENT,
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_CONTAINER_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    ANTHROPIC_THINKING_BLOCKS,
    FAST_AGENT_SAFETY_DETAILS,
    REASONING,
)
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import ModelT
from fast_agent.llm.fastagent_llm import (
    FastAgentLLM,
    RequestParams,
)
from fast_agent.llm.provider.anthropic.cache_planner import AnthropicCachePlanner
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import (
    ANTHROPIC_FILE_ID_META_KEY,
    AnthropicConverter,
)
from fast_agent.llm.provider.anthropic.web_tools import (
    build_web_tool_params,
    dedupe_preserve_order,
    extract_citation_payloads,
    is_server_tool_trace_payload,
    resolve_web_tools,
    serialize_anthropic_block_payload,
    web_tool_progress_label,
)
from fast_agent.llm.provider.error_utils import build_stream_failure_response
from fast_agent.llm.provider.streaming_timeouts import await_stream_start, enter_stream_with_timeout
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import (
    AUTO_REASONING,
    format_reasoning_setting,
    is_auto_reasoning,
    parse_reasoning_setting,
)
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.structured_output_mode import StructuredOutputMode
from fast_agent.llm.task_budget import (
    format_task_budget_tokens,
    parse_task_budget_tokens,
    validate_task_budget_tokens,
)
from fast_agent.llm.tool_call_errors import format_incomplete_tool_call_error
from fast_agent.llm.tool_tracking import ToolCallTracker
from fast_agent.llm.usage_tracking import usage_from_anthropic
from fast_agent.mcp.mime_utils import DOCUMENT_MIME_TYPES, guess_mime_type, normalize_mime_type
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.provider_management import build_anthropic_provider_managed_mcp_payload
from fast_agent.tool_activity_presentation import build_tool_activity_presentation
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.reasoning_chunk_join import ReasoningTextAccumulator
from fast_agent.utils.text import casefold_text, strip_casefold
from fast_agent.utils.type_narrowing import is_str_object_dict

DEFAULT_ANTHROPIC_MODEL = "sonnet"
STRUCTURED_OUTPUT_TOOL_NAME = "return_structured_output"
STRUCTURED_OUTPUT_BETA = "structured-outputs-2025-11-13"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
TASK_BUDGETS_BETA = "task-budgets-2026-03-13"
TEST_REFUSAL_TRIGGER_ENV = "FAST_AGENT_TEST_REFUSAL_TRIGGERS"

# Explicit 1M context is still opt-in for pre-4.6 Anthropic models.
LONG_CONTEXT_BETA = "context-1m-2025-08-07"

# Beta for fine-grained tool streaming - enables incremental tool input streaming
# https://docs.anthropic.com/en/docs/build-with-claude/tool-use#streaming-tool-inputs
FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
MCP_CLIENT_BETA = "mcp-client-2025-11-20"

# Stream capture mode - when enabled, saves all streaming chunks to files for debugging
# Set FAST_AGENT_LLM_TRACE=1 (or any non-empty value) to enable
STREAM_CAPTURE_ENABLED = bool(os.environ.get("FAST_AGENT_LLM_TRACE"))
STREAM_CAPTURE_DIR = Path("stream-debug")

# Type alias for system field - can be string or list of text blocks with cache control
SystemParam = str | list[BetaTextBlockParam]
CacheTTL = Literal["5m", "1h"]

logger = get_logger(__name__)

_OTEL_STREAM_WRAPPER_WARNED = False
_UNSET_TASK_BUDGET = object()


@runtime_checkable
class _ModelDumpable(Protocol):
    def model_dump(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class _AnthropicToolInputBuffer:
    chunks: list[str] = field(default_factory=list)

    def append(self, chunk: str) -> None:
        self.chunks.append(chunk)

    def joined(self) -> str:
        return "".join(self.chunks)


@dataclass(slots=True)
class _AnthropicCompletionRequest:
    client: Any
    params: RequestParams
    messages: list[BetaMessageParam]
    message_param: BetaMessageParam


@dataclass(slots=True)
class _AnthropicStructuredMode:
    effective_schema: dict[str, Any] | None
    mode: StructuredOutputMode | None
    auto_tool_use_fallback: bool


@dataclass(slots=True)
class _AnthropicStopResult:
    stop_reason: LlmStopReason
    tool_calls: dict[str, CallToolRequest] | None = None
    structured_blocks: list[ContentBlock] = field(default_factory=list)


@dataclass(slots=True)
class _AnthropicProviderPayloads:
    raw_assistant: list[TextContent] = field(default_factory=list)
    server_tools: list[TextContent] = field(default_factory=list)
    citations: list[TextContent] = field(default_factory=list)


@dataclass(slots=True)
class _AnthropicStreamState:
    estimated_tokens: int = 0
    tool_tracker: ToolCallTracker = field(default_factory=ToolCallTracker)
    tool_input_buffers: dict[str, _AnthropicToolInputBuffer] = field(default_factory=dict)
    provider_tool_names: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    thinking_segments: ReasoningTextAccumulator = field(default_factory=ReasoningTextAccumulator)
    streamed_text_segments: list[str] = field(default_factory=list)
    thinking_indices: set[int] = field(default_factory=set)


def _server_tool_preview_chunk(tool_input: object) -> str:
    if not tool_input:
        return "…"
    try:
        preview_chunk = json.dumps(tool_input)
    except Exception:
        return "…"
    if len(preview_chunk) > 120:
        return f"{preview_chunk[:117]}..."
    return preview_chunk


def _mcp_tool_preview_chunk(tool_input: object) -> str:
    if tool_input is None:
        return "…"
    try:
        preview_chunk = json.dumps(tool_input)
    except Exception:
        return "…"
    if len(preview_chunk) > 120:
        return f"{preview_chunk[:117]}..."
    return preview_chunk


def _mcp_tool_result_preview_chunk(result_content: object) -> str:
    if isinstance(result_content, Sequence) and not isinstance(result_content, (str, bytes)):
        for item in result_content:
            text_value = getattr(item, "text", None)
            if isinstance(text_value, str) and text_value:
                preview_chunk = text_value
                break
            if isinstance(item, Mapping):
                item_map = {key: value for key, value in item.items() if isinstance(key, str)}
                mapped_text = item_map.get("text")
                if isinstance(mapped_text, str) and mapped_text:
                    preview_chunk = mapped_text
                    break
        else:
            try:
                preview_chunk = json.dumps(result_content)
            except Exception:
                return "…"
    else:
        try:
            preview_chunk = json.dumps(result_content)
        except Exception:
            return "…"

    if len(preview_chunk) > 120:
        return f"{preview_chunk[:117]}..."
    return preview_chunk or "…"


def _provider_tool_display_name(
    *,
    tool_name: str,
    server_name: str | None = None,
    phase: Literal["call", "result"] = "call",
) -> str:
    if server_name is None:
        return web_tool_progress_label(tool_name)
    combined_name = f"{server_name}/{tool_name}" if server_name else tool_name
    return build_tool_activity_presentation(
        tool_name=combined_name,
        phase="result" if phase == "result" else "call",
        server_name=server_name,
        remote=bool(server_name),
    ).display_name


def _is_beta_text_block_validation_error(error: Exception) -> bool:
    """Return True when Anthropic SDK rejects a text block with null text."""
    detail = casefold_text(f"{type(error).__name__}: {error}")
    return (
        "betatextblock" in detail
        and "input should be a valid string" in detail
        and "text" in detail
    )


def _stream_capture_filename(turn: int) -> Path | None:
    """Generate filename for stream capture. Returns None if capture is disabled."""
    if not STREAM_CAPTURE_ENABLED:
        return None
    STREAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STREAM_CAPTURE_DIR / f"anthropic_{timestamp}_turn{turn}"


def _serialize_for_trace(value: Any) -> Any:
    """Serialize request payloads safely for stream tracing."""
    if isinstance(value, _ModelDumpable):
        try:
            serialized = value.model_dump(warnings="none")
        except TypeError:
            serialized = value.model_dump()
        except Exception:
            serialized = str(value)
        else:
            if serialized is value:
                return str(value)
            serialized = _serialize_for_trace(serialized)
    elif isinstance(value, dict):
        serialized = {key: _serialize_for_trace(item) for key, item in value.items()}
    elif isinstance(value, list):
        serialized = [_serialize_for_trace(item) for item in value]
    elif isinstance(value, (str, int, float, bool)) or value is None:
        serialized = value
    else:
        serialized = str(value)
    return serialized


def _save_stream_request(filename_base: Path | None, arguments: dict[str, Any]) -> None:
    """Save the outgoing request payload for debugging."""
    if not filename_base:
        return
    try:
        request_file = filename_base.with_name(f"{filename_base.name}.request.json")
        payload = _serialize_for_trace(arguments)
        payload = {
            "captured_at": datetime.now().isoformat(),
            "arguments": payload,
        }
        with request_file.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.debug(f"Failed to save stream request: {e}")


def _start_fallback_stream_span(model: str) -> Span:
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span("anthropic.chat")
    if span.is_recording():
        span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "Anthropic")
        span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
        span.set_attribute(
            SpanAttributes.LLM_REQUEST_TYPE,
            LLMRequestTypeValues.COMPLETION.value,
        )
    return span


def _finalize_fallback_stream_span(
    span: Span,
    response: BetaMessage | None,
    had_error: bool,
) -> None:
    if not span.is_recording():
        span.end()
        return
    if response is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_ID, response.id)
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.model)
        if response.usage:
            usage = usage_from_anthropic(
                response.usage,
                provider=Provider.ANTHROPIC,
                model=response.model,
            )
            if usage.prompt.total is not None:
                span.set_attribute(
                    GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                    usage.prompt.total,
                )
            if usage.completion.total is not None:
                span.set_attribute(
                    GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                    usage.completion.total,
                )
            if usage.total is not None:
                span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total)
    if not had_error:
        span.set_status(Status(StatusCode.OK))
    span.end()


def _otel_stream_wrapper_uses_awrap(wrapper: Any) -> bool:
    closure = getattr(wrapper, "__closure__", None)
    if not isinstance(closure, tuple):
        return False
    for cell in closure:
        candidate = cell.cell_contents
        module_name = getattr(candidate, "__module__", None)
        if (
            getattr(candidate, "__name__", None) == "_awrap"
            and isinstance(module_name, str)
            and module_name.startswith("opentelemetry.instrumentation.anthropic")
        ):
            return True
    return False


def _maybe_unwrap_otel_beta_stream(stream_method: Any) -> Any:
    """Bypass a broken OTel anthropic wrapper for beta async streaming.

    The opentelemetry-instrumentation-anthropic wrapper uses an async wrapper
    that awaits the sync beta stream method, which raises
    `TypeError: object BetaAsyncMessageStreamManager can't be used in 'await' expression`.
    If detected, fall back to the original stream method to avoid the error.
    """

    wrapper = getattr(stream_method, "_self_wrapper", None)
    if wrapper is None:
        return stream_method
    wrapper_module = getattr(wrapper, "__module__", None)
    if not isinstance(wrapper_module, str):
        return stream_method
    if wrapper_module != "opentelemetry.instrumentation.anthropic":
        return stream_method

    wrapped = getattr(stream_method, "__wrapped__", None)
    if wrapped is None or inspect.iscoroutinefunction(wrapped):
        return stream_method
    if not _otel_stream_wrapper_uses_awrap(wrapper):
        return stream_method

    global _OTEL_STREAM_WRAPPER_WARNED
    if not _OTEL_STREAM_WRAPPER_WARNED:
        logger.warning(
            "Detected OpenTelemetry anthropic beta stream wrapper that awaits a sync "
            "method. Falling back to the unwrapped stream call to avoid runtime errors."
        )
        _OTEL_STREAM_WRAPPER_WARNED = True

    return wrapped


def _save_stream_chunk(filename_base: Path | None, chunk: Any) -> None:
    """Save a streaming chunk to file when capture mode is enabled."""
    if not filename_base:
        return
    try:
        chunk_file = filename_base.with_name(f"{filename_base.name}.jsonl")
        try:
            payload: Any = chunk.model_dump(warnings="none")
        except TypeError:
            payload = chunk.model_dump()
        except Exception:
            payload = {"type": type(chunk).__name__, "str": str(chunk)}
        with chunk_file.open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        logger.debug(f"Failed to save stream chunk: {e}")


def _transform_anthropic_schema(schema: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """Return an Anthropic-compatible schema using the SDK's schema transformer.

    Anthropic accepts both Pydantic models and raw JSON-schema dicts through the
    same SDK transformer. Keep fast-agent out of this normalization path so the
    model and raw-schema routes match Anthropic SDK behavior exactly.
    """
    return transform_schema(schema)


class AnthropicLLM(FastAgentLLM[BetaMessageParam, BetaMessage]):
    CONVERSATION_CACHE_WALK_DISTANCE = 6
    MAX_CONVERSATION_CACHE_BLOCKS = 2
    # Anthropic-specific parameter exclusions
    ANTHROPIC_EXCLUDE_FIELDS: ClassVar[set[str]] = {
        FastAgentLLM.PARAM_MESSAGES,
        FastAgentLLM.PARAM_MODEL,
        FastAgentLLM.PARAM_SYSTEM_PROMPT,
        FastAgentLLM.PARAM_STOP_SEQUENCES,
        FastAgentLLM.PARAM_MAX_TOKENS,
        FastAgentLLM.PARAM_METADATA,
        FastAgentLLM.PARAM_USE_HISTORY,
        FastAgentLLM.PARAM_MAX_ITERATIONS,
        FastAgentLLM.PARAM_PARALLEL_TOOL_CALLS,
        FastAgentLLM.PARAM_TEMPLATE_VARS,
        FastAgentLLM.PARAM_MCP_METADATA,
        "response_format",
    }

    def __init__(self, **kwargs) -> None:
        # Initialize logger - keep it simple without name reference
        kwargs.pop("provider", None)
        structured_override = kwargs.pop("structured_output_mode", None)
        long_context_requested = kwargs.pop("long_context", False)
        web_search_override = kwargs.pop("web_search", None)
        web_fetch_override = kwargs.pop("web_fetch", None)
        raw_task_budget = kwargs.pop("task_budget_tokens", _UNSET_TASK_BUDGET)
        super().__init__(provider=self.provider_identity(), **kwargs)
        self._structured_output_mode_override: StructuredOutputMode | None = structured_override
        self._web_search_override: bool | None = (
            bool(web_search_override) if isinstance(web_search_override, bool) else None
        )
        self._web_fetch_override: bool | None = (
            bool(web_fetch_override) if isinstance(web_fetch_override, bool) else None
        )
        self._file_id_cache: dict[str, str] = {}

        raw_setting = kwargs.get("reasoning_effort")
        config = self.context.config.anthropic if self.context and self.context.config else None
        model_name = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL
        reasoning_source = self._configure_anthropic_reasoning(
            raw_setting,
            config,
            model_name,
        )

        self._task_budget_tokens: int | None = None
        task_budget_source = self._configure_anthropic_task_budget(
            raw_task_budget,
            config,
            model_name,
        )
        self._log_anthropic_reasoning_resolution(model_name, reasoning_source)
        self._log_anthropic_task_budget_resolution(model_name, task_budget_source)

        # Explicit long-context (1M) opt-in setup for pre-4.6 models.
        self._configure_anthropic_long_context(long_context_requested, model_name)

    def _configure_anthropic_reasoning(
        self,
        raw_setting: Any,
        config: Any,
        model_name: str,
    ) -> str | None:
        reasoning_source = "llm_kwargs" if raw_setting is not None else None
        if raw_setting is None and config:
            raw_setting = config.reasoning
            if raw_setting is not None:
                reasoning_source = "config_reasoning"

        reasoning_mode = self._get_model_reasoning(model_name)
        spec = self._get_model_reasoning_effort_spec(model_name)

        if raw_setting is not None and reasoning_mode != "anthropic_thinking":
            self.logger.warning(
                "Reasoning setting ignored for model without Anthropic thinking support."
            )
            raw_setting = None
            reasoning_source = None

        if raw_setting is None and reasoning_mode == "anthropic_thinking":
            if spec and spec.kind == "effort" and spec.default:
                raw_setting = spec.default
            elif spec and spec.kind == "effort" and spec.allow_auto:
                raw_setting = AUTO_REASONING
            else:
                raw_setting = spec.default if spec and spec.default else 1024
            if raw_setting is not None:
                reasoning_source = "model_default"

        setting = parse_reasoning_setting(raw_setting)
        if setting is None:
            self.set_reasoning_effort(None)
            return reasoning_source

        try:
            self.set_reasoning_effort(setting)
        except ValueError as exc:
            self.logger.warning(f"Invalid reasoning setting: {exc}")
            if spec and spec.default:
                self.set_reasoning_effort(spec.default)
                return "model_default"
            self.set_reasoning_effort(None)
        return reasoning_source

    def _configure_anthropic_task_budget(
        self,
        raw_task_budget: Any,
        config: Any,
        model_name: str,
    ) -> str | None:
        task_budget_source: str | None = None
        if raw_task_budget is not _UNSET_TASK_BUDGET:
            task_budget_source = "llm_kwargs"
        elif config:
            raw_task_budget = config.task_budget
            if raw_task_budget is not None:
                task_budget_source = "config_task_budget"

        if raw_task_budget is _UNSET_TASK_BUDGET:
            return task_budget_source

        if not self._supports_task_budget(model_name):
            self.logger.warning(
                "Task budget ignored for model without Anthropic task budget support."
            )
            return task_budget_source

        try:
            parsed_task_budget = parse_task_budget_tokens(
                raw_task_budget if isinstance(raw_task_budget, (int, str)) else None
            )
            self.set_task_budget_tokens(parsed_task_budget)
        except ValueError as exc:
            self.logger.warning(f"Invalid task budget setting: {exc}")
            self.set_task_budget_tokens(None)
            return None
        return task_budget_source

    def _log_anthropic_reasoning_resolution(
        self,
        model_name: str,
        reasoning_source: str | None,
    ) -> None:
        if self._get_model_reasoning(model_name) != "anthropic_thinking":
            return

        thinking_enabled = self._is_thinking_enabled(model_name)
        payload = {
            "model": model_name,
            "setting": format_reasoning_setting(self.reasoning_effort),
            "reasoning_source": reasoning_source or "unknown",
            "thinking_enabled": thinking_enabled,
            "config_path": (
                self.context.config._config_file if self.context and self.context.config else None
            ),
        }
        self.logger.event(
            "info" if thinking_enabled else "warning",
            "anthropic_reasoning",
            "Anthropic reasoning resolved" if thinking_enabled else "Anthropic reasoning disabled",
            None,
            payload,
        )

    def _log_anthropic_task_budget_resolution(
        self,
        model_name: str,
        task_budget_source: str | None,
    ) -> None:
        if task_budget_source is None and self.task_budget_tokens is None:
            return
        self.logger.event(
            "info",
            "anthropic_task_budget",
            "Anthropic task budget resolved",
            None,
            {
                "model": model_name,
                "task_budget": format_task_budget_tokens(self.task_budget_tokens),
                "task_budget_source": task_budget_source or "unknown",
            },
        )

    def _configure_anthropic_long_context(
        self,
        long_context_requested: bool,
        model_name: str,
    ) -> None:
        self._long_context = False
        if not long_context_requested:
            return

        from fast_agent.llm.model_database import ModelDatabase

        base_context_window = self._get_model_context_window(model_name)
        long_context_window = self._get_model_long_context_window(model_name)
        if (
            base_context_window is not None
            and base_context_window >= ModelDatabase.ANTHROPIC_LONG_CONTEXT_WINDOW
        ):
            self.logger.debug(
                f"Long context query ignored for model '{model_name}' — "
                f"{base_context_window:,} context is already enabled by default"
            )
        elif long_context_window is not None:
            self._long_context = True
            self._context_window_override = long_context_window
            self._usage_accumulator.set_context_window_size(long_context_window)
            self.logger.info(
                f"Long context ({long_context_window:,}) enabled for model '{model_name}'"
            )
        else:
            supported = ", ".join(self._list_supported_long_context_models())
            self.logger.warning(
                f"Long context (context=1m) is not supported for model "
                f"'{model_name}'. Ignoring. Supported models: "
                f"{supported}"
            )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Anthropic-specific default parameters"""
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_ANTHROPIC_MODEL)

    @classmethod
    def provider_identity(cls) -> Provider:
        return Provider.ANTHROPIC

    def _list_supported_long_context_models(self) -> list[str]:
        """Return models that support explicit long-context overrides."""
        from fast_agent.llm.model_database import ModelDatabase

        return ModelDatabase.list_long_context_models()

    def _provider_base_url(self) -> str | None:
        assert self.context.config
        return self.context.config.anthropic.base_url if self.context.config.anthropic else None

    def _provider_default_headers(self) -> dict[str, str] | None:
        """Get custom default headers from configuration."""
        assert self.context.config
        return (
            self.context.config.anthropic.default_headers if self.context.config.anthropic else None
        )

    def _provider_api_key(self):
        from fast_agent.llm.provider_key_manager import ProviderKeyManager

        return ProviderKeyManager.get_api_key(
            self.provider.config_name,
            self.context.config,
        )

    def _configured_api_key(self) -> str | None:
        if self._init_api_key is not None:
            return self._init_api_key

        from fast_agent.llm.provider_key_manager import ProviderKeyManager

        api_key = ProviderKeyManager.get_optional_api_key(
            self.provider.config_name,
            self.context.config,
        )
        return api_key if api_key else None

    def _initialize_anthropic_client(self) -> Any:
        base_url = self._base_url()
        default_headers = self._default_headers()

        if base_url and base_url.endswith("/v1"):
            base_url = base_url.rstrip("/v1")
        client_args: dict[str, Any] = {
            "base_url": base_url,
            "default_headers": default_headers,
        }
        api_key = self._configured_api_key()
        if api_key is not None:
            client_args["api_key"] = api_key
        return AsyncAnthropic(**client_args)

    def validate_provider_credentials(self) -> None:
        if self._configured_api_key() is not None:
            return
        client = self._initialize_anthropic_client()
        if (
            client.api_key is not None
            or client.auth_token is not None
            or client.credentials is not None
        ):
            return
        self._provider_api_key()

    def supports_files_api(self) -> bool:
        return True

    def supports_document_uploads(self) -> bool:
        return self.supports_files_api()

    def supports_web_tools(self) -> bool:
        return True

    def supports_direct_anthropic_beta(self, feature: str) -> bool:
        del feature
        return True

    def _get_cache_mode(self) -> str:
        """Get the cache mode configuration."""
        cache_mode = "auto"  # Default to auto
        if self.context.config and self.context.config.anthropic:
            cache_mode = self.context.config.anthropic.cache_mode
        return cache_mode

    @staticmethod
    def _anthropic_file_cache_key(data: bytes, filename: str, mime_type: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        return f"{mime_type}:{filename}:{digest}"

    async def _upload_anthropic_file_bytes(
        self,
        anthropic: Any,
        *,
        data: bytes,
        filename: str,
        mime_type: str,
    ) -> str | None:
        files_api = getattr(getattr(anthropic, "beta", None), "files", None)
        upload = getattr(files_api, "upload", None)
        if not callable(upload):
            return None

        cache_key = self._anthropic_file_cache_key(data, filename, mime_type)
        cached = self._file_id_cache.get(cache_key)
        if cached:
            return cached

        file_metadata = await upload(file=(filename, data, mime_type))
        file_id = getattr(file_metadata, "id", None)
        if not isinstance(file_id, str) or not file_id:
            return None

        self._file_id_cache[cache_key] = file_id
        return file_id

    @staticmethod
    def _anthropic_document_mime_type(resource: BlobResourceContents) -> str | None:
        mime_type = normalize_mime_type(resource.mimeType)
        if not mime_type and getattr(resource, "uri", None):
            mime_type = guess_mime_type(str(resource.uri))
        if mime_type not in DOCUMENT_MIME_TYPES or mime_type == "application/pdf":
            return None
        return mime_type

    @staticmethod
    def _has_anthropic_file_id(resource: BlobResourceContents) -> bool:
        meta = dict(getattr(resource, "meta", None) or {})
        existing = meta.get(ANTHROPIC_FILE_ID_META_KEY)
        return isinstance(existing, str) and bool(existing)

    @staticmethod
    def _decode_anthropic_document_blob(
        resource: BlobResourceContents,
        mime_type: str,
    ) -> bytes | None:
        try:
            return base64.b64decode(resource.blob)
        except Exception:
            logger.warning(
                "Unable to decode Anthropic document upload bytes",
                data={"mime_type": mime_type, "uri": str(getattr(resource, "uri", ""))},
            )
            return None

    @staticmethod
    def _anthropic_document_filename(resource: BlobResourceContents) -> str:
        from fast_agent.mcp.resource_utils import extract_title_from_uri

        if getattr(resource, "uri", None):
            return extract_title_from_uri(resource.uri) or "document"
        return "document"

    async def _prepare_anthropic_document_resource(
        self,
        anthropic: Any,
        resource: BlobResourceContents,
    ) -> None:
        mime_type = self._anthropic_document_mime_type(resource)
        if not mime_type or self._has_anthropic_file_id(resource):
            return

        data = self._decode_anthropic_document_blob(resource, mime_type)
        if data is None:
            return

        file_id = await self._upload_anthropic_file_bytes(
            anthropic,
            data=data,
            filename=self._anthropic_document_filename(resource),
            mime_type=mime_type,
        )
        if file_id:
            meta = dict(getattr(resource, "meta", None) or {})
            meta[ANTHROPIC_FILE_ID_META_KEY] = file_id
            resource.meta = meta

    async def _prepare_anthropic_file_resources(
        self,
        anthropic: Any,
        messages: Sequence[PromptMessageExtended],
    ) -> None:
        if not self.supports_document_uploads():
            return

        for message in messages:
            for content in message.content:
                if not isinstance(content, EmbeddedResource):
                    continue

                resource = content.resource
                if not isinstance(resource, BlobResourceContents):
                    continue

                await self._prepare_anthropic_document_resource(anthropic, resource)

    def _get_cache_ttl(self) -> CacheTTL:
        """Get the cache TTL configuration ('5m' or '1h')."""
        cache_ttl: CacheTTL = "5m"  # Default to 5 minutes
        if self.context.config and self.context.config.anthropic:
            cache_ttl = self.context.config.anthropic.cache_ttl
        return cache_ttl

    def _supports_adaptive_thinking(self, model: str) -> bool:
        """Return True when model uses adaptive thinking instead of manual budgets."""

        if self._get_model_reasoning(model) != "anthropic_thinking":
            return False
        spec = self._get_model_reasoning_effort_spec(model)
        return bool(spec and spec.kind == "effort")

    def _is_thinking_enabled(self, model: str) -> bool:
        """Check if extended thinking should be enabled for this request."""

        if self._get_model_reasoning(model) != "anthropic_thinking":
            return False
        setting = self.reasoning_effort
        if setting is None:
            return False

        match setting.kind:
            case _ if is_auto_reasoning(setting):
                enabled = self._supports_adaptive_thinking(model)
            case "toggle" | "budget":
                enabled = bool(setting.value)
            case "effort":
                enabled = strip_casefold(
                    str(setting.value)
                ) != "none" and self._supports_adaptive_thinking(model)
            case _:
                enabled = False
        return enabled

    def _uses_summarized_thinking_display(self, model: str) -> bool:
        """Return True when summarized thinking should be requested explicitly."""
        return self._normalize_model_name(model) == "claude-opus-4-7"

    def _requires_explicit_thinking_field(self, model: str) -> bool:
        return self._get_model_anthropic_thinking_field_required(model)

    def _supports_thinking_disable(self, model: str) -> bool:
        return self._get_model_anthropic_thinking_disable_supported(model)

    def _supports_task_budget(self, model: str) -> bool:
        """Return True when Anthropic task budgets are supported for the model/provider."""
        return self.provider_identity() in {
            Provider.ANTHROPIC,
            Provider.ANTHROPIC_VERTEX,
        } and self._get_model_anthropic_task_budget_supported(model)

    @property
    def task_budget_supported(self) -> bool:
        model_name = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL
        return self._supports_task_budget(model_name)

    @property
    def task_budget_tokens(self) -> int | None:
        return self._task_budget_tokens

    def set_task_budget_tokens(self, value: int | None) -> None:
        if value is None:
            self._task_budget_tokens = None
            return
        if not self.task_budget_supported:
            raise ValueError("Current model does not support task budget configuration.")
        self._task_budget_tokens = validate_task_budget_tokens(value)

    def _resolve_adaptive_effort(self, model: str) -> str | None:
        """Resolve adaptive effort for Anthropic output_config."""
        setting = self.reasoning_effort
        if setting is None or setting.kind != "effort":
            return None
        if is_auto_reasoning(setting):
            return None
        effort = strip_casefold(str(setting.value))
        if effort == "xhigh":
            spec = self._get_model_reasoning_effort_spec(model)
            if spec and "xhigh" in (spec.allowed_efforts or []):
                return "xhigh"
            return "max"
        if effort == "none":
            return None
        return effort

    def _get_thinking_budget(self) -> int:
        """Get the thinking budget tokens (minimum 1024)."""
        setting = self.reasoning_effort
        if setting and setting.kind == "budget" and isinstance(setting.value, int):
            return max(1024, setting.value)
        return 1024

    def _resolve_thinking_arguments(
        self,
        model: str,
        max_tokens: int | None,
        structured_mode: StructuredOutputMode | None,
    ) -> tuple[dict[str, Any], bool]:
        """Build Anthropic thinking/output_config arguments for this request."""
        args: dict[str, Any] = {}
        thinking_enabled = self._is_thinking_enabled(model)
        adaptive_supported = self._supports_adaptive_thinking(model)

        if (
            thinking_enabled
            and structured_mode == "tool_use"
            and self._requires_explicit_thinking_field(model)
        ):
            if max_tokens is not None:
                args["max_tokens"] = max_tokens
            return args, False

        if not thinking_enabled:
            if self._supports_thinking_disable(model):
                setting = self.reasoning_effort
                if setting and setting.kind == "toggle" and setting.value is False:
                    args["thinking"] = {"type": "disabled"}
            if max_tokens is not None:
                args["max_tokens"] = max_tokens
            return args, False

        if adaptive_supported:
            if self._requires_explicit_thinking_field(model):
                thinking: dict[str, str] = {"type": "adaptive"}
                if self._uses_summarized_thinking_display(model):
                    thinking["display"] = "summarized"
                args["thinking"] = thinking
            effort = self._resolve_adaptive_effort(model)
            if effort:
                args["output_config"] = {"effort": effort}
            args["max_tokens"] = max_tokens if max_tokens is not None else 16000
            return args, True

        thinking_budget = self._get_thinking_budget()
        args["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        current_max = max_tokens if max_tokens is not None else 16000
        if current_max <= thinking_budget:
            args["max_tokens"] = thinking_budget + 8192
        else:
            args["max_tokens"] = current_max
        return args, True

    def _resolve_structured_output_mode(
        self,
        model: str,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None = None,
    ) -> StructuredOutputMode | None:
        if structured_model is None and structured_schema is None:
            return None
        if self._structured_output_mode_override is not None:
            return self._structured_output_mode_override
        config = self.context.config.anthropic if self.context and self.context.config else None
        if config and config.structured_output_mode != "auto":
            return config.structured_output_mode

        json_mode = self._get_model_json_mode(model)
        if json_mode == "schema":
            if self.supports_direct_anthropic_beta("structured_output"):
                return "json"
            return "tool_use"
        return "tool_use"

    def _resolve_structured_tool_policy(
        self,
        request_params: RequestParams,
    ) -> Literal["always", "defer", "no_tools"]:
        if request_params.structured_tool_policy != "auto":
            return request_params.structured_tool_policy

        model_name = request_params.model or self.default_request_params.model or self._model_name
        if model_name:
            structured_mode = self._resolve_structured_output_mode(
                model_name,
                None,
                request_params.structured_schema,
            )
            if structured_mode == "tool_use":
                return "no_tools"

        return super()._resolve_structured_tool_policy(request_params)

    def _is_auto_tool_use_structured_fallback(
        self,
        model: str,
        structured_mode: StructuredOutputMode | None,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None = None,
    ) -> bool:
        if structured_mode != "tool_use":
            return False
        if structured_model is None and structured_schema is None:
            return False
        if self._structured_output_mode_override is not None:
            return False
        config = self.context.config.anthropic if self.context and self.context.config else None
        if config and config.structured_output_mode != "auto":
            return False
        return self._get_model_json_mode(model) != "schema"

    def _build_output_format(
        self,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if structured_schema is not None:
            schema = _transform_anthropic_schema(structured_schema)
        elif structured_model is not None:
            schema = _transform_anthropic_schema(cast("type[BaseModel]", structured_model))
        else:
            schema = _transform_anthropic_schema({"type": "object"})
        return {"type": "json_schema", "schema": schema}

    async def _prepare_tools(
        self,
        model: str,
        structured_model: type[ModelT] | None = None,
        structured_schema: dict[str, Any] | None = None,
        tools: list[Tool] | None = None,
        structured_mode: StructuredOutputMode | None = None,
        auto_tool_use_fallback: bool = False,
    ) -> list[BetaToolParam]:
        """Prepare tools based on whether we're in structured output mode."""
        regular_tools = [
            BetaToolParam(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in tools or []
        ]
        if (structured_model or structured_schema) and structured_mode == "tool_use":
            if auto_tool_use_fallback and regular_tools:
                logger.warning(
                    "Anthropic structured output fell back to legacy tool_use mode; "
                    "normal tools will be suppressed for this structured request.",
                    model=model,
                    structured_mode=structured_mode,
                    tool_count=len(regular_tools),
                )
            schema: dict[str, object]
            if structured_schema is not None:
                schema = cast("dict[str, object]", _transform_anthropic_schema(structured_schema))
            elif structured_model is not None:
                schema = cast(
                    "dict[str, object]",
                    _transform_anthropic_schema(cast("type[BaseModel]", structured_model)),
                )
            else:
                schema = cast("dict[str, object]", _transform_anthropic_schema({"type": "object"}))
            return [
                BetaToolParam(
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    description="Return the response in the required JSON format",
                    input_schema=schema,
                    strict=True,
                )
            ]
        if structured_model or structured_schema:
            return regular_tools
        return regular_tools

    def _prepare_web_tools(self, model: str) -> tuple[list[BetaToolParam], tuple[str, ...]]:
        if not self.supports_web_tools():
            return [], ()

        anthropic_settings = self.context.config.anthropic if self.context.config else None
        resolved = resolve_web_tools(
            anthropic_settings,
            web_search_override=self._web_search_override,
            web_fetch_override=self._web_fetch_override,
        )
        return build_web_tool_params(
            resolved_tools=resolved,
            search_version=self._get_model_anthropic_web_search_version(model),
            fetch_version=self._get_model_anthropic_web_fetch_version(model),
            required_betas=self._get_model_anthropic_required_betas(model),
        )

    @property
    def web_tools_enabled(self) -> tuple[bool, bool]:
        """Return (search_enabled, fetch_enabled) for toolbar display."""
        if not self.supports_web_tools():
            return False, False
        anthropic_settings = self.context.config.anthropic if self.context.config else None
        resolved = resolve_web_tools(
            anthropic_settings,
            web_search_override=self._web_search_override,
            web_fetch_override=self._web_fetch_override,
        )
        return resolved.search_enabled, resolved.fetch_enabled

    @property
    def web_search_supported(self) -> bool:
        if not self.supports_web_tools():
            return False
        model_name = self.model_name
        if not model_name:
            return False
        return self._get_model_anthropic_web_search_version(model_name) is not None

    def set_web_search_enabled(self, value: bool | None) -> None:
        if value is None:
            self._web_search_override = None
            return
        if not self.web_search_supported:
            raise ValueError("Current model does not support web search configuration.")
        self._web_search_override = value

    @property
    def web_fetch_supported(self) -> bool:
        if not self.supports_web_tools():
            return False
        model_name = self.model_name
        if not model_name:
            return False
        return self._get_model_anthropic_web_fetch_version(model_name) is not None

    @property
    def web_fetch_enabled(self) -> bool:
        _, fetch_enabled = self.web_tools_enabled
        return fetch_enabled

    def set_web_fetch_enabled(self, value: bool | None) -> None:
        if value is None:
            self._web_fetch_override = None
            return
        if not self.web_fetch_supported:
            raise ValueError("Current model does not support web fetch configuration.")
        self._web_fetch_override = value

    @property
    def web_search_enabled(self) -> bool:
        """Whether Anthropic web search is enabled for this LLM instance."""
        search_enabled, _ = self.web_tools_enabled
        return search_enabled

    def _apply_system_cache(self, base_args: dict, cache_mode: str) -> int:
        """Apply cache control to system prompt if cache mode allows it."""
        system_content: SystemParam | None = base_args.get("system")

        if cache_mode != "off" and system_content:
            cache_ttl = self._get_cache_ttl()
            # Convert string to list format with cache control
            if isinstance(system_content, str):
                base_args["system"] = [
                    BetaTextBlockParam(
                        type="text",
                        text=system_content,
                        cache_control={"type": "ephemeral", "ttl": cache_ttl},
                    )
                ]
                logger.debug(
                    "Applied cache_control to system prompt (caches tools+system in one block)"
                )
                return 1
            # If it's already a list (shouldn't happen in current flow but type-safe)
            if isinstance(system_content, list):
                logger.debug("System prompt already in list format")
            else:
                logger.debug(f"Unexpected system prompt type: {type(system_content)}")

        return 0

    @staticmethod
    def _apply_cache_control_to_message(message: BetaMessageParam, ttl: CacheTTL = "5m") -> bool:
        """Apply cache control to the last content block of a message."""
        if not is_str_object_dict(message) or "content" not in message:
            return False

        content_list = message["content"]
        if not isinstance(content_list, list) or not content_list:
            return False

        for content_block in reversed(content_list):
            if is_str_object_dict(content_block):
                content_block["cache_control"] = {"type": "ephemeral", "ttl": ttl}
                return True

        return False

    def _is_structured_output_request(self, tool_uses: list[Any]) -> bool:
        """
        Check if the tool uses contain a structured output request.

        Args:
            tool_uses: List of tool use blocks from the response

        Returns:
            True if any tool is the structured output tool
        """
        return any(tool.name == STRUCTURED_OUTPUT_TOOL_NAME for tool in tool_uses)

    def _build_tool_calls_dict(
        self, tool_uses: list[BetaToolUseBlock]
    ) -> dict[str, CallToolRequest]:
        """
        Convert Anthropic tool use blocks into our CallToolRequest.

        Args:
            tool_uses: List of tool use blocks from Anthropic response

        Returns:
            Dictionary mapping tool_use_id to CallToolRequest objects
        """
        tool_calls = {}
        for tool_use in tool_uses:
            tool_call = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name=tool_use.name,
                    arguments=cast("dict[str, Any] | None", tool_use.input),
                ),
            )
            tool_calls[tool_use.id] = tool_call
        return tool_calls

    async def _handle_structured_output_response(
        self,
        tool_use_block: BetaToolUseBlock,
        structured_model: type[ModelT] | None,
        messages: list[BetaMessageParam],
    ) -> tuple[LlmStopReason, list[ContentBlock]]:
        """
        Handle a structured output tool response from Anthropic.

        This handles the special case where Anthropic's model was forced to use
        a 'return_structured_output' tool via tool_choice. The tool input contains
        the JSON data we want, so we extract it and format it for display.

        Even though we don't call an external tool, we must create a CallToolResult
        to satisfy Anthropic's API requirement that every tool_use has a corresponding
        tool_result in the next message.

        Args:
            tool_use_block: The tool use block containing structured output
            structured_model: The model class for structured output, when one was supplied
            messages: The message list to append tool results to

        Returns:
            Tuple of (stop_reason, response_content_blocks)
        """
        tool_args = tool_use_block.input
        tool_use_id = tool_use_block.id

        # Create the content for responses
        structured_content = TextContent(type="text", text=json.dumps(tool_args))

        tool_result = CallToolResult(isError=False, content=[structured_content])
        messages.append(
            AnthropicConverter.create_tool_results_message([(tool_use_id, tool_result)])
        )

        logger.debug("Structured output received, treating as END_TURN")

        return LlmStopReason.END_TURN, [structured_content]

    def _start_client_tool_stream(
        self,
        content_block: BetaToolUseBlock,
        event: BetaRawContentBlockStartEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        tool_state = state.tool_tracker.register(
            tool_use_id=content_block.id,
            name=content_block.name,
            index=event.index,
        )
        state.tool_input_buffers.setdefault(tool_state.tool_use_id, _AnthropicToolInputBuffer())
        self._notify_tool_stream_listeners(
            "start",
            {
                "tool_name": tool_state.name,
                "tool_use_id": tool_state.tool_use_id,
                "index": event.index,
            },
        )
        self.logger.info(
            "Model started streaming tool input",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": tool_state.name,
                "tool_use_id": tool_state.tool_use_id,
                "tool_event": "start",
            },
        )

    def _start_server_tool_stream(
        self,
        content_block: BetaServerToolUseBlock,
        event: BetaRawContentBlockStartEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        tool_state = state.tool_tracker.register(
            tool_use_id=content_block.id,
            name=content_block.name,
            index=event.index,
            kind="server_tool",
        )
        presentation = build_tool_activity_presentation(
            tool_name=tool_state.name,
            phase="call",
        )
        progress_label = _provider_tool_display_name(tool_name=tool_state.name)
        self._notify_tool_stream_listeners(
            "start",
            {
                "tool_name": tool_state.name,
                "presentation_family": presentation.family,
                "preserve_details": False,
                "tool_display_name": progress_label,
                "chunk": _server_tool_preview_chunk(content_block.input),
                "tool_use_id": tool_state.tool_use_id,
                "index": event.index,
            },
        )
        self.logger.info(
            "Anthropic server tool started",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": tool_state.name,
                "tool_use_id": tool_state.tool_use_id,
                "tool_event": "start",
                "details": progress_label,
            },
        )

    def _start_mcp_tool_stream(
        self,
        content_block: BetaMCPToolUseBlock,
        event: BetaRawContentBlockStartEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        combined_name = f"{content_block.server_name}/{content_block.name}"
        state.provider_tool_names[content_block.id] = (
            content_block.name,
            content_block.server_name,
        )
        tool_state = state.tool_tracker.register(
            tool_use_id=content_block.id,
            name=combined_name,
            index=event.index,
            kind="server_tool",
        )
        progress_label = _provider_tool_display_name(
            tool_name=content_block.name,
            server_name=content_block.server_name,
        )
        self._notify_tool_stream_listeners(
            "start",
            {
                "tool_name": combined_name,
                "server_name": content_block.server_name,
                "presentation_family": "remote_tool",
                "preserve_details": True,
                "tool_display_name": progress_label,
                "chunk": _mcp_tool_preview_chunk(content_block.input),
                "tool_use_id": tool_state.tool_use_id,
                "index": event.index,
            },
        )
        self.logger.info(
            "Anthropic MCP tool started",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": combined_name,
                "tool_use_id": tool_state.tool_use_id,
                "tool_event": "start",
                "details": progress_label,
            },
        )

    def _handle_mcp_tool_result_start(
        self,
        content_block: BetaMCPToolResultBlock,
        event: BetaRawContentBlockStartEvent,
        state: _AnthropicStreamState,
    ) -> None:
        raw_tool_name, server_name = state.provider_tool_names.get(
            content_block.tool_use_id,
            ("mcp_tool", None),
        )
        combined_name = f"{server_name}/{raw_tool_name}" if server_name else raw_tool_name
        display_name = _provider_tool_display_name(
            tool_name=raw_tool_name,
            server_name=server_name,
            phase="result",
        )
        result_tool_use_id = f"{content_block.tool_use_id}:result"
        self._notify_tool_stream_listeners(
            "replace",
            {
                "tool_name": combined_name,
                "server_name": server_name,
                "presentation_family": "remote_tool",
                "preserve_details": True,
                "tool_display_name": display_name,
                "tool_use_id": result_tool_use_id,
                "index": event.index,
                "chunk": _mcp_tool_result_preview_chunk(content_block.content),
            },
        )
        self._notify_tool_stream_listeners(
            "stop",
            {
                "tool_name": combined_name,
                **({"server_name": server_name} if server_name else {}),
                "presentation_family": "remote_tool",
                "preserve_details": True,
                "tool_display_name": display_name,
                "tool_use_id": result_tool_use_id,
                "index": event.index,
            },
        )

    def _handle_anthropic_content_block_start(
        self,
        event: BetaRawContentBlockStartEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        content_block = event.content_block
        if isinstance(content_block, (BetaThinkingBlock, BetaRedactedThinkingBlock)):
            state.thinking_indices.add(event.index)
        elif isinstance(content_block, BetaToolUseBlock):
            self._start_client_tool_stream(content_block, event, model, state)
        elif isinstance(content_block, BetaServerToolUseBlock):
            self._start_server_tool_stream(content_block, event, model, state)
        elif isinstance(content_block, BetaMCPToolUseBlock):
            self._start_mcp_tool_stream(content_block, event, model, state)
        elif isinstance(content_block, BetaMCPToolResultBlock):
            self._handle_mcp_tool_result_start(content_block, event, state)

    def _handle_thinking_delta(
        self,
        delta: BetaThinkingDelta,
        state: _AnthropicStreamState,
    ) -> None:
        if not delta.thinking:
            return
        self._notify_stream_listeners(StreamChunk(text=delta.thinking, is_reasoning=True))
        state.thinking_segments.append(delta.thinking)

    def _handle_input_json_delta(
        self,
        event: BetaRawContentBlockDeltaEvent,
        delta: BetaInputJSONDelta,
        state: _AnthropicStreamState,
    ) -> None:
        tool_state = state.tool_tracker.resolve_open(index=event.index)
        if tool_state is None or tool_state.kind != "tool":
            return

        chunk = delta.partial_json or ""
        state.tool_input_buffers.setdefault(
            tool_state.tool_use_id,
            _AnthropicToolInputBuffer(),
        ).append(chunk)
        preview = chunk if len(chunk) <= 80 else chunk[:77] + "..."
        self._notify_tool_stream_listeners(
            "delta",
            {
                "tool_name": tool_state.name,
                "tool_use_id": tool_state.tool_use_id,
                "index": event.index,
                "chunk": chunk,
            },
        )
        self.logger.debug(
            "Streaming tool input delta",
            data={
                "tool_name": tool_state.name,
                "tool_use_id": tool_state.tool_use_id,
                "chunk": preview,
            },
        )

    def _handle_text_delta(
        self,
        event: BetaRawContentBlockDeltaEvent,
        delta: BetaTextDelta,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        self._notify_stream_listeners(StreamChunk(text=delta.text, is_reasoning=False))
        if delta.text:
            state.streamed_text_segments.append(delta.text)
        state.estimated_tokens = self._update_streaming_progress(
            delta.text, model, state.estimated_tokens
        )
        self._notify_tool_stream_listeners(
            "text",
            {
                "chunk": delta.text,
                "index": event.index,
            },
        )

    def _handle_anthropic_content_block_delta(
        self,
        event: BetaRawContentBlockDeltaEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        delta = event.delta
        if isinstance(delta, BetaThinkingDelta):
            self._handle_thinking_delta(delta, state)
        elif isinstance(delta, BetaSignatureDelta):
            return
        elif isinstance(delta, BetaInputJSONDelta):
            self._handle_input_json_delta(event, delta, state)
        elif isinstance(delta, BetaTextDelta):
            self._handle_text_delta(event, delta, model, state)

    def _stop_client_tool_stream(
        self,
        event: BetaRawContentBlockStopEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        tool_state = state.tool_tracker.close(index=event.index)
        if tool_state is None:
            return

        preview_raw = state.tool_input_buffers.get(
            tool_state.tool_use_id, _AnthropicToolInputBuffer()
        ).joined()
        if preview_raw:
            preview = preview_raw if len(preview_raw) <= 120 else preview_raw[:117] + "..."
            self.logger.debug(
                "Completed tool input stream",
                data={
                    "tool_name": tool_state.name,
                    "tool_use_id": tool_state.tool_use_id,
                    "input_preview": preview,
                },
            )
        self._notify_tool_stream_listeners(
            "stop",
            {
                "tool_name": tool_state.name,
                "tool_use_id": tool_state.tool_use_id,
                "index": event.index,
            },
        )
        self.logger.info(
            "Model finished streaming tool input",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": tool_state.name,
                "tool_use_id": tool_state.tool_use_id,
                "tool_event": "stop",
                "tool_terminal": True,
            },
        )

    def _server_tool_stop_details(
        self,
        event: BetaRawContentBlockStopEvent,
        default_tool_name: str,
    ) -> tuple[str, str | None, Any | None]:
        content_block = getattr(event, "content_block", None)
        server_name: str | None = None
        tool_name = default_tool_name
        raw_input: Any | None = None
        content_block_type = (
            content_block.get("type")
            if isinstance(content_block, Mapping)
            else getattr(content_block, "type", None)
        )

        if content_block_type == "mcp_tool_use":
            raw_server_name = (
                content_block.get("server_name")
                if isinstance(content_block, Mapping)
                else getattr(content_block, "server_name", None)
            )
            raw_tool_name = (
                content_block.get("name")
                if isinstance(content_block, Mapping)
                else getattr(content_block, "name", None)
            )
            raw_input = (
                content_block.get("input")
                if isinstance(content_block, Mapping)
                else getattr(content_block, "input", None)
            )
            if isinstance(raw_server_name, str) and isinstance(raw_tool_name, str):
                server_name = raw_server_name
                tool_name = f"{raw_server_name}/{raw_tool_name}"
        elif isinstance(content_block, BetaServerToolUseBlock):
            tool_name = content_block.name

        return tool_name, server_name, raw_input

    def _stop_server_tool_stream(
        self,
        event: BetaRawContentBlockStopEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        tool_state = state.tool_tracker.close(index=event.index)
        if tool_state is None:
            return

        tool_name, server_name, raw_input = self._server_tool_stop_details(event, tool_state.name)
        if raw_input is not None:
            self._notify_tool_stream_listeners(
                "replace",
                {
                    "tool_name": tool_name,
                    "server_name": server_name,
                    "presentation_family": "remote_tool",
                    "preserve_details": True,
                    "tool_use_id": tool_state.tool_use_id,
                    "index": event.index,
                    "chunk": _mcp_tool_preview_chunk(raw_input),
                },
            )

        presentation = build_tool_activity_presentation(tool_name=tool_name, phase="call")
        progress_label = _provider_tool_display_name(
            tool_name=tool_name.split("/", 1)[-1],
            server_name=server_name,
        )
        self._notify_tool_stream_listeners(
            "stop",
            {
                "tool_name": tool_name,
                **({"server_name": server_name} if server_name else {}),
                "presentation_family": "remote_tool" if server_name else presentation.family,
                "preserve_details": bool(server_name),
                "tool_display_name": progress_label,
                "tool_use_id": tool_state.tool_use_id,
                "index": event.index,
            },
        )
        self.logger.info(
            "Anthropic server tool completed",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": tool_name,
                "tool_use_id": tool_state.tool_use_id,
                "tool_event": "stop",
                "tool_terminal": True,
                "details": progress_label,
            },
        )

    def _handle_anthropic_content_block_stop(
        self,
        event: BetaRawContentBlockStopEvent,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        if event.index in state.thinking_indices:
            state.thinking_indices.discard(event.index)
            return

        tool_state = state.tool_tracker.resolve_open(index=event.index)
        if tool_state is None:
            return
        if tool_state.kind == "tool":
            self._stop_client_tool_stream(event, model, state)
        elif tool_state.kind == "server_tool":
            self._stop_server_tool_stream(event, model, state)

    def _handle_anthropic_message_delta(
        self,
        event: BetaRawMessageDeltaEvent,
        model: str,
    ) -> None:
        if not event.usage.output_tokens:
            return
        token_str = str(event.usage.output_tokens).rjust(5)
        logger.info(
            "Streaming progress",
            data={
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            },
        )

    def _raise_for_incomplete_anthropic_tools(
        self,
        model: str,
        state: _AnthropicStreamState,
    ) -> None:
        incomplete_tools = state.tool_tracker.incomplete()
        if not incomplete_tools:
            return

        tool_labels = [f"{tool.name}:{tool.tool_use_id}" for tool in incomplete_tools]
        logger.error(
            "Anthropic stream ended with incomplete tool state",
            data={
                "model": model,
                "incomplete_tools": tool_labels,
                "tool_count": len(tool_labels),
            },
        )
        raise RuntimeError(format_incomplete_tool_call_error(tool_labels))

    async def _final_anthropic_stream_message(
        self,
        stream: BetaAsyncMessageStream,
        model: str,
        state: _AnthropicStreamState,
    ) -> BetaMessage:
        try:
            return await stream.get_final_message()
        except Exception as error:
            if not (_is_beta_text_block_validation_error(error) and state.streamed_text_segments):
                raise

            logger.warning(
                "Anthropic final message validation failed; falling back to streamed text",
                data={
                    "model": model,
                    "streamed_text_chunks": len(state.streamed_text_segments),
                    "error": str(error),
                },
            )
            if os.environ.get("FAST_AGENT_WEBDEBUG"):
                print(
                    "[webdebug] final message validation failed; "
                    "using streamed text fallback "
                    f"model={model} chunks={len(state.streamed_text_segments)}"
                )

            return BetaMessage.model_construct(
                id="msg_stream_fallback",
                type="message",
                role="assistant",
                content=[],
                model=model,
                stop_reason="end_turn",
                usage=None,
            )

    async def _process_stream(
        self,
        stream: BetaAsyncMessageStream,
        model: str,
        capture_filename: Path | None = None,
    ) -> tuple[BetaMessage, list[str], list[str]]:
        """Process the streaming response and display real-time token usage."""
        # Track estimated output tokens by counting text chunks
        state = _AnthropicStreamState()

        try:
            # Process the raw event stream to get token counts.
            # Cancellation is handled via asyncio.Task.cancel() which raises CancelledError.
            async for event in stream:
                _save_stream_chunk(capture_filename, event)

                if isinstance(event, BetaRawContentBlockStartEvent):
                    self._handle_anthropic_content_block_start(event, model, state)
                elif isinstance(event, BetaRawContentBlockDeltaEvent):
                    self._handle_anthropic_content_block_delta(event, model, state)
                elif isinstance(event, BetaRawContentBlockStopEvent):
                    self._handle_anthropic_content_block_stop(event, model, state)
                elif isinstance(event, BetaRawMessageDeltaEvent) and event.usage.output_tokens:
                    self._handle_anthropic_message_delta(event, model)

            self._raise_for_incomplete_anthropic_tools(model, state)
            message = await self._final_anthropic_stream_message(stream, model, state)
            if message.usage:
                logger.info(
                    f"Streaming complete - Model: {model}, Input tokens: {message.usage.input_tokens}, Output tokens: {message.usage.output_tokens}"
                )

            return message, state.thinking_segments.parts(), state.streamed_text_segments
        except APIError as error:
            logger.error("Streaming APIError during Anthropic completion", exc_info=error)
            raise
        except Exception as error:
            logger.error("Unexpected error during Anthropic stream processing", exc_info=error)
            raise

    def _handle_retry_failure(self, error: Exception) -> PromptMessageExtended | None:
        """Return the legacy error-channel response when retries are exhausted."""
        if isinstance(error, APIError):
            model_name = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL
            return build_stream_failure_response(self.provider, error, model_name)
        return None

    def _build_request_messages(
        self,
        params: RequestParams,
        message_param: BetaMessageParam,
        pre_messages: list[BetaMessageParam] | None = None,
        history: list[PromptMessageExtended] | None = None,
    ) -> list[BetaMessageParam]:
        """
        Build the list of Anthropic message parameters for the next request.

        Ensures that the current user message is only included once when history
        is enabled, which prevents duplicate tool_result blocks from being sent.
        """
        messages: list[BetaMessageParam] = list(pre_messages) if pre_messages else []

        history_messages: list[BetaMessageParam] = []
        if params.use_history and history:
            history_messages = self._convert_to_provider_format(history)
            messages.extend(history_messages)

        include_current = not params.use_history or not history_messages
        if include_current:
            messages.append(message_param)

        return messages

    @staticmethod
    def _container_id_from_channels(
        channels: Mapping[str, Sequence[ContentBlock]] | None,
    ) -> str | None:
        if not channels:
            return None

        raw_container = channels.get(ANTHROPIC_CONTAINER_CHANNEL)
        if not raw_container:
            return None

        for block in raw_container:
            if not isinstance(block, TextContent):
                continue

            raw_text = block.text
            if not raw_text:
                continue

            payload: object = raw_text
            try:
                payload = json.loads(raw_text)
            except Exception:
                payload = raw_text

            if isinstance(payload, dict):
                container_id = payload.get("id")
                if isinstance(container_id, str) and container_id:
                    return container_id
            elif isinstance(payload, str) and payload:
                return payload

        return None

    def _resolve_container_id_for_request(
        self,
        history: list[PromptMessageExtended] | None,
        current_extended: PromptMessageExtended | None,
    ) -> str | None:
        if history:
            for message in reversed(history):
                container_id = self._container_id_from_channels(message.channels)
                if container_id:
                    return container_id

        if current_extended is not None:
            return self._container_id_from_channels(current_extended.channels)

        return None

    def _build_anthropic_base_args(
        self,
        *,
        model: str,
        messages: list[BetaMessageParam],
        params: RequestParams,
        history: list[PromptMessageExtended] | None,
        current_extended: PromptMessageExtended | None,
        request_tools: list[BetaToolParam],
        structured_mode: StructuredOutputMode | None,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None = None,
        auto_tool_use_fallback: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        base_args: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stop_sequences": params.stopSequences,
        }
        container_id = self._resolve_container_id_for_request(history, current_extended)
        if container_id:
            base_args["container"] = container_id

        if request_tools:
            base_args["tools"] = request_tools

        if self.instruction or params.systemPrompt:
            base_args["system"] = self.instruction or params.systemPrompt

        if structured_mode == "tool_use":
            if self._is_thinking_enabled(model) and self._requires_explicit_thinking_field(model):
                if auto_tool_use_fallback:
                    logger.warning(
                        "Anthropic structured output fell back to legacy tool_use mode; "
                        "extended thinking is not compatible and will be disabled for this request.",
                        model=model,
                        structured_mode=structured_mode,
                    )
                else:
                    logger.warning(
                        "Extended thinking is incompatible with tool-forced structured output. "
                        "Disabling thinking for this request."
                    )
            base_args["tool_choice"] = {
                "type": "tool",
                "name": STRUCTURED_OUTPUT_TOOL_NAME,
            }

        thinking_args, thinking_enabled = self._resolve_thinking_arguments(
            model=model,
            max_tokens=params.maxTokens,
            structured_mode=structured_mode,
        )
        base_args.update(thinking_args)
        if self.task_budget_tokens is not None and self._supports_task_budget(model):
            output_config = dict(base_args.get("output_config") or {})
            output_config["task_budget"] = {
                "type": "tokens",
                "total": self.task_budget_tokens,
            }
            base_args["output_config"] = output_config
        if structured_mode == "json" and (structured_model or structured_schema):
            output_config = dict(base_args.get("output_config") or {})
            output_config["format"] = self._build_output_format(
                structured_model,
                structured_schema,
            )
            base_args["output_config"] = output_config
        return base_args, thinking_enabled

    def prepare_provider_arguments(
        self,
        base_args: dict,
        request_params: RequestParams,
        exclude_fields: set | None = None,
    ) -> dict:
        arguments = super().prepare_provider_arguments(base_args, request_params, exclude_fields)
        if self._normalize_model_name(str(arguments.get("model", ""))) not in {
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5",
            "claude-sonnet-5",
        }:
            return arguments

        removed_fields = [
            key for key in ("temperature", "top_p", "top_k") if arguments.pop(key, None) is not None
        ]
        if removed_fields:
            removed = ", ".join(removed_fields)
            self.logger.warning(
                f"Anthropic model ignores unsupported sampling parameters; removed {removed}."
            )
        return arguments

    def _resolve_anthropic_beta_flags(
        self,
        *,
        model: str,
        structured_mode: StructuredOutputMode | None,
        thinking_enabled: bool,
        request_tools: list[BetaToolParam],
        web_tool_betas: Sequence[str],
        provider_mcp_enabled: bool = False,
    ) -> list[str]:
        beta_flags: list[str] = []
        adaptive_thinking = self._supports_adaptive_thinking(model)
        if structured_mode == "json" and self.supports_direct_anthropic_beta("structured_output"):
            beta_flags.append(STRUCTURED_OUTPUT_BETA)
        if (
            thinking_enabled
            and request_tools
            and not adaptive_thinking
            and self.supports_direct_anthropic_beta("interleaved_thinking")
        ):
            beta_flags.append(INTERLEAVED_THINKING_BETA)
        if self._long_context and self.supports_direct_anthropic_beta("long_context"):
            beta_flags.append(LONG_CONTEXT_BETA)
        if request_tools and self.supports_direct_anthropic_beta("fine_grained_tool_streaming"):
            beta_flags.append(FINE_GRAINED_TOOL_STREAMING_BETA)
        if self.supports_direct_anthropic_beta("web_tools"):
            beta_flags.extend(web_tool_betas)
        if provider_mcp_enabled:
            beta_flags.append(MCP_CLIENT_BETA)
        if self.task_budget_tokens is not None and self._supports_task_budget(model):
            beta_flags.append(TASK_BUDGETS_BETA)
        return dedupe_preserve_order(beta_flags)

    def _apply_anthropic_cache_plan(
        self,
        *,
        arguments: dict[str, Any],
        messages: list[BetaMessageParam],
        params: RequestParams,
        cache_mode: str,
        history: list[PromptMessageExtended] | None,
        current_extended: PromptMessageExtended | None,
    ) -> None:
        system_cache_applied = self._apply_system_cache(arguments, cache_mode)

        planner = AnthropicCachePlanner(
            self.CONVERSATION_CACHE_WALK_DISTANCE, self.MAX_CONVERSATION_CACHE_BLOCKS
        )
        plan_messages: list[PromptMessageExtended] = []
        include_current = not params.use_history or not history
        if params.use_history and history:
            plan_messages.extend(history)
        if include_current and current_extended:
            plan_messages.append(current_extended)

        from fast_agent.history.process_poll_folding import (
            managed_process_poll_cache_boundary,
        )

        process_poll_boundary = managed_process_poll_cache_boundary(plan_messages)
        cache_indices = planner.plan_indices(
            plan_messages,
            cache_mode=cache_mode,
            system_cache_blocks=system_cache_applied,
            process_poll_boundary=process_poll_boundary,
        )
        cache_ttl = self._get_cache_ttl()
        for idx in cache_indices:
            if 0 <= idx < len(messages):
                self._apply_cache_control_to_message(messages[idx], ttl=cache_ttl)

    async def _execute_anthropic_stream(
        self,
        *,
        anthropic: Any,
        arguments: dict[str, Any],
        model: str,
        capture_filename: Path | None,
        timeout_seconds: float | None,
    ) -> tuple[BetaMessage, list[str], list[str]]:
        otel_span: Span | None = None
        otel_span_error = False
        response: BetaMessage | None = None

        try:
            stream_method = _maybe_unwrap_otel_beta_stream(anthropic.beta.messages.stream)
            if stream_method is not anthropic.beta.messages.stream:
                otel_span = _start_fallback_stream_span(model)

            stream_call = stream_method(**arguments)
            stream_manager = (
                await await_stream_start(
                    stream_call,
                    timeout_seconds=timeout_seconds,
                    timeout_message=f"Anthropic stream did not start within {timeout_seconds} seconds.",
                )
                if asyncio.iscoroutine(stream_call)
                else stream_call
            )
            stream_manager = cast("BetaAsyncMessageStream", stream_manager)

            if otel_span is not None:
                with trace.use_span(otel_span, end_on_exit=False):
                    async with enter_stream_with_timeout(
                        stream_manager,
                        timeout_seconds=timeout_seconds,
                        timeout_message=(
                            f"Anthropic stream did not start within {timeout_seconds} seconds."
                        ),
                    ) as stream:
                        (
                            response,
                            thinking_segments,
                            streamed_text_segments,
                        ) = await self._process_stream(stream, model, capture_filename)
            else:
                async with enter_stream_with_timeout(
                    stream_manager,
                    timeout_seconds=timeout_seconds,
                    timeout_message=f"Anthropic stream did not start within {timeout_seconds} seconds.",
                ) as stream:
                    (
                        response,
                        thinking_segments,
                        streamed_text_segments,
                    ) = await self._process_stream(stream, model, capture_filename)
        except APIError as error:
            if otel_span is not None and otel_span.is_recording():
                otel_span.record_exception(error)
                otel_span.set_status(Status(StatusCode.ERROR))
                otel_span_error = True
            logger.error("Streaming APIError during Anthropic completion", exc_info=error)
            raise
        except Exception as error:
            if otel_span is not None and otel_span.is_recording():
                otel_span.record_exception(error)
                otel_span.set_status(Status(StatusCode.ERROR))
                otel_span_error = True
            raise
        finally:
            if otel_span is not None:
                _finalize_fallback_stream_span(otel_span, response, otel_span_error)

        if response is None:
            raise RuntimeError("Anthropic stream completed without a final message.")
        return response, thinking_segments, streamed_text_segments

    def _anthropic_response_text_blocks(
        self,
        response: BetaMessage,
        model: str,
    ) -> list[ContentBlock]:
        response_content_blocks: list[ContentBlock] = []
        for index, content_block in enumerate(response.content or []):
            if not isinstance(content_block, BetaTextBlock):
                continue
            text_value = content_block.text
            if not isinstance(text_value, str):
                logger.warning(
                    "Skipping Anthropic text block with non-string text in final response",
                    data={
                        "model": model,
                        "index": index,
                        "text_type": type(text_value).__name__,
                    },
                )
                if os.environ.get("FAST_AGENT_WEBDEBUG"):
                    print(
                        "[webdebug] skipped invalid final text block "
                        f"model={model} index={index} "
                        f"text_type={type(text_value).__name__}"
                    )
                continue
            response_content_blocks.append(TextContent(type="text", text=text_value))
        return response_content_blocks

    @staticmethod
    def _apply_streamed_text_fallback(
        response_content_blocks: list[ContentBlock],
        streamed_text_segments: list[str],
    ) -> list[ContentBlock]:
        if not streamed_text_segments:
            return response_content_blocks

        streamed_text = "".join(streamed_text_segments)
        if not streamed_text.strip():
            return response_content_blocks

        provider_text = "".join(
            block.text
            for block in response_content_blocks
            if isinstance(block, TextContent) and block.text
        )
        if provider_text.strip() and provider_text.strip() == streamed_text.strip():
            return response_content_blocks
        return [TextContent(type="text", text=streamed_text)]

    async def _anthropic_stop_result(
        self,
        response: BetaMessage,
        structured_mode: StructuredOutputMode | None,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None,
        messages: list[BetaMessageParam],
    ) -> _AnthropicStopResult:
        match response.stop_reason:
            case "stop_sequence":
                return _AnthropicStopResult(LlmStopReason.STOP_SEQUENCE)
            case "max_tokens":
                return _AnthropicStopResult(LlmStopReason.MAX_TOKENS)
            case "refusal":
                return _AnthropicStopResult(LlmStopReason.SAFETY)
            case "pause" | "pause_turn":
                return _AnthropicStopResult(LlmStopReason.PAUSE)
            case "tool_use":
                return await self._anthropic_tool_stop_result(
                    response,
                    structured_mode,
                    structured_model,
                    structured_schema,
                    messages,
                )
        return _AnthropicStopResult(LlmStopReason.END_TURN)

    async def _anthropic_tool_stop_result(
        self,
        response: BetaMessage,
        structured_mode: StructuredOutputMode | None,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None,
        messages: list[BetaMessageParam],
    ) -> _AnthropicStopResult:
        tool_uses = [
            content for content in response.content if isinstance(content, BetaToolUseBlock)
        ]
        if not tool_uses:
            return _AnthropicStopResult(LlmStopReason.END_TURN)

        if (
            structured_mode == "tool_use"
            and (structured_model is not None or structured_schema is not None)
            and self._is_structured_output_request(tool_uses)
        ):
            stop_reason, structured_blocks = await self._handle_structured_output_response(
                tool_uses[0], structured_model, messages
            )
            return _AnthropicStopResult(
                stop_reason,
                structured_blocks=structured_blocks,
            )

        return _AnthropicStopResult(
            LlmStopReason.TOOL_USE,
            tool_calls=self._build_tool_calls_dict(tool_uses),
        )

    @staticmethod
    def _add_anthropic_channel(
        channels: dict[str, list[Any]] | None,
        key: str,
        values: list[Any],
    ) -> dict[str, list[Any]]:
        if channels is None:
            channels = {}
        channels[key] = values
        return channels

    @staticmethod
    def _anthropic_reasoning_channel(
        response: BetaMessage,
        thinking_segments: list[str],
    ) -> dict[str, list[Any]] | None:
        if thinking_segments:
            return {REASONING: [TextContent(type="text", text="".join(thinking_segments))]}

        thinking_texts = [
            block.thinking
            for block in response.content or []
            if isinstance(block, BetaThinkingBlock) and block.thinking
        ]
        if not thinking_texts:
            return None
        return {REASONING: [TextContent(type="text", text="".join(thinking_texts))]}

    @staticmethod
    def _raw_anthropic_thinking_blocks(
        response: BetaMessage,
    ) -> list[BetaThinkingBlock | BetaRedactedThinkingBlock]:
        return [
            block
            for block in response.content or []
            if isinstance(block, (BetaThinkingBlock, BetaRedactedThinkingBlock))
        ]

    @staticmethod
    def _serialized_thinking_blocks(
        raw_thinking_blocks: list[BetaThinkingBlock | BetaRedactedThinkingBlock],
    ) -> list[TextContent]:
        serialized_blocks: list[TextContent] = []
        for block in raw_thinking_blocks:
            try:
                payload: dict[str, Any] = block.model_dump()
            except Exception:
                payload = {"type": block.type}
                if isinstance(block, BetaThinkingBlock):
                    payload["thinking"] = block.thinking
                    payload["signature"] = block.signature
                elif isinstance(block, BetaRedactedThinkingBlock):
                    payload["data"] = block.data
            serialized_blocks.append(TextContent(type="text", text=json.dumps(payload)))
        return serialized_blocks

    @staticmethod
    def _safety_details_channel(response: BetaMessage) -> list[TextContent]:
        stop_details = response.stop_details
        if stop_details is None:
            return []
        if isinstance(stop_details, _ModelDumpable):
            stop_payload = stop_details.model_dump(mode="json", exclude_none=True)
        else:
            stop_payload = stop_details
        payload: dict[str, Any] = {"provider": "anthropic"}
        if isinstance(stop_payload, dict):
            reason = stop_payload.get("type")
            if isinstance(reason, str):
                payload["reason"] = reason
            category = stop_payload.get("category")
            if isinstance(category, str) and category:
                payload["category"] = category
            explanation = stop_payload.get("explanation")
            if isinstance(explanation, str) and explanation:
                payload["explanation"] = explanation
            fallback_credit_token = stop_payload.get("fallback_credit_token")
            if isinstance(fallback_credit_token, str) and fallback_credit_token:
                payload["fallback_credit_token"] = fallback_credit_token
            fallback_has_prefill_claim = stop_payload.get("fallback_has_prefill_claim")
            if isinstance(fallback_has_prefill_claim, bool):
                payload["fallback_has_prefill_claim"] = fallback_has_prefill_claim
            recommended_model = stop_payload.get("recommended_model")
            if isinstance(recommended_model, str) and recommended_model:
                payload["recommended_model"] = recommended_model
        else:
            payload["details"] = str(stop_payload)
        return [TextContent(type="text", text=json.dumps(payload))]

    @staticmethod
    def _test_refusal_category() -> str | None:
        raw_value = os.environ.get(TEST_REFUSAL_TRIGGER_ENV)
        if raw_value is None:
            return None
        category = strip_casefold(raw_value)
        if not category or category in {"0", "false", "off", "no"}:
            return None
        allowed = {"bio", "cyber", "frontier_llm", "reasoning_extraction", "none", "null"}
        if category not in allowed:
            logger.warning(
                "Ignoring invalid Anthropic test refusal trigger category",
                data={
                    "env": TEST_REFUSAL_TRIGGER_ENV,
                    "category": raw_value,
                    "allowed": sorted(allowed),
                },
            )
            return None
        return category

    @staticmethod
    def _test_refusal_stop_details(category: str) -> list[TextContent]:
        payload: dict[str, Any] = {"provider": "anthropic", "reason": "refusal"}
        if category not in {"none", "null"}:
            payload["category"] = category
        return [TextContent(type="text", text=json.dumps(payload))]

    @staticmethod
    def _anthropic_provider_payloads(response: BetaMessage) -> _AnthropicProviderPayloads:
        payloads = _AnthropicProviderPayloads()
        for block in response.content or []:
            payload = serialize_anthropic_block_payload(block)
            if payload is not None:
                try:
                    payloads.raw_assistant.append(
                        TextContent(type="text", text=json.dumps(payload))
                    )
                except Exception as error:
                    logger.warning(
                        "Skipping non-serializable assistant block payload",
                        data={
                            "payload_type": payload.get("type"),
                            "error": str(error),
                        },
                    )

            if payload is not None and is_server_tool_trace_payload(payload):
                payloads.server_tools.append(TextContent(type="text", text=json.dumps(payload)))

            if isinstance(block, BetaTextBlock) and block.citations:
                extracted = extract_citation_payloads(block.citations)
                payloads.citations.extend(
                    TextContent(type="text", text=json.dumps(citation_payload))
                    for citation_payload in extracted
                )
        return payloads

    def _anthropic_response_channels(
        self,
        response: BetaMessage,
        model: str,
        thinking_segments: list[str],
        tool_calls: dict[str, CallToolRequest] | None,
    ) -> dict[str, list[Any]] | None:
        channels = self._anthropic_reasoning_channel(response, thinking_segments)
        raw_thinking_blocks = self._raw_anthropic_thinking_blocks(response)
        if raw_thinking_blocks:
            channels = self._add_anthropic_channel(
                channels,
                ANTHROPIC_THINKING_BLOCKS,
                self._serialized_thinking_blocks(raw_thinking_blocks),
            )

        payloads = self._anthropic_provider_payloads(response)
        if os.environ.get("FAST_AGENT_WEBDEBUG"):
            print(
                "[webdebug]"
                f" model={model}"
                f" response_blocks={len(response.content or [])}"
                f" server_tool_payloads={len(payloads.server_tools)}"
                f" citation_payloads={len(payloads.citations)}"
            )

        if payloads.server_tools:
            channels = self._add_anthropic_channel(
                channels, ANTHROPIC_SERVER_TOOLS_CHANNEL, payloads.server_tools
            )
        if payloads.raw_assistant and (
            raw_thinking_blocks or payloads.server_tools or tool_calls is not None
        ):
            channels = self._add_anthropic_channel(
                channels, ANTHROPIC_ASSISTANT_RAW_CONTENT, payloads.raw_assistant
            )
        if payloads.citations:
            channels = self._add_anthropic_channel(
                channels, ANTHROPIC_CITATIONS_CHANNEL, payloads.citations
            )
        if stop_details := self._safety_details_channel(response):
            channels = self._add_anthropic_channel(
                channels,
                FAST_AGENT_SAFETY_DETAILS,
                stop_details,
            )
        if response.container and response.container.id:
            channels = self._add_anthropic_channel(
                channels,
                ANTHROPIC_CONTAINER_CHANNEL,
                [TextContent(type="text", text=json.dumps({"id": response.container.id}))],
            )
        return channels

    async def _finalize_anthropic_response(
        self,
        *,
        response: BetaMessage,
        model: str,
        messages: list[BetaMessageParam],
        thinking_segments: list[str],
        streamed_text_segments: list[str],
        structured_mode: StructuredOutputMode | None,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None = None,
    ) -> PromptMessageExtended:
        response_as_message = self.convert_message_to_message_param(response)
        messages.append(response_as_message)
        response_content_blocks = self._anthropic_response_text_blocks(response, model)
        response_content_blocks = self._apply_streamed_text_fallback(
            response_content_blocks,
            streamed_text_segments,
        )
        stop_result = await self._anthropic_stop_result(
            response,
            structured_mode,
            structured_model,
            structured_schema,
            messages,
        )
        response_content_blocks.extend(stop_result.structured_blocks)
        channels = self._anthropic_response_channels(
            response,
            model,
            thinking_segments,
            stop_result.tool_calls,
        )
        if test_refusal_category := self._test_refusal_category():
            stop_result = _AnthropicStopResult(LlmStopReason.SAFETY)
            channels = self._add_anthropic_channel(
                channels,
                FAST_AGENT_SAFETY_DETAILS,
                self._test_refusal_stop_details(test_refusal_category),
            )

        return PromptMessageExtended(
            role="assistant",
            content=response_content_blocks,
            tool_calls=stop_result.tool_calls,
            channels=channels,
            stop_reason=stop_result.stop_reason,
        )

    async def _prepare_anthropic_completion_request(
        self,
        anthropic: Any,
        message_param: BetaMessageParam,
        request_params: RequestParams | None,
        pre_messages: list[BetaMessageParam] | None,
        history: list[PromptMessageExtended] | None,
        current_extended: PromptMessageExtended | None,
    ) -> _AnthropicCompletionRequest:
        params = self.get_request_params(request_params)
        messages_to_prepare: list[PromptMessageExtended] = []
        if history:
            messages_to_prepare.extend(history)
        if current_extended is not None:
            messages_to_prepare.append(current_extended)
        if messages_to_prepare:
            await self._prepare_anthropic_file_resources(anthropic, messages_to_prepare)
        if current_extended is not None:
            message_param = AnthropicConverter.convert_to_anthropic(current_extended)
        messages = self._build_request_messages(
            params,
            message_param,
            pre_messages,
            history=history,
        )
        return _AnthropicCompletionRequest(
            client=anthropic,
            params=params,
            messages=messages,
            message_param=message_param,
        )

    def _resolve_anthropic_structured_mode(
        self,
        model: str,
        params: RequestParams,
        structured_model: type[ModelT] | None,
        structured_schema: dict[str, Any] | None,
    ) -> _AnthropicStructuredMode:
        effective_schema = (
            params.structured_schema if params.structured_schema is not None else structured_schema
        )
        structured_mode = self._resolve_structured_output_mode(
            model,
            structured_model,
            effective_schema,
        )
        auto_tool_use_fallback = self._is_auto_tool_use_structured_fallback(
            model,
            structured_mode,
            structured_model,
            effective_schema,
        )
        return _AnthropicStructuredMode(
            effective_schema=effective_schema,
            mode=structured_mode,
            auto_tool_use_fallback=auto_tool_use_fallback,
        )

    async def _anthropic_request_tools(
        self,
        model: str,
        structured_model: type[ModelT] | None,
        structured: _AnthropicStructuredMode,
        tools: list[Tool] | None,
    ) -> tuple[list[BetaToolParam], list[str], Any]:
        available_tools = await self._prepare_tools(
            model,
            structured_model,
            structured.effective_schema,
            tools,
            structured_mode=structured.mode,
            auto_tool_use_fallback=structured.auto_tool_use_fallback,
        )
        web_tools, web_tool_betas = self._prepare_web_tools(model)
        request_tools = [*available_tools, *web_tools]
        provider_mcp_payload = build_anthropic_provider_managed_mcp_payload(
            self.provider_managed_mcp_state
        )
        if provider_mcp_payload.tools:
            request_tools.extend(cast("list[BetaToolParam]", provider_mcp_payload.tools))
        return request_tools, list(web_tool_betas), provider_mcp_payload

    def _track_anthropic_usage(self, response: BetaMessage, model: str) -> None:
        if not response.usage:
            return
        try:
            turn_usage = usage_from_anthropic(
                response.usage,
                provider=self.provider,
                model=model or DEFAULT_ANTHROPIC_MODEL,
            )
            self._finalize_turn_usage(turn_usage)
        except Exception as exc:
            logger.warning(f"Failed to track usage: {exc}")

    async def _anthropic_completion(
        self,
        message_param,
        request_params: RequestParams | None = None,
        structured_model: type[ModelT] | None = None,
        structured_schema: dict[str, Any] | None = None,
        tools: list[Tool] | None = None,
        pre_messages: list[BetaMessageParam] | None = None,
        history: list[PromptMessageExtended] | None = None,
        current_extended: PromptMessageExtended | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using an LLM and available tools.
        Override this method to use a different LLM.
        """

        try:
            anthropic = self._initialize_anthropic_client()
            request = await self._prepare_anthropic_completion_request(
                anthropic,
                message_param,
                request_params,
                pre_messages,
                history,
                current_extended,
            )
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from e

        # Get cache mode configuration
        cache_mode = self._get_cache_mode()
        logger.debug(f"Anthropic cache_mode: {cache_mode}")

        model = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL

        structured = self._resolve_anthropic_structured_mode(
            model,
            request.params,
            structured_model,
            structured_schema,
        )
        request_tools, web_tool_betas, provider_mcp_payload = await self._anthropic_request_tools(
            model,
            structured_model,
            structured,
            tools,
        )

        base_args, thinking_enabled = self._build_anthropic_base_args(
            model=model,
            messages=request.messages,
            params=request.params,
            history=history,
            current_extended=current_extended,
            request_tools=request_tools,
            structured_mode=structured.mode,
            structured_model=structured_model,
            structured_schema=structured.effective_schema,
            auto_tool_use_fallback=structured.auto_tool_use_fallback,
        )

        beta_flags = self._resolve_anthropic_beta_flags(
            model=model,
            structured_mode=structured.mode,
            thinking_enabled=thinking_enabled,
            request_tools=request_tools,
            web_tool_betas=web_tool_betas,
            provider_mcp_enabled=bool(provider_mcp_payload.servers),
        )
        if beta_flags:
            base_args["betas"] = beta_flags
        if provider_mcp_payload.servers:
            base_args["mcp_servers"] = provider_mcp_payload.servers

        self._log_chat_progress(self.chat_turn(), model=model)
        # Use the base class method to prepare all arguments with Anthropic-specific exclusions
        # Do this BEFORE applying cache control so metadata doesn't override cached fields
        arguments = self.prepare_provider_arguments(
            base_args, request.params, self.ANTHROPIC_EXCLUDE_FIELDS
        )

        self._apply_anthropic_cache_plan(
            arguments=arguments,
            messages=request.messages,
            params=request.params,
            cache_mode=cache_mode,
            history=history,
            current_extended=current_extended,
        )

        logger.debug(f"{arguments}")

        # Generate stream capture filename once (before streaming starts)
        capture_filename = _stream_capture_filename(self.chat_turn())
        _save_stream_request(capture_filename, arguments)

        try:
            (
                response,
                thinking_segments,
                streamed_text_segments,
            ) = await self._execute_anthropic_stream(
                anthropic=request.client,
                arguments=arguments,
                model=model,
                capture_filename=capture_filename,
                timeout_seconds=request.params.streaming_timeout,
            )
        except asyncio.CancelledError as e:
            reason = str(e) if e.args else "cancelled"
            logger.info(f"Anthropic completion cancelled: {reason}")
            # Return a response indicating cancellation
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )

        # Track usage if response is valid and has usage data
        self._track_anthropic_usage(response, model)

        if isinstance(response, AuthenticationError):
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from response
        if isinstance(response, BaseException):
            # This path shouldn't be reached anymore since we handle APIError above,
            # but keeping for backward compatibility
            logger.error(f"Unexpected error type: {type(response).__name__}", exc_info=response)
            return build_stream_failure_response(self.provider, response, model)

        logger.debug(
            f"{model} response:",
            data=response,
        )

        result = await self._finalize_anthropic_response(
            response=response,
            model=model,
            messages=request.messages,
            thinking_segments=thinking_segments,
            streamed_text_segments=streamed_text_segments,
            structured_mode=structured.mode,
            structured_model=structured_model,
            structured_schema=structured.effective_schema,
        )

        # Update diagnostic snapshot (never read again)
        # This provides a snapshot of what was sent to the provider for debugging
        self.history.set(request.messages)

        self._log_chat_finished(model=model)
        return result

    def _prepare_structured_request(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> tuple[list[PromptMessageExtended], RequestParams]:
        if not self._should_suppress_structured_schema_for_tools(messages, request_params, tools):
            return messages, request_params
        return messages, request_params.model_copy(update={"structured_schema": None})

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
        # Check the last message role
        last_message = multipart_messages[-1]

        if last_message.role == "user":
            logger.debug("Last message in prompt is from user, generating assistant response")
            message_param = AnthropicConverter.convert_to_anthropic(last_message)
            # No need to pass pre_messages - conversion happens in _anthropic_completion
            # via _convert_to_provider_format()
            return await self._anthropic_completion(
                message_param,
                request_params,
                tools=tools,
                pre_messages=None,
                history=multipart_messages,
                current_extended=last_message,
            )
        # For assistant messages: Return the last message content as text
        logger.debug("Last message in prompt is from assistant, returning it directly")
        return last_message

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: list[PromptMessageExtended],
        model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """
        Provider-specific structured output implementation.
        Note: BetaMessage history is managed by base class and converted via
        _convert_to_provider_format() on each call.
        """
        request_params = self.get_request_params(request_params)

        # Check the last message role
        last_message = multipart_messages[-1]

        if last_message.role == "user":
            logger.debug("Last message in prompt is from user, generating structured response")
            message_param = AnthropicConverter.convert_to_anthropic(last_message)

            # Call _anthropic_completion with the structured model
            result: PromptMessageExtended = await self._anthropic_completion(
                message_param,
                request_params,
                structured_model=model,
                history=multipart_messages,
                current_extended=last_message,
            )
            return self._structured_from_multipart(result, model)
        # For assistant messages: Return the last message content
        logger.debug("Last message in prompt is from assistant, returning it directly")
        return None, last_message

    async def _apply_prompt_provider_specific_structured_schema(
        self,
        multipart_messages: list[PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        request_params = self.get_request_params(request_params)
        last_message = multipart_messages[-1]

        if last_message.role == "user":
            logger.debug(
                "Last message in prompt is from user, generating structured schema response"
            )
            message_param = AnthropicConverter.convert_to_anthropic(last_message)
            result = await self._anthropic_completion(
                message_param,
                request_params,
                structured_schema=schema,
                history=multipart_messages,
                current_extended=last_message,
            )
            return self._structured_schema_from_multipart(result, schema)

        logger.debug("Last message in prompt is from assistant, returning it directly")
        return self._structured_schema_from_multipart(last_message, schema)

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[BetaMessageParam]:
        """
        Convert PromptMessageExtended list to Anthropic BetaMessageParam format.
        This is called fresh on every API call from _convert_to_provider_format().

        Args:
            messages: List of PromptMessageExtended objects

        Returns:
            List of Anthropic BetaMessageParam objects
        """
        return [AnthropicConverter.convert_to_anthropic(msg) for msg in messages]

    @classmethod
    def convert_message_to_message_param(cls, message: BetaMessage, **kwargs) -> BetaMessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content: list[BetaContentBlockParam] = []

        for content_block in message.content:
            if isinstance(content_block, BetaTextBlock):
                text_value = getattr(content_block, "text", None)
                if not isinstance(text_value, str):
                    logger.warning(
                        "Skipping Anthropic text block with non-string text while converting message",
                        data={"text_type": type(text_value).__name__},
                    )
                    continue
                content.append(BetaTextBlockParam(type="text", text=text_value))
            elif isinstance(content_block, BetaToolUseBlock):
                content.append(
                    BetaToolUseBlockParam(
                        type="tool_use",
                        name=content_block.name,
                        input=content_block.input,
                        id=content_block.id,
                    )
                )
            elif isinstance(content_block, BetaServerToolUseBlock):
                payload = serialize_anthropic_block_payload(content_block)
                if payload is not None and is_server_tool_trace_payload(payload):
                    content.append(cast("BetaContentBlockParam", payload))
            else:
                payload = serialize_anthropic_block_payload(content_block)
                if payload is not None and is_server_tool_trace_payload(payload):
                    content.append(cast("BetaContentBlockParam", payload))

        return BetaMessageParam(role="assistant", content=content, **kwargs)
