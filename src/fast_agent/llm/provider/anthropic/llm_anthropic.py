import asyncio
import inspect
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Type, Union, cast

from anthropic import APIError, AsyncAnthropic, AuthenticationError, transform_schema
from anthropic.lib.streaming import BetaAsyncMessageStream
from mcp import Tool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    TextContent,
)
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Span, Status, StatusCode

from fast_agent.constants import (
    ANTHROPIC_ASSISTANT_RAW_CONTENT,
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_CONTAINER_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    ANTHROPIC_THINKING_BLOCKS,
    REASONING,
)
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import ModelT
from fast_agent.llm.fastagent_llm import (
    FastAgentLLM,
    RequestParams,
)
from fast_agent.llm.provider.anthropic.beta_types import (
    InputJSONDelta,
    Message,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RedactedThinkingBlock,
    ServerToolUseBlock,
    SignatureDelta,
    TextBlock,
    TextBlockParam,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolParam,
    ToolUseBlock,
    ToolUseBlockParam,
    Usage,
)
from fast_agent.llm.provider.anthropic.cache_planner import AnthropicCachePlanner
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import (
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
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import (
    AUTO_REASONING,
    format_reasoning_setting,
    is_auto_reasoning,
    parse_reasoning_setting,
)
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.structured_output_mode import StructuredOutputMode
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

DEFAULT_ANTHROPIC_MODEL = "sonnet"
STRUCTURED_OUTPUT_TOOL_NAME = "return_structured_output"
STRUCTURED_OUTPUT_BETA = "structured-outputs-2025-11-13"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"

# TODO: Remove beta header once Anthropic promotes 1M context to GA.
LONG_CONTEXT_BETA = "context-1m-2025-08-07"

# Beta for fine-grained tool streaming - enables incremental tool input streaming
# https://docs.anthropic.com/en/docs/build-with-claude/tool-use#streaming-tool-inputs
FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"

# Stream capture mode - when enabled, saves all streaming chunks to files for debugging
# Set FAST_AGENT_LLM_TRACE=1 (or any non-empty value) to enable
STREAM_CAPTURE_ENABLED = bool(os.environ.get("FAST_AGENT_LLM_TRACE"))
STREAM_CAPTURE_DIR = Path("stream-debug")

# Type alias for system field - can be string or list of text blocks with cache control
SystemParam = Union[str, list[TextBlockParam]]

logger = get_logger(__name__)

_OTEL_STREAM_WRAPPER_WARNED = False


def _is_beta_text_block_validation_error(error: Exception) -> bool:
    """Return True when Anthropic SDK rejects a text block with null text."""
    detail = f"{type(error).__name__}: {error}".lower()
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
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(warnings="none")
        except TypeError:
            return value.model_dump()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_trace(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_for_trace(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
        with open(request_file, "w") as f:
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
    response: Message | None,
    had_error: bool,
) -> None:
    if not span.is_recording():
        span.end()
        return
    if response is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_ID, response.id)
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.model)
        if response.usage:
            input_tokens = response.usage.input_tokens or 0
            cache_read_tokens = response.usage.cache_read_input_tokens or 0
            cache_creation_tokens = response.usage.cache_creation_input_tokens or 0
            input_total = input_tokens + cache_read_tokens + cache_creation_tokens
            output_tokens = response.usage.output_tokens or 0
            span.set_attribute(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, input_total)
            span.set_attribute(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
            span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, input_total + output_tokens)
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
        with open(chunk_file, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        logger.debug(f"Failed to save stream chunk: {e}")


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure object schemas explicitly set additionalProperties=false."""
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "additionalProperties" not in node:
                node["additionalProperties"] = False

            for key, value in node.items():
                if key in {"properties", "$defs", "definitions", "patternProperties"}:
                    if isinstance(value, dict):
                        for child in value.values():
                            visit(child)
                    continue
                if key in {"items", "anyOf", "oneOf", "allOf"}:
                    if isinstance(value, list):
                        for child in value:
                            visit(child)
                    else:
                        visit(value)
                    continue
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(result)
    return result


class AnthropicLLM(FastAgentLLM[MessageParam, Message]):
    CONVERSATION_CACHE_WALK_DISTANCE = 6
    MAX_CONVERSATION_CACHE_BLOCKS = 2
    # Anthropic-specific parameter exclusions
    ANTHROPIC_EXCLUDE_FIELDS = {
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
        super().__init__(provider=Provider.ANTHROPIC, **kwargs)
        self._structured_output_mode_override: StructuredOutputMode | None = structured_override
        self._web_search_override: bool | None = (
            bool(web_search_override) if isinstance(web_search_override, bool) else None
        )
        self._web_fetch_override: bool | None = (
            bool(web_fetch_override) if isinstance(web_fetch_override, bool) else None
        )

        raw_setting = kwargs.get("reasoning_effort", None)
        reasoning_source: str | None = None
        if raw_setting is not None:
            reasoning_source = "llm_kwargs"
        config = self.context.config.anthropic if self.context and self.context.config else None
        model_name = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL
        if raw_setting is None and config:
            raw_setting = config.reasoning
            if raw_setting is not None:
                reasoning_source = "config_reasoning"

        from fast_agent.llm.model_database import ModelDatabase

        reasoning_mode = ModelDatabase.get_reasoning(model_name)
        spec = ModelDatabase.get_reasoning_effort_spec(model_name)

        if raw_setting is not None and reasoning_mode != "anthropic_thinking":
            self.logger.warning(
                "Reasoning setting ignored for model without Anthropic thinking support."
            )
            raw_setting = None
            reasoning_source = None

        if raw_setting is None and reasoning_mode == "anthropic_thinking":
            if spec and spec.kind == "effort" and spec.allow_auto:
                # Adaptive-thinking model: use "auto" so the API omits the
                # effort parameter and lets the provider choose automatically.
                raw_setting = AUTO_REASONING
            else:
                raw_setting = spec.default if spec and spec.default else 1024
            if raw_setting is not None:
                reasoning_source = "model_default"

        setting = parse_reasoning_setting(raw_setting)
        if setting is not None:
            try:
                self.set_reasoning_effort(setting)
            except ValueError as exc:
                self.logger.warning(f"Invalid reasoning setting: {exc}")
                if spec and spec.default:
                    self.set_reasoning_effort(spec.default)
                    reasoning_source = "model_default"
                else:
                    self.set_reasoning_effort(None)
        else:
            self.set_reasoning_effort(None)

        if ModelDatabase.get_reasoning(model_name) == "anthropic_thinking":
            resolved_setting = self.reasoning_effort
            thinking_enabled = self._is_thinking_enabled(model_name)
            payload = {
                "model": model_name,
                "setting": format_reasoning_setting(resolved_setting),
                "reasoning_source": reasoning_source or "unknown",
                "thinking_enabled": thinking_enabled,
                "config_path": (
                    self.context.config._config_file
                    if self.context and self.context.config
                    else None
                ),
            }
            if thinking_enabled:
                self.logger.event(
                    "info",
                    "anthropic_reasoning",
                    "Anthropic reasoning resolved",
                    None,
                    payload,
                )
            else:
                self.logger.event(
                    "warning",
                    "anthropic_reasoning",
                    "Anthropic reasoning disabled",
                    None,
                    payload,
                )

        # Long context (1M) setup
        self._long_context = False
        if long_context_requested:
            model_name = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL
            long_context_window = ModelDatabase.get_long_context_window(model_name)
            if long_context_window is not None:
                self._long_context = True
                self._context_window_override = long_context_window
                self._usage_accumulator.set_context_window_override(long_context_window)
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

    def _list_supported_long_context_models(self) -> list[str]:
        """Return models that support explicit long-context overrides."""
        from fast_agent.llm.model_database import ModelDatabase

        return ModelDatabase.list_long_context_models()

    def _base_url(self) -> str | None:
        assert self.context.config
        return self.context.config.anthropic.base_url if self.context.config.anthropic else None

    def _default_headers(self) -> dict[str, str] | None:
        """Get custom default headers from configuration."""
        assert self.context.config
        return (
            self.context.config.anthropic.default_headers if self.context.config.anthropic else None
        )

    def _get_cache_mode(self) -> str:
        """Get the cache mode configuration."""
        cache_mode = "auto"  # Default to auto
        if self.context.config and self.context.config.anthropic:
            cache_mode = self.context.config.anthropic.cache_mode
        return cache_mode

    def _get_cache_ttl(self) -> str:
        """Get the cache TTL configuration ('5m' or '1h')."""
        cache_ttl = "5m"  # Default to 5 minutes
        if self.context.config and self.context.config.anthropic:
            cache_ttl = self.context.config.anthropic.cache_ttl
        return cache_ttl

    def _supports_adaptive_thinking(self, model: str) -> bool:
        """Return True when model uses adaptive thinking instead of manual budgets."""
        from fast_agent.llm.model_database import ModelDatabase

        if ModelDatabase.get_reasoning(model) != "anthropic_thinking":
            return False
        spec = ModelDatabase.get_reasoning_effort_spec(model)
        return bool(spec and spec.kind == "effort")

    def _is_thinking_enabled(self, model: str) -> bool:
        """Check if extended thinking should be enabled for this request."""
        from fast_agent.llm.model_database import ModelDatabase

        if ModelDatabase.get_reasoning(model) != "anthropic_thinking":
            return False
        setting = self.reasoning_effort
        if setting is None:
            return False
        if is_auto_reasoning(setting):
            return self._supports_adaptive_thinking(model)
        if setting.kind == "toggle":
            return bool(setting.value)
        if setting.kind == "budget":
            return bool(setting.value)
        if setting.kind == "effort":
            if str(setting.value).lower() == "none":
                return False
            return self._supports_adaptive_thinking(model)
        return False

    def _resolve_adaptive_effort(self) -> str | None:
        """Resolve adaptive effort for Anthropic output_config."""
        setting = self.reasoning_effort
        if setting is None or setting.kind != "effort":
            return None
        if is_auto_reasoning(setting):
            return None
        effort = str(setting.value).lower()
        if effort == "xhigh":
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

        if thinking_enabled and structured_mode == "tool_use":
            if max_tokens is not None:
                args["max_tokens"] = max_tokens
            return args, False

        if not thinking_enabled:
            if max_tokens is not None:
                args["max_tokens"] = max_tokens
            return args, False

        if adaptive_supported:
            args["thinking"] = {"type": "adaptive"}
            effort = self._resolve_adaptive_effort()
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
        self, model: str, structured_model: Type[ModelT] | None
    ) -> StructuredOutputMode | None:
        if structured_model is None:
            return None
        if self._structured_output_mode_override is not None:
            return self._structured_output_mode_override
        config = self.context.config.anthropic if self.context and self.context.config else None
        if config and config.structured_output_mode != "auto":
            return config.structured_output_mode
        from fast_agent.llm.model_database import ModelDatabase

        json_mode = ModelDatabase.get_json_mode(model)
        if json_mode == "schema":
            return "json"
        return "tool_use"

    def _build_output_format(self, structured_model: Type[ModelT]) -> dict[str, Any]:
        try:
            schema = transform_schema(structured_model)
        except Exception:
            schema = structured_model.model_json_schema()
        return {"type": "json_schema", "schema": schema}

    async def _prepare_tools(
        self,
        structured_model: Type[ModelT] | None = None,
        tools: list[Tool] | None = None,
        structured_mode: StructuredOutputMode | None = None,
    ) -> list[ToolParam]:
        """Prepare tools based on whether we're in structured output mode."""
        if structured_model and structured_mode == "tool_use":
            schema = _ensure_additional_properties_false(structured_model.model_json_schema())
            return [
                ToolParam(
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    description="Return the response in the required JSON format",
                    input_schema=schema,
                    strict=True,
                )
            ]
        if structured_model:
            return []
        # Regular mode - use tools from aggregator
        return [
            ToolParam(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in tools or []
        ]

    def _prepare_web_tools(self, model: str) -> tuple[list[ToolParam], tuple[str, ...]]:
        anthropic_settings = self.context.config.anthropic if self.context.config else None
        resolved = resolve_web_tools(
            anthropic_settings,
            web_search_override=self._web_search_override,
            web_fetch_override=self._web_fetch_override,
        )
        return build_web_tool_params(model, resolved_tools=resolved)

    @property
    def web_tools_enabled(self) -> tuple[bool, bool]:
        """Return (search_enabled, fetch_enabled) for toolbar display."""
        anthropic_settings = self.context.config.anthropic if self.context.config else None
        resolved = resolve_web_tools(
            anthropic_settings,
            web_search_override=self._web_search_override,
            web_fetch_override=self._web_fetch_override,
        )
        return resolved.search_enabled, resolved.fetch_enabled

    @property
    def web_search_supported(self) -> bool:
        from fast_agent.llm.model_database import ModelDatabase

        model_name = self.model_name
        if not model_name:
            return False
        return ModelDatabase.get_anthropic_web_search_version(model_name) is not None

    def set_web_search_enabled(self, value: bool | None) -> None:
        if value is None:
            self._web_search_override = None
            return
        if not self.web_search_supported:
            raise ValueError("Current model does not support web search configuration.")
        self._web_search_override = value

    @property
    def web_fetch_supported(self) -> bool:
        from fast_agent.llm.model_database import ModelDatabase

        model_name = self.model_name
        if not model_name:
            return False
        return ModelDatabase.get_anthropic_web_fetch_version(model_name) is not None

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
        """Whether any Anthropic web tooling is enabled for this LLM instance."""
        search_enabled, fetch_enabled = self.web_tools_enabled
        return search_enabled or fetch_enabled

    def _apply_system_cache(self, base_args: dict, cache_mode: str) -> int:
        """Apply cache control to system prompt if cache mode allows it."""
        system_content: SystemParam | None = base_args.get("system")

        if cache_mode != "off" and system_content:
            cache_ttl = self._get_cache_ttl()
            # Convert string to list format with cache control
            if isinstance(system_content, str):
                base_args["system"] = [
                    TextBlockParam(
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
            elif isinstance(system_content, list):
                logger.debug("System prompt already in list format")
            else:
                logger.debug(f"Unexpected system prompt type: {type(system_content)}")

        return 0

    @staticmethod
    def _apply_cache_control_to_message(message: MessageParam, ttl: str = "5m") -> bool:
        """Apply cache control to the last content block of a message."""
        if not isinstance(message, dict) or "content" not in message:
            return False

        content_list = message["content"]
        if not isinstance(content_list, list) or not content_list:
            return False

        for content_block in reversed(content_list):
            if isinstance(content_block, dict):
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

    def _build_tool_calls_dict(self, tool_uses: list[ToolUseBlock]) -> dict[str, CallToolRequest]:
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
        tool_use_block: ToolUseBlock,
        structured_model: Type[ModelT],
        messages: list[MessageParam],
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
            structured_model: The model class for structured output
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

    async def _process_stream(
        self,
        stream: BetaAsyncMessageStream,
        model: str,
        capture_filename: Path | None = None,
    ) -> tuple[Message, list[str], list[str]]:
        """Process the streaming response and display real-time token usage."""
        # Track estimated output tokens by counting text chunks
        estimated_tokens = 0
        tool_streams: dict[int, dict[str, Any]] = {}
        server_tool_streams: dict[int, dict[str, Any]] = {}
        thinking_segments: list[str] = []
        streamed_text_segments: list[str] = []
        thinking_indices: set[int] = set()

        try:
            # Process the raw event stream to get token counts
            # Cancellation is handled via asyncio.Task.cancel() which raises CancelledError
            async for event in stream:
                # Save chunk if stream capture is enabled
                _save_stream_chunk(capture_filename, event)

                if isinstance(event, RawContentBlockStartEvent):
                    content_block = event.content_block
                    if isinstance(content_block, (ThinkingBlock, RedactedThinkingBlock)):
                        thinking_indices.add(event.index)
                        continue
                    if isinstance(content_block, ToolUseBlock):
                        tool_streams[event.index] = {
                            "name": content_block.name,
                            "id": content_block.id,
                            "buffer": [],
                        }
                        self._notify_tool_stream_listeners(
                            "start",
                            {
                                "tool_name": content_block.name,
                                "tool_use_id": content_block.id,
                                "index": event.index,
                            },
                        )
                        self.logger.info(
                            "Model started streaming tool input",
                            data={
                                "progress_action": ProgressAction.CALLING_TOOL,
                                "agent_name": self.name,
                                "model": model,
                                "tool_name": content_block.name,
                                "tool_use_id": content_block.id,
                                "tool_event": "start",
                            },
                        )
                        continue
                    if isinstance(content_block, ServerToolUseBlock):
                        server_tool_streams[event.index] = {
                            "name": content_block.name,
                            "id": content_block.id,
                        }
                        progress_label = web_tool_progress_label(content_block.name)
                        preview_chunk = "…"
                        if content_block.input:
                            try:
                                preview_chunk = json.dumps(content_block.input)
                            except Exception:
                                preview_chunk = "…"
                            if len(preview_chunk) > 120:
                                preview_chunk = f"{preview_chunk[:117]}..."
                        self._notify_tool_stream_listeners(
                            "start",
                            {
                                "tool_name": content_block.name,
                                "tool_display_name": progress_label,
                                "chunk": preview_chunk,
                                "tool_use_id": content_block.id,
                                "index": event.index,
                            },
                        )
                        self.logger.info(
                            "Anthropic server tool started",
                            data={
                                "progress_action": ProgressAction.CALLING_TOOL,
                                "agent_name": self.name,
                                "model": model,
                                "tool_name": content_block.name,
                                "tool_use_id": content_block.id,
                                "tool_event": "start",
                                "details": progress_label,
                            },
                        )
                        continue

                if isinstance(event, RawContentBlockDeltaEvent):
                    delta = event.delta
                    if isinstance(delta, ThinkingDelta):
                        if delta.thinking:
                            self._notify_stream_listeners(
                                StreamChunk(text=delta.thinking, is_reasoning=True)
                            )
                            thinking_segments.append(delta.thinking)
                        continue
                    if isinstance(delta, SignatureDelta):
                        continue
                    if isinstance(delta, InputJSONDelta):
                        info = tool_streams.get(event.index)
                        if info is not None:
                            chunk = delta.partial_json or ""
                            info["buffer"].append(chunk)
                            preview = chunk if len(chunk) <= 80 else chunk[:77] + "..."
                            self._notify_tool_stream_listeners(
                                "delta",
                                {
                                    "tool_name": info.get("name"),
                                    "tool_use_id": info.get("id"),
                                    "index": event.index,
                                    "chunk": chunk,
                                },
                            )
                            self.logger.debug(
                                "Streaming tool input delta",
                                data={
                                    "tool_name": info.get("name"),
                                    "tool_use_id": info.get("id"),
                                    "chunk": preview,
                                },
                            )
                        continue

                if isinstance(event, RawContentBlockStopEvent) and event.index in thinking_indices:
                    thinking_indices.discard(event.index)
                    continue

                if isinstance(event, RawContentBlockStopEvent) and event.index in tool_streams:
                    info = tool_streams.pop(event.index)
                    preview_raw = "".join(info.get("buffer", []))
                    if preview_raw:
                        preview = (
                            preview_raw if len(preview_raw) <= 120 else preview_raw[:117] + "..."
                        )
                        self.logger.debug(
                            "Completed tool input stream",
                            data={
                                "tool_name": info.get("name"),
                                "tool_use_id": info.get("id"),
                                "input_preview": preview,
                            },
                        )
                    self._notify_tool_stream_listeners(
                        "stop",
                        {
                            "tool_name": info.get("name"),
                            "tool_use_id": info.get("id"),
                            "index": event.index,
                        },
                    )
                    self.logger.info(
                        "Model finished streaming tool input",
                        data={
                            "progress_action": ProgressAction.CALLING_TOOL,
                            "agent_name": self.name,
                            "model": model,
                            "tool_name": info.get("name"),
                            "tool_use_id": info.get("id"),
                            "tool_event": "stop",
                        },
                    )
                    continue

                if (
                    isinstance(event, RawContentBlockStopEvent)
                    and event.index in server_tool_streams
                ):
                    info = server_tool_streams.pop(event.index)
                    tool_name = str(info.get("name") or "tool")
                    tool_id = str(info.get("id") or "")
                    progress_label = web_tool_progress_label(tool_name)
                    self._notify_tool_stream_listeners(
                        "stop",
                        {
                            "tool_name": tool_name,
                            "tool_display_name": progress_label,
                            "tool_use_id": tool_id,
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
                            "tool_use_id": tool_id,
                            "tool_event": "stop",
                            "details": progress_label,
                        },
                    )
                    continue

                # Count tokens in real-time from content_block_delta events
                if isinstance(event, RawContentBlockDeltaEvent):
                    delta = event.delta
                    if isinstance(delta, TextDelta):
                        # Notify stream listeners for UI streaming
                        self._notify_stream_listeners(
                            StreamChunk(text=delta.text, is_reasoning=False)
                        )
                        if delta.text:
                            streamed_text_segments.append(delta.text)
                        # Use base class method for token estimation and progress emission
                        estimated_tokens = self._update_streaming_progress(
                            delta.text, model, estimated_tokens
                        )
                        self._notify_tool_stream_listeners(
                            "text",
                            {
                                "chunk": delta.text,
                                "index": event.index,
                            },
                        )

                # Also check for final message_delta events with actual usage info
                elif isinstance(event, RawMessageDeltaEvent) and event.usage.output_tokens:
                    actual_tokens = event.usage.output_tokens
                    # Emit final progress with actual token count
                    token_str = str(actual_tokens).rjust(5)
                    data = {
                        "progress_action": ProgressAction.STREAMING,
                        "model": model,
                        "agent_name": self.name,
                        "chat_turn": self.chat_turn(),
                        "details": token_str.strip(),
                    }
                    logger.info("Streaming progress", data=data)

            # Get the final message with complete usage data
            try:
                message = await stream.get_final_message()
            except Exception as error:
                if _is_beta_text_block_validation_error(error) and streamed_text_segments:
                    logger.warning(
                        "Anthropic final message validation failed; falling back to streamed text",
                        data={
                            "model": model,
                            "streamed_text_chunks": len(streamed_text_segments),
                            "error": str(error),
                        },
                    )
                    if os.environ.get("FAST_AGENT_WEBDEBUG"):
                        print(
                            "[webdebug] final message validation failed; "
                            "using streamed text fallback "
                            f"model={model} chunks={len(streamed_text_segments)}"
                        )

                    fallback_message = Message.model_construct(
                        id="msg_stream_fallback",
                        type="message",
                        role="assistant",
                        content=[],
                        model=model,
                        stop_reason="end_turn",
                        usage=None,
                    )
                    return fallback_message, thinking_segments, streamed_text_segments
                raise

            # Log final usage information
            if hasattr(message, "usage") and message.usage:
                logger.info(
                    f"Streaming complete - Model: {model}, Input tokens: {message.usage.input_tokens}, Output tokens: {message.usage.output_tokens}"
                )

            return message, thinking_segments, streamed_text_segments
        except APIError as error:
            logger.error("Streaming APIError during Anthropic completion", exc_info=error)
            raise  # Re-raise to be handled by _anthropic_completion
        except Exception as error:
            logger.error("Unexpected error during Anthropic stream processing", exc_info=error)
            # Re-raise for consistent handling - caller handles the error
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
        message_param: MessageParam,
        pre_messages: list[MessageParam] | None = None,
        history: list[PromptMessageExtended] | None = None,
    ) -> list[MessageParam]:
        """
        Build the list of Anthropic message parameters for the next request.

        Ensures that the current user message is only included once when history
        is enabled, which prevents duplicate tool_result blocks from being sent.
        """
        messages: list[MessageParam] = list(pre_messages) if pre_messages else []

        history_messages: list[MessageParam] = []
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

    async def _anthropic_completion(
        self,
        message_param,
        request_params: RequestParams | None = None,
        structured_model: Type[ModelT] | None = None,
        tools: list[Tool] | None = None,
        pre_messages: list[MessageParam] | None = None,
        history: list[PromptMessageExtended] | None = None,
        current_extended: PromptMessageExtended | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using an LLM and available tools.
        Override this method to use a different LLM.
        """

        api_key = self._api_key()
        base_url = self._base_url()
        if base_url and base_url.endswith("/v1"):
            base_url = base_url.rstrip("/v1")
        default_headers = self._default_headers()

        try:
            anthropic = AsyncAnthropic(
                api_key=api_key, base_url=base_url, default_headers=default_headers
            )
            params = self.get_request_params(request_params)
            messages = self._build_request_messages(
                params, message_param, pre_messages, history=history
            )
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from e

        # Get cache mode configuration
        cache_mode = self._get_cache_mode()
        logger.debug(f"Anthropic cache_mode: {cache_mode}")

        response_content_blocks: list[ContentBlock] = []
        tool_calls: dict[str, CallToolRequest] | None = None
        model = self.default_request_params.model or DEFAULT_ANTHROPIC_MODEL

        structured_mode = self._resolve_structured_output_mode(model, structured_model)
        available_tools = await self._prepare_tools(
            structured_model, tools, structured_mode=structured_mode
        )
        web_tools, web_tool_betas = self._prepare_web_tools(model)
        request_tools = [*available_tools, *web_tools]

        # Create base arguments dictionary
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

        if structured_mode:
            if structured_mode == "tool_use":
                if self._is_thinking_enabled(model):
                    logger.warning(
                        "Extended thinking is incompatible with tool-forced structured output. "
                        "Disabling thinking for this request."
                    )
                base_args["tool_choice"] = {
                    "type": "tool",
                    "name": STRUCTURED_OUTPUT_TOOL_NAME,
                }
            if structured_mode == "json" and structured_model:
                base_args["output_format"] = self._build_output_format(structured_model)

        thinking_args, thinking_enabled = self._resolve_thinking_arguments(
            model=model,
            max_tokens=params.maxTokens,
            structured_mode=structured_mode,
        )
        base_args.update(thinking_args)

        beta_flags: list[str] = []
        adaptive_thinking = self._supports_adaptive_thinking(model)
        if structured_mode:
            beta_flags.append(STRUCTURED_OUTPUT_BETA)
        if thinking_enabled and request_tools and not adaptive_thinking:
            beta_flags.append(INTERLEAVED_THINKING_BETA)
        if self._long_context:
            beta_flags.append(LONG_CONTEXT_BETA)
        # Enable fine-grained tool streaming when tools are present
        if request_tools:
            beta_flags.append(FINE_GRAINED_TOOL_STREAMING_BETA)
        beta_flags.extend(web_tool_betas)
        beta_flags = dedupe_preserve_order(beta_flags)
        if beta_flags:
            base_args["betas"] = beta_flags

        self._log_chat_progress(self.chat_turn(), model=model)
        # Use the base class method to prepare all arguments with Anthropic-specific exclusions
        # Do this BEFORE applying cache control so metadata doesn't override cached fields
        arguments = self.prepare_provider_arguments(
            base_args, params, self.ANTHROPIC_EXCLUDE_FIELDS
        )

        # Apply cache control to system prompt AFTER merging arguments
        system_cache_applied = self._apply_system_cache(arguments, cache_mode)

        # Apply cache_control markers using planner
        planner = AnthropicCachePlanner(
            self.CONVERSATION_CACHE_WALK_DISTANCE, self.MAX_CONVERSATION_CACHE_BLOCKS
        )
        plan_messages: list[PromptMessageExtended] = []
        include_current = not params.use_history or not history
        if params.use_history and history:
            plan_messages.extend(history)
        if include_current and current_extended:
            plan_messages.append(current_extended)

        cache_indices = planner.plan_indices(
            plan_messages, cache_mode=cache_mode, system_cache_blocks=system_cache_applied
        )
        cache_ttl = self._get_cache_ttl()
        for idx in cache_indices:
            if 0 <= idx < len(messages):
                self._apply_cache_control_to_message(messages[idx], ttl=cache_ttl)

        logger.debug(f"{arguments}")

        # Generate stream capture filename once (before streaming starts)
        capture_filename = _stream_capture_filename(self.chat_turn())
        _save_stream_request(capture_filename, arguments)

        # Use streaming API with helper
        otel_span: Span | None = None
        otel_span_error = False
        response: Message | None = None
        streamed_text_segments: list[str] = []
        try:
            # OpenTelemetry instrumentation wraps the stream() call and returns a coroutine
            # that must be awaited before using as context manager. When the wrapper is
            # known-broken for beta streams, we bypass it to avoid await errors.
            stream_method = _maybe_unwrap_otel_beta_stream(anthropic.beta.messages.stream)
            otel_wrapper_bypassed = stream_method is not anthropic.beta.messages.stream
            if otel_wrapper_bypassed:
                otel_span = _start_fallback_stream_span(model)

            stream_call = stream_method(**arguments)
            # Check if it's a coroutine (OpenTelemetry is installed)
            if asyncio.iscoroutine(stream_call):
                stream_manager = await stream_call
            else:
                stream_manager = stream_call
            # Type annotation: stream_manager is BetaAsyncMessageStream ats runtime
            stream_manager = cast("BetaAsyncMessageStream", stream_manager)
            if otel_span is not None:
                with trace.use_span(otel_span, end_on_exit=False):
                    async with stream_manager as stream:
                        # Process the stream
                        (
                            response,
                            thinking_segments,
                            streamed_text_segments,
                        ) = await self._process_stream(stream, model, capture_filename)
            else:
                async with stream_manager as stream:
                    # Process the stream
                    (
                        response,
                        thinking_segments,
                        streamed_text_segments,
                    ) = await self._process_stream(stream, model, capture_filename)
        except asyncio.CancelledError as e:
            reason = str(e) if e.args else "cancelled"
            logger.info(f"Anthropic completion cancelled: {reason}")
            # Return a response indicating cancellation
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )
        except APIError as error:
            if otel_span is not None and otel_span.is_recording():
                otel_span.record_exception(error)
                otel_span.set_status(Status(StatusCode.ERROR))
                otel_span_error = True
            logger.error("Streaming APIError during Anthropic completion", exc_info=error)
            raise error
        except Exception as error:
            if otel_span is not None and otel_span.is_recording():
                otel_span.record_exception(error)
                otel_span.set_status(Status(StatusCode.ERROR))
                otel_span_error = True
            raise
        finally:
            if otel_span is not None:
                _finalize_fallback_stream_span(otel_span, response, otel_span_error)

        # Track usage if response is valid and has usage data
        if (
            hasattr(response, "usage")
            and response.usage
            and not isinstance(response, BaseException)
        ):
            try:
                turn_usage = TurnUsage.from_anthropic(
                    response.usage, model or DEFAULT_ANTHROPIC_MODEL
                )
                self._finalize_turn_usage(turn_usage)
            except Exception as e:
                logger.warning(f"Failed to track usage: {e}")

        if isinstance(response, AuthenticationError):
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\nPlease check that your API key is valid and not expired.",
            ) from response
        elif isinstance(response, BaseException):
            # This path shouldn't be reached anymore since we handle APIError above,
            # but keeping for backward compatibility
            logger.error(f"Unexpected error type: {type(response).__name__}", exc_info=response)
            return build_stream_failure_response(self.provider, response, model)

        logger.debug(
            f"{model} response:",
            data=response,
        )

        response_as_message = self.convert_message_to_message_param(response)
        messages.append(response_as_message)
        if response.content:
            for index, content_block in enumerate(response.content):
                if isinstance(content_block, TextBlock):
                    text_value = getattr(content_block, "text", None)
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
        if streamed_text_segments:
            streamed_text = "".join(streamed_text_segments)
            if streamed_text.strip():
                provider_text = "".join(
                    block.text
                    for block in response_content_blocks
                    if isinstance(block, TextContent) and block.text
                )
                # In some Anthropic server-tool turns, streamed text can be richer than
                # the final response content blocks. Prefer the streamed transcript
                # when they differ so post-stream re-render matches what users saw.
                if not provider_text.strip() or provider_text.strip() != streamed_text.strip():
                    response_content_blocks = [TextContent(type="text", text=streamed_text)]

        stop_reason: LlmStopReason = LlmStopReason.END_TURN

        match response.stop_reason:
            case "stop_sequence":
                stop_reason = LlmStopReason.STOP_SEQUENCE
            case "max_tokens":
                stop_reason = LlmStopReason.MAX_TOKENS
            case "refusal":
                stop_reason = LlmStopReason.SAFETY
            case "pause" | "pause_turn":
                stop_reason = LlmStopReason.PAUSE
            case "tool_use":
                tool_uses: list[ToolUseBlock] = [
                    c for c in response.content if isinstance(c, ToolUseBlock)
                ]
                if (
                    structured_mode == "tool_use"
                    and structured_model
                    and self._is_structured_output_request(tool_uses)
                ):
                    stop_reason, structured_blocks = await self._handle_structured_output_response(
                        tool_uses[0], structured_model, messages
                    )
                    response_content_blocks.extend(structured_blocks)
                elif tool_uses:
                    stop_reason = LlmStopReason.TOOL_USE
                    tool_calls = self._build_tool_calls_dict(tool_uses)
                else:
                    # Anthropic server tools (web_search/web_fetch) run provider-side and
                    # must not be surfaced as MCP tool calls.
                    stop_reason = LlmStopReason.END_TURN

        # Update diagnostic snapshot (never read again)
        # This provides a snapshot of what was sent to the provider for debugging
        self.history.set(messages)

        self._log_chat_finished(model=model)

        channels: dict[str, list[Any]] | None = None
        if thinking_segments:
            channels = {REASONING: [TextContent(type="text", text="".join(thinking_segments))]}
        elif response.content:
            thinking_texts = [
                block.thinking
                for block in response.content
                if isinstance(block, ThinkingBlock) and block.thinking
            ]
            if thinking_texts:
                channels = {REASONING: [TextContent(type="text", text="".join(thinking_texts))]}

        raw_thinking_blocks = []
        if response.content:
            raw_thinking_blocks = [
                block
                for block in response.content
                if isinstance(block, (ThinkingBlock, RedactedThinkingBlock))
            ]
        if raw_thinking_blocks:
            if channels is None:
                channels = {}
            serialized_blocks = []
            for block in raw_thinking_blocks:
                try:
                    payload = block.model_dump()
                except Exception:
                    payload = {"type": getattr(block, "type", "thinking")}
                    if isinstance(block, ThinkingBlock):
                        payload.update({"thinking": block.thinking, "signature": block.signature})
                    elif isinstance(block, RedactedThinkingBlock):
                        payload.update({"data": block.data})
                serialized_blocks.append(TextContent(type="text", text=json.dumps(payload)))
            channels[ANTHROPIC_THINKING_BLOCKS] = serialized_blocks

        if response.content:
            raw_assistant_payloads: list[TextContent] = []
            server_tool_payloads: list[TextContent] = []
            citation_payloads: list[TextContent] = []
            for block in response.content:
                payload = serialize_anthropic_block_payload(block)
                if payload is not None:
                    try:
                        raw_assistant_payloads.append(
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
                    server_tool_payloads.append(TextContent(type="text", text=json.dumps(payload)))

                if isinstance(block, TextBlock) and block.citations:
                    extracted = extract_citation_payloads(block.citations)
                    for citation_payload in extracted:
                        citation_payloads.append(
                            TextContent(type="text", text=json.dumps(citation_payload))
                        )

            if os.environ.get("FAST_AGENT_WEBDEBUG"):
                print(
                    "[webdebug]"
                    f" model={model}"
                    f" response_blocks={len(response.content)}"
                    f" server_tool_payloads={len(server_tool_payloads)}"
                    f" citation_payloads={len(citation_payloads)}"
                )

            if server_tool_payloads:
                if channels is None:
                    channels = {}
                channels[ANTHROPIC_SERVER_TOOLS_CHANNEL] = server_tool_payloads

            if raw_assistant_payloads and (
                raw_thinking_blocks or server_tool_payloads or tool_calls is not None
            ):
                if channels is None:
                    channels = {}
                channels[ANTHROPIC_ASSISTANT_RAW_CONTENT] = raw_assistant_payloads

            if citation_payloads:
                if channels is None:
                    channels = {}
                channels[ANTHROPIC_CITATIONS_CHANNEL] = citation_payloads

        if response.container and response.container.id:
            if channels is None:
                channels = {}
            channels[ANTHROPIC_CONTAINER_CHANNEL] = [
                TextContent(type="text", text=json.dumps({"id": response.container.id}))
            ]

        return PromptMessageExtended(
            role="assistant",
            content=response_content_blocks,
            tool_calls=tool_calls,
            channels=channels,
            stop_reason=stop_reason,
        )

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
        else:
            # For assistant messages: Return the last message content as text
            logger.debug("Last message in prompt is from assistant, returning it directly")
            return last_message

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: list[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:  # noqa: F821
        """
        Provider-specific structured output implementation.
        Note: Message history is managed by base class and converted via
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
        else:
            # For assistant messages: Return the last message content
            logger.debug("Last message in prompt is from assistant, returning it directly")
            return None, last_message

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[MessageParam]:
        """
        Convert PromptMessageExtended list to Anthropic MessageParam format.
        This is called fresh on every API call from _convert_to_provider_format().

        Args:
            messages: List of PromptMessageExtended objects

        Returns:
            List of Anthropic MessageParam objects
        """
        return [AnthropicConverter.convert_to_anthropic(msg) for msg in messages]

    @classmethod
    def convert_message_to_message_param(cls, message: Message, **kwargs) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []

        for content_block in message.content:
            if isinstance(content_block, TextBlock):
                text_value = getattr(content_block, "text", None)
                if not isinstance(text_value, str):
                    logger.warning(
                        "Skipping Anthropic text block with non-string text while converting message",
                        data={"text_type": type(text_value).__name__},
                    )
                    continue
                content.append(TextBlockParam(type="text", text=text_value))
            elif isinstance(content_block, ToolUseBlock):
                content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        name=content_block.name,
                        input=content_block.input,
                        id=content_block.id,
                    )
                )
            elif isinstance(content_block, ServerToolUseBlock):
                payload = serialize_anthropic_block_payload(content_block)
                if payload is not None and is_server_tool_trace_payload(payload):
                    content.append(payload)
            else:
                payload = serialize_anthropic_block_payload(content_block)
                if payload is not None and is_server_tool_trace_payload(payload):
                    content.append(payload)

        return MessageParam(role="assistant", content=content, **kwargs)

    def _show_usage(self, raw_usage: Usage, turn_usage: TurnUsage) -> None:
        """This is a debug routine, leaving in for convenience"""
        # Print raw usage for debugging
        print(f"\n=== USAGE DEBUG ({turn_usage.model}) ===")
        print(f"Raw usage: {raw_usage}")
        print(
            f"Turn usage: input={turn_usage.input_tokens}, output={turn_usage.output_tokens}, current_context={turn_usage.current_context_tokens}"
        )
        print(
            f"Cache: read={turn_usage.cache_usage.cache_read_tokens}, write={turn_usage.cache_usage.cache_write_tokens}"
        )
        print(f"Effective input: {turn_usage.effective_input_tokens}")
        print(
            f"Accumulator: total_turns={self.usage_accumulator.turn_count}, cumulative_billing={self.usage_accumulator.cumulative_billing_tokens}, current_context={self.usage_accumulator.current_context_tokens}"
        )
        if self.usage_accumulator.context_usage_percentage:
            print(
                f"Context usage: {self.usage_accumulator.context_usage_percentage:.1f}% of {self.usage_accumulator.context_window_size}"
            )
        if self.usage_accumulator.cache_hit_rate:
            print(f"Cache hit rate: {self.usage_accumulator.cache_hit_rate:.1f}%")
        print("===========================\n")
