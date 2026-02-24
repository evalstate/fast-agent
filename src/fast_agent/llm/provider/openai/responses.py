import asyncio
import json
import os
from typing import Any, Literal

from mcp import Tool
from mcp.types import ContentBlock, TextContent
from openai import APIError, AsyncOpenAI, AuthenticationError, DefaultAioHttpClient

from fast_agent.constants import (
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    OPENAI_REASONING_ENCRYPTED,
    REASONING,
)
from fast_agent.core.exceptions import ModelConfigError, ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.error_utils import build_stream_failure_response
from fast_agent.llm.provider.openai._stream_capture import (
    save_stream_request as _save_stream_request,
)
from fast_agent.llm.provider.openai._stream_capture import (
    stream_capture_filename as _stream_capture_filename,
)
from fast_agent.llm.provider.openai.responses_content import ResponsesContentMixin
from fast_agent.llm.provider.openai.responses_files import ResponsesFileMixin
from fast_agent.llm.provider.openai.responses_output import ResponsesOutputMixin
from fast_agent.llm.provider.openai.responses_streaming import ResponsesStreamingMixin
from fast_agent.llm.provider.openai.responses_websocket import (
    ManagedWebSocketConnection,
    ResponsesWebSocketError,
    ResponsesWsRequestPlanner,
    StatefulContinuationResponsesWsPlanner,
    WebSocketConnectionManager,
    WebSocketResponsesStream,
    build_ws_headers,
    connect_websocket,
    resolve_responses_ws_url,
    send_response_request,
)
from fast_agent.llm.provider.openai.schema_sanitizer import (
    sanitize_tool_input_schema,
    should_strip_tool_schema_defaults,
)
from fast_agent.llm.provider.openai.web_tools import build_web_search_tool, resolve_web_search
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import format_reasoning_setting, parse_reasoning_setting
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.text_verbosity import parse_text_verbosity
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

_logger = get_logger(__name__)

DEFAULT_RESPONSES_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "medium"
MIN_RESPONSES_MAX_TOKENS = 16
DEFAULT_RESPONSES_BASE_URL = "https://api.openai.com/v1"
RESPONSES_DIAGNOSTICS_CHANNEL = "fast-agent-provider-diagnostics"
RESPONSE_INCLUDE_REASONING = "reasoning.encrypted_content"
RESPONSE_INCLUDE_WEB_SEARCH_SOURCES = "web_search_call.action.sources"

ResponsesTransport = Literal["sse", "websocket", "auto"]


class ResponsesLLM(
    ResponsesContentMixin,
    ResponsesFileMixin,
    ResponsesOutputMixin,
    ResponsesStreamingMixin,
    FastAgentLLM[dict[str, Any], Any],
):
    """LLM implementation for OpenAI's Responses models."""

    config_section: str | None = None

    RESPONSES_EXCLUDE_FIELDS = {
        FastAgentLLM.PARAM_MESSAGES,
        FastAgentLLM.PARAM_MODEL,
        FastAgentLLM.PARAM_MAX_TOKENS,
        FastAgentLLM.PARAM_SYSTEM_PROMPT,
        FastAgentLLM.PARAM_STOP_SEQUENCES,
        FastAgentLLM.PARAM_USE_HISTORY,
        FastAgentLLM.PARAM_MAX_ITERATIONS,
        FastAgentLLM.PARAM_TEMPLATE_VARS,
        FastAgentLLM.PARAM_MCP_METADATA,
        FastAgentLLM.PARAM_PARALLEL_TOOL_CALLS,
        "response_format",
    }

    def __init__(self, provider: Provider = Provider.RESPONSES, **kwargs) -> None:
        web_search_override = kwargs.pop("web_search", None)
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)
        self._tool_call_id_map: dict[str, str] = {}
        self._seen_tool_call_ids: set[str] = set()
        self._tool_call_diagnostics: dict[str, Any] | None = None
        self._last_ws_request_type: str | None = None
        self._last_ws_request_mode: Literal["create", "continuation"] | None = None
        self._file_id_cache: dict[str, str] = {}
        self._transport: ResponsesTransport = "sse"
        self._last_transport_used: Literal["sse", "websocket"] | None = None
        self._ws_connections = WebSocketConnectionManager(idle_timeout_seconds=300.0)
        self._ws_debug_inline = os.getenv("FAST_AGENT_DEBUG_RESPONSES_WS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._web_search_override: bool | None = (
            bool(web_search_override) if isinstance(web_search_override, bool) else None
        )

        raw_setting = kwargs.get("reasoning_effort", None)
        settings = self._get_provider_config()
        if settings and raw_setting is None:
            raw_setting = getattr(settings, "reasoning", None)
            if raw_setting is None and hasattr(settings, "reasoning_effort"):
                raw_setting = settings.reasoning_effort
                if (
                    raw_setting is not None
                    and "reasoning_effort" in settings.model_fields_set
                    and settings.reasoning_effort
                    != settings.model_fields["reasoning_effort"].default
                ):
                    self.logger.warning(
                        "Responses config 'reasoning_effort' is deprecated; use 'reasoning'."
                    )

        setting = parse_reasoning_setting(raw_setting)
        if setting is not None:
            try:
                self.set_reasoning_effort(setting)
            except ValueError as exc:
                self.logger.warning(f"Invalid reasoning setting: {exc}")

        raw_text_verbosity = kwargs.get("text_verbosity", None)
        if settings and raw_text_verbosity is None:
            raw_text_verbosity = getattr(settings, "text_verbosity", None)
        if raw_text_verbosity is not None:
            parsed_verbosity = parse_text_verbosity(str(raw_text_verbosity))
            if parsed_verbosity is None:
                self.logger.warning(f"Invalid text verbosity setting: {raw_text_verbosity}")
            else:
                try:
                    self.set_text_verbosity(parsed_verbosity)
                except ValueError as exc:
                    self.logger.warning(f"Invalid text verbosity setting: {exc}")

        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning_mode = ModelDatabase.get_reasoning(chosen_model) if chosen_model else None
        self._reasoning = self._reasoning_mode == "openai"
        if self._reasoning_mode:
            self.logger.info(
                f"Using Responses model '{chosen_model}' (mode='{self._reasoning_mode}') with "
                f"'{format_reasoning_setting(self.reasoning_effort)}' reasoning effort"
            )

        self._transport = self._resolve_transport_setting(kwargs.get("transport"), settings)
        self._validate_transport_support(chosen_model, self._transport)

    @property
    def active_transport(self) -> Literal["sse", "websocket"] | None:
        """Return the transport used by the most recent completion call."""
        return self._last_transport_used

    @property
    def configured_transport(self) -> ResponsesTransport:
        """Return configured transport preference for this LLM instance."""
        return self._transport

    def _resolve_transport_setting(self, raw_value: Any, settings: Any) -> ResponsesTransport:
        value = raw_value
        if value is None and settings is not None:
            value = getattr(settings, "transport", None)
        if value is None:
            return "sse"

        normalized = str(value).strip().lower()
        transport_aliases: dict[str, ResponsesTransport] = {
            "ws": "websocket",
            "sse": "sse",
            "websocket": "websocket",
            "auto": "auto",
        }
        normalized_transport = transport_aliases.get(normalized)
        if normalized_transport is not None:
            return normalized_transport

        self.logger.warning(
            "Invalid Responses transport setting; defaulting to SSE",
            data={"transport": value},
        )
        return "sse"

    def _supports_websocket_transport(self) -> bool:
        """Provider-level websocket support flag (opt-in while experimental)."""
        return True

    def _validate_transport_support(
        self,
        model_name: str | None,
        transport: ResponsesTransport,
    ) -> None:
        if transport not in {"websocket", "auto"}:
            return

        model_to_check = model_name or self.default_request_params.model
        if not model_to_check:
            raise ModelConfigError("WebSocket transport requires a resolved model name.")

        if self.provider == Provider.RESPONSES:
            if not self._supports_websocket_transport():
                raise ModelConfigError(
                    "WebSocket transport is experimental and not enabled for this provider."
                )
            return

        response_transports = ModelDatabase.get_response_transports(model_to_check)
        if not response_transports or "websocket" not in response_transports:
            raise ModelConfigError(
                f"Transport '{transport}' is not supported for model '{model_to_check}'."
            )
        websocket_providers = ModelDatabase.get_response_websocket_providers(model_to_check)
        if websocket_providers is not None and self.provider not in websocket_providers:
            raise ModelConfigError(
                f"Transport '{transport}' is not supported for model '{model_to_check}' "
                f"with provider '{self.provider.value}'."
            )
        if not self._supports_websocket_transport():
            raise ModelConfigError(
                "WebSocket transport is experimental and not enabled for this provider."
            )

    def _effective_transport(self) -> ResponsesTransport:
        return self._transport

    def _base_responses_url(self) -> str:
        return self._base_url() or DEFAULT_RESPONSES_BASE_URL

    def _build_websocket_headers(self) -> dict[str, str]:
        return build_ws_headers(api_key=self._api_key(), default_headers=self._default_headers())

    async def _create_websocket_connection(
        self,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> ManagedWebSocketConnection:
        return await connect_websocket(url=url, headers=headers, timeout_seconds=timeout_seconds)

    def _new_ws_request_planner(self) -> ResponsesWsRequestPlanner:
        return StatefulContinuationResponsesWsPlanner()

    def _websocket_retry_diagnostics(
        self,
        connection: ManagedWebSocketConnection,
        error: ResponsesWebSocketError,
    ) -> dict[str, Any]:
        now = asyncio.get_running_loop().time()
        idle_age_seconds: float | None = None
        if connection.last_used_monotonic > 0.0:
            idle_age_seconds = max(0.0, now - connection.last_used_monotonic)

        websocket = connection.websocket
        close_code = getattr(websocket, "close_code", None)
        exception_obj = websocket.exception()

        diagnostics: dict[str, Any] = {
            "stream_started": error.stream_started,
            "session_closed": connection.session.closed,
            "websocket_closed": websocket.closed,
            "websocket_close_code": close_code,
            "websocket_exception": str(exception_obj) if exception_obj else None,
        }
        if idle_age_seconds is not None:
            diagnostics["idle_age_seconds"] = round(idle_age_seconds, 3)
        return diagnostics

    def _ws_input_count(self, payload: dict[str, Any]) -> int | None:
        input_items = payload.get("input")
        if not isinstance(input_items, list):
            return None
        return len(input_items)

    def _payload_size_bytes(self, payload: dict[str, Any]) -> int | None:
        try:
            compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return None
        return len(compact.encode("utf-8"))

    def _report_ws_request_plan(
        self,
        *,
        model_name: str,
        ws_url: str,
        planned_request: Any,
        full_arguments: dict[str, Any],
        reused_existing_connection: bool,
    ) -> None:
        continuation_id = planned_request.arguments.get("previous_response_id")
        request_mode: Literal["create", "continuation"] = (
            "continuation" if isinstance(continuation_id, str) and continuation_id else "create"
        )
        self._last_ws_request_mode = request_mode
        sent_input_count = self._ws_input_count(planned_request.arguments)
        full_input_count = self._ws_input_count(full_arguments)
        sent_payload_bytes = self._payload_size_bytes(planned_request.arguments)
        full_payload_bytes = self._payload_size_bytes(full_arguments)
        payload_saved_bytes: int | None = None
        payload_saved_ratio: float | None = None
        if sent_payload_bytes is not None and full_payload_bytes and full_payload_bytes > 0:
            payload_saved_bytes = max(0, full_payload_bytes - sent_payload_bytes)
            payload_saved_ratio = payload_saved_bytes / full_payload_bytes

        self.logger.info(
            "Responses websocket request plan",
            data={
                "model": model_name,
                "url": ws_url,
                "request_type": planned_request.event_type,
                "request_mode": request_mode,
                "reused_connection": reused_existing_connection,
                "sent_input_items": sent_input_count,
                "total_input_items": full_input_count,
                "sent_payload_bytes": sent_payload_bytes,
                "total_payload_bytes": full_payload_bytes,
                "payload_saved_bytes": payload_saved_bytes,
                "payload_saved_ratio": payload_saved_ratio,
                "uses_previous_response_id": request_mode == "continuation",
                "previous_response_id": continuation_id if request_mode == "continuation" else None,
            },
        )

        if not self._ws_debug_inline:
            return

        try:
            from rich.text import Text

            item_counts = ""
            if sent_input_count is not None and full_input_count is not None:
                item_counts = f" {sent_input_count}/{full_input_count} items"
            byte_counts = ""
            if sent_payload_bytes is not None and full_payload_bytes is not None:
                if payload_saved_bytes is not None and payload_saved_ratio is not None:
                    percent_saved = round(payload_saved_ratio * 100.0)
                    byte_counts = (
                        f" {sent_payload_bytes}/{full_payload_bytes}B"
                        f" ({percent_saved}% saved)"
                    )
                else:
                    byte_counts = f" {sent_payload_bytes}/{full_payload_bytes}B"
            prev_suffix = f" prev={continuation_id}" if request_mode == "continuation" else ""
            reuse_suffix = " reused-conn" if reused_existing_connection else ""
            self.display.show_status_message(
                Text.from_markup(
                    f"[dim]WS {request_mode}{item_counts}{byte_counts}{prev_suffix}{reuse_suffix}[/dim]"
                )
            )
        except Exception:
            # UI status notification should never affect completion flow.
            pass

    def _resolve_reasoning_effort(self) -> str | None:
        setting = self.reasoning_effort
        if setting is None:
            return DEFAULT_REASONING_EFFORT
        if setting.kind == "effort":
            return str(setting.value)
        if setting.kind == "toggle":
            return None if setting.value is False else DEFAULT_REASONING_EFFORT
        if setting.kind == "budget":
            self.logger.warning("Ignoring budget reasoning setting for Responses models.")
            return DEFAULT_REASONING_EFFORT
        return DEFAULT_REASONING_EFFORT

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_RESPONSES_MODEL)

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        return ("openai",)

    def _openai_settings(self):
        return self._get_provider_config()

    @property
    def web_search_supported(self) -> bool:
        """Responses-family models currently expose the web_search server tool."""
        return True

    @property
    def web_search_enabled(self) -> bool:
        """Whether Responses web_search is enabled for this LLM instance."""
        resolved_web_search = resolve_web_search(
            self._openai_settings(),
            web_search_override=self._web_search_override,
        )
        return resolved_web_search.enabled

    def set_web_search_enabled(self, value: bool | None) -> None:
        self._web_search_override = value

    @property
    def web_fetch_supported(self) -> bool:
        """Responses-family models do not expose web_fetch."""
        return False

    @property
    def web_fetch_enabled(self) -> bool:
        """Responses-family models do not expose web_fetch."""
        return False

    def set_web_fetch_enabled(self, value: bool | None) -> None:
        super().set_web_fetch_enabled(value)

    def _base_url(self) -> str | None:
        settings = self._openai_settings()
        return settings.base_url if settings else None

    def _default_headers(self) -> dict[str, str] | None:
        settings = self._openai_settings()
        return settings.default_headers if settings else None

    def _responses_client(self) -> AsyncOpenAI:
        try:
            kwargs: dict[str, Any] = {
                "api_key": self._api_key(),
                "base_url": self._base_url(),
                "http_client": DefaultAioHttpClient(),
            }
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

    def _adjust_schema(self, input_schema: dict[str, Any], model_name: str) -> dict[str, Any]:
        result = (
            sanitize_tool_input_schema(input_schema)
            if should_strip_tool_schema_defaults(model_name)
            else input_schema
        )
        if "properties" in result:
            return result
        result = result.copy()
        result["properties"] = {}
        return result

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        req_params = self.get_request_params(request_params)

        last_message = multipart_messages[-1]
        if last_message.role == "assistant":
            return last_message

        input_items = self._convert_to_provider_format(multipart_messages)
        if not input_items:
            input_items = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": ""}],
                }
            ]

        return await self._responses_completion(input_items, req_params, tools)

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        model = request_params.model or self.default_request_params.model or DEFAULT_RESPONSES_MODEL
        base_args: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
            "include": [RESPONSE_INCLUDE_REASONING],
            "parallel_tool_calls": request_params.parallel_tool_calls,
        }

        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            base_args["instructions"] = system_prompt

        if tools:
            base_args["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": self._adjust_schema(tool.inputSchema, model),
                }
                for tool in tools
            ]

        resolved_web_search = resolve_web_search(
            self._openai_settings(),
            web_search_override=self._web_search_override,
        )
        web_search_tool = build_web_search_tool(resolved_web_search)
        if web_search_tool is not None:
            tools_payload = base_args.setdefault("tools", [])
            if isinstance(tools_payload, list):
                tools_payload.append(web_search_tool)

            include_payload = base_args.get("include")
            if isinstance(include_payload, list):
                if RESPONSE_INCLUDE_WEB_SEARCH_SOURCES not in include_payload:
                    include_payload.append(RESPONSE_INCLUDE_WEB_SEARCH_SOURCES)

        if self._reasoning:
            effort = self._resolve_reasoning_effort()
            if effort:
                base_args["reasoning"] = {
                    "summary": "auto",
                    "effort": effort,
                }

        if request_params.maxTokens is not None:
            max_tokens = request_params.maxTokens
            if max_tokens < MIN_RESPONSES_MAX_TOKENS:
                self.logger.debug(
                    "Clamping max_output_tokens to Responses minimum",
                    data={
                        "requested": max_tokens,
                        "minimum": MIN_RESPONSES_MAX_TOKENS,
                    },
                )
                max_tokens = MIN_RESPONSES_MAX_TOKENS
            base_args["max_output_tokens"] = max_tokens

        if request_params.response_format:
            base_args["text"] = {
                "format": self._normalize_text_format(request_params.response_format)
            }

        text_verbosity_spec = self.text_verbosity_spec
        if text_verbosity_spec:
            text_payload = base_args.get("text")
            if not isinstance(text_payload, dict):
                text_payload = {}
            text_payload["verbosity"] = self.text_verbosity or text_verbosity_spec.default
            base_args["text"] = text_payload

        return self.prepare_provider_arguments(
            base_args, request_params, self.RESPONSES_EXCLUDE_FIELDS
        )

    async def _responses_completion(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        model_name = (
            request_params.model or self.default_request_params.model or DEFAULT_RESPONSES_MODEL
        )
        transport = self._effective_transport()
        self._validate_transport_support(model_name, transport)

        display_model = model_name
        if transport in {"websocket", "auto"}:
            display_model = f"{model_name} [ws]"

        self._log_chat_progress(self.chat_turn(), model=display_model)
        self._last_ws_request_type = None
        self._last_ws_request_mode = None

        try:
            if transport == "sse":
                response, streamed_summary, input_items = await self._responses_completion_sse(
                    input_items=input_items,
                    request_params=request_params,
                    tools=tools,
                    model_name=model_name,
                )
                self._last_transport_used = "sse"
            else:
                response, streamed_summary, input_items = await self._responses_completion_ws(
                    input_items=input_items,
                    request_params=request_params,
                    tools=tools,
                    model_name=model_name,
                )
                self._last_transport_used = "websocket"
        except ResponsesWebSocketError as error:
            should_fallback_to_sse = transport == "auto" and not error.stream_started
            if should_fallback_to_sse:
                self.logger.warning(
                    "WebSocket transport failed before stream start; falling back to SSE "
                    "(auto transport safeguard)",
                    data={
                        "model": model_name,
                        "requested_transport": transport,
                        "error": str(error),
                    },
                )
                try:
                    from rich.text import Text

                    self.display.show_status_message(
                        Text.from_markup(
                            "[yellow]⚠ WebSocket transport unavailable for this turn; using SSE fallback.[/yellow]"
                        )
                    )
                except Exception:
                    # UI status notification should never affect completion flow.
                    pass
                response, streamed_summary, input_items = await self._responses_completion_sse(
                    input_items=input_items,
                    request_params=request_params,
                    tools=tools,
                    model_name=model_name,
                )
                self._last_transport_used = "sse"
            else:
                raise
        except asyncio.CancelledError:
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )

        if response is None:
            raise RuntimeError("Responses stream did not return a final response")

        self._log_chat_finished(model=model_name)

        response_content_blocks: list[ContentBlock] = []
        channels: dict[str, list[ContentBlock]] | None = None
        reasoning_blocks = self._extract_reasoning_summary(response, streamed_summary)
        encrypted_blocks = self._extract_encrypted_reasoning(response)
        if reasoning_blocks or encrypted_blocks:
            channels = {}
            if reasoning_blocks:
                channels[REASONING] = reasoning_blocks
            if encrypted_blocks:
                channels[OPENAI_REASONING_ENCRYPTED] = encrypted_blocks

        tool_calls = self._extract_tool_calls(response)
        tool_call_diagnostics = self._consume_tool_call_diagnostics()
        if tool_call_diagnostics:
            diagnostics_payload = dict(tool_call_diagnostics)
            diagnostics_payload["transport"] = self._last_transport_used or "unknown"
            if self._last_transport_used == "websocket" and self._last_ws_request_type:
                diagnostics_payload["websocket_request_type"] = self._last_ws_request_type
                if self._last_ws_request_mode is not None:
                    diagnostics_payload["websocket_request_mode"] = self._last_ws_request_mode
            if channels is None:
                channels = {}
            channels[RESPONSES_DIAGNOSTICS_CHANNEL] = [
                TextContent(type="text", text=json.dumps(diagnostics_payload))
            ]
        if tool_calls:
            stop_reason = LlmStopReason.TOOL_USE
        else:
            stop_reason = self._map_response_stop_reason(response)

        for output_item in getattr(response, "output", []) or []:
            if getattr(output_item, "type", None) != "message":
                continue
            for part in getattr(output_item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    response_content_blocks.append(
                        TextContent(type="text", text=getattr(part, "text", ""))
                    )

        if not response_content_blocks:
            output_text = getattr(response, "output_text", None)
            if output_text:
                response_content_blocks.append(TextContent(type="text", text=output_text))

        if getattr(response, "usage", None):
            self._record_usage(response.usage, model_name)

        web_tool_payloads, citation_payloads = self._extract_web_search_metadata(response)
        if web_tool_payloads:
            if channels is None:
                channels = {}
            channels[ANTHROPIC_SERVER_TOOLS_CHANNEL] = web_tool_payloads
        if citation_payloads:
            if channels is None:
                channels = {}
            channels[ANTHROPIC_CITATIONS_CHANNEL] = citation_payloads

        self.history.set(input_items)

        return PromptMessageExtended(
            role="assistant",
            content=response_content_blocks,
            tool_calls=tool_calls,
            channels=channels,
            stop_reason=stop_reason,
        )

    async def _responses_completion_sse(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        model_name: str,
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        try:
            async with self._responses_client() as client:
                normalized_input = await self._normalize_input_files(client, input_items)
                arguments = self._build_response_args(normalized_input, request_params, tools)
                self.logger.debug("Responses request", data=arguments)
                capture_filename = _stream_capture_filename(self.chat_turn())
                _save_stream_request(capture_filename, arguments)
                async with client.responses.stream(**arguments) as stream:
                    timeout = request_params.streaming_timeout
                    if timeout is None:
                        response, streamed_summary = await self._process_stream(
                            stream, model_name, capture_filename
                        )
                    else:
                        try:
                            response, streamed_summary = await asyncio.wait_for(
                                self._process_stream(stream, model_name, capture_filename),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError as exc:
                            self.logger.error(
                                "Streaming timeout while waiting for Responses",
                                data={
                                    "model": model_name,
                                    "timeout_seconds": timeout,
                                },
                            )
                            raise TimeoutError(
                                f"Streaming did not complete within {timeout} seconds."
                            ) from exc
                return response, streamed_summary, normalized_input
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e
        except APIError as error:
            self.logger.error("Streaming APIError during Responses completion", exc_info=error)
            raise

    async def _responses_completion_ws(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        model_name: str,
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        async with self._responses_client() as client:
            normalized_input = await self._normalize_input_files(client, input_items)

        arguments = self._build_response_args(normalized_input, request_params, tools)
        self.logger.debug("Responses websocket request", data=arguments)
        capture_filename = _stream_capture_filename(self.chat_turn())
        _save_stream_request(capture_filename, arguments)

        ws_url = resolve_responses_ws_url(self._base_responses_url())
        ws_headers = self._build_websocket_headers()
        timeout = request_params.streaming_timeout

        async def _create_connection() -> ManagedWebSocketConnection:
            return await self._create_websocket_connection(ws_url, ws_headers, timeout)

        last_error: ResponsesWebSocketError | None = None
        reconnected = False
        for attempt in range(2):
            connection, is_reusable = await self._ws_connections.acquire(_create_connection)
            reused_existing_connection = is_reusable and connection.last_used_monotonic > 0.0
            planner = connection.session_state.request_planner
            if planner is None:
                planner = self._new_ws_request_planner()
                connection.session_state.request_planner = planner
            keep_connection = False
            stream: WebSocketResponsesStream | None = None
            retry_after_release = False
            reconnect_diagnostics: dict[str, Any] | None = None

            try:
                self.logger.info(
                    "Using Responses websocket transport",
                    data={"model": model_name, "url": ws_url},
                )
                planned_request = planner.plan(arguments)
                self._last_ws_request_type = planned_request.event_type
                self._report_ws_request_plan(
                    model_name=model_name,
                    ws_url=ws_url,
                    planned_request=planned_request,
                    full_arguments=arguments,
                    reused_existing_connection=reused_existing_connection,
                )
                await send_response_request(connection.websocket, planned_request)
                stream = WebSocketResponsesStream(connection.websocket)
                if timeout is None:
                    response, streamed_summary = await self._process_stream(
                        stream, model_name, capture_filename
                    )
                else:
                    try:
                        response, streamed_summary = await asyncio.wait_for(
                            self._process_stream(stream, model_name, capture_filename),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError as exc:
                        self.logger.error(
                            "Streaming timeout while waiting for Responses websocket",
                            data={
                                "model": model_name,
                                "timeout_seconds": timeout,
                            },
                        )
                        raise TimeoutError(
                            f"Streaming did not complete within {timeout} seconds."
                        ) from exc
                planner.commit(arguments, planned_request, response)
                keep_connection = True
                if reconnected:
                    try:
                        from rich.text import Text

                        self.display.show_status_message(
                            Text.from_markup("[dim]WebSocket reconnected[/dim]")
                        )
                    except Exception:
                        # UI status notification should never affect completion flow.
                        pass
                elif reused_existing_connection:
                    try:
                        from rich.text import Text

                        self.display.show_status_message(
                            Text.from_markup("[dim]WebSocket reused[/dim]")
                        )
                    except Exception:
                        # UI status notification should never affect completion flow.
                        pass
                return response, streamed_summary, normalized_input
            except ResponsesWebSocketError as error:
                planner.rollback(error, stream_started=error.stream_started)
                last_error = error
                retry_after_release = (
                    attempt == 0
                    and not error.stream_started
                    and (
                        reused_existing_connection
                        or error.error_code
                        in {
                            "previous_response_not_found",
                            "websocket_connection_limit_reached",
                        }
                    )
                )
                if retry_after_release:
                    reconnect_diagnostics = self._websocket_retry_diagnostics(connection, error)
                if not retry_after_release:
                    raise
            except TimeoutError as error:
                planner.rollback(
                    error,
                    stream_started=stream.stream_started if stream is not None else False,
                )
                raise
            except Exception as exc:
                stream_started = stream.stream_started if stream is not None else False
                planner.rollback(exc, stream_started=stream_started)
                wrapped_error = ResponsesWebSocketError(
                    str(exc),
                    stream_started=stream_started,
                )
                last_error = wrapped_error
                retry_after_release = (
                    attempt == 0 and not stream_started and reused_existing_connection
                )
                if retry_after_release:
                    reconnect_diagnostics = self._websocket_retry_diagnostics(
                        connection,
                        wrapped_error,
                    )
                if not retry_after_release:
                    raise wrapped_error from exc
            finally:
                if not (is_reusable and keep_connection):
                    planner.reset()
                    connection.session_state.request_planner = None
                await self._ws_connections.release(
                    connection,
                    reusable=is_reusable,
                    keep=keep_connection,
                )

            if retry_after_release:
                reconnected = True
                retry_data: dict[str, Any] = {
                    "model": model_name,
                    "url": ws_url,
                    "attempt": attempt + 1,
                    "error": str(last_error) if last_error else None,
                }
                if reconnect_diagnostics is not None:
                    retry_data.update(reconnect_diagnostics)
                self.logger.info(
                    "Reusable Responses websocket connection unavailable; re-establishing connection",
                    data=retry_data,
                )
                continue

        if last_error is not None:
            raise last_error

        raise ResponsesWebSocketError(
            "WebSocket transport failed without an explicit error.",
            stream_started=False,
        )

    def _handle_retry_failure(self, error: Exception) -> PromptMessageExtended | None:
        """Return the legacy error-channel response when retries are exhausted."""
        if isinstance(error, APIError):
            model_name = self.default_request_params.model or DEFAULT_RESPONSES_MODEL
            return build_stream_failure_response(self.provider, error, model_name)
        return None

    async def close(self) -> None:
        """Release long-lived websocket resources used by Responses transport."""

        await self._ws_connections.close()

    def clear(self, *, clear_prompts: bool = False) -> None:
        super().clear(clear_prompts=clear_prompts)
        self._tool_call_id_map.clear()
        self._seen_tool_call_ids.clear()
        self._tool_call_diagnostics = None
        self._last_ws_request_type = None
        self._last_ws_request_mode = None
