import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from mcp import Tool
from mcp.types import ContentBlock, TextContent
from openai import APIError, AsyncOpenAI, AuthenticationError, DefaultAioHttpClient

from fast_agent.constants import (
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    OPENAI_ASSISTANT_MESSAGE_ITEMS,
    OPENAI_MCP_LIST_TOOLS_ITEMS,
    OPENAI_REASONING_ENCRYPTED,
    REASONING,
)
from fast_agent.core.exceptions import ModelConfigError, ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.fastagent_llm import FastAgentLLM
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
from fast_agent.llm.provider.openai.streaming_utils import with_stream_idle_timeout
from fast_agent.llm.provider.openai.structured_output import OpenAIStructuredOutputMixin
from fast_agent.llm.provider.openai.web_tools import (
    ResolvedOpenAIWebSearch,
    build_web_search_tool,
    resolve_web_search,
)
from fast_agent.llm.provider.reasoning_config import reasoning_setting_from_config
from fast_agent.llm.provider.streaming_timeouts import enter_stream_with_timeout
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import format_reasoning_setting, parse_reasoning_setting
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.text_verbosity import parse_text_verbosity
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.provider_management import build_openai_provider_managed_mcp_tools
from fast_agent.tools.apply_patch_tool import get_openai_responses_custom_tool_payload
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.env import env_flag
from fast_agent.utils.text import strip_casefold

_logger = get_logger(__name__)

DEFAULT_RESPONSES_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "medium"
MIN_RESPONSES_MAX_TOKENS = 16
DEFAULT_RESPONSES_BASE_URL = "https://api.openai.com/v1"
RESPONSES_DIAGNOSTICS_CHANNEL = "fast-agent-provider-diagnostics"
RESPONSE_INCLUDE_REASONING = "reasoning.encrypted_content"
RESPONSE_INCLUDE_WEB_SEARCH_SOURCES = "web_search_call.action.sources"
OPENAI_TOOL_SEARCH_TOOL = {"type": "tool_search", "execution": "server"}

ResponsesTransport = Literal["sse", "websocket", "auto"]
ResponsesActiveTransport = Literal["sse", "websocket"]
ResponsesServiceTier = Literal["fast", "flex"]
ResponsesWsTurnOutcome = Literal["fresh", "reused", "reconnected"]

RESPONSES_TRANSPORT_SSE: ResponsesActiveTransport = "sse"
RESPONSES_TRANSPORT_WEBSOCKET: ResponsesActiveTransport = "websocket"
RESPONSES_TRANSPORT_AUTO: ResponsesTransport = "auto"
RESPONSES_WS_RECONNECT_OUTCOME: ResponsesWsTurnOutcome = "reconnected"
RESPONSES_WS_REUSED_OUTCOME: ResponsesWsTurnOutcome = "reused"
RESPONSES_WS_FRESH_OUTCOME: ResponsesWsTurnOutcome = "fresh"
RESPONSES_WS_TURN_INDICATORS: dict[ResponsesWsTurnOutcome, str] = {
    RESPONSES_WS_RECONNECT_OUTCOME: "↗",
    RESPONSES_WS_REUSED_OUTCOME: "↔",
    RESPONSES_WS_FRESH_OUTCOME: "↗",
}
RESPONSES_WS_SUPPORTED_TRANSPORTS = {
    RESPONSES_TRANSPORT_WEBSOCKET,
    RESPONSES_TRANSPORT_AUTO,
}


@dataclass(slots=True)
class _ResponsesCompletionResult:
    response: Any
    streamed_summary: list[str]
    input_items: list[dict[str, Any]]


@dataclass(slots=True)
class _ResponsesCompletionContext:
    model_name: str
    transport: ResponsesTransport
    display_model: str


@dataclass(slots=True)
class _ResponsesWsContext:
    model_name: str
    normalized_input: list[dict[str, Any]]
    arguments: dict[str, Any]
    capture_filename: Any
    ws_url: str
    ws_headers: dict[str, str]
    timeout: float | None
    started_at: float
    phase_timings: dict[str, float]


@dataclass(slots=True)
class _ResponsesWsAttemptState:
    attempt: int
    connection: ManagedWebSocketConnection
    is_reusable: bool
    reused_existing_connection: bool
    planner: ResponsesWsRequestPlanner
    keep_connection: bool = False
    retry_after_release: bool = False
    reconnect_diagnostics: dict[str, Any] | None = None
    stream: WebSocketResponsesStream | None = None


class ResponsesLLM(
    ResponsesContentMixin,
    ResponsesFileMixin,
    ResponsesOutputMixin,
    ResponsesStreamingMixin,
    OpenAIStructuredOutputMixin,
    FastAgentLLM[dict[str, Any], Any],
):
    """LLM implementation for OpenAI's Responses models."""

    config_section: str | None = None

    RESPONSES_EXCLUDE_FIELDS: ClassVar[set[str]] = {
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
        settings = self._get_provider_config()

        self._initialize_response_state(web_search_override)
        self._configure_reasoning(kwargs, settings)
        self._configure_text_verbosity(kwargs, settings)
        self._configure_service_tier(settings)
        chosen_model = self._configure_reasoning_mode()
        self._configure_transport(kwargs, settings, chosen_model)

    def _initialize_response_state(self, web_search_override: Any) -> None:
        self._tool_call_id_map: dict[str, str] = {}
        self._tool_name_map: dict[str, str] = {}
        self._tool_kind_map: dict[str, Literal["function", "custom"]] = {}
        self._seen_tool_call_ids: set[str] = set()
        self._tool_call_diagnostics: dict[str, Any] | None = None
        self._last_ws_request_type: str | None = None
        self._last_ws_request_mode: Literal["create", "continuation"] | None = None
        self._last_ws_turn_outcome: ResponsesWsTurnOutcome | None = None
        self._last_ws_phase_timings_ms: dict[str, float] | None = None
        self._ws_turn_counters: dict[str, int] = {
            "total": 0,
            RESPONSES_WS_FRESH_OUTCOME: 0,
            RESPONSES_WS_REUSED_OUTCOME: 0,
            RESPONSES_WS_RECONNECT_OUTCOME: 0,
        }
        self._file_id_cache: dict[str, str] = {}
        self._transport: ResponsesTransport = self._default_transport_setting()
        self._last_transport_used: ResponsesActiveTransport | None = None
        self._ws_connections = WebSocketConnectionManager(idle_timeout_seconds=300.0)
        self._ws_debug_inline = env_flag("FAST_AGENT_DEBUG_RESPONSES_WS")
        self._web_search_override: bool | None = (
            bool(web_search_override) if isinstance(web_search_override, bool) else None
        )

    def _configure_reasoning(self, kwargs: dict[str, Any], settings: Any) -> None:
        raw_setting = kwargs.get("reasoning_effort")
        if settings and raw_setting is None:
            raw_setting, warn_deprecated_reasoning_effort = reasoning_setting_from_config(settings)
            if warn_deprecated_reasoning_effort:
                self.logger.warning(
                    "Responses config 'reasoning_effort' is deprecated; use 'reasoning'."
                )

        setting = parse_reasoning_setting(raw_setting)
        if setting is not None:
            try:
                self.set_reasoning_effort(setting)
            except ValueError as exc:
                self.logger.warning(f"Invalid reasoning setting: {exc}")

    def _configure_text_verbosity(self, kwargs: dict[str, Any], settings: Any) -> None:
        raw_text_verbosity = kwargs.get("text_verbosity")
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

    def _configure_service_tier(self, settings: Any) -> None:
        if self.default_request_params.service_tier is None and settings is not None:
            configured_service_tier = self._normalize_service_tier(
                getattr(settings, "service_tier", None)
            )
            if configured_service_tier is not None:
                self.default_request_params.service_tier = configured_service_tier

        self.default_request_params.service_tier = self._ensure_supported_service_tier(
            self.default_request_params.service_tier,
            source="initial configuration",
        )

    def _configure_reasoning_mode(self) -> str | None:
        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning_mode = self._get_model_reasoning(chosen_model)
        self._reasoning = self._reasoning_mode == "openai"
        if self._reasoning_mode:
            self.logger.info(
                f"Using Responses model '{chosen_model}' (mode='{self._reasoning_mode}') with "
                f"'{format_reasoning_setting(self.reasoning_effort)}' reasoning effort"
            )
        return chosen_model

    def _configure_transport(
        self,
        kwargs: dict[str, Any],
        settings: Any,
        chosen_model: str | None,
    ) -> None:
        self._transport = self._resolve_transport_setting(kwargs.get("transport"), settings)
        self._validate_transport_support(chosen_model, self._transport)

    @property
    def active_transport(self) -> ResponsesActiveTransport | None:
        """Return the transport used by the most recent completion call."""
        return self._last_transport_used

    @property
    def configured_transport(self) -> ResponsesTransport:
        """Return configured transport preference for this LLM instance."""
        return self._transport

    @property
    def websocket_turn_indicator(self) -> str | None:
        """Small glyph representing the websocket outcome for the last turn."""
        if self._last_ws_turn_outcome is None:
            return None
        return RESPONSES_WS_TURN_INDICATORS.get(self._last_ws_turn_outcome)

    @property
    def websocket_turn_metrics(self) -> dict[str, int] | None:
        """Cumulative websocket turn counters for this LLM instance."""
        if self._ws_turn_counters["total"] <= 0:
            return None
        return dict(self._ws_turn_counters)

    def _record_ws_turn_outcome(self, outcome: ResponsesWsTurnOutcome) -> None:
        self._last_ws_turn_outcome = outcome
        self._ws_turn_counters["total"] += 1
        self._ws_turn_counters[outcome] += 1

    def _websocket_diagnostics_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"transport": self._last_transport_used or "unknown"}
        if self._last_transport_used != RESPONSES_TRANSPORT_WEBSOCKET:
            return payload
        if self._last_ws_request_type:
            payload["websocket_request_type"] = self._last_ws_request_type
        if self._last_ws_request_mode is not None:
            payload["websocket_request_mode"] = self._last_ws_request_mode
        if self._last_ws_turn_outcome is not None:
            payload["websocket_turn_outcome"] = self._last_ws_turn_outcome
        if metrics := self.websocket_turn_metrics:
            payload["websocket_turn_metrics"] = metrics
        if self._last_ws_phase_timings_ms:
            payload["websocket_phase_ms"] = self._last_ws_phase_timings_ms
        return payload

    def _parse_service_tier(self, raw_value: Any) -> ResponsesServiceTier | None:
        if raw_value is None:
            return None

        normalized = strip_casefold(str(raw_value))
        if normalized == "fast":
            return "fast"
        if normalized == "flex":
            return "flex"

        self.logger.warning(f"Invalid service tier setting: {raw_value}")
        return None

    def _available_service_tiers_for_model(
        self, model_name: str | None
    ) -> tuple[ResponsesServiceTier, ...]:
        if self.provider == Provider.CODEX_RESPONSES:
            return ("fast",)
        if model_name:
            configured_tiers = self._get_model_response_service_tiers(model_name)
            if configured_tiers is not None:
                return configured_tiers
        return ("fast", "flex")

    def _normalize_service_tier(self, raw_value: Any) -> ResponsesServiceTier | None:
        normalized = self._parse_service_tier(raw_value)
        if normalized is None:
            return None
        if normalized not in self.available_service_tiers:
            self.logger.warning(
                f"Service tier '{normalized}' is not supported for provider '{self.provider.value}'."
            )
            return None
        return normalized

    def _ensure_supported_service_tier(
        self,
        service_tier: ResponsesServiceTier | None,
        *,
        source: str,
        model_name: str | None = None,
    ) -> ResponsesServiceTier | None:
        if service_tier is None:
            return None
        available_tiers = self._available_service_tiers_for_model(model_name)
        if service_tier in available_tiers:
            return service_tier
        allowed = ", ".join(available_tiers) or "standard"
        target_model = f" for model '{model_name}'" if model_name else ""
        raise ModelConfigError(
            f"Provider '{self.provider.value}' does not support service tier '{service_tier}' "
            f"from {source}{target_model}. Allowed values: {allowed} or unset (standard)."
        )

    def _resolve_service_tier(
        self,
        request_params: RequestParams,
        model_name: str | None,
    ) -> ResponsesServiceTier | None:
        if "service_tier" in request_params.model_fields_set:
            return self._ensure_supported_service_tier(
                request_params.service_tier,
                source="request params",
                model_name=model_name,
            )
        return self._ensure_supported_service_tier(
            self.service_tier,
            source="provider defaults",
            model_name=model_name,
        )

    @staticmethod
    def _map_service_tier_to_wire_value(service_tier: ResponsesServiceTier) -> str:
        if service_tier == "fast":
            return "priority"
        return "flex"

    def _default_transport_setting(self) -> ResponsesTransport:
        if self.provider in {Provider.RESPONSES, Provider.CODEX_RESPONSES}:
            return RESPONSES_TRANSPORT_AUTO
        return RESPONSES_TRANSPORT_SSE

    def _resolve_transport_setting(self, raw_value: Any, settings: Any) -> ResponsesTransport:
        value = raw_value
        if value is None and settings is not None:
            model_fields_set = getattr(settings, "model_fields_set", set())
            if "transport" in model_fields_set:
                value = getattr(settings, "transport", None)
        if value is None:
            return self._default_transport_setting()

        normalized = strip_casefold(str(value))
        transport_aliases: dict[str, ResponsesTransport] = {
            "ws": RESPONSES_TRANSPORT_WEBSOCKET,
            RESPONSES_TRANSPORT_SSE: RESPONSES_TRANSPORT_SSE,
            RESPONSES_TRANSPORT_WEBSOCKET: RESPONSES_TRANSPORT_WEBSOCKET,
            RESPONSES_TRANSPORT_AUTO: RESPONSES_TRANSPORT_AUTO,
        }
        normalized_transport = transport_aliases.get(normalized)
        if normalized_transport is not None:
            return normalized_transport

        default_transport = self._default_transport_setting()
        self.logger.warning(
            f"Invalid Responses transport setting; defaulting to {default_transport}",
            data={"transport": value},
        )
        return default_transport

    def _supports_websocket_transport(self) -> bool:
        """Provider-level websocket support flag (opt-in while experimental)."""
        return True

    def _validate_transport_support(
        self,
        model_name: str | None,
        transport: ResponsesTransport,
    ) -> None:
        if transport not in RESPONSES_WS_SUPPORTED_TRANSPORTS:
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

        response_transports = self._get_model_response_transports(model_to_check)
        if not response_transports or RESPONSES_TRANSPORT_WEBSOCKET not in response_transports:
            raise ModelConfigError(
                f"Transport '{transport}' is not supported for model '{model_to_check}'."
            )
        websocket_providers = self._get_model_response_websocket_providers(model_to_check)
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
                        f" {sent_payload_bytes}/{full_payload_bytes}B ({percent_saved}% saved)"
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

    @property
    def service_tier_supported(self) -> bool:
        """Responses-family models expose service tier selection."""
        return bool(self.available_service_tiers)

    @property
    def available_service_tiers(self) -> tuple[ResponsesServiceTier, ...]:
        return self._available_service_tiers_for_model(self.default_request_params.model)

    @property
    def service_tier(self) -> ResponsesServiceTier | None:
        return self.default_request_params.service_tier

    def set_service_tier(self, value: ResponsesServiceTier | None) -> None:
        parsed_value = self._parse_service_tier(value)
        if value is not None and parsed_value is None:
            raise ValueError("Current model does not support the requested service tier.")
        if parsed_value is not None and parsed_value not in self.available_service_tiers:
            allowed = ", ".join(self.available_service_tiers) or "standard"
            raise ValueError(
                f"Current model supports only {allowed} or unset (standard) service tier."
            )
        self.default_request_params.service_tier = parsed_value

    def _provider_base_url(self) -> str | None:
        settings = self._openai_settings()
        return settings.base_url if settings else None

    def _provider_default_headers(self) -> dict[str, str] | None:
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

    def _build_web_search_tool(
        self,
        resolved_web_search: ResolvedOpenAIWebSearch,
    ) -> dict[str, Any] | None:
        return build_web_search_tool(resolved_web_search)

    def _build_declared_tools_payload(
        self,
        tools: list[Tool] | None,
        model: str,
    ) -> list[dict[str, Any]]:
        tools_payload: list[dict[str, Any]] = []
        for tool in tools or []:
            custom_payload = get_openai_responses_custom_tool_payload(tool)
            if custom_payload is not None:
                tools_payload.append(custom_payload)
                continue

            tools_payload.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": self._adjust_schema(tool.inputSchema, model),
                }
            )
        return tools_payload

    def _tools_payload(self, base_args: dict[str, Any]) -> list[dict[str, Any]]:
        tools_payload = base_args.setdefault("tools", [])
        if not isinstance(tools_payload, list):
            tools_payload = []
            base_args["tools"] = tools_payload
        return tools_payload

    def _append_provider_managed_tools(self, base_args: dict[str, Any]) -> None:
        if not self.provider_managed_mcp_state.has_servers():
            return
        if self.provider != Provider.RESPONSES:
            raise ModelConfigError(
                "Provider-managed MCP is not supported for this Responses-family provider."
            )

        tools_payload = self._tools_payload(base_args)
        tools_payload.extend(
            build_openai_provider_managed_mcp_tools(self.provider_managed_mcp_state)
        )
        if self._needs_tool_search_for_deferred_provider_tools(tools_payload):
            tools_payload.append(dict(OPENAI_TOOL_SEARCH_TOOL))

    def _needs_tool_search_for_deferred_provider_tools(
        self,
        tools_payload: list[dict[str, Any]],
    ) -> bool:
        has_deferred_attachment = any(
            attachment.defer_loading for attachment in self.provider_managed_mcp_state.attachments
        )
        has_tool_search = any(
            isinstance(tool_payload, dict) and tool_payload.get("type") == "tool_search"
            for tool_payload in tools_payload
        )
        return has_deferred_attachment and not has_tool_search

    def _append_web_search_tool(self, base_args: dict[str, Any]) -> None:
        resolved_web_search = resolve_web_search(
            self._openai_settings(),
            web_search_override=self._web_search_override,
        )
        web_search_tool = self._build_web_search_tool(resolved_web_search)
        if web_search_tool is None:
            return

        self._tools_payload(base_args).append(web_search_tool)
        include_payload = base_args.get("include")
        if (
            isinstance(include_payload, list)
            and RESPONSE_INCLUDE_WEB_SEARCH_SOURCES not in include_payload
        ):
            include_payload.append(RESPONSE_INCLUDE_WEB_SEARCH_SOURCES)

    def _apply_response_reasoning(self, base_args: dict[str, Any]) -> None:
        if not self._reasoning:
            return

        effort = self._resolve_reasoning_effort()
        if effort:
            base_args["reasoning"] = {
                "summary": "auto",
                "effort": effort,
            }

    def _apply_response_max_tokens(
        self,
        base_args: dict[str, Any],
        request_params: RequestParams,
    ) -> None:
        if request_params.maxTokens is None:
            return

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

    def _apply_response_text_options(
        self,
        base_args: dict[str, Any],
        request_params: RequestParams,
    ) -> None:
        if request_params.response_format:
            base_args["text"] = {
                "format": self._normalize_text_format(request_params.response_format)
            }

        text_verbosity_spec = self.text_verbosity_spec
        if not text_verbosity_spec:
            return

        text_payload = base_args.get("text")
        if not isinstance(text_payload, dict):
            text_payload = {}
        text_payload["verbosity"] = self.text_verbosity or text_verbosity_spec.default
        base_args["text"] = text_payload

    def _apply_response_service_tier(
        self,
        base_args: dict[str, Any],
        request_params: RequestParams,
        model: str,
    ) -> None:
        service_tier = self._resolve_service_tier(request_params, model)
        if service_tier is not None:
            base_args["service_tier"] = self._map_service_tier_to_wire_value(service_tier)

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

        tools_payload = self._build_declared_tools_payload(tools, model)
        if tools_payload:
            base_args["tools"] = tools_payload

        self._append_provider_managed_tools(base_args)
        self._append_web_search_tool(base_args)
        self._apply_response_reasoning(base_args)
        self._apply_response_max_tokens(base_args, request_params)
        self._apply_response_text_options(base_args, request_params)
        self._apply_response_service_tier(base_args, request_params, model)

        return self.prepare_provider_arguments(
            base_args, request_params, self.RESPONSES_EXCLUDE_FIELDS
        )

    def _responses_completion_context(
        self,
        request_params: RequestParams,
    ) -> _ResponsesCompletionContext:
        model_name = (
            request_params.model or self.default_request_params.model or DEFAULT_RESPONSES_MODEL
        )
        transport = self._effective_transport()
        self._validate_transport_support(model_name, transport)
        display_model = (
            f"{model_name} [ws]" if transport in RESPONSES_WS_SUPPORTED_TRANSPORTS else model_name
        )
        return _ResponsesCompletionContext(
            model_name=model_name,
            transport=transport,
            display_model=display_model,
        )

    def _reset_responses_transport_diagnostics(self) -> None:
        self._last_ws_request_type = None
        self._last_ws_request_mode = None
        self._last_ws_turn_outcome = None
        self._last_ws_phase_timings_ms = None

    async def _run_responses_transport(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        context: _ResponsesCompletionContext,
    ) -> _ResponsesCompletionResult:
        if context.transport == RESPONSES_TRANSPORT_SSE:
            response, streamed_summary, normalized_input = await self._responses_completion_sse(
                input_items=input_items,
                request_params=request_params,
                tools=tools,
                model_name=context.model_name,
            )
            self._last_transport_used = RESPONSES_TRANSPORT_SSE
            return _ResponsesCompletionResult(response, streamed_summary, normalized_input)

        try:
            response, streamed_summary, normalized_input = await self._responses_completion_ws(
                input_items=input_items,
                request_params=request_params,
                tools=tools,
                model_name=context.model_name,
            )
            self._last_transport_used = RESPONSES_TRANSPORT_WEBSOCKET
            return _ResponsesCompletionResult(response, streamed_summary, normalized_input)
        except ResponsesWebSocketError as error:
            if context.transport != RESPONSES_TRANSPORT_AUTO or error.stream_started:
                raise
            return await self._run_responses_sse_fallback(
                input_items=input_items,
                request_params=request_params,
                tools=tools,
                context=context,
                error=error,
            )

    async def _run_responses_sse_fallback(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        context: _ResponsesCompletionContext,
        error: ResponsesWebSocketError,
    ) -> _ResponsesCompletionResult:
        self.logger.warning(
            "WebSocket transport failed before stream start; falling back to SSE "
            "(auto transport safeguard)",
            data={
                "model": context.model_name,
                "requested_transport": context.transport,
                "error": str(error),
            },
        )
        self._show_responses_sse_fallback_status()
        response, streamed_summary, normalized_input = await self._responses_completion_sse(
            input_items=input_items,
            request_params=request_params,
            tools=tools,
            model_name=context.model_name,
        )
        self._last_transport_used = RESPONSES_TRANSPORT_SSE
        return _ResponsesCompletionResult(response, streamed_summary, normalized_input)

    def _show_responses_sse_fallback_status(self) -> None:
        try:
            from rich.text import Text

            self.display.show_status_message(
                Text.from_markup(
                    "[yellow]⚠ WebSocket transport unavailable for this turn; using SSE fallback.[/yellow]"
                )
            )
        except Exception:
            pass

    @staticmethod
    def _add_response_channel(
        channels: dict[str, list[ContentBlock]] | None,
        name: str,
        blocks: list[ContentBlock],
    ) -> dict[str, list[ContentBlock]] | None:
        if not blocks:
            return channels
        if channels is None:
            channels = {}
        channels[name] = blocks
        return channels

    def _responses_reasoning_channels(
        self,
        response: Any,
        streamed_summary: list[str],
    ) -> dict[str, list[ContentBlock]] | None:
        channels: dict[str, list[ContentBlock]] | None = None
        channels = self._add_response_channel(
            channels, REASONING, self._extract_reasoning_summary(response, streamed_summary)
        )
        return self._add_response_channel(
            channels, OPENAI_REASONING_ENCRYPTED, self._extract_encrypted_reasoning(response)
        )

    def _responses_raw_item_channels(
        self,
        response: Any,
        channels: dict[str, list[ContentBlock]] | None,
    ) -> tuple[dict[str, list[ContentBlock]] | None, Any]:
        assistant_message_items, message_phase = self._extract_raw_assistant_message_items(response)
        channels = self._add_response_channel(
            channels, OPENAI_ASSISTANT_MESSAGE_ITEMS, assistant_message_items
        )
        channels = self._add_response_channel(
            channels,
            OPENAI_MCP_LIST_TOOLS_ITEMS,
            self._extract_raw_mcp_list_tools_items(response),
        )
        return channels, message_phase

    def _responses_diagnostics_channels(
        self,
        channels: dict[str, list[ContentBlock]] | None,
    ) -> dict[str, list[ContentBlock]] | None:
        tool_call_diagnostics = self._consume_tool_call_diagnostics()
        diagnostics_payload = dict(tool_call_diagnostics) if tool_call_diagnostics else None
        websocket_diagnostics = self._websocket_diagnostics_payload()
        if diagnostics_payload is not None:
            diagnostics_payload.update(websocket_diagnostics)
        elif self._last_transport_used == RESPONSES_TRANSPORT_WEBSOCKET:
            diagnostics_payload = websocket_diagnostics

        if not diagnostics_payload:
            return channels
        return self._add_response_channel(
            channels,
            RESPONSES_DIAGNOSTICS_CHANNEL,
            [TextContent(type="text", text=json.dumps(diagnostics_payload))],
        )

    def _responses_server_tool_channels(
        self,
        response: Any,
        channels: dict[str, list[ContentBlock]] | None,
    ) -> dict[str, list[ContentBlock]] | None:
        tool_search_payloads = self._extract_tool_search_metadata(response)
        web_tool_payloads, citation_payloads = self._extract_web_search_metadata(response)
        provider_mcp_payloads = self._extract_provider_mcp_metadata(response)
        server_tool_payloads = [
            *tool_search_payloads,
            *web_tool_payloads,
            *provider_mcp_payloads,
        ]
        channels = self._add_response_channel(
            channels, ANTHROPIC_SERVER_TOOLS_CHANNEL, server_tool_payloads
        )
        return self._add_response_channel(channels, ANTHROPIC_CITATIONS_CHANNEL, citation_payloads)

    @staticmethod
    def _responses_content_blocks(response: Any) -> list[ContentBlock]:
        response_content_blocks: list[ContentBlock] = []
        for output_item in getattr(response, "output", []) or []:
            if getattr(output_item, "type", None) != "message":
                continue
            response_content_blocks.extend(
                TextContent(type="text", text=getattr(part, "text", ""))
                for part in getattr(output_item, "content", []) or []
                if getattr(part, "type", None) == "output_text"
            )

        if not response_content_blocks:
            output_text = getattr(response, "output_text", None)
            if output_text:
                response_content_blocks.append(TextContent(type="text", text=output_text))
        return response_content_blocks

    def _finalize_responses_completion(
        self,
        *,
        result: _ResponsesCompletionResult,
        context: _ResponsesCompletionContext,
    ) -> PromptMessageExtended:
        response = result.response
        if response is None:
            raise RuntimeError("Responses stream did not return a final response")

        self._log_chat_finished(model=context.model_name)
        channels = self._responses_reasoning_channels(response, result.streamed_summary)
        tool_calls = self._extract_tool_calls(response)
        channels, message_phase = self._responses_raw_item_channels(response, channels)
        channels = self._responses_diagnostics_channels(channels)
        channels = self._responses_server_tool_channels(response, channels)

        if getattr(response, "usage", None):
            self._record_usage(response.usage, context.model_name)
        self.history.set(result.input_items)

        return PromptMessageExtended(
            role="assistant",
            content=self._responses_content_blocks(response),
            tool_calls=tool_calls,
            channels=channels,
            stop_reason=(
                LlmStopReason.TOOL_USE if tool_calls else self._map_response_stop_reason(response)
            ),
            phase=message_phase,
        )

    async def _responses_completion(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        context = self._responses_completion_context(request_params)
        self._log_chat_progress(self.chat_turn(), model=context.display_model)
        self._reset_responses_transport_diagnostics()
        try:
            result = await self._run_responses_transport(
                input_items=input_items,
                request_params=request_params,
                tools=tools,
                context=context,
            )
        except asyncio.CancelledError:
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )
        return self._finalize_responses_completion(result=result, context=context)

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
                timeout = request_params.streaming_timeout
                async with self._response_sse_stream(
                    client=client,
                    arguments=arguments,
                    timeout_seconds=timeout,
                ) as stream:
                    timed_stream = with_stream_idle_timeout(
                        stream,
                        idle_timeout_seconds=timeout,
                        timeout_message=f"Streaming was idle for more than {timeout} seconds.",
                    )
                    try:
                        response, streamed_summary = await self._process_stream(
                            timed_stream, model_name, capture_filename
                        )
                    except TimeoutError:
                        if timeout is None:
                            raise
                        self.logger.error(
                            "Streaming idle timeout while waiting for Responses",
                            data={
                                "model": model_name,
                                "timeout_seconds": timeout,
                            },
                        )
                        raise
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

    @asynccontextmanager
    async def _response_sse_stream(
        self,
        *,
        client: AsyncOpenAI,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ):
        async with enter_stream_with_timeout(
            client.responses.stream(**arguments),
            timeout_seconds=timeout_seconds,
            timeout_message=f"Responses stream did not start within {timeout_seconds} seconds.",
        ) as stream:
            yield stream

    @staticmethod
    def _record_ws_phase(
        phase_timings: dict[str, float],
        name: str,
        phase_started_at: float,
    ) -> None:
        phase_timings[name] = round((time.perf_counter() - phase_started_at) * 1000.0, 2)

    async def _responses_ws_context(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        model_name: str,
    ) -> _ResponsesWsContext:
        started_at = time.perf_counter()
        phase_timings: dict[str, float] = {}
        self._last_ws_phase_timings_ms = phase_timings

        async with self._responses_client() as client:
            phase_started_at = time.perf_counter()
            normalized_input = await self._normalize_input_files(client, input_items)
            self._record_ws_phase(phase_timings, "normalize_input", phase_started_at)

        phase_started_at = time.perf_counter()
        arguments = self._build_response_args(normalized_input, request_params, tools)
        self._record_ws_phase(phase_timings, "build_args", phase_started_at)
        self.logger.debug("Responses websocket request", data=arguments)
        capture_filename = _stream_capture_filename(self.chat_turn())
        _save_stream_request(capture_filename, arguments)

        return _ResponsesWsContext(
            model_name=model_name,
            normalized_input=normalized_input,
            arguments=arguments,
            capture_filename=capture_filename,
            ws_url=resolve_responses_ws_url(self._base_responses_url()),
            ws_headers=self._build_websocket_headers(),
            timeout=request_params.streaming_timeout,
            started_at=started_at,
            phase_timings=phase_timings,
        )

    async def _acquire_responses_ws_attempt(
        self,
        *,
        attempt: int,
        context: _ResponsesWsContext,
    ) -> _ResponsesWsAttemptState:
        async def _create_connection() -> ManagedWebSocketConnection:
            return await self._create_websocket_connection(
                context.ws_url, context.ws_headers, context.timeout
            )

        phase_started_at = time.perf_counter()
        connection, is_reusable = await self._ws_connections.acquire(_create_connection)
        self._record_ws_phase(context.phase_timings, "acquire_connection", phase_started_at)
        reused_existing_connection = is_reusable and connection.last_used_monotonic > 0.0
        planner = connection.session_state.request_planner
        if planner is None:
            planner = self._new_ws_request_planner()
            connection.session_state.request_planner = planner
        return _ResponsesWsAttemptState(
            attempt=attempt,
            connection=connection,
            is_reusable=is_reusable,
            reused_existing_connection=reused_existing_connection,
            planner=planner,
        )

    async def _run_responses_ws_attempt(
        self,
        *,
        context: _ResponsesWsContext,
        attempt_state: _ResponsesWsAttemptState,
        reconnected: bool,
    ) -> _ResponsesCompletionResult:
        self.logger.info(
            "Using Responses websocket transport",
            data={"model": context.model_name, "url": context.ws_url},
        )
        phase_started_at = time.perf_counter()
        planned_request = attempt_state.planner.plan(context.arguments)
        self._record_ws_phase(context.phase_timings, "plan_request", phase_started_at)
        self._last_ws_request_type = planned_request.event_type
        self._report_ws_request_plan(
            model_name=context.model_name,
            ws_url=context.ws_url,
            planned_request=planned_request,
            full_arguments=context.arguments,
            reused_existing_connection=attempt_state.reused_existing_connection,
        )

        phase_started_at = time.perf_counter()
        await send_response_request(attempt_state.connection.websocket, planned_request)
        self._record_ws_phase(context.phase_timings, "send_request", phase_started_at)
        attempt_state.stream = WebSocketResponsesStream(attempt_state.connection.websocket)
        response, streamed_summary = await self._process_responses_ws_stream(
            context, attempt_state.stream
        )
        attempt_state.planner.commit(context.arguments, planned_request, response)
        attempt_state.keep_connection = True
        self._record_responses_ws_success(attempt_state, reconnected, context)
        return _ResponsesCompletionResult(
            response=response,
            streamed_summary=streamed_summary,
            input_items=context.normalized_input,
        )

    async def _process_responses_ws_stream(
        self,
        context: _ResponsesWsContext,
        stream: WebSocketResponsesStream,
    ) -> tuple[Any, list[str]]:
        stream_started_at = time.perf_counter()
        timed_stream = with_stream_idle_timeout(
            stream,
            idle_timeout_seconds=context.timeout,
            timeout_message=f"Streaming was idle for more than {context.timeout} seconds.",
        )
        try:
            response, streamed_summary = await self._process_stream(
                timed_stream, context.model_name, context.capture_filename
            )
        except TimeoutError:
            if context.timeout is None:
                raise
            self.logger.error(
                "Streaming idle timeout while waiting for Responses websocket",
                data={
                    "model": context.model_name,
                    "timeout_seconds": context.timeout,
                },
            )
            raise
        self._record_ws_phase(context.phase_timings, "stream_total", stream_started_at)
        if stream.first_event_monotonic is not None:
            context.phase_timings["first_event"] = round(
                (stream.first_event_monotonic - context.started_at) * 1000.0,
                2,
            )
        return response, streamed_summary

    def _record_responses_ws_success(
        self,
        attempt_state: _ResponsesWsAttemptState,
        reconnected: bool,
        context: _ResponsesWsContext,
    ) -> None:
        if reconnected:
            self._record_ws_turn_outcome(RESPONSES_WS_RECONNECT_OUTCOME)
        elif attempt_state.reused_existing_connection:
            self._record_ws_turn_outcome(RESPONSES_WS_REUSED_OUTCOME)
        else:
            self._record_ws_turn_outcome(RESPONSES_WS_FRESH_OUTCOME)
        context.phase_timings["total"] = round(
            (time.perf_counter() - context.started_at) * 1000.0, 2
        )

    def _handle_responses_ws_error(
        self,
        *,
        error: ResponsesWebSocketError,
        attempt_state: _ResponsesWsAttemptState,
        context: _ResponsesWsContext,
    ) -> ResponsesWebSocketError:
        attempt_state.planner.rollback(error, stream_started=error.stream_started)
        attempt_state.retry_after_release = (
            attempt_state.attempt == 0
            and not error.stream_started
            and (
                attempt_state.reused_existing_connection
                or error.error_code
                in {
                    "previous_response_not_found",
                    "websocket_connection_limit_reached",
                }
            )
        )
        if attempt_state.retry_after_release:
            attempt_state.reconnect_diagnostics = self._websocket_retry_diagnostics(
                attempt_state.connection, error
            )
            return error
        raise error

    def _handle_responses_ws_unexpected_error(
        self,
        *,
        error: Exception,
        attempt_state: _ResponsesWsAttemptState,
    ) -> ResponsesWebSocketError:
        stream_started = (
            attempt_state.stream.stream_started if attempt_state.stream is not None else False
        )
        attempt_state.planner.rollback(error, stream_started=stream_started)
        wrapped_error = ResponsesWebSocketError(str(error), stream_started=stream_started)
        attempt_state.retry_after_release = (
            attempt_state.attempt == 0
            and not stream_started
            and attempt_state.reused_existing_connection
        )
        if attempt_state.retry_after_release:
            attempt_state.reconnect_diagnostics = self._websocket_retry_diagnostics(
                attempt_state.connection,
                wrapped_error,
            )
            return wrapped_error
        raise wrapped_error from error

    async def _release_responses_ws_attempt(
        self,
        attempt_state: _ResponsesWsAttemptState,
    ) -> None:
        if not (attempt_state.is_reusable and attempt_state.keep_connection):
            attempt_state.planner.reset()
            attempt_state.connection.session_state.request_planner = None
        await self._ws_connections.release(
            attempt_state.connection,
            reusable=attempt_state.is_reusable,
            keep=attempt_state.keep_connection,
        )

    def _log_responses_ws_retry(
        self,
        *,
        context: _ResponsesWsContext,
        attempt_state: _ResponsesWsAttemptState,
        last_error: ResponsesWebSocketError | None,
    ) -> None:
        retry_data: dict[str, Any] = {
            "model": context.model_name,
            "url": context.ws_url,
            "attempt": attempt_state.attempt + 1,
            "error": str(last_error) if last_error else None,
        }
        if attempt_state.reconnect_diagnostics is not None:
            retry_data.update(attempt_state.reconnect_diagnostics)
        self.logger.info(
            "Reusable Responses websocket connection unavailable; re-establishing connection",
            data=retry_data,
        )

    async def _responses_completion_ws(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        model_name: str,
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        context = await self._responses_ws_context(
            input_items=input_items,
            request_params=request_params,
            tools=tools,
            model_name=model_name,
        )
        last_error: ResponsesWebSocketError | None = None
        reconnected = False
        for attempt in range(2):
            attempt_state = await self._acquire_responses_ws_attempt(
                attempt=attempt, context=context
            )
            try:
                result = await self._run_responses_ws_attempt(
                    context=context,
                    attempt_state=attempt_state,
                    reconnected=reconnected,
                )
                return result.response, result.streamed_summary, result.input_items
            except ResponsesWebSocketError as error:
                last_error = self._handle_responses_ws_error(
                    error=error, attempt_state=attempt_state, context=context
                )
            except TimeoutError as error:
                attempt_state.planner.rollback(
                    error,
                    stream_started=(
                        attempt_state.stream.stream_started
                        if attempt_state.stream is not None
                        else False
                    ),
                )
                raise
            except Exception as exc:
                last_error = self._handle_responses_ws_unexpected_error(
                    error=exc, attempt_state=attempt_state
                )
            finally:
                await self._release_responses_ws_attempt(attempt_state)

            if attempt_state.retry_after_release:
                reconnected = True
                self._log_responses_ws_retry(
                    context=context,
                    attempt_state=attempt_state,
                    last_error=last_error,
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
        self._tool_kind_map.clear()
        self._seen_tool_call_ids.clear()
        self._tool_call_diagnostics = None
        self._last_ws_request_type = None
        self._last_ws_request_mode = None
        self._last_ws_turn_outcome = None
        self._ws_turn_counters = {
            "total": 0,
            RESPONSES_WS_FRESH_OUTCOME: 0,
            RESPONSES_WS_REUSED_OUTCOME: 0,
            RESPONSES_WS_RECONNECT_OUTCOME: 0,
        }
