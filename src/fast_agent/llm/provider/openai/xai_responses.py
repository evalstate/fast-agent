from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from mcp.types import ContentBlock, TextContent

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ManagedWebSocketConnection,
    ResponsesWebSocketError,
    ResponsesWsRequestPlanner,
    StatelessResponsesWsPlanner,
)
from fast_agent.llm.provider.openai.tool_event_helpers import (
    first_nonempty_string,
    item_type_is_responses_function_tool_call,
    responses_item_tool_use_id,
)
from fast_agent.llm.provider.openai.web_tools import (
    ResolvedOpenAIWebSearch,
    build_xai_web_search_tool,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage, usage_from_responses_compatible

if TYPE_CHECKING:
    from mcp import Tool
    from openai.types.responses import ResponseUsage

    from fast_agent.llm.provider.openai.responses import ResponsesTransport
    from fast_agent.tool_activity_presentation import ToolActivityFamily
    from fast_agent.types import RequestParams

DEFAULT_XAI_MODEL = "grok-4.3"
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_X_SEARCH_INTERNAL_TOOL_NAMES = frozenset(
    {
        "x_keyword_search",
        "x_semantic_search",
        "x_user_search",
        "x_thread_fetch",
    }
)


class XAIResponsesLLM(ResponsesLLM):
    """LLM implementation for xAI's Responses-compatible API."""

    def _translate_responses_usage(
        self,
        usage: ResponseUsage,
        *,
        provider: Provider,
        model: str,
    ) -> TurnUsage:
        return usage_from_responses_compatible(
            usage.model_dump(mode="json"),
            provider=provider,
            model=model,
        )

    config_section: str | None = "xai"

    def __init__(self, provider: Provider = Provider.XAI, **kwargs: Any) -> None:
        x_search_override = kwargs.pop("x_search", None)
        provider = kwargs.pop("provider", provider)
        self.config_section = "xai"
        super().__init__(provider=provider, **kwargs)
        self._x_search_override: bool | None = (
            bool(x_search_override) if isinstance(x_search_override, bool) else None
        )

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        params = self._initialize_default_params_with_model_fallback(
            kwargs,
            DEFAULT_XAI_MODEL,
        )
        params.parallel_tool_calls = True
        return params

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        return ()

    def _default_transport_setting(self) -> ResponsesTransport:
        return "websocket"

    @property
    def web_search_supported(self) -> bool:
        return True

    @property
    def service_tier_supported(self) -> bool:
        return False

    @property
    def x_search_supported(self) -> bool:
        return True

    @property
    def x_search_enabled(self) -> bool:
        if self._x_search_override is not None:
            return self._x_search_override
        settings = self._get_provider_config()
        return bool(getattr(settings, "x_search", False)) if settings else False

    def set_x_search_enabled(self, value: bool | None) -> None:
        self._x_search_override = value

    def _is_provider_managed_function_call(self, name: str) -> bool:
        return self.x_search_enabled and name in XAI_X_SEARCH_INTERNAL_TOOL_NAMES

    def _tool_family_for_responses_item(
        self,
        *,
        item_type: str | None,
        tool_name: str,
    ) -> "ToolActivityFamily":
        if item_type_is_responses_function_tool_call(
            item_type
        ) and self._is_provider_managed_function_call(tool_name):
            return "remote_tool"
        return super()._tool_family_for_responses_item(item_type=item_type, tool_name=tool_name)

    def _extract_provider_mcp_metadata(
        self,
        response: Any,
    ) -> list[ContentBlock]:
        payloads = super()._extract_provider_mcp_metadata(response)
        if not self.x_search_enabled:
            return payloads

        for output_item in getattr(response, "output", []) or []:
            item_type = getattr(output_item, "type", None)
            if not item_type_is_responses_function_tool_call(item_type):
                continue
            name = getattr(output_item, "name", None)
            if not isinstance(name, str) or not self._is_provider_managed_function_call(name):
                continue

            payload: dict[str, Any] = {
                "type": "server_tool_use",
                "provider_tool_type": "x_search_call",
                "name": name,
            }
            tool_use_id = responses_item_tool_use_id(output_item)
            if tool_use_id is not None:
                payload["id"] = tool_use_id
            status = getattr(output_item, "status", None)
            if isinstance(status, str) and status:
                payload["status"] = status
            raw_input = first_nonempty_string(
                getattr(output_item, "input", None),
                getattr(output_item, "arguments", None),
            )
            if raw_input is not None:
                payload["arguments"] = raw_input
                try:
                    parsed_input = json.loads(raw_input)
                except json.JSONDecodeError:
                    parsed_input = None
                if isinstance(parsed_input, dict):
                    payload["input"] = parsed_input
            payloads.append(TextContent(type="text", text=json.dumps(payload)))
        return payloads

    def _resolve_reasoning_effort(self) -> str | None:
        setting = self.reasoning_effort
        if setting is None:
            return "low"
        return super()._resolve_reasoning_effort()

    def _provider_base_url(self) -> str | None:
        base_url: str | None = os.getenv("XAI_BASE_URL", XAI_BASE_URL)
        settings = self._get_provider_config()
        if settings and getattr(settings, "base_url", None):
            base_url = settings.base_url
        return base_url

    def _provider_default_headers(self) -> dict[str, str] | None:
        settings = self._get_provider_config()
        return getattr(settings, "default_headers", None) if settings else None

    def _build_websocket_headers(self) -> dict[str, str]:
        headers = dict(self._default_headers() or {})
        headers.setdefault("Authorization", f"Bearer {self._api_key()}")
        return headers

    def _uses_oauth_credential(self) -> bool:
        if self._init_api_key is not None:
            return False
        from fast_agent.llm.provider_key_manager import ProviderKeyManager

        return (
            ProviderKeyManager.get_config_file_key("xai", self.context.config) is None
            and ProviderKeyManager.get_env_var("xai") is None
        )

    def _new_ws_request_planner(self) -> ResponsesWsRequestPlanner:
        # Live xAI websocket smoke tests currently hang on store=false
        # `previous_response_id` continuations. Keep ZDR/store=false semantics
        # by replaying full context on each websocket turn until xAI's in-memory
        # continuation path behaves as documented.
        return StatelessResponsesWsPlanner()

    async def _create_websocket_connection(
        self,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> ManagedWebSocketConnection:
        try:
            return await super()._create_websocket_connection(url, headers, timeout_seconds)
        except ResponsesWebSocketError as exc:
            if exc.status != 401:
                raise
            if not self._uses_oauth_credential():
                raise
        from fast_agent.llm.provider.openai.xai_oauth import get_xai_access_token

        if get_xai_access_token(force_refresh=True) is None:
            raise ProviderKeyError(
                "xAI OAuth token rejected",
                "Run `fast-agent auth login xai` to reauthenticate.",
            )
        return await super()._create_websocket_connection(
            url,
            self._build_websocket_headers(),
            timeout_seconds,
        )

    def _build_web_search_tool(
        self,
        resolved_web_search: ResolvedOpenAIWebSearch,
    ) -> dict[str, Any] | None:
        return build_xai_web_search_tool(resolved_web_search)

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        args = super()._build_response_args(input_items, request_params, tools)
        # Keep the first pass xAI payload conservative; these are OpenAI-specific
        # Responses extensions and xAI's websocket docs show the portable core.
        args.pop("include", None)
        args.pop("service_tier", None)
        reasoning = args.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            args["reasoning"] = {"effort": effort} if effort else reasoning
        if self.x_search_enabled:
            tools_payload = args.setdefault("tools", [])
            if isinstance(tools_payload, list):
                tools_payload.append({"type": "x_search"})
        return args
