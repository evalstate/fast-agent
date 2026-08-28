from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import uuid4

from mcp_types import ContentBlock, TextContent
from openai import APIError, AsyncOpenAI
from pydantic import ValidationError

from fast_agent.core.exceptions import ModelConfigError, ProviderKeyError
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ManagedWebSocketConnection,
    ResponsesWebSocketError,
    ResponsesWebSocketKeepaliveOptions,
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
from fast_agent.llm.provider.openai.xai_image_uploads import XAIImageUploadManager
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage, usage_from_responses_compatible

if TYPE_CHECKING:
    from mcp import Tool
    from openai.types.responses import ResponseUsage

    from fast_agent.config import XAISettings
    from fast_agent.llm.provider.openai.responses import ResponsesTransport
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.tool_activity_presentation import ToolActivityFamily
    from fast_agent.types import RequestParams

DEFAULT_XAI_MODEL = "grok-4.6"
GROK_EXTENDED_STREAMING_TIMEOUT: Final = 300.0
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_EXPERIMENTAL_STREAMING_MODELS: Final = frozenset({"grok-4.5", "grok-4.6"})
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
        self._apply_reasoning_streaming_timeout_default()
        self._x_search_override: bool | None = (
            bool(x_search_override) if isinstance(x_search_override, bool) else None
        )
        self._prompt_cache_key = uuid4().hex
        settings = self._xai_settings()
        self._image_upload_manager = (
            XAIImageUploadManager(settings.image_upload_ttl_seconds)
            if settings is not None and settings.image_upload_mode == "public_url"
            else None
        )
        self._image_upload_warning_emitted = False

    def _apply_reasoning_streaming_timeout_default(self) -> None:
        params = self.default_request_params
        if (
            self._init_request_params is not None
            and "streaming_timeout" in self._init_request_params.model_fields_set
        ):
            return
        effort = self._resolve_reasoning_effort()
        if (params.model == "grok-4.5" and effort == "high") or (
            params.model == "grok-4.6" and effort in {"high", "xhigh"}
        ):
            params.streaming_timeout = GROK_EXTENDED_STREAMING_TIMEOUT

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
        settings = self._xai_settings()
        return settings.x_search if settings is not None else False

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
            default = self._reasoning_effort_spec.default if self._reasoning_effort_spec else None
            if default is not None and default.kind == "effort" and isinstance(default.value, str):
                return default.value
            return "high"
        return super()._resolve_reasoning_effort()

    def _xai_settings(self) -> XAISettings | None:
        return cast("XAISettings | None", self._get_provider_config())

    def _provider_base_url(self) -> str | None:
        base_url: str | None = os.getenv("XAI_BASE_URL", XAI_BASE_URL)
        settings = self._xai_settings()
        if settings is not None and settings.base_url:
            base_url = settings.base_url
        return base_url

    def _provider_default_headers(self) -> dict[str, str] | None:
        settings = self._xai_settings()
        return settings.default_headers if settings is not None else None

    async def _normalize_input_image_part(
        self,
        client: AsyncOpenAI,
        part: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        manager = self._image_upload_manager
        image_url = part.get("image_url")
        if manager is None or not isinstance(image_url, str):
            return await super()._normalize_input_image_part(client, part)

        try:
            public_url = await manager.public_url(client, image_url)
        except (APIError, ValidationError):
            if not self._image_upload_warning_emitted:
                self.logger.warning(
                    "xAI image upload failed; falling back to inline image data for this session."
                )
                self._image_upload_warning_emitted = True
            return part, False

        if public_url is None:
            return await super()._normalize_input_image_part(client, part)

        normalized: dict[str, Any] = {
            "type": "input_image",
            "image_url": public_url,
        }
        if detail := part.get("detail"):
            normalized["detail"] = detail
        return normalized, True

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

    def _extract_assistant_message_items(
        self,
        msg: PromptMessageExtended,
    ) -> list[dict[str, Any]]:
        items = super()._extract_assistant_message_items(msg)
        # xAI can reuse one message ID for distinct responses on a persistent
        # websocket. The ID is optional and cannot safely identify replay items.
        for item in items:
            item.pop("id", None)
        return items

    def _input_item_dedupe_key(self, item: dict[str, Any]) -> tuple[str, ...] | None:
        key = super()._input_item_dedupe_key(item)
        encrypted_content = item.get("encrypted_content")
        if (
            key is not None
            and item.get("type") == "reasoning"
            and isinstance(encrypted_content, str)
            and encrypted_content
        ):
            return (*key, encrypted_content)
        return key

    def _websocket_keepalive_options(self) -> ResponsesWebSocketKeepaliveOptions:
        # xAI currently doesn't reliably answer client-generated Ping frames.
        # Keep automatic Pong replies enabled while restoring the previous
        # aiohttp behavior of relying on application stream-idle detection.
        return {"ping_interval": None}

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
                "Run `fast-agent auth provider login xai` to reauthenticate.",
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
        settings = self._xai_settings()
        reasoning_summary = settings.reasoning_summary if settings is not None else None
        stream_tool_calls = settings.stream_tool_calls if settings is not None else False
        model = args.get("model")
        if (reasoning_summary is not None or stream_tool_calls) and (
            not isinstance(model, str) or model not in XAI_EXPERIMENTAL_STREAMING_MODELS
        ):
            supported = ", ".join(sorted(XAI_EXPERIMENTAL_STREAMING_MODELS))
            raise ModelConfigError(
                "xAI reasoning summaries and streamed tool arguments are experimental "
                f"and supported only for {supported}; got '{model}'."
            )

        # xAI accepts encrypted reasoning passback for stateless full-context
        # replay. Keep only the include verified by Grok Build and live probes.
        args["include"] = ["reasoning.encrypted_content"]
        args.pop("service_tier", None)
        args["prompt_cache_key"] = self._prompt_cache_key
        reasoning = args.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            args["reasoning"] = {"effort": effort} if effort else reasoning
            if reasoning_summary is not None:
                args["reasoning"]["summary"] = reasoning_summary
        if stream_tool_calls:
            extra_body = args.setdefault("extra_body", {})
            if isinstance(extra_body, dict):
                extra_body["stream_tool_calls"] = True
        if self.x_search_enabled:
            tools_payload = args.setdefault("tools", [])
            if isinstance(tools_payload, list):
                tools_payload.append({"type": "x_search"})
        return args

    def _prepare_websocket_arguments(self, arguments: dict[str, Any]) -> None:
        extra_body = arguments.get("extra_body")
        if not isinstance(extra_body, dict):
            return
        if extra_body.pop("stream_tool_calls", None) is True:
            arguments["stream_tool_calls"] = True
        if not extra_body:
            arguments.pop("extra_body")

    def clear(self, *, clear_prompts: bool = False) -> None:
        super().clear(clear_prompts=clear_prompts)
        self._prompt_cache_key = uuid4().hex
