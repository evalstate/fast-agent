"""Copilot routing for the existing Responses turn and transport implementations."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from fast_agent.llm.provider.copilot import broker
from fast_agent.llm.provider.copilot.models import get_copilot_model
from fast_agent.llm.provider.copilot.policy import (
    apply_policy,
    reject_files,
    reject_overrides,
    request_headers,
)
from fast_agent.llm.provider.openai.responses import (
    ResponsesActiveTransport,
    ResponsesLLM,
    ResponsesTransport,
    _ResponsesWsAttemptState,
    _ResponsesWsContext,
)
from fast_agent.llm.provider.openai.responses_websocket import (
    ManagedWebSocketConnection,
    StatelessResponsesWsPlanner,
    connect_websocket,
    merge_headers_case_insensitive,
    resolve_responses_ws_url,
)
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
    from fast_agent.types import RequestParams


class CopilotResponsesLLM(ResponsesLLM):
    config_section = "copilot"

    def __init__(self, **kwargs: Any) -> None:
        reject_overrides(kwargs)
        if kwargs.get("web_fetch"):
            raise ValueError("Copilot provider web fetch is not supported.")
        self._copilot_owner_id = uuid4().hex
        self._copilot_endpoint: ContextVar[CopilotEndpoint] = ContextVar(
            "copilot_responses_endpoint"
        )
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.COPILOT, **kwargs)
        if get_copilot_model(self.default_request_params.model or "").wire_api != "responses":
            raise ValueError("Copilot Responses requires a Responses model.")
        self.set_web_search_enabled(self._web_search_override)

    def _get_provider_config(self) -> None:
        return None

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        return ()

    def _default_transport_setting(self) -> ResponsesTransport:
        return "websocket"

    def _validate_transport_support(
        self, model_name: str | None, transport: ResponsesTransport
    ) -> None:
        spec = get_copilot_model(model_name or self.default_request_params.model or "")
        if transport != "sse" and "websocket" not in spec.transports:
            raise ValueError("This Copilot model does not support WebSockets.")

    def _new_ws_request_planner(self) -> StatelessResponsesWsPlanner:
        return StatelessResponsesWsPlanner()

    def validate_provider_credentials(self) -> None:
        return None

    async def _prepare_responses_client(
        self, model: str, transport: ResponsesActiveTransport
    ) -> None:
        if get_copilot_model(model).wire_api != "responses":
            raise ValueError("Copilot Responses requires a Responses model.")
        if self.provider_managed_mcp_state.has_servers():
            raise ValueError("Copilot native MCP is not supported.")
        endpoint = await broker.get_copilot_broker(self.context).resolve(
            model, owner_id=self._copilot_owner_id, transport=transport
        )
        self._copilot_endpoint.set(endpoint)

    def _provider_base_url(self) -> str:
        return self._copilot_endpoint.get().base_url

    def _provider_default_headers(self) -> dict[str, str]:
        return dict(self._copilot_endpoint.get().headers)

    def _responses_client(self) -> AsyncOpenAI:
        endpoint = self._copilot_endpoint.get()
        # Match the SDK's spelling so its default User-Agent is replaced, not
        # sent alongside the native fast-agent identity under another casing.
        headers = {
            "User-Agent" if name.lower() == "user-agent" else name: value
            for name, value in endpoint.headers.items()
        }
        client = AsyncOpenAI(
            api_key="",
            base_url=endpoint.base_url,
            default_headers=headers,
            max_retries=0,
            organization="",
            project="",
            admin_api_key="",
            # The broker validates credentials, including header-only bearer auth.
            _enforce_credentials=False,
            http_client=DefaultAsyncHttpxClient(trust_env=False, follow_redirects=False),
        )
        # The SDK merges OPENAI_CUSTOM_HEADERS during construction; only broker
        # headers belong on this client, without changing the process environment.
        client._custom_headers = headers
        return client

    def _base_responses_url(self) -> str:
        return self._provider_base_url()

    def _build_websocket_headers(self) -> dict[str, str]:
        return dict(self._copilot_endpoint.get().headers)

    async def _create_websocket_connection(
        self, url: str, headers: dict[str, str], timeout_seconds: float | None
    ) -> ManagedWebSocketConnection:
        return await connect_websocket(
            url=url,
            headers=headers,
            timeout_seconds=timeout_seconds,
            client=self._responses_client(),
        )

    async def _acquire_responses_ws_attempt(
        self, *, attempt: int, context: _ResponsesWsContext
    ) -> _ResponsesWsAttemptState:
        # The first binding was resolved before file normalization. Reconnects
        # must re-resolve too; a new credential changes the connection reuse key.
        if attempt:
            old_headers = {name.lower() for name in self._build_websocket_headers()}
            extra_headers = {
                name: value
                for name, value in context.ws_headers.items()
                if name.lower() not in old_headers
            }
            await self._prepare_responses_client(context.model_name, "websocket")
            context.ws_url = resolve_responses_ws_url(self._base_responses_url())
            context.ws_headers = merge_headers_case_insensitive(
                extra_headers,
                self._build_websocket_headers(),
                request_headers(context.arguments.get("input")),
            )
        return await super()._acquire_responses_ws_attempt(attempt=attempt, context=context)

    async def _normalize_input_files(
        self, client: AsyncOpenAI, input_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        reject_files(input_items)
        # Inline images are already wire-ready; never invoke OpenAI's Files API.
        return input_items

    @property
    def web_search_supported(self) -> bool:
        return get_copilot_model(self.default_request_params.model or "").supports_web_search

    @property
    def web_search_enabled(self) -> bool:
        return self.web_search_supported and super().web_search_enabled

    def set_web_search_enabled(self, value: bool | None) -> None:
        if value and not self.web_search_supported:
            raise ValueError("Copilot web search is not verified for this model.")
        super().set_web_search_enabled(value)

    def _append_web_search_tool(self, base_args: dict[str, Any]) -> None:
        if self.web_search_enabled:
            super()._append_web_search_tool(base_args)

    def _apply_response_max_tokens(
        self, base_args: dict[str, Any], request_params: RequestParams
    ) -> None:
        if request_params.max_tokens is not None:
            base_args["max_output_tokens"] = request_params.max_tokens

    def prepare_provider_arguments(
        self,
        base_args: dict,
        request_params: RequestParams,
        exclude_fields: set | None = None,
    ) -> dict:
        arguments = super().prepare_provider_arguments(base_args, request_params, exclude_fields)
        endpoint = self._copilot_endpoint.get()
        return apply_policy(arguments, get_copilot_model(endpoint.model_id), endpoint.headers)
