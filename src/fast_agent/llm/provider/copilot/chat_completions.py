"""Copilot routing for the existing OpenAI-compatible Chat Completions loop."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fast_agent.llm.provider.copilot import broker
from fast_agent.llm.provider.copilot.models import get_copilot_model
from fast_agent.llm.provider.copilot.openai_client import create_openai_client
from fast_agent.llm.provider.copilot.policy import apply_policy, reject_files, reject_overrides
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )

    from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
    from fast_agent.llm.provider.openai.llm_openai import (
        _OpenAICompletionRequest,
        _OpenAICompletionResponse,
    )
    from fast_agent.types import RequestParams


class CopilotChatCompletionsLLM(OpenAILLM):
    config_section = "copilot"

    def __init__(self, **kwargs: Any) -> None:
        reject_overrides(kwargs)
        if kwargs.get("web_search") or kwargs.get("web_fetch"):
            raise ValueError("Copilot Chat Completions provider web tools are not supported.")
        if kwargs.get("transport") not in (None, "sse"):
            raise ValueError("Copilot Chat Completions only supports SSE.")
        self._copilot_owner_id = uuid4().hex
        self._copilot_endpoint: ContextVar[CopilotEndpoint] = ContextVar(
            "copilot_chat_completions_endpoint"
        )
        super().__init__(provider=Provider.COPILOT, **kwargs)
        if (
            get_copilot_model(self.default_request_params.model or "").wire_api
            != "chat_completions"
        ):
            raise ValueError("Copilot Chat Completions requires a Chat Completions model.")

    def _openai_reasoning_settings(self) -> None:
        return None

    def _resolve_reasoning_effort(self) -> str | None:
        # The catalog lists accepted efforts, but does not advertise a default.
        setting = self.reasoning_effort
        return str(setting.value) if setting is not None and setting.kind == "effort" else None

    def _get_provider_config(self) -> None:
        return None

    def validate_provider_credentials(self) -> None:
        # Credentials are resolved asynchronously by the endpoint broker.
        return None

    def supports_files_api(self) -> bool:
        return False

    def _prepare_api_request(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
        spec = get_copilot_model(self.default_request_params.model or "")
        # Validate before resolving credentials; bind request headers only once the
        # broker's protected headers are known.
        apply_policy(dict(arguments), spec, {})
        return arguments

    async def _run_openai_completion_request(
        self, request: _OpenAICompletionRequest
    ) -> _OpenAICompletionResponse:
        if self.provider_managed_mcp_state.has_servers():
            raise ValueError("Copilot native MCP is not supported.")
        endpoint = await broker.get_copilot_broker(self.context).resolve(
            request.model_name, owner_id=self._copilot_owner_id, transport="sse"
        )
        if endpoint.wire_api != "chat_completions":
            raise ValueError("Copilot Chat Completions requires a Chat Completions model.")
        self._copilot_endpoint.set(endpoint)
        request.arguments = apply_policy(
            request.arguments, get_copilot_model(endpoint.model_id), endpoint.headers
        )
        return await super()._run_openai_completion_request(request)

    def _openai_client(self) -> AsyncOpenAI:
        return create_openai_client(self._copilot_endpoint.get())

    async def _normalize_chat_completion_files(
        self,
        client: AsyncOpenAI,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        # Never use the direct provider Files API, including for URL attachments.
        reject_files(messages)
        return messages
