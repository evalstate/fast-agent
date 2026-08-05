from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fast_agent.constants import REASONING
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.web_tools import (
    ResolvedOpenAIWebSearch,
    build_web_search_tool,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mcp import Tool
    from mcp.types import ContentBlock

    from fast_agent.config import DeepSeekSettings
    from fast_agent.llm.provider.openai.responses import ResponsesTransport
    from fast_agent.types import RequestParams

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "max"


class DeepSeekResponsesLLM(ResponsesLLM):
    """LLM implementation for DeepSeek's stateless Responses API."""

    config_section: str | None = "deepseek"

    def __init__(self, provider: Provider = Provider.DEEPSEEK, **kwargs: Any) -> None:
        provider = kwargs.pop("provider", provider)
        self.config_section = "deepseek"
        super().__init__(provider=provider, **kwargs)

        model = self.default_request_params.model
        if model != DEFAULT_DEEPSEEK_MODEL:
            raise ModelConfigError(
                "DeepSeek Responses currently supports only "
                f"'{DEFAULT_DEEPSEEK_MODEL}', got '{model}'."
            )

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        return self._initialize_default_params_with_model_fallback(
            kwargs,
            DEFAULT_DEEPSEEK_MODEL,
        )

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        return ()

    def _default_transport_setting(self) -> ResponsesTransport:
        return "sse"

    @property
    def service_tier_supported(self) -> bool:
        return False

    def _deepseek_settings(self) -> DeepSeekSettings | None:
        return cast("DeepSeekSettings | None", self._get_provider_config())

    def _provider_base_url(self) -> str:
        settings = self._deepseek_settings()
        if settings is not None and settings.base_url:
            return settings.base_url
        return DEEPSEEK_BASE_URL

    def _provider_default_headers(self) -> dict[str, str] | None:
        settings = self._deepseek_settings()
        return settings.default_headers if settings is not None else None

    def _build_web_search_tool(
        self,
        resolved_web_search: ResolvedOpenAIWebSearch,
    ) -> dict[str, Any] | None:
        payload = build_web_search_tool(resolved_web_search)
        return {"type": "web_search"} if payload is not None else None

    def _resolve_reasoning_effort(self) -> str | None:
        setting = self.reasoning_effort
        if setting is None:
            return DEFAULT_DEEPSEEK_REASONING_EFFORT
        if setting.kind == "toggle":
            return "none" if setting.value is False else DEFAULT_DEEPSEEK_REASONING_EFFORT
        if setting.kind == "budget":
            self.logger.warning("Ignoring budget reasoning setting for DeepSeek.")
            return DEFAULT_DEEPSEEK_REASONING_EFFORT
        return super()._resolve_reasoning_effort()

    def _apply_response_reasoning(self, base_args: dict[str, Any]) -> None:
        effort = self._resolve_reasoning_effort()
        base_args["reasoning"] = {"effort": effort or "none"}

    def _extract_encrypted_reasoning_items(
        self,
        channels: Mapping[str, Iterable[ContentBlock]] | None,
    ) -> list[dict[str, Any]]:
        if not channels:
            return []
        reasoning_blocks = channels.get(REASONING)
        if not reasoning_blocks:
            return []

        content = [
            {"type": "reasoning_text", "text": text}
            for block in reasoning_blocks
            if (text := get_text(block))
        ]
        return [{"type": "reasoning", "content": content}] if content else []

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        args = super()._build_response_args(input_items, request_params, tools)
        # DeepSeek is stateless and silently ignores these OpenAI-only controls.
        for field in ("include", "parallel_tool_calls", "service_tier", "store"):
            args.pop(field, None)
        return args
