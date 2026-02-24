from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, AuthenticationError, DefaultAioHttpClient

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.openai.codex_oauth import parse_chatgpt_account_id
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ResponsesWsRequestPlanner,
    StatefulContinuationResponsesWsPlanner,
)
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from mcp import Tool

    from fast_agent.llm.request_params import RequestParams

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


class CodexResponsesLLM(ResponsesLLM):
    """LLM implementation for Codex responses via ChatGPT OAuth tokens."""

    config_section: str | None = "codexresponses"

    def __init__(self, provider: Provider = Provider.CODEX_RESPONSES, **kwargs: Any) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

    def _display_model(self, model: str | None) -> str | None:
        if not model:
            return model
        return f"{model} ($)"

    def _log_chat_progress(self, chat_turn: int | None = None, model: str | None = None) -> None:
        super()._log_chat_progress(chat_turn=chat_turn, model=self._display_model(model))

    def _log_chat_finished(self, model: str | None = None) -> None:
        super()._log_chat_finished(model=self._display_model(model))

    def _update_streaming_progress(self, content: str, model: str, estimated_tokens: int) -> int:
        display_model = self._display_model(model) or model
        return super()._update_streaming_progress(content, display_model, estimated_tokens)

    def _base_url(self) -> str | None:
        settings = self._get_provider_config()
        if settings and getattr(settings, "base_url", None):
            return settings.base_url
        return CODEX_BASE_URL

    def _responses_client(self) -> AsyncOpenAI:
        try:
            token = self._api_key()
            account_id = parse_chatgpt_account_id(token)
            if not account_id:
                raise ProviderKeyError(
                    "Codex OAuth token invalid",
                    "The Codex access token did not contain a chatgpt_account_id. "
                    "Run `fast-agent auth codex-login` to refresh your token.",
                )
            default_headers = dict(self._default_headers() or {})
            default_headers["chatgpt-account-id"] = account_id
            default_headers.setdefault("originator", "fast-agent")
            try:
                app_version = version("fast-agent-mcp")
            except Exception:
                app_version = "unknown"
            default_headers.setdefault("User-Agent", f"fast-agent/{app_version}")
            return AsyncOpenAI(
                api_key=token,
                base_url=self._base_url(),
                http_client=DefaultAioHttpClient(),
                default_headers=default_headers,
            )
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Codex OAuth token",
                "The configured Codex OAuth token was rejected. "
                "Run `fast-agent auth codex-login` to reauthenticate.",
            ) from e

    def _supports_websocket_transport(self) -> bool:
        return True

    def _new_ws_request_planner(self) -> ResponsesWsRequestPlanner:
        """Use response-id continuation on websocket turns."""
        return StatefulContinuationResponsesWsPlanner()

    def _build_websocket_headers(self) -> dict[str, str]:
        token = self._api_key()
        account_id = parse_chatgpt_account_id(token)
        if not account_id:
            raise ProviderKeyError(
                "Codex OAuth token invalid",
                "The Codex access token did not contain a chatgpt_account_id. "
                "Run `fast-agent auth codex-login` to refresh your token.",
            )
        default_headers = dict(self._default_headers() or {})
        default_headers["chatgpt-account-id"] = account_id
        default_headers.setdefault("originator", "fast-agent")
        try:
            app_version = version("fast-agent-mcp")
        except Exception:
            app_version = "unknown"
        default_headers.setdefault("User-Agent", f"fast-agent/{app_version}")
        return default_headers | super()._build_websocket_headers()

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        args = super()._build_response_args(input_items, request_params, tools)
        if "max_output_tokens" in args:
            args.pop("max_output_tokens", None)
            self.logger.debug(
                "Dropping max_output_tokens for Codex responses; parameter unsupported by API"
            )
        args["tool_choice"] = "auto"
        return args
