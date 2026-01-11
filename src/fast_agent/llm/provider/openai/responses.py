import asyncio
from typing import Any

from mcp import Tool
from mcp.types import ContentBlock, TextContent
from openai import APIError, AsyncOpenAI, AuthenticationError, DefaultAioHttpClient

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL, OPENAI_REASONING_ENCRYPTED, REASONING
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.responses_content import ResponsesContentMixin
from fast_agent.llm.provider.openai.responses_files import ResponsesFileMixin
from fast_agent.llm.provider.openai.responses_output import ResponsesOutputMixin
from fast_agent.llm.provider.openai.responses_streaming import (
    ResponsesStreamingMixin,
    _save_stream_request,
    _stream_capture_filename,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

_logger = get_logger(__name__)

DEFAULT_RESPONSES_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "medium"
MIN_RESPONSES_MAX_TOKENS = 16


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
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)
        self._tool_call_id_map: dict[str, str] = {}
        self._file_id_cache: dict[str, str] = {}

        self._reasoning_effort = kwargs.get("reasoning_effort", None)
        if self.context and self.context.config and self.context.config.openai:
            if self._reasoning_effort is None and hasattr(
                self.context.config.openai, "reasoning_effort"
            ):
                self._reasoning_effort = self.context.config.openai.reasoning_effort

        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning_mode = ModelDatabase.get_reasoning(chosen_model) if chosen_model else None
        self._reasoning = self._reasoning_mode == "openai"
        if self._reasoning_mode:
            self.logger.info(
                f"Using Responses model '{chosen_model}' (mode='{self._reasoning_mode}') with "
                f"'{self._reasoning_effort}' reasoning effort"
            )

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        base_params = super()._initialize_default_params(kwargs)
        chosen_model = kwargs.get("model", DEFAULT_RESPONSES_MODEL)
        base_params.model = chosen_model
        return base_params

    def _get_provider_config(self):
        if not self.context or not self.context.config:
            return None
        section_name = self.config_section or getattr(self.provider, "value", None)
        if section_name and hasattr(self.context.config, section_name):
            return getattr(self.context.config, section_name)
        return getattr(self.context.config, "openai", None)

    def _openai_settings(self):
        return self._get_provider_config()

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

    def _adjust_schema(self, input_schema: dict[str, Any]) -> dict[str, Any]:
        if "properties" in input_schema:
            return input_schema
        result = input_schema.copy()
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
        model = self.default_request_params.model or DEFAULT_RESPONSES_MODEL
        base_args: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
            "include": ["reasoning.encrypted_content"],
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
                    "parameters": self._adjust_schema(tool.inputSchema),
                }
                for tool in tools
            ]

        if self._reasoning:
            base_args["reasoning"] = {
                "summary": "auto",
                "effort": self._reasoning_effort or DEFAULT_REASONING_EFFORT,
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

        return self.prepare_provider_arguments(
            base_args, request_params, self.RESPONSES_EXCLUDE_FIELDS
        )

    async def _responses_completion(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        response_content_blocks: list[ContentBlock] = []
        model_name = self.default_request_params.model or DEFAULT_RESPONSES_MODEL

        self._log_chat_progress(self.chat_turn(), model=model_name)

        try:
            async with self._responses_client() as client:
                input_items = await self._normalize_input_files(client, input_items)
                arguments = self._build_response_args(input_items, request_params, tools)
                self.logger.debug("Responses request", data=arguments)
                capture_filename = _stream_capture_filename(self.chat_turn())
                _save_stream_request(capture_filename, arguments)
                async with client.responses.stream(**arguments) as stream:
                    response, streamed_summary = await self._process_stream(
                        stream, model_name, capture_filename
                    )
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e
        except APIError as error:
            self.logger.error("Streaming APIError during Responses completion", exc_info=error)
            raise
        except asyncio.CancelledError:
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )

        if response is None:
            raise RuntimeError("Responses stream did not return a final response")

        self._log_chat_finished(model=model_name)

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

        self.history.set(input_items)

        return PromptMessageExtended(
            role="assistant",
            content=response_content_blocks,
            tool_calls=tool_calls,
            channels=channels,
            stop_reason=stop_reason,
        )

    def _stream_failure_response(self, error: APIError, model_name: str) -> PromptMessageExtended:
        """Convert streaming API errors into a graceful assistant reply."""
        provider_label = (
            self.provider.value if isinstance(self.provider, Provider) else str(self.provider)
        )
        detail = getattr(error, "message", None) or str(error)
        detail = detail.strip() if isinstance(detail, str) else ""

        parts: list[str] = [f"{provider_label} request failed"]
        if model_name:
            parts.append(f"for model '{model_name}'")
        code = getattr(error, "code", None)
        if code:
            parts.append(f"(code: {code})")
        status = getattr(error, "status_code", None)
        if status:
            parts.append(f"(status={status})")

        message = " ".join(parts)
        if detail:
            message = f"{message}: {detail}"

        user_summary = " ".join(message.split()) if message else ""
        if user_summary and len(user_summary) > 280:
            user_summary = user_summary[:277].rstrip() + "..."

        if user_summary:
            assistant_text = f"I hit an internal error while calling the model: {user_summary}"
            if not assistant_text.endswith((".", "!", "?")):
                assistant_text += "."
            assistant_text += " See fast-agent-error for additional details."
        else:
            assistant_text = (
                "I hit an internal error while calling the model; see fast-agent-error for details."
            )

        assistant_block = text_content(assistant_text)
        error_block = text_content(message)

        return PromptMessageExtended(
            role="assistant",
            content=[assistant_block],
            channels={FAST_AGENT_ERROR_CHANNEL: [error_block]},
            stop_reason=LlmStopReason.ERROR,
        )

    def _handle_retry_failure(self, error: Exception) -> PromptMessageExtended | None:
        """Return the legacy error-channel response when retries are exhausted."""
        if isinstance(error, APIError):
            model_name = self.default_request_params.model or DEFAULT_RESPONSES_MODEL
            return self._stream_failure_response(error, model_name)
        return None
