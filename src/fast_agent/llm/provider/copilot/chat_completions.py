"""Copilot routing for the existing OpenAI-compatible Chat Completions loop."""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import uuid4

from mcp.types import TextContent

from fast_agent.constants import (
    COPILOT_REASONING_OPAQUE,
    DEFAULT_HIGH_REASONING_STREAMING_TIMEOUT,
    REASONING,
)
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.prompt_cache import copilot_gemini_observed_cache_ttl
from fast_agent.llm.provider.copilot import broker
from fast_agent.llm.provider.copilot.models import CopilotModelSpec, get_copilot_model
from fast_agent.llm.provider.copilot.policy import apply_policy, reject_files, reject_overrides
from fast_agent.llm.provider.openai.llm_openai import EmptyStreamError, OpenAILLM
from fast_agent.llm.provider.request_images import ChatRequestImageFitter, ImageBudget
from fast_agent.llm.provider.streaming_timeouts import StreamIdleTimeoutError
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from mcp.types import ContentBlock, Tool
    from openai import AsyncOpenAI
    from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )

    from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
    from fast_agent.llm.provider.openai.llm_openai import (
        _OpenAICompletionRequest,
        _OpenAICompletionResponse,
    )
    from fast_agent.types import PromptMessageExtended, RequestParams

_GEMINI_SPECIAL_TOKEN_RE: Final = re.compile(
    r"<(?:end_of_image|start_of_image|eos|bos|pad|end_of_turn|start_of_turn|unused\d+)>"
)
_COPILOT_GEMINI_STREAMING_TIMEOUT: Final = DEFAULT_HIGH_REASONING_STREAMING_TIMEOUT
_COPILOT_GEMINI_IMAGE_MAX_DIMENSION: Final = 2048
_COPILOT_GEMINI_IMAGE_REENCODE_MIN_BYTES: Final = 200_000
_COPILOT_GEMINI_EMPTY_RECOVERY_NOTE: Final = (
    "Your previous response contained no usable output. "
    "Continue with the next tool call or final answer."
)
_COPILOT_GEMINI_TIMEOUT_RECOVERY_NOTE: Final = (
    "Your previous attempt timed out before producing any output. "
    "Continue incrementally with a smaller, focused next step."
)


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
        self._copilot_recovery_state: ContextVar[dict[str, str | None] | None] = ContextVar(
            "copilot_chat_recovery_state",
            default=None,
        )
        self._image_fitter = ChatRequestImageFitter()
        super().__init__(provider=Provider.COPILOT, **kwargs)
        if (
            get_copilot_model(self.default_request_params.model or "").wire_api
            != "chat_completions"
        ):
            raise ValueError("Copilot Chat Completions requires a Chat Completions model.")

    def clear(self, *, clear_prompts: bool = False) -> None:
        super().clear(clear_prompts=clear_prompts)
        self._image_fitter.reset_watermark()

    def _openai_reasoning_settings(self) -> None:
        return None

    def _resolve_reasoning_effort(self) -> str | None:
        # The catalog lists accepted efforts, but does not advertise a default.
        setting = self.reasoning_effort
        return str(setting.value) if setting is not None and setting.kind == "effort" else None

    def get_request_params(self, request_params: RequestParams | None = None) -> RequestParams:
        resolved = super().get_request_params(request_params)
        model_name = resolved.model or self.default_request_params.model or ""
        if (
            model_name.startswith("gemini-")
            and "streaming_timeout" not in resolved.model_fields_set
            and (
                resolved.streaming_timeout is not None
                and resolved.streaming_timeout < _COPILOT_GEMINI_STREAMING_TIMEOUT
            )
        ):
            # Copilot Gemini buffers reasoning server-side before emitting the first SSE event.
            updated = resolved.model_copy(
                update={"streaming_timeout": _COPILOT_GEMINI_STREAMING_TIMEOUT}
            )
            updated.model_fields_set.discard("streaming_timeout")
            return updated
        return resolved

    def _get_provider_config(self) -> None:
        return None

    def validate_provider_credentials(self) -> None:
        # Credentials are resolved asynchronously by the endpoint broker.
        return None

    def supports_files_api(self) -> bool:
        return False

    @property
    def cache_retention_seconds(self) -> int | None:
        model = self.default_request_params.model
        return copilot_gemini_observed_cache_ttl(model) if model else None

    def _fallback_to_non_streaming_on_empty_stream(self, model: str) -> bool:
        # Copilot Gemini serves empty completions identically on non-streaming after a long
        # reasoning stall; surface EmptyStreamError directly to the bounded retry path.
        return not model.startswith("gemini-")

    def _sanitize_completion_text(self, model: str, text: str) -> str:
        marker = "<end_of_image>"
        if not model.startswith("gemini-") or marker not in text:
            return text
        stripped = text.rstrip()
        if not stripped.endswith(marker):
            return text
        while stripped.endswith(marker):
            stripped = stripped[: -len(marker)].rstrip()
        return stripped if stripped else text

    def _unusable_completion_text(self, model: str, text: str) -> str | None:
        # Gemini's Copilot gateway sometimes emits only raw special tokens (most commonly
        # repeated <end_of_image>) with finish_reason="stop" and no tool calls.
        if not model.startswith("gemini-"):
            return None
        marker = "<end_of_image>"
        if marker in text and not text.replace(marker, "").strip():
            return "only <end_of_image> markers"
        if (
            _GEMINI_SPECIAL_TOKEN_RE.search(text)
            and not _GEMINI_SPECIAL_TOKEN_RE.sub("", text).strip()
        ):
            return "only Gemini special tokens"
        return None

    def _extra_openai_response_channels(
        self, message: ChatCompletionMessage
    ) -> dict[str, list[ContentBlock]]:
        extra = message.model_extra or {}
        opaque = extra.get("reasoning_opaque")
        if isinstance(opaque, str) and opaque:
            return {COPILOT_REASONING_OPAQUE: [TextContent(type="text", text=opaque)]}
        return {}

    def _apply_reasoning_replay(
        self,
        openai_msgs: list[ChatCompletionMessageParam],
        msg: PromptMessageExtended,
        reasoning_mode: str | None,
    ) -> None:
        super()._apply_reasoning_replay(openai_msgs, msg, reasoning_mode)
        if msg.role != "assistant" or not msg.channels or not openai_msgs:
            return
        opaque_blocks = msg.channels.get(COPILOT_REASONING_OPAQUE)
        if not opaque_blocks:
            return
        opaque = "".join(text for block in opaque_blocks if (text := get_text(block)))
        if not opaque:
            return
        target = cast("dict[str, Any]", openai_msgs[0])
        target["reasoning_opaque"] = opaque
        reasoning_blocks = msg.channels.get(REASONING)
        if reasoning_blocks:
            reasoning_text = "".join(
                text for block in reasoning_blocks if (text := get_text(block))
            )
            if reasoning_text:
                target["reasoning_text"] = reasoning_text

    async def _openai_completion(
        self,
        message: list[ChatCompletionMessageParam] | None,
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        state = self._copilot_recovery_state.get()
        if state is not None:
            return await super()._openai_completion(message, request_params, tools)
        token = self._copilot_recovery_state.set({"reason": None})
        try:
            return await super()._openai_completion(message, request_params, tools)
        finally:
            self._copilot_recovery_state.reset(token)

    async def _finalize_openai_completion(
        self,
        request: _OpenAICompletionRequest,
        completion: _OpenAICompletionResponse,
    ) -> PromptMessageExtended:
        result = await super()._finalize_openai_completion(request, completion)
        state = self._copilot_recovery_state.get()
        if state is not None and request.model_name.startswith("gemini-"):
            from fast_agent.llm.provider.error_utils import is_empty_completion

            if is_empty_completion(result):
                state["reason"] = "empty_completion"
        return result

    @staticmethod
    def _append_recovery_nudge(
        messages: list[ChatCompletionMessageParam],
        note: str,
    ) -> list[ChatCompletionMessageParam]:
        last = messages[-1]
        if last.get("role") not in ("tool", "assistant"):
            return messages
        return [
            *messages,
            cast("ChatCompletionMessageParam", {"role": "user", "content": note}),
        ]

    @staticmethod
    def _image_budget(spec: CopilotModelSpec) -> ImageBudget | None:
        if spec.max_prompt_images is None and spec.max_prompt_image_bytes is None:
            return None
        return ImageBudget(
            max_dimension=_COPILOT_GEMINI_IMAGE_MAX_DIMENSION,
            reencode_min_bytes=_COPILOT_GEMINI_IMAGE_REENCODE_MIN_BYTES,
            max_request_bytes=None,
            max_images=spec.max_prompt_images,
            max_image_bytes=spec.max_prompt_image_bytes,
        )

    def _fit_copilot_images(
        self,
        request: _OpenAICompletionRequest,
        spec: CopilotModelSpec,
    ) -> _OpenAICompletionRequest:
        budget = self._image_budget(spec)
        if budget is None:
            return request
        arguments, report = self._image_fitter.fit(request.arguments, budget)
        if report.changed:
            self.logger.info(
                "Fitted inline images to Copilot model limits",
                data={
                    "model": request.model_name,
                    "images": report.images,
                    "resized": report.resized,
                    "reencoded": report.reencoded,
                    "omitted": report.omitted,
                    "max_images": budget.max_images,
                    "max_image_bytes": budget.max_image_bytes,
                    "request_bytes_before": report.request_bytes_before,
                    "request_bytes_after": report.request_bytes_after,
                    "dimensions": report.dimensions,
                },
            )
            return replace(request, arguments=arguments)
        return request

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

    def _apply_gemini_retry_recovery(
        self, request: _OpenAICompletionRequest
    ) -> tuple[_OpenAICompletionRequest, bool]:
        state = self._copilot_recovery_state.get()
        if state is None or not request.model_name.startswith("gemini-"):
            return request, False
        reason = state.get("reason")
        if reason is None:
            return request, False
        messages = request.arguments.get("messages")
        if not isinstance(messages, list) or not messages:
            return request, False
        last_role = messages[-1].get("role") if isinstance(messages[-1], dict) else None
        agent_initiated = last_role in ("tool", "assistant")
        note = (
            _COPILOT_GEMINI_TIMEOUT_RECOVERY_NOTE
            if reason == "start_timeout"
            else _COPILOT_GEMINI_EMPTY_RECOVERY_NOTE
        )
        nudged_messages = self._append_recovery_nudge(messages, note)
        arguments = {**request.arguments, "messages": nudged_messages}
        if reason == "start_timeout" and arguments.get("reasoning_effort") in (
            "medium",
            "high",
            "xhigh",
        ):
            arguments["reasoning_effort"] = "low"
        return replace(
            request, arguments=arguments
        ), agent_initiated and nudged_messages is not messages

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        state = self._copilot_recovery_state.get()
        if state is not None:
            return await super()._apply_prompt_provider_specific(
                multipart_messages, request_params, tools, is_template
            )
        token = self._copilot_recovery_state.set({"reason": None})
        try:
            return await super()._apply_prompt_provider_specific(
                multipart_messages, request_params, tools, is_template
            )
        finally:
            self._copilot_recovery_state.reset(token)

    async def _execute_with_retry(
        self,
        func: Any,
        *args: Any,
        on_final_error: Any = None,
        **kwargs: Any,
    ) -> Any:
        token = self._copilot_recovery_state.set({"reason": None})
        try:
            return await super()._execute_with_retry(
                func, *args, on_final_error=on_final_error, **kwargs
            )
        finally:
            self._copilot_recovery_state.reset(token)

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
        spec = get_copilot_model(endpoint.model_id)
        fitted_request = self._fit_copilot_images(request, spec)
        recovered_request, preserve_agent_initiator = self._apply_gemini_retry_recovery(
            fitted_request
        )
        recovered_request.arguments = apply_policy(
            recovered_request.arguments,
            spec,
            endpoint.headers,
            agent_initiated=preserve_agent_initiator or self.verb == ProgressAction.WARMING_CACHE,
        )
        state = self._copilot_recovery_state.get()
        try:
            return await super()._run_openai_completion_request(recovered_request)
        except EmptyStreamError:
            if state is not None:
                state["reason"] = "empty_stream"
            raise
        except TimeoutError as error:
            if state is not None and not isinstance(error, StreamIdleTimeoutError):
                state["reason"] = "start_timeout"
            raise

    def _openai_client(self) -> AsyncOpenAI:
        from fast_agent.llm.provider.copilot.openai_client import create_openai_client

        return create_openai_client(self._copilot_endpoint.get())

    async def _normalize_chat_completion_files(
        self,
        client: AsyncOpenAI,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        # Never use the direct provider Files API, including for URL attachments.
        reject_files(messages)
        return messages
