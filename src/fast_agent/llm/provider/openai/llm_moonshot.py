import base64
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from fast_agent.llm.provider.openai.llm_openai import EmptyStreamError
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider.openai.multipart_converter_openai import (
    OPENAI_CHAT_VIDEO_MIME_TYPES,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MOONSHOT_MODEL = "kimi-k3"
MOONSHOT_REASONING_EFFORTS = frozenset(("low", "high", "max"))
MOONSHOT_IMAGE_MIME_TYPES = frozenset(
    (
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/heic",
        "image/heif",
    )
)
MOONSHOT_FIXED_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


class MoonshotLLM(OpenAICompatibleLLM):
    """Native Moonshot Chat Completions provider."""

    def __init__(self, **kwargs: Any) -> None:
        explicit_reasoning_effort = "reasoning_effort" in kwargs
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.MOONSHOT, **kwargs)
        if not explicit_reasoning_effort:
            self.set_reasoning_effort(None)

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_MOONSHOT_MODEL)

    def _provider_base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.moonshot:
            base_url = self.context.config.moonshot.base_url
        return base_url if base_url else MOONSHOT_BASE_URL

    def _resolve_reasoning_effort(self) -> str:
        setting = self.reasoning_effort
        if setting is None:
            return "max"
        if setting.kind == "effort" and isinstance(setting.value, str):
            if setting.value in MOONSHOT_REASONING_EFFORTS:
                return setting.value
        self.logger.warning("Kimi K3 always reasons; using the default max reasoning effort.")
        return "max"

    def _prepare_api_request(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
        if self._reasoning_mode != "reasoning_content":
            return arguments

        arguments["reasoning_effort"] = self._resolve_reasoning_effort()
        arguments.pop("parallel_tool_calls", None)
        ignored_sampling_fields = [
            field for field in MOONSHOT_FIXED_SAMPLING_FIELDS if field in arguments
        ]
        if ignored_sampling_fields:
            self.logger.warning(
                "Ignoring fixed or unsupported Kimi K3 sampling parameters.",
                data={"fields": ignored_sampling_fields},
            )
            for field in ignored_sampling_fields:
                arguments.pop(field)
        arguments.pop("max_tokens", None)
        if request_params.maxTokens is not None:
            arguments["max_completion_tokens"] = request_params.maxTokens
        return arguments

    async def _normalize_chat_completion_files(
        self,
        client: AsyncOpenAI,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        normalized = await super()._normalize_chat_completion_files(client, messages)
        return [await self._embed_remote_media(message) for message in normalized]

    async def _embed_remote_media(
        self,
        message: ChatCompletionMessageParam,
    ) -> ChatCompletionMessageParam:
        content = message.get("content")
        if not isinstance(content, list):
            return message

        updated_content: list[Any] = []
        changed = False
        for part in content:
            if not isinstance(part, dict):
                updated_content.append(part)
                continue
            part_type = part.get("type")
            media_key = (
                "image_url"
                if part_type == "image_url"
                else "video_url"
                if part_type == "video_url"
                else None
            )
            if media_key is None:
                updated_content.append(part)
                continue
            media = part.get(media_key)
            if not isinstance(media, dict):
                updated_content.append(part)
                continue
            url = media.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                updated_content.append(part)
                continue

            data, mime_type = await self._download_remote_file(url)
            supported_mime_types = (
                MOONSHOT_IMAGE_MIME_TYPES
                if media_key == "image_url"
                else OPENAI_CHAT_VIDEO_MIME_TYPES
            )
            if data is None or mime_type not in supported_mime_types:
                raise ValueError(f"Moonshot could not embed supported {part_type} media from {url}")
            encoded = base64.b64encode(data).decode("ascii")
            updated_content.append(
                {
                    **part,
                    media_key: {
                        **media,
                        "url": f"data:{mime_type};base64,{encoded}",
                    },
                }
            )
            changed = True

        if not changed:
            return message
        return cast("ChatCompletionMessageParam", {**message, "content": updated_content})

    async def _process_stream_manual(
        self,
        stream: Any,
        model: str,
        capture_filename: Path | None = None,
    ) -> tuple[Any, list[str]]:
        completion, reasoning = await super()._process_stream_manual(
            stream,
            model,
            capture_filename,
        )
        if not completion.choices or completion.choices[0].finish_reason is None:
            raise EmptyStreamError("Moonshot stream ended without a finish reason")
        return completion, reasoning
