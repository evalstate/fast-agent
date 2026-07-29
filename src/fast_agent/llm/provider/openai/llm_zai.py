from typing import Any

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
DEFAULT_ZAI_MODEL = "glm-5.2"


class ZaiLLM(OpenAICompatibleLLM):
    """Native Z.ai Chat Completions provider."""

    def __init__(self, **kwargs: Any) -> None:
        explicit_reasoning_effort = "reasoning_effort" in kwargs
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.ZAI, **kwargs)
        if not explicit_reasoning_effort:
            # Z.ai uses the OpenAI-compatible transport, but its reasoning default
            # comes from model metadata rather than the OpenAI provider config.
            self.set_reasoning_effort(None)

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_ZAI_MODEL)

    def _provider_base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.zai:
            base_url = self.context.config.zai.base_url
        return base_url if base_url else ZAI_BASE_URL

    def _resolve_reasoning_effort(self) -> str | None:
        setting = self.reasoning_effort
        if setting is None:
            return "max"
        if setting.kind == "toggle":
            return "max" if setting.value else None
        if setting.kind == "budget":
            self.logger.warning("Ignoring budget reasoning setting for Z.ai models.")
            return "max"
        if setting.kind == "effort" and setting.value == "none":
            return None
        return setting.value if isinstance(setting.value, str) else "max"

    def _prepare_api_request(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}
        if tools:
            extra_body["tool_stream"] = True
        if self._reasoning_mode != "reasoning_content":
            if extra_body:
                arguments["extra_body"] = extra_body
            return arguments

        effort = self._resolve_reasoning_effort()
        extra_body["thinking"] = {"type": "enabled" if effort else "disabled"}
        arguments["extra_body"] = extra_body
        if effort:
            arguments["reasoning_effort"] = effort
        else:
            arguments.pop("reasoning_effort", None)
        return arguments

    @staticmethod
    def _prepare_non_streaming_request(arguments: dict[str, Any]) -> dict[str, Any]:
        non_stream_args = OpenAICompatibleLLM._prepare_non_streaming_request(arguments)
        extra_body_raw = non_stream_args.get("extra_body")
        if not isinstance(extra_body_raw, dict) or "tool_stream" not in extra_body_raw:
            return non_stream_args

        extra_body = dict(extra_body_raw)
        extra_body.pop("tool_stream")
        if extra_body:
            non_stream_args["extra_body"] = extra_body
        else:
            non_stream_args.pop("extra_body")
        return non_stream_args
