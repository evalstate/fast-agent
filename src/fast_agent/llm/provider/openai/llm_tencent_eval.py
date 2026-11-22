import os
from typing import Any

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from fast_agent.config import TENCENT_EVAL_DEFAULT_MODEL as DEFAULT_TENCENT_EVAL_MODEL
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

TENCENT_EVAL_BASE_URL = "https://eval.hunyuan.tencent.com/openapi/v2/data"


class TencentEvalLLM(OpenAICompatibleLLM):
    """Native Tencent Eval Chat Completions provider."""

    config_section: str | None = "tencent_eval"

    def __init__(self, **kwargs: Any) -> None:
        explicit_reasoning_effort = "reasoning_effort" in kwargs
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.TENCENT_EVAL, **kwargs)
        if not explicit_reasoning_effort:
            self.set_reasoning_effort(None)

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        default_params = self._initialize_default_params_with_model_fallback(
            kwargs, DEFAULT_TENCENT_EVAL_MODEL
        )
        explicit_max_tokens = kwargs.get("maxTokens", kwargs.get("max_tokens"))
        default_params.max_tokens = explicit_max_tokens if explicit_max_tokens is not None else None
        return default_params

    def _provider_base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.tencent_eval:
            base_url = self.context.config.tencent_eval.base_url
        if base_url:
            return base_url
        return os.getenv("TENCENT_EVAL_BASE_URL") or TENCENT_EVAL_BASE_URL

    def _preserves_raw_reasoning_history(self, model: str) -> bool:
        return model.casefold() == "hy4-flash-dev"

    def _resolve_reasoning_effort(self) -> str:
        setting = self.reasoning_effort
        if setting is None:
            return "high"
        if setting.kind == "toggle" and not setting.value:
            return "no_think"
        if setting.kind == "effort" and setting.value == "none":
            return "no_think"
        return "high"

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
        return arguments
