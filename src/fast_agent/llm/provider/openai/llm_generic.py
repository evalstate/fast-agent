import os
from typing import Any

from fast_agent.constants import DEFAULT_MAX_ITERATIONS
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"


class GenericLLM(OpenAILLM):
    _EXTRA_BODY_SAMPLING_KEYS = ("top_k", "min_p", "repetition_penalty")

    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.GENERIC, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Generic  parameters"""
        chosen_model = self._resolve_default_model_name(
            kwargs.get("model"),
            DEFAULT_OLLAMA_MODEL,
        )

        return RequestParams(
            model=chosen_model,
            system_prompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            use_history=True,
        )

    def _provider_base_url(self) -> str | None:
        base_url: str | None = os.getenv("GENERIC_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        if self.context.config and self.context.config.generic:
            base_url = self.context.config.generic.base_url

        return base_url

    def _prepare_api_request(
        self,
        messages,
        tools: list | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
        extra_body_raw = arguments.get("extra_body")
        extra_body = dict(extra_body_raw) if isinstance(extra_body_raw, dict) else {}
        for key in self._EXTRA_BODY_SAMPLING_KEYS:
            value = arguments.pop(key, None)
            if value is not None:
                extra_body[key] = value
        if extra_body:
            arguments["extra_body"] = extra_body
        return arguments
