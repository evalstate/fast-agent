from typing import Any

from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.types import RequestParams

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "qwen/qwen3-32b"

# Groq accepts only `reasoning_effort="default"` (thinking on) or `"none"` (off);
# standard low/medium/high levels are rejected by the API.
_GROQ_REASONING_DEFAULT = "default"
_GROQ_REASONING_NONE = "none"


def _normalize_groq_reasoning_setting(
    setting: ReasoningEffortSetting,
) -> ReasoningEffortSetting:
    """Collapse effort levels to Groq's binary on/off reasoning toggle."""
    if setting.kind == "effort":
        return ReasoningEffortSetting(kind="toggle", value=setting.value != "none")
    return setting


### There is some big refactorings to be had quite easily here now:
### - combining the structured output type handling
### - deduplicating between this and the deepseek llm


class GroqLLM(OpenAICompatibleLLM):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.GROQ, **kwargs)

    def set_reasoning_effort(self, setting: ReasoningEffortSetting | None) -> None:
        if setting is not None:
            setting = _normalize_groq_reasoning_setting(setting)
        super().set_reasoning_effort(setting)

    def _resolve_reasoning_effort(self) -> str | None:
        setting = self.reasoning_effort
        if setting is None:
            return _GROQ_REASONING_DEFAULT
        if setting.kind == "toggle":
            return _GROQ_REASONING_DEFAULT if setting.value else _GROQ_REASONING_NONE
        return _GROQ_REASONING_DEFAULT

    def _prepare_api_request(
        self,
        messages,
        tools: list | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
        # Only shape the reasoning wire contract for separate-field ("stream")
        # reasoning models; tag-based Groq reasoners keep their inline behavior.
        if self._reasoning_mode != "stream":
            return arguments
        effort = self._resolve_reasoning_effort()
        arguments["reasoning_effort"] = effort
        # `reasoning_format` is a Groq extension the OpenAI SDK does not model as a
        # typed parameter, so it rides in extra_body. `parsed` surfaces reasoning
        # in a separate `reasoning` delta that fast-agent streams; omit it when
        # thinking is off so nothing extra is sent.
        if effort == _GROQ_REASONING_DEFAULT:
            extra_body_raw = arguments.get("extra_body", {})
            extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}
            extra_body["reasoning_format"] = "parsed"
            arguments["extra_body"] = extra_body
        return arguments

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Groq default parameters"""
        base_params = self._initialize_default_params_with_model_fallback(
            kwargs, DEFAULT_GROQ_MODEL
        )
        base_params.parallel_tool_calls = False

        return base_params

    def _supports_structured_prompt(self) -> bool:
        llm_model = (
            self.default_request_params.model if self.default_request_params else DEFAULT_GROQ_MODEL
        )
        if not llm_model:
            return False
        json_mode = self._get_model_json_mode(llm_model)
        return json_mode == "object"

    def _provider_base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.groq:
            base_url = self.context.config.groq.base_url

        return base_url if base_url else GROQ_BASE_URL
