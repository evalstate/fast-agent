from typing import Any, Type

from fast_agent.interfaces import ModelT
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.types import RequestParams

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class DeepSeekLLM(OpenAICompatibleLLM):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.DEEPSEEK, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Deepseek-specific default parameters"""
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_DEEPSEEK_MODEL)

    def _provider_base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.deepseek:
            base_url = self.context.config.deepseek.base_url

        return base_url if base_url else DEEPSEEK_BASE_URL

    def set_reasoning_effort(self, setting: ReasoningEffortSetting | None) -> None:
        if setting is not None and setting.kind == "effort":
            if setting.value in {"none"}:
                setting = ReasoningEffortSetting(kind="toggle", value=False)
            elif setting.value in {"minimal", "low", "medium", "high"}:
                setting = ReasoningEffortSetting(kind="effort", value="high")
            elif setting.value in {"xhigh", "max"}:
                setting = ReasoningEffortSetting(kind="effort", value="max")
        super().set_reasoning_effort(setting)

    def _resolve_reasoning_effort(self) -> str | None:
        setting = self.reasoning_effort
        if setting is None:
            return "high"
        if setting.kind == "toggle":
            return None if setting.value is False else "high"
        if setting.kind == "budget":
            self.logger.warning("Ignoring budget reasoning setting for DeepSeek models.")
            return "high"
        effort = str(setting.value)
        if effort == "none":
            return None
        if effort in {"minimal", "low", "medium", "high"}:
            return "high"
        if effort in {"xhigh", "max"}:
            return "max"
        return "high"

    def _prepare_api_request(
        self,
        messages,
        tools: list | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
        if self._reasoning_mode != "reasoning_content":
            return arguments

        effort = self._resolve_reasoning_effort()
        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}
        extra_body["thinking"] = {"type": "enabled" if effort else "disabled"}
        arguments["extra_body"] = extra_body
        if effort:
            arguments["reasoning_effort"] = effort
        else:
            arguments.pop("reasoning_effort", None)
        return arguments

    def _build_structured_prompt_instruction(self, model: Type[ModelT]) -> str | None:
        full_schema = model.model_json_schema()
        properties = full_schema.get("properties", {})
        required_fields = set(full_schema.get("required", []))

        format_lines = ["{"] 
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            description = field_info.get("description", "")
            line = f'  "{field_name}": "{field_type}"'
            if description:
                line += f"  // {description}"
            if field_name in required_fields:
                line += "  // REQUIRED"
            format_lines.append(line)
        format_lines.append("}")
        format_description = "\n".join(format_lines)

        return f"""YOU MUST RESPOND WITH A JSON OBJECT IN EXACTLY THIS FORMAT:
{format_description}

IMPORTANT RULES:
- Respond ONLY with the JSON object, no other text
- Do NOT include "properties" or "schema" wrappers
- Do NOT use code fences or markdown
- The response must be valid JSON that matches the format above
- All required fields must be included"""
