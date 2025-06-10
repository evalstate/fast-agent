from copy import copy
from typing import List, Tuple, Type, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
)

from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseekchat"  # current Deepseek only has two type models


class DeepSeekAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.DEEPSEEK, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Deepseek-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_DEEPSEEK_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.deepseek:
            base_url = self.context.config.deepseek.base_url

        return base_url if base_url else DEEPSEEK_BASE_URL

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:  # noqa: F821
        request_params = self.get_request_params(request_params)

        request_params.response_format = {"type": "json_object"}

        # Get the full schema and extract just the properties
        full_schema = model.model_json_schema()
        properties = full_schema.get("properties", {})
        required_fields = full_schema.get("required", [])

        # Create a cleaner format description
        format_description = "{\n"
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            description = field_info.get("description", "")
            format_description += f'  "{field_name}": "{field_type}"'
            if description:
                format_description += f"  // {description}"
            if field_name in required_fields:
                format_description += "  // REQUIRED"
            format_description += "\n"
        format_description += "}"

        multipart_messages[-1].add_text(
            f"""YOU MUST RESPOND WITH A JSON OBJECT IN EXACTLY THIS FORMAT:
            {format_description}

            IMPORTANT RULES:
            - Respond ONLY with the JSON object, no other text
            - Do NOT include "properties" or "schema" wrappers
            - Do NOT use code fences or markdown
            - The response must be valid JSON that matches the format above
            - All required fields must be included"""
        )

        result: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )
        return self._structured_from_multipart(result, model)

    @classmethod
    def convert_message_to_message_param(cls, message: ChatCompletionMessage, **kwargs) -> ChatCompletionAssistantMessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        if hasattr(message, "reasoning_content"):
            message = copy(message)
            del message.reasoning_content
        return cast("ChatCompletionAssistantMessageParam", message)
