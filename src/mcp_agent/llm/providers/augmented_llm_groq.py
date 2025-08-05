from typing import List, Tuple, Type, cast

from pydantic_core import from_json

from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.model_database import ModelDatabase
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import get_text, split_thinking_content
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "moonshotai/kimi-k2-instruct"

### There is some big refactorings to be had quite easily here now:
### - combining the structured output type handling
### - deduplicating between this and the deepseek llm


class GroqAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GROQ, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Groq default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with Groq-specific settings
        chosen_model = kwargs.get("model", DEFAULT_GROQ_MODEL)
        base_params.model = chosen_model
        base_params.parallel_tool_calls = False

        return base_params

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:  # noqa: F821
        request_params = self.get_request_params(request_params)

        assert self.default_request_params
        llm_model = self.default_request_params.model or DEFAULT_GROQ_MODEL
        json_mode: str | None = ModelDatabase.get_json_mode(llm_model)
        if "json_object" == json_mode:
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
        reasoning_mode: str | None = ModelDatabase.get_reasoning(llm_model)
        try:
            text = get_text(result.content[-1]) or ""
            if "tags" == reasoning_mode:
                _, text = split_thinking_content(text)
            json_data = from_json(text, allow_partial=True)
            validated_model = model.model_validate(json_data)
            return cast("ModelT", validated_model), result
        except ValueError as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to parse structured response: {str(e)}")
            return None, result

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.groq:
            base_url = self.context.config.groq.base_url

        return base_url if base_url else GROQ_BASE_URL
