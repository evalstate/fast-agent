from typing import List, Tuple, Type, cast

from pydantic_core import from_json

from fast_agent.interfaces import ModelT
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.helpers.content_helpers import get_text, split_thinking_content
from fast_agent.types import PromptMessageExtended, RequestParams
from mcp_agent.logging.logger import get_logger

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "moonshotai/kimi-k2-instruct"

### There is some big refactorings to be had quite easily here now:
### - combining the structured output type handling
### - deduplicating between this and the deepseek llm


class GroqLLM(OpenAILLM):
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
        multipart_messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:  # noqa: F821
        request_params = self.get_request_params(request_params)

        assert self.default_request_params
        llm_model = self.default_request_params.model or DEFAULT_GROQ_MODEL
        json_mode: str | None = ModelDatabase.get_json_mode(llm_model)
        if "object" == json_mode:
            request_params.response_format = {"type": "json_object"}

            # Create a cleaner format description from full schema
            full_schema = model.model_json_schema()
            format_description = self._schema_to_json_object(full_schema, full_schema.get("$defs"))

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

        result: PromptMessageExtended = await self._apply_prompt_provider_specific(
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

    def _schema_to_json_object(
        self, schema: dict, defs: dict | None = None, visited: set | None = None
    ) -> str:
        visited = visited or set()

        if id(schema) in visited:
            return '"<recursive>"'
        visited.add(id(schema))

        if "$ref" in schema:
            ref = schema.get("$ref", "")
            if ref.startswith("#/$defs/"):
                target = ref.split("/")[-1]
                if defs and target in defs:
                    return self._schema_to_json_object(defs[target], defs, visited)
            return f'"<ref:{ref}>"'

        schema_type = schema.get("type")
        description = schema.get("description", "")
        required = schema.get("required", [])

        if schema_type == "object":
            props = schema.get("properties", {})
            result = "{\n"
            for prop_name, prop_schema in props.items():
                is_required = prop_name in required
                prop_str = self._schema_to_json_object(prop_schema, defs, visited)
                if is_required:
                    prop_str += " // REQUIRED"
                result += f'  "{prop_name}": {prop_str},\n'
            result += "}"
            return result
        elif schema_type == "array":
            items = schema.get("items", {})
            items_str = self._schema_to_json_object(items, defs, visited)
            return f"[{items_str}]"
        elif schema_type:
            comment = f" // {description}" if description else ""
            return f'"{schema_type}"' + comment

        return '"<unknown>"'
