from typing import Any, Type

from mcp import Tool

from fast_agent.interfaces import ModelT
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.mcp.helpers.content_helpers import split_thinking_content
from fast_agent.types import PromptMessageExtended, RequestParams


class OpenAICompatibleLLM(OpenAILLM):
    """Shared helpers for OpenAI-compatible providers that need structured output prompting."""

    STRUCTURED_PROMPT_TEMPLATE = """YOU MUST RESPOND WITH A JSON OBJECT IN EXACTLY THIS FORMAT:
{format_description}

IMPORTANT RULES:
- Respond ONLY with the JSON object, no other text
- Do NOT include "properties" or "schema" wrappers
- Do NOT use code fences or markdown
- The response must be valid JSON that matches the format above
- All required fields must be included"""

    def _prepare_structured_request(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> tuple[list[PromptMessageExtended], RequestParams]:
        if not request_params.structured_schema:
            return messages, request_params
        if self._should_defer_structured_schema_for_tools(messages, request_params, tools):
            return messages, request_params.model_copy(update={"structured_schema": None})

        prepared_params = request_params
        json_mode = self._structured_json_mode(request_params)
        if json_mode == "schema" and not request_params.response_format:
            return messages, request_params.model_copy(
                update={
                    "response_format": self.schema_to_response_format(
                        request_params.structured_schema
                    )
                }
            )

        if not self._supports_structured_prompt():
            return messages, request_params

        if json_mode == "object" and not request_params.response_format:
            prepared_params = request_params.model_copy(
                update={"response_format": {"type": "json_object"}}
            )

        if not messages or messages[-1].role != "user":
            return messages, prepared_params

        instructions = self._build_structured_prompt_instruction_from_schema(
            request_params.structured_schema
        )
        if not instructions:
            return messages, prepared_params

        prepared_messages = list(messages)
        last_message = prepared_messages[-1].model_copy(deep=True)
        last_message.add_text(instructions)
        prepared_messages[-1] = last_message
        return prepared_messages, prepared_params

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: list[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        if not self._supports_structured_prompt():
            return await super()._apply_prompt_provider_specific_structured(
                multipart_messages, model, request_params
            )

        request_params = self.get_request_params(request_params)

        if multipart_messages and multipart_messages[-1].role == "assistant":
            return await super()._apply_prompt_provider_specific_structured(
                multipart_messages,
                model,
                request_params,
            )

        json_mode = self._structured_json_mode(request_params)
        if json_mode == "schema" and not request_params.response_format:
            schema = self.model_to_response_format(model)
            if schema:
                request_params.response_format = schema
            return await super()._apply_prompt_provider_specific_structured(
                multipart_messages, model, request_params
            )

        if json_mode == "object" and not request_params.response_format:
            request_params.response_format = {"type": "json_object"}

        instructions = self._build_structured_prompt_instruction(model)
        if instructions:
            multipart_messages[-1].add_text(instructions)

        if json_mode is None:
            result = await self._apply_prompt_provider_specific(multipart_messages, request_params)
            return self._structured_from_multipart(result, model)

        return await super()._apply_prompt_provider_specific_structured(
            multipart_messages, model, request_params
        )

    async def _apply_prompt_provider_specific_structured_schema(
        self,
        multipart_messages: list[PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended | tuple[Any | None, PromptMessageExtended]:
        if not self._supports_structured_prompt():
            return await super()._apply_prompt_provider_specific_structured_schema(
                multipart_messages,
                schema,
                request_params,
            )

        request_params = self.get_request_params(request_params)

        if multipart_messages and multipart_messages[-1].role == "assistant":
            return await FastAgentLLM._apply_prompt_provider_specific_structured_schema(
                self,
                multipart_messages,
                schema,
                request_params,
            )

        json_mode = self._structured_json_mode(request_params)
        if json_mode == "schema" and not request_params.response_format:
            request_params.response_format = self.schema_to_response_format(schema)
            return await FastAgentLLM._apply_prompt_provider_specific_structured_schema(
                self,
                multipart_messages,
                schema,
                request_params,
            )

        if json_mode == "object" and not request_params.response_format:
            request_params.response_format = {"type": "json_object"}

        instructions = self._build_structured_prompt_instruction_from_schema(schema)
        if instructions:
            multipart_messages[-1].add_text(instructions)

        return await FastAgentLLM._apply_prompt_provider_specific_structured_schema(
            self,
            multipart_messages,
            schema,
            request_params,
        )

    def _supports_structured_prompt(self) -> bool:
        """Allow subclasses to opt-out of shared structured prompting."""
        return True

    def _structured_prompt_format(self) -> str | None:
        """Return the response_format type this provider expects."""
        return "json_object"

    def _structured_json_mode(self, request_params: RequestParams | None = None) -> str | None:
        model_name = (
            request_params.model
            if request_params and request_params.model
            else self.default_request_params.model
            if self.default_request_params
            else self._model_name
        )
        if not model_name:
            return self._structured_prompt_format()
        try:
            params = self._get_model_params(model_name)
        except Exception:
            params = ModelDatabase.get_model_params(model_name)
        if params is not None:
            return params.json_mode
        return self._structured_prompt_format()

    def _build_structured_prompt_instruction(self, model: Type[ModelT]) -> str | None:
        return self._build_structured_prompt_instruction_from_schema(model.model_json_schema())

    def _build_structured_prompt_instruction_from_schema(
        self,
        schema: dict[str, Any],
    ) -> str | None:
        template = self._structured_prompt_template()
        if not template:
            return None

        format_description = self._schema_to_json_object(schema, schema.get("$defs"))
        return template.format(format_description=format_description)

    def _structured_prompt_template(self) -> str | None:
        return self.STRUCTURED_PROMPT_TEMPLATE

    def _prepare_structured_text(self, text: str) -> str:
        reasoning_mode = self._structured_reasoning_mode()
        if reasoning_mode == "tags":
            thinking, trimmed = split_thinking_content(text)
            if thinking is None:
                closing_tag = "</think>"
                closing_index = text.find(closing_tag)
                if closing_index != -1:
                    trimmed = text[closing_index + len(closing_tag) :].lstrip()
                else:
                    trimmed = text
            return trimmed

        if "</think>" in text:
            logger = getattr(self, "logger", None)
            if logger:
                logger.warning(
                    "Model emitted reasoning tags without 'tags' reasoning mode",
                    data={
                        "model": getattr(self.default_request_params, "model", None),
                        "text_preview": text[:200],
                    },
                )
        return text

    def _structured_reasoning_mode(self) -> str | None:
        model_name = self.default_request_params.model if self.default_request_params else None
        return self._get_model_reasoning(model_name)

    def _schema_to_json_object(
        self, schema: dict, defs: dict | None = None, visited: set | None = None
    ) -> str:
        """Render a compact, human-friendly shape of the JSON schema."""
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
