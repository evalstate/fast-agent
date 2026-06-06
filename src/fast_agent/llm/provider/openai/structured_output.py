from typing import Any, cast

from mcp import Tool

from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider.openai.schema_sanitizer import sanitize_response_format_schema
from fast_agent.llm.request_params import RequestParams
from fast_agent.types import PromptMessageExtended


class OpenAIStructuredOutputMixin:
    """Shared structured-output helpers for OpenAI chat and responses providers."""

    def _prepare_structured_request(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> tuple[list[PromptMessageExtended], RequestParams]:
        if not request_params.structured_schema or request_params.response_format:
            return messages, request_params
        llm = cast("FastAgentLLM[Any, Any]", self)
        if llm._should_defer_structured_schema_for_tools(messages, request_params, tools):
            return messages, request_params.model_copy(update={"structured_schema": None})
        return messages, request_params.model_copy(
            update={
                "response_format": self.schema_to_response_format(request_params.structured_schema)
            }
        )

    async def _apply_prompt_provider_specific_structured_schema(
        self,
        multipart_messages: list[PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended | tuple[Any | None, PromptMessageExtended]:
        llm = cast("FastAgentLLM[Any, Any]", self)
        request_params = llm.get_request_params(request_params)
        if not request_params.response_format:
            request_params.response_format = self.schema_to_response_format(schema)
        return await FastAgentLLM._apply_prompt_provider_specific_structured_schema(
            llm,
            multipart_messages,
            schema,
            request_params,
        )

    def schema_to_response_format(
        self,
        schema: dict[str, Any],
        *,
        name: str = "structured_output",
        strict: bool = True,
    ) -> dict[str, Any]:
        return FastAgentLLM.schema_to_response_format(
            sanitize_response_format_schema(schema) if strict else schema,
            name=name,
            strict=strict,
        )
