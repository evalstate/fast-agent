from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.interfaces import ModelT
from typing import List, Tuple, Type

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
            response_format={
                                "type": "json_object"
                            }
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

        # TODO - convert this to use Tool Calling convention for Anthropic Structured outputs
        multipart_messages[-1].add_text(
            """YOU MUST RESPOND IN THE FOLLOWING FORMAT:
            {schema}
            RESPOND ONLY WITH THE JSON, NO PREAMBLE, CODE FENCES OR 'properties' ARE PERMISSABLE """.format(
                schema=model.model_json_schema()
            )
        )

        result: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )
        return self._structured_from_multipart(result, model)
