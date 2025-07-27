from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


class HuggingFaceAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.HUGGINGFACE, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize HuggingFace"""
        chosen_model = kwargs.get("model", DEFAULT_HF_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=20,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.huggingface:
            base_url = self.context.config.huggingface.base_url

        return base_url if base_url else HUGGINGFACE_BASE_URL

    async def _process_stream(self, stream, model: str):
        """Process stream and fix timestamp issues from HuggingFace router."""

        async def fixed_stream():
            async for chunk in stream:
                # Fix the created timestamp if it's a float
                if hasattr(chunk, "created") and isinstance(chunk.created, float):
                    # Convert float timestamp to int
                    chunk.created = int(chunk.created)
                yield chunk

        # Use the parent's stream processing with our fixed stream
        return await super()._process_stream(fixed_stream(), model)
