from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.llm.augmented_llm import RequestParams


DEFAULT_DEEP_INFRA_MODEL = "deepseek-ai/DeepSeek-V3-0324"

class DeepInfraAugmentedLLM(OpenAIAugmentedLLM):

    def __init__(self, provider: Provider = Provider.DEEPINFRA, *args, **kwargs) -> None:
        super().__init__(provider, *args, **kwargs)


    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenAI-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_DEEP_INFRA_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        return "https://api.deepinfra.com/v1/openai"


    def _api_key(self):
        from mcp_agent.llm.provider_key_manager import ProviderKeyManager

        assert self.provider
        return ProviderKeyManager.get_api_key(self.provider.value, self.context.config)