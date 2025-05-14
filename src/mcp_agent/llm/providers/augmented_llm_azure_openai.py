from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from openai import AzureOpenAI, AuthenticationError
from mcp_agent.core.exceptions import ProviderKeyError

DEFAULT_AZURE_OPENAI_MODEL = "azure_openai"
DEFAULT_AZURE_API_VERSION = "2025-01-01"

class AzureOpenAIAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.AZURE_OPENAI, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Azure OpenAI default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_AZURE_OPENAI_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=20,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = None
        if self.context and self.context.config and hasattr(self.context.config, "azure_openai"):
            if self.context.config.azure_openai and hasattr(self.context.config.azure_openai, "base_url"):
                base_url = self.context.config.azure_openai.base_url
        return base_url or ""

    def _openai_client(self) -> AzureOpenAI:
        """Create an Azure OpenAI client with the appropriate configuration"""
        try:
            api_key = self._api_key()
            api_version = DEFAULT_AZURE_API_VERSION
            azure_endpoint = self._base_url()
            
            # Safely get api_version if available
            if (self.context and self.context.config and 
                hasattr(self.context.config, "azure_openai") and 
                self.context.config.azure_openai):
                
                if hasattr(self.context.config.azure_openai, "api_version") and self.context.config.azure_openai.api_version:
                    api_version = self.context.config.azure_openai.api_version
            
            return AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e