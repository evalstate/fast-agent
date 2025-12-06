from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "moonshotai/kimi-k2-instruct"

### There is some big refactorings to be had quite easily here now:
### - combining the structured output type handling
### - deduplicating between this and the deepseek llm


class GroqLLM(OpenAICompatibleLLM):
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

    def _supports_structured_prompt(self) -> bool:
        llm_model = (
            self.default_request_params.model if self.default_request_params else DEFAULT_GROQ_MODEL
        )
        json_mode: str | None = ModelDatabase.get_json_mode(llm_model)
        return json_mode == "object"

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.groq:
            base_url = self.context.config.groq.base_url

        return base_url if base_url else GROQ_BASE_URL
