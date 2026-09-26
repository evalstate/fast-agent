import os

from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"
# OrcaRouter is an OpenAI-compatible routing gateway; model ids are namespaced
# (e.g. "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-6", "orcarouter/auto").
# The default is pinned to a fixed chat model rather than the "orcarouter/auto"
# router because fast-agent sends tool calls on every request and some upstreams
# in the auto pool do not support tool calling.
DEFAULT_ORCAROUTER_MODEL = "openai/gpt-4o-mini"


def _ensure_orcarouter_namespace(model_name: str) -> str:
    """Prefix a bare model name with the OrcaRouter namespace.

    OrcaRouter routes on the full namespaced model id and rejects bare names
    (e.g. "auto" -> 503). Only bare names are prefixed; names that already
    contain a namespace (e.g. "openai/gpt-4o-mini") are passed through.
    """
    if "/" in model_name:
        return model_name
    return f"orcarouter/{model_name}"


class OrcaRouterLLM(OpenAICompatibleLLM):
    """LLM provider for OrcaRouter, an OpenAI-compatible model routing gateway."""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.ORCAROUTER, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OrcaRouter default parameters."""
        base_params = self._initialize_default_params_with_model_fallback(
            kwargs, DEFAULT_ORCAROUTER_MODEL
        )
        if base_params.model:
            base_params.model = _ensure_orcarouter_namespace(base_params.model)
        return base_params

    def _provider_base_url(self) -> str:
        """Retrieve the OrcaRouter base URL from env/config or use the default."""
        base_url = os.getenv("ORCAROUTER_BASE_URL", ORCAROUTER_BASE_URL)
        config = self.context.config

        # Check config file for override
        if config and config.orcarouter:
            config_base_url = config.orcarouter.base_url
            if config_base_url:
                base_url = config_base_url

        return base_url
