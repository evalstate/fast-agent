from typing import Any, List, Optional, Dict
from uuid import UUID

from mcp_agent.agents.agent import Agent
from mcp_agent.context import Context
from mcp_agent.core.exceptions import ModelConfigError, ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.logger import get_logger


class TensorZeroAugmentedLLM(OpenAIAugmentedLLM):
    """
    AugmentedLLM implementation for TensorZero using OpenAI compatibility layer.
    Inherits most logic from OpenAIAugmentedLLM, overrides config and potentially request prep.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize TensorZero LLM.
        Parses 'model' kwarg (function name) and sets T0-specific model identifier.
        Initializes base class, relying on it to set self.context correctly.
        """
        self.t0_function_name: Optional[str] = kwargs.get("model")
        self._episode_id: Optional[str] = kwargs.get("episode_id")

        if not self.t0_function_name:
            raise ModelConfigError(
                "TensorZero provider selected, but no function name model was provided in kwargs"
            )

        t0_model_identifier = f"tensorzero::function_name::{self.t0_function_name}"
        kwargs["model"] = t0_model_identifier
        # --- Model Name Processing --- END

        super().__init__(*args, provider=Provider.TENSORZERO, **kwargs)

        self.logger.debug(
            f"TensorZero LLM initialized. Base model: {self.default_request_params.model}"
        )

    # --- Overrides for Configuration --- START
    def _base_url(self) -> str:
        """Get T0 gateway URL from config, raising error if not found."""
        base_url = None
        uri = None
        if not self.context or not self.context.config:
            raise ModelConfigError(
                "Context or config not available when resolving TensorZero base URL"
            )
        if hasattr(self.context.config, "tensorzero") and self.context.config.tensorzero:
            t0_config = self.context.config.tensorzero
            base_url = getattr(t0_config, "base_url", None)
            if not base_url:
                uri = getattr(t0_config, "uri", None)
        else:
            self.logger.debug("'tensorzero' configuration block not found in self.context.config")
        resolved_url = base_url or uri
        if not resolved_url:
            raise ModelConfigError(
                "TensorZero base URL/URI not configured.",
                "Please configure 'base_url' or 'uri' under 'tensorzero' in config/secrets.",
            )
        return resolved_url

    def _api_key(self) -> str:
        """Return T0 API key from config/secrets if configured, otherwise a dummy key."""
        api_key = None
        if not self.context or not self.context.config:
            # Logger might not be ready if called during super init
            # print("Warning: Context/config unavailable for T0 API key. Using dummy.")
            return "dummy-key-for-t0"
        if hasattr(self.context.config, "tensorzero") and self.context.config.tensorzero:
            t0_config = self.context.config.tensorzero
            api_key = getattr(t0_config, "api_key", None)
        if api_key:
            return api_key
        else:
            return "dummy-key-for-t0"
