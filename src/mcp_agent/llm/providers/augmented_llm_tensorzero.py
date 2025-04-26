from typing import Any, List, Optional
from uuid import UUID

from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.core.request_params import RequestParams

from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.logger import get_logger
from mcp_agent.agents.agent import Agent
from mcp_agent.context import Context


class TensorZeroAugmentedLLM(OpenAIAugmentedLLM):
    """
    AugmentedLLM implementation for TensorZero using OpenAI compatibility layer.
    Leverages TensorZero's OpenAI API compatibility.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize TensorZero LLM.
        Relies on base classes to set up context.
        Parses model name and constructs T0 identifier for base class.
        """
        # --- Model Name Processing --- START
        self.t0_function_name: Optional[str] = None # Store the original function name
        self._episode_id: Optional[str] = kwargs.get("episode_id")
        model_arg = kwargs.get("model") # This is the function name (e.g., "chat") from factory

        if model_arg and isinstance(model_arg, str):
            # Store the function name
            self.t0_function_name = model_arg
            # Construct the T0-specific identifier expected by the T0 OpenAI layer
            t0_model_identifier = f"tensorzero::function_name::{self.t0_function_name}"
            # Update kwargs['model'] BEFORE calling super, so base class uses the right ID
            kwargs["model"] = t0_model_identifier
            # Cannot log here yet, logger initialized in super()
        else:
            # Factory should have passed a model name if provider is T0
            raise ModelConfigError("TensorZeroAugmentedLLM initialized without a valid model name in kwargs")
        # --- Model Name Processing --- END

        # Initialize base classes. ContextDependent will try to find/set self.context.
        # OpenAIAugmentedLLM will use the modified kwargs['model'].
        # It will call our overridden _base_url and _api_key during its init.
        super().__init__(*args, provider=Provider.TENSORZERO, **kwargs)

        # Logger is now initialized
        self.logger.debug(f"TensorZero LLM initialized. Base model: {self.default_request_params.model}")

    # --- Use self.context.config to access config/secrets --- START
    def _base_url(self) -> str:
        """Get T0 gateway URL from config, raising error if not found."""
        base_url = None
        uri = None

        if not self.context or not self.context.config:
             raise ModelConfigError("Context or config not available when resolving TensorZero base URL")

        # Check config object for tensorzero settings
        if hasattr(self.context.config, 'tensorzero') and self.context.config.tensorzero:
            t0_config = self.context.config.tensorzero
            # 1. Check for base_url attribute (from config.yaml)
            base_url = getattr(t0_config, 'base_url', None)
            # 2. Check for uri attribute (from secrets.yaml merged into config)
            if not base_url:
                 uri = getattr(t0_config, 'uri', None)
        else:
             self.logger.debug("'tensorzero' configuration block not found in self.context.config")

        resolved_url = base_url or uri # Prioritize base_url if both exist

        if not resolved_url:
              raise ModelConfigError(
                  "TensorZero base URL/URI not configured.",
                  "Please configure 'base_url' or 'uri' under 'tensorzero' in fastagent.config.yaml or fastagent.secrets.yaml."
              )
        return resolved_url

    def _api_key(self) -> str:
        """
        Return T0 API key from config/secrets if configured, otherwise a dummy key.
        """
        api_key = None

        if not self.context or not self.context.config:
             self.logger.warning("Context or config not available when resolving TensorZero API key. Using dummy key.")
             return "dummy-key-for-t0"

        # Check config object for tensorzero settings and api_key attribute
        if hasattr(self.context.config, 'tensorzero') and self.context.config.tensorzero:
            t0_config = self.context.config.tensorzero
            api_key = getattr(t0_config, 'api_key', None)
        else:
             self.logger.debug("'tensorzero' configuration block not found when checking for API key.")

        if api_key:
            self.logger.debug("Using configured TensorZero API key.")
            return api_key
        else:
            self.logger.debug("No TensorZero API key configured, using dummy key.")
            return "dummy-key-for-t0"
    # --- Use self.context.config to access config/secrets --- END

    def _prepare_api_request(
        self, messages, tools, request_params: RequestParams
    ) -> dict[str, Any]:
        """Prepare API request, adding T0 episode_id via extra_body if available."""
        args = super()._prepare_api_request(messages, tools, request_params)

        # Add episode_id using extra_body (relies on underlying openai client support)
        if self._episode_id:
            self.logger.debug(f"Adding episode_id to T0 request: {self._episode_id}")
            extra_body = args.get("extra_body", {})
            # Use a namespaced key as suggested by T0 example, verify if needed
            extra_body["tensorzero::episode_id"] = str(self._episode_id)
            args["extra_body"] = extra_body

        return args

    async def _openai_completion(self, *args, **kwargs) -> List[Any]:
        """Execute completion and capture T0 episode_id from response object."""
        # Execute the completion using the base class method
        responses = await super()._openai_completion(*args, **kwargs)
        last_response = None
        if hasattr(self, "executor") and hasattr(self.executor, "last_response"):
            last_response = self.executor.last_response

        if last_response and hasattr(last_response, "episode_id"):
            new_episode_id = getattr(last_response, "episode_id")
            # Check if it's a UUID or string and convert if necessary
            if isinstance(new_episode_id, UUID):
                 new_episode_id_str = str(new_episode_id)
            elif isinstance(new_episode_id, str):
                 new_episode_id_str = new_episode_id
            else:
                 new_episode_id_str = None
                 self.logger.warning(f"Unexpected type for episode_id in response: {type(new_episode_id)}")

            if new_episode_id_str and new_episode_id_str != self._episode_id:
                self._episode_id = new_episode_id_str
                self.logger.debug(f"Captured/Updated episode_id from T0 response: {self._episode_id}")
            elif not new_episode_id_str:
                 pass
            # else: episode_id hasn't changed

        return responses

    # Inherits generate(), structured(), shutdown() etc. from OpenAIAugmentedLLM
    # Base class uses OpenAIConverter for message mapping and handles streaming/tools.
