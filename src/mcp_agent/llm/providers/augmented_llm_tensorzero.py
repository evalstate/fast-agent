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
        Extracts context from agent kwarg and passes it explicitly to super().
        Parses 'model' for 'tensorzero.function_name' format.
        Constructs the T0-specific model identifier.
        Sets up logger and episode ID tracking.
        """
        # --- Get Agent and Context early --- START
        agent: Optional[Agent] = kwargs.get('agent')
        resolved_context: Optional[Context] = None
        temp_logger = get_logger(__name__) # Use temp logger until self.logger is set by super

        if agent and hasattr(agent, 'app') and agent.app and hasattr(agent.app, 'context') and agent.app.context:
             resolved_context = agent.app.context
             temp_logger.debug(f"Retrieved context from agent: {type(resolved_context)}")
        else:
             # Log an error, initialization will likely fail in super or later
             temp_logger.error("Could not retrieve context from agent during TensorZeroAugmentedLLM init! Configuration/secrets will be unavailable.")
             # Pass None to super, let ContextDependent try global fallback (which might also fail)
        # --- Get Agent and Context early --- END

        # --- Model Name Parsing --- START
        self.t0_function_name: Optional[str] = None
        self._episode_id: Optional[str] = kwargs.get("episode_id")
        model_arg = kwargs.get("model")
        if model_arg and isinstance(model_arg, str):
            parts = model_arg.split(".", 1)
            provider_prefix = Provider.TENSORZERO.value
            if len(parts) > 1 and parts[0].lower() == provider_prefix:
                if parts[1]:
                    self.t0_function_name = parts[1]
                    t0_model_identifier = f"tensorzero::function_name::{self.t0_function_name}"
                    kwargs["model"] = t0_model_identifier # Modify model in kwargs for superclass
                    temp_logger.info(f"Using T0 function '{self.t0_function_name}' via ID '{t0_model_identifier}'.") # Use temp logger
                else:
                    raise ModelConfigError(
                         f"TensorZero provider specified, but function name is missing: '{model_arg}'. "
                         f"Expected: {provider_prefix}.<function_name>"
                    )
        # --- Model Name Parsing --- END

        # Add/Update the resolved context in kwargs BEFORE calling super
        kwargs['context'] = resolved_context

        # Initialize base class - DO NOT pass context explicitly here.
        # It's now included in kwargs.
        super().__init__(*args, provider=Provider.TENSORZERO, **kwargs)

        # Logger is now initialized by base class, can use self.logger
        self.logger.debug(f"TensorZero LLM initialized. Base model: {self.default_request_params.model}")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize TensorZero-specific default parameters"""
        # Use the model identifier set in __init__ (either original or T0-formatted)
        chosen_model = kwargs.get("model")
        if not chosen_model:
             raise ModelConfigError(
                 "Could not determine model identifier for TensorZeroAugmentedLLM. "
                 "Ensure model is specified correctly (e.g., 'tensorzero.my-function')."
            )
        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        """Get T0 gateway URL from config or secrets, raising error if not found."""
        base_url = None
        # Check context availability
        if not self.context:
             raise ModelConfigError("Context is not available in TensorZero LLM for retrieving base URL.")

        # 1. Check config: self.context.config.tensorzero.base_url
        if self.context.config and hasattr(self.context.config, 'tensorzero') and self.context.config.tensorzero:
            base_url = getattr(self.context.config.tensorzero, 'base_url', None)

        # 2. Check secrets if not found in config: self.context.secrets['tensorzero']['uri']
        if not base_url:
             if hasattr(self.context, 'secrets') and self.context.secrets:
                t0_secrets = self.context.secrets.get("tensorzero", {})
                base_url = t0_secrets.get("uri")
             else:
                 self.logger.warning("Context secrets attribute not available for checking T0 URI.")

        # 3. Raise error if not found in either
        if not base_url:
              raise ModelConfigError(
                  "TensorZero base URL/URI not configured.",
                  "Please configure 'base_url' under 'tensorzero' in fastagent.config.yaml "
                  "or 'uri' under 'tensorzero' in fastagent.secrets.yaml."
              )
        return base_url

    def _api_key(self) -> str:
        """
        Return T0 API key from secrets if configured, otherwise a dummy key.
        """
        api_key = None
        # Check context availability
        if not self.context:
             self.logger.warning("Context is not available in TensorZero LLM for retrieving API key. Using dummy key.")
             return "dummy-key-for-t0"

        # Check secrets: self.context.secrets['tensorzero']['api_key']
        if hasattr(self.context, 'secrets') and self.context.secrets:
            t0_secrets = self.context.secrets.get("tensorzero", {})
            api_key = t0_secrets.get("api_key")
        else:
             self.logger.warning("Context secrets attribute not available for checking T0 API key.")

        if api_key:
            self.logger.debug("Using configured TensorZero API key.")
            return api_key
        else:
            self.logger.debug("No TensorZero API key configured, using dummy key.")
            return "dummy-key-for-t0"

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

        # --- Attempt to capture episode_id from response object ---
        # Access the last response potentially stored by the executor/base class
        # Note: Accessing internals like self.executor.last_response might be fragile.
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
