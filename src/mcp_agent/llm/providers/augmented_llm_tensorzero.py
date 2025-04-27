from typing import Any, List, Optional, Dict
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
        self.t0_function_name: Optional[str] = None  # Store the original function name
        self._episode_id: Optional[str] = kwargs.get("episode_id")
        model_arg = kwargs.get("model")  # This is the function name (e.g., "chat") from factory

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
            raise ModelConfigError(
                "TensorZeroAugmentedLLM initialized without a valid model name in kwargs"
            )

        # Initialize base classes. ContextDependent will try to find/set self.context.
        # OpenAIAugmentedLLM will use the modified kwargs['model'].
        # It will call our overridden _base_url and _api_key during its init.
        super().__init__(*args, provider=Provider.TENSORZERO, **kwargs)

        # Logger is now initialized
        self.logger.debug(
            f"TensorZero LLM initialized. Base model: {self.default_request_params.model}"
        )

    # --- Use self.context.config to access config/secrets --- START
    def _base_url(self) -> str:
        """Get T0 gateway URL from config, raising error if not found."""
        base_url = None
        uri = None

        if not self.context or not self.context.config:
            raise ModelConfigError(
                "Context or config not available when resolving TensorZero base URL"
            )

        # Check config object for tensorzero settings
        if hasattr(self.context.config, "tensorzero") and self.context.config.tensorzero:
            t0_config = self.context.config.tensorzero
            # 1. Check for base_url attribute (from config.yaml)
            base_url = getattr(t0_config, "base_url", None)
            # 2. Check for uri attribute (from secrets.yaml merged into config)
            if not base_url:
                uri = getattr(t0_config, "uri", None)
        else:
            self.logger.debug("'tensorzero' configuration block not found in self.context.config")

        resolved_url = base_url or uri  # Prioritize base_url if both exist

        if not resolved_url:
            raise ModelConfigError(
                "TensorZero base URL/URI not configured.",
                "Please configure 'base_url' or 'uri' under 'tensorzero' in fastagent.config.yaml or fastagent.secrets.yaml.",
            )
        return resolved_url

    def _api_key(self) -> str:
        """
        Return T0 API key from config/secrets if configured, otherwise a dummy key.
        """
        api_key = None

        if not self.context or not self.context.config:
            self.logger.warning(
                "Context or config not available when resolving TensorZero API key. Using dummy key."
            )
            return "dummy-key-for-t0"

        # Check config object for tensorzero settings and api_key attribute
        if hasattr(self.context.config, "tensorzero") and self.context.config.tensorzero:
            t0_config = self.context.config.tensorzero
            api_key = getattr(t0_config, "api_key", None)
        else:
            self.logger.debug(
                "'tensorzero' configuration block not found when checking for API key."
            )

        if api_key:
            self.logger.debug("Using configured TensorZero API key.")
            return api_key
        else:
            self.logger.debug("No TensorZero API key configured, using dummy key.")
            return "dummy-key-for-t0"

    def _prepare_api_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        request_params: RequestParams,
    ) -> Dict[str, Any]:
        """
        Prepare API request for T0 OpenAI compatibility layer.
        Always formats the system prompt using tensorzero::arguments structure.
        Injects episode_id if available.
        """
        # Get standard args prepared by base class (messages list might contain a simple system prompt)
        args: Dict[str, Any] = super()._prepare_api_request(messages, tools, request_params)

        # --- Always format system prompt for T0 --- START
        # Get base instruction and metadata args
        base_instruction = (
            self.instruction or "You are a helpful assistant."
        )  # Default if agent has no instruction
        metadata_args = (
            request_params.metadata.get("tensorzero_arguments", {})
            if request_params.metadata
            else {}
        )
        if not isinstance(metadata_args, dict):
            self.logger.warning(
                f"'tensorzero_arguments' in metadata is not a dict, ignoring. Value: {metadata_args}"
            )
            metadata_args = {}

        # Define defaults for required T0 args
        t0_final_args = {
            "BASE_INSTRUCTIONS": base_instruction,
            "DISCLAIMER_TEXT": "",
            "BIBLE": "",
            "USER_PORTFOLIO_DESCRIPTION": "",
            "USER_PROFILE_DATA": "{}",  # Default empty JSON string
            "CONTEXT": "",
        }

        # Merge metadata args over defaults
        t0_final_args.update(metadata_args)
        self.logger.debug(f"Using final tensorzero::arguments: {t0_final_args}")

        # Construct the T0 system message
        t0_system_message = {
            "role": "system",
            "content": [{"type": "text", "tensorzero::arguments": t0_final_args}],
        }

        # Rebuild messages list: T0 system message + non-system messages from original
        formatted_messages: List[Dict[str, Any]] = [t0_system_message]  # Start with T0 system msg
        original_messages: List[Dict[str, Any]] = args.get("messages", [])

        if isinstance(original_messages, list):
            for msg in original_messages:
                # Check message format is dict before accessing role
                if isinstance(msg, dict) and msg.get("role") != "system":
                    formatted_messages.append(msg)
        else:
            self.logger.warning("Base class did not return 'messages' as a list in args")

        # Replace messages in args dict
        args["messages"] = formatted_messages
        # --- Always format system prompt for T0 --- END

        # Add episode_id using extra_body
        if self._episode_id:
            self.logger.debug(f"Adding episode_id to T0 request: {self._episode_id}")
            extra_body = args.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            extra_body["tensorzero::episode_id"] = str(self._episode_id)
            args["extra_body"] = extra_body

        self.logger.debug(f"Final prepared T0 API request args: {args}")
        return args

    async def _openai_completion(self, *args, **kwargs) -> List[Any]:
        """Execute completion using base class. Episode ID capture removed due to fragility."""
        # Execute the completion using the base class method
        responses = await super()._openai_completion(*args, **kwargs)
        return responses
