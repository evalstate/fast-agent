from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam

from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM


class TensorZeroOpenAIAugmentedLLM(OpenAIAugmentedLLM):
    """
    An LLM augmentation that interacts with TensorZero's OpenAI-compatible inference endpoint.
    This class extends the base OpenAIAugmentedLLM to handle TensorZero-specific
    features, such as system template variables and custom parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the TensorZeroOpenAIAugmentedLLM.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._t0_episode_id = kwargs.pop("episode_id", None)
        self._t0_function_name = kwargs.get("model", "")

        super().__init__(*args, provider=Provider.TENSORZERO, **kwargs)
        self.logger.info("TensorZeroOpenAIAugmentedLLM initialized.")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """
        Initializes TensorZero-specific default parameters. Ensures the model name
        is correctly prefixed for the TensorZero API.
        """
        model = kwargs.get("model", "")
        if not model.startswith("tensorzero::"):
            model = f"tensorzero::function_name::{model}"

        self.logger.debug(f"Initializing with TensorZero model: {model}")

        return RequestParams(
            model=model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        """
        Constructs the TensorZero OpenAI-compatible endpoint URL.
        """
        default_url = "http://localhost:3000/openai/v1"
        if self.context and self.context.config and hasattr(self.context.config, "tensorzero"):
            base_url = getattr(self.context.config.tensorzero, "base_url", default_url)
            # Ensure the path is correctly appended
            if not base_url.endswith('/openai/v1'):
                base_url = f"{base_url.rstrip('/')}/openai/v1"
            self.logger.debug(f"Using TensorZero base URL from config: {base_url}")
            return base_url
        self.logger.debug(f"Using default TensorZero base URL: {default_url}")
        return default_url

    def _prepare_api_request(
            self,
            messages: List[ChatCompletionMessageParam],
            tools: Optional[List[Any]],
            request_params: RequestParams
    ) -> Dict[str, Any]:
        """
        Prepares the API request for the TensorZero OpenAI-compatible endpoint.
        This method injects system template variables and other TensorZero-specific
        parameters into the request. It also handles multimodal inputs.
        """
        self.logger.debug("Preparing API request for TensorZero OpenAI endpoint.")

        # Start with the base arguments from the parent class
        arguments = super()._prepare_api_request(messages, tools, request_params)

        # Handle system template variables
        if request_params.template_vars:
            self.logger.debug(f"Injecting template variables: {request_params.template_vars}")
            system_message_found = False
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    # If content is a string, convert it to the TensorZero format
                    if isinstance(msg.get("content"), str):
                        messages[i] = ChatCompletionSystemMessageParam(
                            role="system",
                            content=[request_params.template_vars]
                        )
                    elif isinstance(msg.get("content"), list):
                        # If content is already a list, merge the template vars
                        msg["content"][0].update(request_params.template_vars)
                    system_message_found = True
                    break

            if not system_message_found:
                # If no system message exists, create one
                messages.insert(0, ChatCompletionSystemMessageParam(
                    role="system",
                    content=[request_params.template_vars]
                ))

        # Add TensorZero-specific extra body parameters
        extra_body = arguments.get("extra_body", {})

        if self._t0_episode_id:
            extra_body["tensorzero::episode_id"] = str(self._t0_episode_id)
            self.logger.debug(f"Added tensorzero::episode_id: {self._t0_episode_id}")

        # Merge metadata arguments
        if request_params.metadata and isinstance(request_params.metadata, dict):
            t0_args = request_params.metadata.get("tensorzero_arguments")
            if t0_args:
                self.logger.debug(f"Merging tensorzero_arguments from metadata: {t0_args}")
                for msg in messages:
                    if msg.get("role") == "system" and isinstance(msg.get("content"), list):
                        msg["content"][0].update(t0_args)
                        break

        if extra_body:
            arguments["extra_body"] = extra_body

        self.logger.debug(f"Final API request arguments: {arguments}")
        return arguments