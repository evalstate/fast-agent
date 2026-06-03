from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam

from fast_agent.constants import DEFAULT_MAX_ITERATIONS
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams


class TensorZeroOpenAILLM(OpenAILLM):
    """
    An LLM augmentation that interacts with TensorZero's OpenAI-compatible inference endpoint.
    This class extends the base OpenAIAugmentedLLM to handle TensorZero-specific
    features, such as system template variables and custom parameters.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the TensorZeroOpenAIAugmentedLLM.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self._t0_episode_id = kwargs.pop("episode_id", None)
        self._t0_function_name = kwargs.get("model", "")
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.TENSORZERO, **kwargs)
        self.logger.info("TensorZeroOpenAILLM initialized.")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """
        Initializes TensorZero-specific default parameters. Ensures the model name
        is correctly prefixed for the TensorZero API.
        """
        model = self._resolve_default_model_name(kwargs.get("model"), "") or ""
        if not model.startswith("tensorzero::"):
            model = f"tensorzero::function_name::{model}"

        self.logger.debug(f"Initializing with TensorZero model: {model}")

        return RequestParams(
            model=model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            use_history=True,
        )

    def _provider_base_url(self) -> str:
        """
        Constructs the TensorZero OpenAI-compatible endpoint URL.
        """
        default_url = "http://localhost:3000/openai/v1"
        if self.context and self.context.config and self.context.config.tensorzero:
            base_url = self.context.config.tensorzero.base_url or default_url
            # Ensure the path is correctly appended
            if not base_url.endswith("/openai/v1"):
                base_url = f"{base_url.rstrip('/')}/openai/v1"
            self.logger.debug(f"Using TensorZero base URL from config: {base_url}")
            return base_url
        self.logger.debug(f"Using default TensorZero base URL: {default_url}")
        return default_url

    @staticmethod
    def _tensorzero_system_message(
        template_vars: dict[str, Any],
    ) -> ChatCompletionSystemMessageParam:
        return cast(
            "ChatCompletionSystemMessageParam",
            {"role": "system", "content": [template_vars]},
        )

    @staticmethod
    def _first_system_content_dict(
        messages: list[ChatCompletionMessageParam],
    ) -> dict[str, Any] | None:
        for msg in messages:
            msg_dict = cast("dict[str, Any]", msg)
            content = msg_dict.get("content")
            if msg_dict.get("role") == "system" and isinstance(content, list) and content:
                first_part = content[0]
                if isinstance(first_part, dict):
                    return first_part
        return None

    def _apply_template_vars(
        self,
        messages: list[ChatCompletionMessageParam],
        template_vars: dict[str, Any] | None,
    ) -> None:
        if not template_vars:
            return

        self.logger.debug(f"Injecting template variables: {template_vars}")
        for i, msg in enumerate(messages):
            msg_dict = cast("dict[str, Any]", msg)
            if msg_dict.get("role") != "system":
                continue

            content = msg_dict.get("content")
            if isinstance(content, str):
                messages[i] = self._tensorzero_system_message(template_vars)
            elif isinstance(content, list) and content and isinstance(content[0], dict):
                content[0].update(template_vars)
            return

        messages.insert(0, self._tensorzero_system_message(template_vars))

    def _merge_metadata_arguments(
        self,
        messages: list[ChatCompletionMessageParam],
        metadata: Any,
    ) -> None:
        if not isinstance(metadata, dict):
            return

        t0_args = metadata.get("tensorzero_arguments")
        if not t0_args:
            return

        self.logger.debug(f"Merging tensorzero_arguments from metadata: {t0_args}")
        system_content = self._first_system_content_dict(messages)
        if system_content is not None:
            system_content.update(t0_args)

    def _tensorzero_extra_body(self, arguments: dict[str, Any]) -> dict[str, Any]:
        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = (
            extra_body_raw if isinstance(extra_body_raw, dict) else {}
        )
        if self._t0_episode_id:
            extra_body["tensorzero::episode_id"] = str(self._t0_episode_id)
            self.logger.debug(f"Added tensorzero::episode_id: {self._t0_episode_id}")
        return extra_body

    def _prepare_api_request(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[Any] | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        """
        Prepares the API request for the TensorZero OpenAI-compatible endpoint.
        This method injects system template variables and other TensorZero-specific
        parameters into the request. It also handles multimodal inputs.
        """
        self.logger.debug("Preparing API request for TensorZero OpenAI endpoint.")

        # Start with the base arguments from the parent class
        arguments = super()._prepare_api_request(messages, tools, request_params)

        self._apply_template_vars(messages, request_params.template_vars)
        extra_body = self._tensorzero_extra_body(arguments)
        self._merge_metadata_arguments(messages, request_params.metadata)

        if extra_body:
            arguments["extra_body"] = extra_body

        self.logger.debug(f"Final API request arguments: {arguments}")
        return arguments
