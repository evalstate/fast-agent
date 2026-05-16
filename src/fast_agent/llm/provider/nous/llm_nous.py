"""Nous Research (Nous Portal) provider for fast-agent.

Mirrors the Hermes Nous Portal provider in /hermes-agent/plugins/model-providers/nous/.
Key differences from raw OpenAI:
  - Nous Portal product-attribution tags injected into ``extra_body`` on every call
  - Base URL defaults to ``https://inference.nousresearch.com/v1``
  - Auth via NOUS_API_KEY env var or ``nous.api_key`` in fast-agent config

Usage::

    fast-agent --model nous.hermes-3-405b
    # or rely on default_model from config:
    # default_model: nous.hermes-3-405b
"""

from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOUS_BASE_URL = "https://inference.nousresearch.com/v1"
NOUS_DEFAULT_MODEL = "hermes-3-405b"


# ---------------------------------------------------------------------------
# NousLLM
# ---------------------------------------------------------------------------


class NousLLM(OpenAICompatibleLLM):
    """Nous Portal provider — OpenAI-compatible calls with Portal attribution tags.

    The base ``OpenAICompatibleLLM`` handles the standard OpenAI protocol.
    Nous adds two things:
    1. ``extra_body["tags"]`` — product=fast-agent + client version tags for portal attribution
    2. Nous-specific base URL (``https://inference.nousresearch.com/v1``)
    """

    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.NOUS, **kwargs)

    # ------------------------------------------------------------------
    # Base URL
    # ------------------------------------------------------------------

    def _provider_base_url(self) -> str | None:
        """Return Nous Portal base URL, honouring an explicit config override."""
        if self.context.config and self.context.config.nous:
            override = self.context.config.nous.base_url
            if override:
                return override
        return NOUS_BASE_URL

    # ------------------------------------------------------------------
    # Default model (when no model is specified)
    # ------------------------------------------------------------------

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Nous default parameters when no model is explicitly set."""
        return self._initialize_default_params_with_model_fallback(
            kwargs, NOUS_DEFAULT_MODEL
        )

    # ------------------------------------------------------------------
    # Portal attribution tags injected on every request
    # ------------------------------------------------------------------

    @staticmethod
    def _nous_portal_tags() -> list[str]:
        """Return fast-agent product-attribution tags for Nous Portal requests.

        Shape: ``["product=fast-agent", "client=fast-agent-client-v{__version__}"]``.

        The version tag allows Nous to bucketed usage by agent toolkit release.
        """
        try:
            from fast_agent import __version__ as _ver

            client_tag = f"client=fast-agent-client-v{_ver}"
        except Exception:
            client_tag = "client=fast-agent-client-vunknown"
        return ["product=fast-agent", client_tag]

    def _prepare_api_request(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[Any] | None,
        request_params: RequestParams,
    ) -> dict[str, Any]:
        """Build keyword arguments for the OpenAI-compatible chat completion.

        Calls ``super()._prepare_api_request()`` for the standard path, then
        adds ``tags`` to ``extra_body`` for Nous Portal product attribution.
        """
        arguments: dict[str, Any] = super()._prepare_api_request(
            messages, tools, request_params
        )

        # Nous Portal attribution tags — idempotent merge
        tags = self._nous_portal_tags()
        existing = arguments.get("extra_body")
        if isinstance(existing, dict):
            existing.setdefault("tags", []).extend(tags)
        else:
            arguments["extra_body"] = {"tags": tags}

        return arguments
