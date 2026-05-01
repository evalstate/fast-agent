"""LiteLLM provider — routes through the LiteLLM SDK to 100+ underlying providers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended  # noqa: F401


# Maps fast-agent config sections to the env vars LiteLLM reads when calling
# the corresponding backing provider. Letting users put credentials in
# `fastagent.config.yaml` (e.g. `anthropic: { api_key: ... }`) and have them
# transparently picked up by LiteLLM avoids re-declaring keys in two places.
# Only well-known backings are bridged; less common LiteLLM providers
# (cohere, mistral, perplexity, ...) still rely on the user exporting the
# matching `*_API_KEY` env var directly, matching standard LiteLLM convention.
_CONFIG_TO_LITELLM_ENV: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("anthropic", (("api_key", "ANTHROPIC_API_KEY"), ("base_url", "ANTHROPIC_BASE_URL"))),
    ("openai", (("api_key", "OPENAI_API_KEY"), ("base_url", "OPENAI_BASE_URL"))),
    ("google", (("api_key", "GEMINI_API_KEY"),)),
    ("xai", (("api_key", "XAI_API_KEY"),)),
    ("groq", (("api_key", "GROQ_API_KEY"),)),
    ("deepseek", (("api_key", "DEEPSEEK_API_KEY"),)),
    ("openrouter", (("api_key", "OPENROUTER_API_KEY"),)),
)


def _bridge_fastagent_config_to_litellm_env(config: Any) -> None:
    """Export config-stored backing creds as env vars LiteLLM understands.

    Only sets each env var when it's not already present, so a user who
    explicitly exports `ANTHROPIC_API_KEY` always wins over the config file.
    """
    if config is None:
        return
    for section_name, mappings in _CONFIG_TO_LITELLM_ENV:
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for attr, env_key in mappings:
            value = getattr(section, attr, None)
            if not value or not isinstance(value, str):
                continue
            if os.getenv(env_key):
                continue
            os.environ[env_key] = value


class _LiteLLMCompletions:
    """Mimics the `client.chat.completions` namespace by dispatching to litellm.acompletion."""

    def __init__(self, parent: "_LiteLLMClientShim") -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        import litellm

        merged: dict[str, Any] = dict(kwargs)
        if self._parent.api_key and "api_key" not in merged:
            merged["api_key"] = self._parent.api_key
        if self._parent.base_url and "base_url" not in merged:
            # LiteLLM expects "api_base" for proxy routing
            merged["api_base"] = self._parent.base_url
        if self._parent.default_headers and "extra_headers" not in merged:
            merged["extra_headers"] = self._parent.default_headers
        if self._parent.timeout is not None and "timeout" not in merged:
            merged["timeout"] = self._parent.timeout
        merged.setdefault("drop_params", self._parent.drop_params)

        return await litellm.acompletion(**merged)


class _LiteLLMChat:
    def __init__(self, parent: "_LiteLLMClientShim") -> None:
        self.completions = _LiteLLMCompletions(parent)


class _LiteLLMFiles:
    """Stub for `client.files.*` — LiteLLM routes do not own file uploads."""

    async def create(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        raise NotImplementedError(
            "File uploads via the LiteLLM provider are not supported. "
            "Inline content (base64 / URL) is preferred for multimodal inputs."
        )


class _LiteLLMClientShim:
    """`AsyncOpenAI`-shaped facade over `litellm.acompletion`.

    Reused by `LiteLLMLLM` so the entire OpenAI streaming, tool-call,
    structured-output, and reasoning pipeline in `OpenAILLM` works against
    LiteLLM responses, which are normalized to OpenAI shape by design.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        default_headers: dict[str, str] | None,
        drop_params: bool,
        timeout: float | int | None,
    ) -> None:
        self.api_key = api_key or None
        self.base_url = base_url or None
        self.default_headers = default_headers or None
        self.drop_params = drop_params
        self.timeout = timeout
        self.chat = _LiteLLMChat(self)
        self.files = _LiteLLMFiles()

    async def __aenter__(self) -> "_LiteLLMClientShim":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # noqa: ARG002
        return None


DEFAULT_LITELLM_MODEL = "openai/gpt-4o-mini"


class LiteLLMLLM(OpenAILLM):
    """Native LiteLLM SDK provider.

    Inherits the OpenAI streaming/tool-call/structured-output stack and only
    swaps the underlying client to call `litellm.acompletion` — which returns
    OpenAI-shape responses for every backing provider.

    Supports two modes:

    - **Embedded SDK (default)**: model spec like `litellm.anthropic/claude-sonnet-4-5`
      resolves backing-provider credentials from env vars (`ANTHROPIC_API_KEY`,
      `OPENAI_API_KEY`, etc.) per LiteLLM's own resolution rules.
    - **Proxy**: set `litellm.api_base` and optionally `litellm.api_key` in
      config (or `LITELLM_API_KEY` env var) to route every call through a
      LiteLLM proxy server.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.LITELLM, **kwargs)

        cfg = None
        if self.context and self.context.config:
            cfg = getattr(self.context.config, "litellm", None)
            # Bridge backing-provider creds from fastagent.config.yaml into env
            # vars so LiteLLM's per-provider auth resolution picks them up.
            _bridge_fastagent_config_to_litellm_env(self.context.config)

        self._litellm_api_base: str | None = getattr(cfg, "api_base", None) if cfg else None
        self._litellm_drop_params: bool = bool(getattr(cfg, "drop_params", True)) if cfg else True
        self._litellm_extra_kwargs: dict[str, Any] = (
            dict(getattr(cfg, "extra_kwargs", {}) or {}) if cfg else {}
        )

    def _initialize_default_params(self, kwargs: dict) -> Any:
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_LITELLM_MODEL)

    def _api_key(self) -> str:
        try:
            return ProviderKeyManager.get_api_key("litellm", self.context.config)
        except Exception:
            return ""

    def _base_url(self) -> str | None:
        return self._litellm_api_base

    def _openai_client(self):  # type: ignore[override]
        timeout: float | int | None = None
        if (
            self.default_request_params
            and getattr(self.default_request_params, "timeout", None) is not None
        ):
            timeout = self.default_request_params.timeout

        return _LiteLLMClientShim(
            api_key=self._api_key() or None,
            base_url=self._base_url(),
            default_headers=self._provider_default_headers(),
            drop_params=self._litellm_drop_params,
            timeout=timeout,
        )

    async def _normalize_chat_completion_files(self, client: Any, messages: list[Any]) -> list[Any]:
        # OpenAI's file-search file-upload path is not portable through LiteLLM.
        # Skip file normalization and pass messages through unchanged.
        return messages

    def _prepare_api_request(
        self,
        messages: Any,
        available_tools: Any,
        request_params: Any,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = super()._prepare_api_request(
            messages, available_tools, request_params
        )
        if self._litellm_extra_kwargs:
            for key, value in self._litellm_extra_kwargs.items():
                arguments.setdefault(key, value)
        return arguments
