"""
Provider API key management for various LLM providers.
Centralizes API key handling logic to make provider implementations more generic.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.utils.huggingface_hub import get_huggingface_hub_token
from fast_agent.utils.text import strip_casefold

PROVIDER_ENVIRONMENT_MAP: dict[str, str] = {
    # default behaviour in _get_env_key_name is to capitalize the
    # provider name and suffix "_API_KEY" - so no specific mapping needed unless overriding
    "hf": "HF_TOKEN",
    "responses": "OPENAI_API_KEY",  # Temporary workaround
    "openresponses": "OPENRESPONSES_API_KEY",
    "codexresponses": "CODEX_API_KEY",
    "metaai": "META_AI_API_KEY",
}
PROVIDER_CONFIG_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    # HuggingFace historically used "huggingface" (full name) in config files,
    # while the provider id is "hf". Support both spellings.
    "hf": ("hf", "huggingface"),
    "huggingface": ("huggingface", "hf"),
    # Responses shares OpenAI credentials; allow reading openai.api_key when
    # responses.api_key is omitted.
    "responses": ("openai",),
}
API_KEY_HINT_TEXT = "<your-api-key-here>"
API_KEYLESS_PROVIDERS: frozenset[str] = frozenset({"anthropic-vertex"})


@dataclass(frozen=True)
class _ServeOAuthProviderPolicy:
    model_provider_names: frozenset[str]
    display_name: str
    operator_credential: str


_SERVE_OAUTH_PROVIDER_POLICIES: Final[Mapping[str, _ServeOAuthProviderPolicy]] = {
    "huggingface": _ServeOAuthProviderPolicy(
        model_provider_names=frozenset({"hf", "huggingface"}),
        display_name="Hugging Face",
        operator_credential="HF_TOKEN",
    ),
}


@runtime_checkable
class _ConfigGetter(Protocol):
    def get(self, key: str, default: object = None) -> object: ...


def _get_config_value(config: object, key: str) -> object | None:
    if isinstance(config, BaseModel):
        config = config.model_dump()
    if isinstance(config, Mapping):
        values = cast("Mapping[object, object]", config)
        return values.get(key)
    if isinstance(config, _ConfigGetter):
        return config.get(key)
    return None


class ProviderKeyManager:
    """
    Manages API keys for different providers centrally.
    This class abstracts away the provider-specific key access logic,
    making the provider implementations more generic.
    """

    @staticmethod
    def get_env_var(provider_name: str) -> str | None:
        env_key_name = ProviderKeyManager.get_env_key_name(provider_name)
        if not env_key_name:
            return None
        return os.getenv(env_key_name)

    @staticmethod
    def get_env_key_name(provider_name: str) -> str | None:
        normalized_provider = strip_casefold(provider_name)
        if normalized_provider in API_KEYLESS_PROVIDERS:
            return None
        return PROVIDER_ENVIRONMENT_MAP.get(
            normalized_provider,
            f"{normalized_provider.upper()}_API_KEY",
        )

    @staticmethod
    def get_config_file_key(provider_name: str, config: object) -> str | None:
        provider_name = strip_casefold(provider_name)
        provider_keys = ProviderKeyManager._get_provider_config_keys(provider_name)
        for key in provider_keys:
            provider_settings = _get_config_value(config, key)
            if not provider_settings:
                continue
            api_key = _get_config_value(provider_settings, "api_key") or API_KEY_HINT_TEXT
            if api_key == API_KEY_HINT_TEXT:
                return None
            return api_key if isinstance(api_key, str) else None

        return None

    @staticmethod
    def _get_provider_config_keys(provider_name: str) -> list[str]:
        """Return config key candidates for a provider (provider id + aliases)."""
        keys = [provider_name]
        for alias in PROVIDER_CONFIG_KEY_ALIASES.get(provider_name, ()):
            if alias not in keys:
                keys.append(alias)
        return keys

    @staticmethod
    def _uses_no_api_key(provider_name: str, config: Any) -> bool:
        if provider_name == "fast-agent" or provider_name in API_KEYLESS_PROVIDERS:
            return True
        if provider_name != "google":
            return False
        return ProviderKeyManager._google_vertex_enabled(config)

    @staticmethod
    def _google_vertex_enabled(config: Any) -> bool:
        try:
            cfg = config.model_dump() if isinstance(config, BaseModel) else config
            return isinstance(cfg, dict) and bool(
                (cfg.get("google") or {}).get("vertex_ai", {}).get("enabled")
            )
        except Exception:
            return False

    @staticmethod
    def _request_scoped_api_key(provider_name: str) -> str | None:
        if ProviderKeyManager._serve_oauth_policy(provider_name) is None:
            return None

        # Check for request-scoped token first (token passthrough from MCP server).
        from fast_agent.mcp.auth.context import request_bearer_token

        return request_bearer_token.get()

    @staticmethod
    def _serve_oauth_policy(
        provider_name: str,
    ) -> tuple[str, _ServeOAuthProviderPolicy] | None:
        normalized_provider = strip_casefold(provider_name)
        return next(
            (
                (oauth_provider, policy)
                for oauth_provider, policy in _SERVE_OAUTH_PROVIDER_POLICIES.items()
                if normalized_provider in policy.model_provider_names
            ),
            None,
        )

    @staticmethod
    def _active_serve_oauth_policy(
        provider_name: str,
    ) -> tuple[str, _ServeOAuthProviderPolicy] | None:
        resolved = ProviderKeyManager._serve_oauth_policy(provider_name)
        if resolved is None:
            return None

        oauth_provider, _ = resolved
        from fast_agent.mcp.server.common import normalize_serve_oauth_provider

        configured_provider = normalize_serve_oauth_provider(
            os.getenv("FAST_AGENT_SERVE_OAUTH")
        )
        return resolved if configured_provider == oauth_provider else None

    @staticmethod
    def serve_oauth_requires_request_token(provider_name: str) -> bool:
        """Return whether calls for this provider require a caller OAuth token."""
        return ProviderKeyManager._active_serve_oauth_policy(provider_name) is not None

    @staticmethod
    def _raise_missing_request_token(
        oauth_provider: str,
        policy: _ServeOAuthProviderPolicy,
    ) -> NoReturn:
        raise ProviderKeyError(
            f"{policy.display_name} caller token missing",
            f"This server is configured with FAST_AGENT_SERVE_OAUTH={oauth_provider}, "
            f"so {policy.display_name} model calls must use the caller's forwarded "
            "bearer token. "
            "No request-scoped token was present for this call; refusing to fall "
            f"back to the server's {policy.operator_credential}.",
        )

    @staticmethod
    def _configured_or_environment_key(provider_name: str, config: Any) -> str | None:
        return ProviderKeyManager.get_config_file_key(
            provider_name, config
        ) or ProviderKeyManager.get_env_var(provider_name)

    @staticmethod
    def _provider_specific_fallback_key(provider_name: str) -> str | None:
        if provider_name == "codexresponses":
            # Codex OAuth tokens stored in keyring (if no env/config key supplied).
            from fast_agent.llm.provider.openai.codex_oauth import get_codex_access_token

            return get_codex_access_token()

        if provider_name in {"hf", "huggingface"}:
            # HuggingFace also supports tokens managed by huggingface_hub
            # (e.g. `hf auth login`) when env/config keys are absent.
            return get_huggingface_hub_token()

        if provider_name == "generic":
            return "ollama"

        return None

    @staticmethod
    def _raise_missing_api_key(provider_name: str) -> NoReturn:
        from fast_agent.llm.provider_types import Provider

        if provider_name == "codexresponses":
            raise ProviderKeyError(
                "Codex OAuth token not configured",
                "Run `fast-agent auth codex-login` to authenticate, or set the CODEX_API_KEY environment variable.",
            )

        try:
            provider_enum = Provider(provider_name)
        except ValueError as exc:
            raise ProviderKeyError(
                f"Invalid provider: {provider_name}",
                f"'{provider_name}' is not a valid provider name.",
            ) from exc

        display_name = provider_enum.display_name
        env_key_name = ProviderKeyManager.get_env_key_name(provider_name)
        env_hint = f" or set the {env_key_name} environment variable." if env_key_name else "."
        raise ProviderKeyError(
            f"{display_name} API key not configured",
            f"The {display_name} API key is required but not set.\n"
            f"Add it to your configuration file under {provider_name}.api_key{env_hint}",
        )

    @staticmethod
    def get_api_key(
        provider_name: str,
        config: Any,
    ) -> str:
        """
        Gets the API key for the specified provider.

        Args:
            provider_name: Name of the provider (e.g., "anthropic", "openai")
            config: The application configuration object

        Returns:
            The API key as a string

        Raises:
            ProviderKeyError: If the API key is not found or is invalid
        """
        provider_name = strip_casefold(provider_name)

        if ProviderKeyManager._uses_no_api_key(provider_name, config):
            return ""

        api_key = ProviderKeyManager.get_optional_api_key(provider_name, config)
        if not api_key:
            ProviderKeyManager._raise_missing_api_key(provider_name)

        return api_key

    @staticmethod
    def get_optional_api_key(
        provider_name: str,
        config: Any,
    ) -> str | None:
        """Return a configured provider API key if one is available, otherwise None."""
        provider_name = strip_casefold(provider_name)

        if ProviderKeyManager._uses_no_api_key(provider_name, config):
            return ""

        request_scoped = ProviderKeyManager._request_scoped_api_key(provider_name)
        if request_scoped:
            return request_scoped

        # Fail closed rather than fall back to the server's own credentials when
        # serve OAuth requires the caller's token.
        serve_oauth = ProviderKeyManager._active_serve_oauth_policy(provider_name)
        if serve_oauth:
            ProviderKeyManager._raise_missing_request_token(*serve_oauth)

        return ProviderKeyManager._configured_or_environment_key(
            provider_name, config
        ) or ProviderKeyManager._provider_specific_fallback_key(provider_name)
