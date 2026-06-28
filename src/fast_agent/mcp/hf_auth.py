"""HuggingFace authentication utilities for hosted and remote connections."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from fast_agent.utils.huggingface_hub import get_huggingface_hub_token
from fast_agent.utils.text import strip_casefold

# Type alias for token provider functions
TokenProvider = Callable[[], str | None]
HFAuthHeader = Literal["Authorization", "X-HF-Authorization"]
_HF_HOSTNAMES = frozenset({"hf.co", "huggingface.co"})
_HF_AUTH_HEADER_NAMES = frozenset({"authorization", "x-hf-authorization"})


def _default_hub_token_provider() -> str | None:
    """Default token provider that uses huggingface_hub.get_token()."""
    return get_huggingface_hub_token()


def _is_hf_space_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False

    parts = hostname.split(".")
    if len(parts) != 3 or parts[-2:] != ["hf", "space"]:
        return False

    space_name = parts[0]
    return (
        bool(space_name)
        and space_name != "-"
        and not space_name.startswith(".")
        and not space_name.endswith(".")
        and " " not in space_name
    )


def _has_hf_auth_header(headers: Mapping[str, str] | None) -> bool:
    if not headers:
        return False
    return any(strip_casefold(header_name) in _HF_AUTH_HEADER_NAMES for header_name in headers)


def is_huggingface_url(url: str) -> bool:
    """
    Check if a URL is a HuggingFace URL that should receive HF_TOKEN authentication.

    Args:
        url: The URL to check

    Returns:
        True if the URL is a HuggingFace URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname is None:
            return False

        if hostname in _HF_HOSTNAMES:
            return True

        return _is_hf_space_hostname(hostname)
    except Exception:
        return False


def is_hf_space_url(url: str) -> bool:
    """Return True when ``url`` is a validated Hugging Face Space hostname."""
    if not is_huggingface_url(url):
        return False
    try:
        hostname = urlparse(url).hostname
    except Exception:
        return False
    return bool(hostname and _is_hf_space_hostname(hostname))


def get_hf_token_from_env(
    hub_token_provider: TokenProvider | None = None,
) -> str | None:
    """
    Get the HuggingFace token from the HF_TOKEN environment variable.

    Falls back to `huggingface_hub.get_token()` when available, so users who have
    authenticated via `hf auth login` don't need to manually export HF_TOKEN.

    Args:
        hub_token_provider: Optional callable that returns a token. Defaults to
            using huggingface_hub.get_token(). Pass a custom provider for testing.

    Returns:
        The HF_TOKEN value if set, None otherwise
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    provider = hub_token_provider if hub_token_provider is not None else _default_hub_token_provider
    return provider()


def _bearer(value: str) -> str:
    return f"Bearer {value}"


@dataclass(frozen=True)
class HuggingFaceAuthPolicy:
    """Policy for attaching Hugging Face bearer credentials to outbound requests."""

    hf_space_header: HFAuthHeader

    def add_ambient_hf_token(
        self,
        url: str,
        headers: dict[str, str] | None,
        hub_token_provider: TokenProvider | None = None,
    ) -> dict[str, str] | None:
        if not is_huggingface_url(url) or _has_hf_auth_header(headers):
            return headers

        hf_token = get_hf_token_from_env(hub_token_provider)
        if hf_token is None:
            return headers

        return self.add_bearer_token(url, headers, hf_token)

    def add_bearer_token(
        self,
        url: str,
        headers: dict[str, str] | None,
        token: str,
    ) -> dict[str, str]:
        result_headers = dict(headers) if headers else {}
        result_headers[self.header_for_url(url)] = _bearer(token)
        return result_headers

    def header_for_url(self, url: str) -> HFAuthHeader:
        return self.hf_space_header if is_hf_space_url(url) else "Authorization"


HF_CLI_AMBIENT_AUTH_POLICY = HuggingFaceAuthPolicy(
    hf_space_header="X-HF-Authorization",
)
HF_EXPLICIT_BEARER_AUTH_POLICY = HuggingFaceAuthPolicy(
    hf_space_header="Authorization",
)
HF_REQUEST_PASSTHROUGH_AUTH_POLICY = HuggingFaceAuthPolicy(
    hf_space_header="X-HF-Authorization",
)


def add_hf_auth_header(
    url: str,
    headers: dict[str, str] | None,
    hub_token_provider: TokenProvider | None = None,
) -> dict[str, str] | None:
    """
    Add HuggingFace authentication header if appropriate.

    Args:
        url: The URL to check
        headers: Existing headers dictionary (may be None)
        hub_token_provider: Optional callable that returns a token. Defaults to
            using huggingface_hub.get_token(). Pass a custom provider for testing.

    Returns:
        Updated headers dictionary with HF auth if appropriate, or original headers
    """
    return HF_CLI_AMBIENT_AUTH_POLICY.add_ambient_hf_token(
        url,
        headers,
        hub_token_provider,
    )

def add_explicit_bearer_auth_header(
    url: str,
    headers: dict[str, str] | None,
    token: str,
) -> dict[str, str]:
    """Add explicit bearer auth for a target endpoint."""
    return HF_EXPLICIT_BEARER_AUTH_POLICY.add_bearer_token(url, headers, token)


def add_forwarded_hf_auth_header(url: str, headers: dict[str, str] | None) -> dict[str, str] | None:
    """Add the request-scoped bearer token to Hugging Face URLs."""
    if not is_huggingface_url(url):
        return headers

    if _has_hf_auth_header(headers):
        return headers

    from fast_agent.mcp.auth.context import request_bearer_token

    token = request_bearer_token.get()
    if not token:
        return headers

    return HF_REQUEST_PASSTHROUGH_AUTH_POLICY.add_bearer_token(url, headers, token)
