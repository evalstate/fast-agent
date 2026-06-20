"""Hugging Face bearer token verifier for server authentication."""

from typing import Any

import httpx
from fastmcp.server.auth import AccessToken, TokenVerifier


class HuggingFaceTokenVerifier(TokenVerifier):
    """
    Verify Hugging Face bearer tokens against the Hub API.

    This is used when fast-agent is deployed as an MCP/A2A server with
    ``FAST_AGENT_SERVE_OAUTH=huggingface``. A syntactically valid Bearer header is
    not enough: the token must be accepted by Hugging Face before tool execution
    can proceed.
    """

    def __init__(
        self,
        provider: str = "huggingface",
        scopes: list[str] | None = None,
        *,
        base_url: str | None = None,
        hub_base_url: str = "https://huggingface.co",
        timeout_seconds: float = 5.0,
    ) -> None:
        """
        Initialize the Hugging Face token verifier.

        Args:
            provider: Name of the OAuth provider (for logging/debugging).
            scopes: List of scopes to assign to valid tokens. Defaults to ["access"].
            base_url: Optional protected resource base URL for auth metadata.
            hub_base_url: Hugging Face Hub base URL used for token validation.
            timeout_seconds: Timeout for Hub token validation requests.
        """
        super().__init__(base_url=base_url, required_scopes=scopes or ["access"])
        self.provider = provider
        self.scopes = scopes or ["access"]
        self.hub_base_url = hub_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verify that a token is present and accepted by Hugging Face.

        Args:
            token: The bearer token to verify.

        Returns:
            AccessToken if Hugging Face accepts the token, None otherwise.
        """
        if not token or not token.strip():
            return None

        token_info = await self._fetch_token_info(token.strip())
        if token_info is None:
            return None

        client_id = _string_field(token_info, "sub") or _string_field(token_info, "name")
        if client_id is None:
            client_id = _string_field(token_info, "preferred_username") or "huggingface-user"

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=self.scopes,
            subject=client_id,
            claims=token_info,
        )

    async def _fetch_token_info(self, token: str) -> dict[str, Any] | None:
        url = f"{self.hub_base_url}/api/whoami-v2"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            return None

        if response.status_code != 200:
            return None

        try:
            payload = response.json()
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None


def _string_field(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None
