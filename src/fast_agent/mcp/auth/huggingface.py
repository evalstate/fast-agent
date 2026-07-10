"""Hugging Face token verification extensions."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import httpx
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.huggingface import (
    DEFAULT_HUGGINGFACE_SCOPES,
    HUGGINGFACE_WHOAMI_ENDPOINT,
    HuggingFaceTokenVerifier,
)
from fastmcp.utilities.auth import parse_scopes
from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceOAuthOrHubTokenVerifier(HuggingFaceTokenVerifier):
    """Accept Hugging Face OAuth access tokens and ordinary Hub tokens."""

    async def verify_token(self, token: str) -> AccessToken | None:
        token = token.strip()
        if not token:
            return None

        access_token = await super().verify_token(token)
        if access_token is not None:
            return access_token

        try:
            async with (
                contextlib.nullcontext(self._http_client)
                if self._http_client is not None
                else httpx.AsyncClient(timeout=self.timeout_seconds)
            ) as client:
                response = await client.get(
                    HUGGINGFACE_WHOAMI_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "FastMCP-HuggingFace-OAuth",
                    },
                )
        except httpx.RequestError as exc:
            logger.debug("Failed to verify Hugging Face Hub token: %s", exc)
            return None

        if response.status_code != 200:
            return None
        payload: Any = response.json()
        if not isinstance(payload, dict):
            return None

        client_id = _identity(payload)
        if client_id is None:
            return None
        scopes = _hub_scopes(payload) or list(DEFAULT_HUGGINGFACE_SCOPES)
        if self.required_scopes and not set(self.required_scopes).issubset(scopes):
            return None

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=None,
            subject=client_id,
            claims={
                "sub": payload.get("sub") or client_id,
                "name": payload.get("name") or payload.get("fullname"),
                "preferred_username": payload.get("name"),
                "email": payload.get("email"),
                "email_verified": payload.get("emailVerified"),
                "huggingface_userinfo": None,
                "huggingface_whoami": payload,
            },
        )


def _identity(payload: dict[str, Any]) -> str | None:
    for key in ("sub", "id", "name", "preferred_username"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _hub_scopes(payload: Mapping[str, Any]) -> list[str]:
    auth = payload.get("auth")
    if not isinstance(auth, Mapping):
        return []
    access_token = auth.get("accessToken")
    if not isinstance(access_token, Mapping):
        return []
    value = access_token.get("scopes") or access_token.get("scope")
    if isinstance(value, str):
        return parse_scopes(value) or []
    if not isinstance(value, list):
        return []
    return [
        str(scope.get("name") if isinstance(scope, Mapping) else scope).strip()
        for scope in value
        if str(scope.get("name") if isinstance(scope, Mapping) else scope).strip()
    ]


__all__ = ["HuggingFaceOAuthOrHubTokenVerifier"]
