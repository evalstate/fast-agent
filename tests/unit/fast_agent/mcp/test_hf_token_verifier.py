from __future__ import annotations

import httpx
import pytest

import fast_agent.mcp.auth as auth_exports
import fast_agent.mcp.auth.presence as presence_module
from fast_agent.mcp.auth.presence import HuggingFaceTokenVerifier
from fast_agent.mcp.auth.providers._huggingface_compat import (
    HUGGINGFACE_USERINFO_ENDPOINT,
    HUGGINGFACE_WHOAMI_ENDPOINT,
)


def test_presence_token_verifier_alias_is_not_exported() -> None:
    assert "PresenceTokenVerifier" not in auth_exports.__all__
    assert "PresenceTokenVerifier" not in vars(presence_module)


@pytest.mark.asyncio
async def test_hf_token_verifier_accepts_hub_whoami_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer hf_valid"
        if str(request.url) == HUGGINGFACE_USERINFO_ENDPOINT:
            return httpx.Response(401)
        if str(request.url) == HUGGINGFACE_WHOAMI_ENDPOINT:
            return httpx.Response(200, json={"id": "user-123", "name": "shaun"})
        raise AssertionError(f"unexpected URL: {request.url}")

    verifier = HuggingFaceTokenVerifier(
        scopes=["openid", "profile"],
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    access_token = await verifier.verify_token("hf_valid")

    assert access_token is not None
    assert access_token.token == "hf_valid"
    assert access_token.client_id == "user-123"
    assert access_token.scopes == ["openid", "profile"]
    assert access_token.claims["name"] == "shaun"
    assert access_token.claims["huggingface_userinfo"] is None
    assert access_token.claims["huggingface_whoami"] == {"id": "user-123", "name": "shaun"}


@pytest.mark.asyncio
async def test_hf_token_verifier_accepts_oauth_userinfo_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer oauth_valid"
        if str(request.url) == HUGGINGFACE_USERINFO_ENDPOINT:
            return httpx.Response(
                200,
                json={"sub": "oauth-user-123", "name": "shaun", "scope": "openid profile"},
            )
        raise AssertionError(f"unexpected URL: {request.url}")

    verifier = HuggingFaceTokenVerifier(
        scopes=["openid", "profile"],
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    access_token = await verifier.verify_token("oauth_valid")

    assert access_token is not None
    assert access_token.client_id == "oauth-user-123"
    assert access_token.scopes == ["openid", "profile"]
    assert access_token.claims["huggingface_userinfo"]["sub"] == "oauth-user-123"


@pytest.mark.asyncio
async def test_hf_token_verifier_rejects_invalid_hub_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test"
        if str(request.url) in {HUGGINGFACE_USERINFO_ENDPOINT, HUGGINGFACE_WHOAMI_ENDPOINT}:
            return httpx.Response(401)
        raise AssertionError(f"unexpected URL: {request.url}")

    verifier = HuggingFaceTokenVerifier(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert await verifier.verify_token("test") is None


@pytest.mark.asyncio
async def test_hf_token_verifier_rejects_blank_token() -> None:
    verifier = HuggingFaceTokenVerifier()

    assert await verifier.verify_token("   ") is None
