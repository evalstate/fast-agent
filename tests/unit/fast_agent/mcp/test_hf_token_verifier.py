from __future__ import annotations

import httpx
import pytest
from fastmcp.server.auth.providers.huggingface import (
    HUGGINGFACE_USERINFO_ENDPOINT,
    HUGGINGFACE_WHOAMI_ENDPOINT,
)

import fast_agent.mcp.auth as auth_exports
from fast_agent.mcp.auth.huggingface import HuggingFaceOAuthOrHubTokenVerifier


def test_legacy_token_verifier_is_not_exported() -> None:
    assert "HuggingFaceTokenVerifier" not in auth_exports.__all__


@pytest.mark.asyncio
async def test_hf_token_verifier_accepts_hub_whoami_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer hf_valid"
        if str(request.url) == HUGGINGFACE_USERINFO_ENDPOINT:
            return httpx.Response(401)
        if str(request.url) == HUGGINGFACE_WHOAMI_ENDPOINT:
            return httpx.Response(200, json={"id": "user-123", "name": "shaun"})
        raise AssertionError(f"unexpected URL: {request.url}")

    verifier = HuggingFaceOAuthOrHubTokenVerifier(
        required_scopes=["openid", "profile"],
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
async def test_hf_token_verifier_uses_hub_token_scopes():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == HUGGINGFACE_USERINFO_ENDPOINT:
            return httpx.Response(401)
        if str(request.url) == HUGGINGFACE_WHOAMI_ENDPOINT:
            return httpx.Response(
                200,
                json={
                    "id": "user-123",
                    "name": "shaun",
                    "auth": {
                        "accessToken": {
                            "scopes": [
                                {"name": "openid"},
                                {"name": "profile"},
                                {"name": "inference-api"},
                            ]
                        }
                    },
                },
            )
        raise AssertionError(f"unexpected URL: {request.url}")

    verifier = HuggingFaceOAuthOrHubTokenVerifier(
        required_scopes=["inference-api"],
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    access_token = await verifier.verify_token("hf_valid")

    assert access_token is not None
    assert access_token.scopes == ["openid", "profile", "inference-api"]


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

    verifier = HuggingFaceOAuthOrHubTokenVerifier(
        required_scopes=["openid", "profile"],
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

    verifier = HuggingFaceOAuthOrHubTokenVerifier(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert await verifier.verify_token("test") is None


@pytest.mark.asyncio
async def test_hf_token_verifier_rejects_blank_token() -> None:
    verifier = HuggingFaceOAuthOrHubTokenVerifier()

    assert await verifier.verify_token("   ") is None
