from __future__ import annotations

import pytest

import fast_agent.mcp.auth as auth_exports
import fast_agent.mcp.auth.presence as presence_module
from fast_agent.mcp.auth.presence import HuggingFaceTokenVerifier


def test_presence_token_verifier_alias_is_not_exported() -> None:
    assert "PresenceTokenVerifier" not in auth_exports.__all__
    assert "PresenceTokenVerifier" not in vars(presence_module)


@pytest.mark.asyncio
async def test_hf_token_verifier_accepts_hub_whoami_response(monkeypatch: pytest.MonkeyPatch):
    verifier = HuggingFaceTokenVerifier(scopes=["openid", "profile"])

    async def fetch_token_info(token: str) -> dict[str, object] | None:
        assert token == "hf_valid"
        return {"sub": "user-123", "name": "shaun"}

    monkeypatch.setattr(verifier, "_fetch_token_info", fetch_token_info)

    access_token = await verifier.verify_token("hf_valid")

    assert access_token is not None
    assert access_token.token == "hf_valid"
    assert access_token.client_id == "user-123"
    assert access_token.scopes == ["openid", "profile"]
    assert access_token.claims["name"] == "shaun"


@pytest.mark.asyncio
async def test_hf_token_verifier_rejects_invalid_hub_token(monkeypatch: pytest.MonkeyPatch):
    verifier = HuggingFaceTokenVerifier()

    async def fetch_token_info(token: str) -> dict[str, object] | None:
        assert token == "test"
        return None

    monkeypatch.setattr(verifier, "_fetch_token_info", fetch_token_info)

    assert await verifier.verify_token("test") is None


@pytest.mark.asyncio
async def test_hf_token_verifier_rejects_blank_token() -> None:
    verifier = HuggingFaceTokenVerifier()

    assert await verifier.verify_token("   ") is None
