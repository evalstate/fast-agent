from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import httpx

from fast_agent.auth.credentials import OAuthCredential, save_oauth_credential
from fast_agent.llm.provider.openai.xai_oauth import (
    XAI_DEVICE_CODE_URL,
    XAI_TOKEN_URL,
    get_xai_access_token,
    login_xai_oauth,
)
from fast_agent.llm.provider_key_manager import ProviderKeyManager

if TYPE_CHECKING:
    from pathlib import Path


class XaiOAuthSimulator:
    def __init__(self) -> None:
        self.polls = 0
        self.refreshes = 0
        self.forms: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        form = dict(httpx.QueryParams(request.content.decode()))
        self.forms.append(form)
        if str(request.url) == XAI_DEVICE_CODE_URL:
            return httpx.Response(
                200,
                json={
                    "device_code": "device",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://accounts.x.ai/device",
                    "expires_in": 900,
                    "interval": 1,
                },
            )
        if str(request.url) != XAI_TOKEN_URL:
            return httpx.Response(404)
        if form["grant_type"] == "refresh_token":
            self.refreshes += 1
            return httpx.Response(
                200,
                json={
                    "access_token": "refreshed-access",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        self.polls += 1
        if self.polls == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        return httpx.Response(
            200,
            json={
                "access_token": "initial-access",
                "refresh_token": "initial-refresh",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )


def test_xai_device_login_persists_portable_credential(
    monkeypatch,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "xai.auth.json"
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))
    simulator = XaiOAuthSimulator()

    with httpx.Client(transport=httpx.MockTransport(simulator)) as client:
        credential = login_xai_oauth(client=client, sleep=lambda _: None)

    document = json.loads(auth_path.read_text())
    assert credential.access_token == "initial-access"
    assert document["providers"]["xai"]["refresh_token"] == "initial-refresh"
    assert simulator.polls == 2
    assert simulator.forms[0]["referrer"] == "fast-agent"
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert ProviderKeyManager.get_api_key("xai", {}) == "initial-access"


def test_expired_xai_credential_refreshes_and_preserves_refresh_token(
    monkeypatch,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "xai.auth.json"
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))
    save_oauth_credential(
        "xai",
        OAuthCredential(
            access_token="expired",
            refresh_token="keep-refresh",
            expires_at=time.time() - 1,
        ),
    )
    simulator = XaiOAuthSimulator()

    with httpx.Client(transport=httpx.MockTransport(simulator)) as client:
        access_token = get_xai_access_token(client=client)

    document = json.loads(auth_path.read_text())
    assert access_token == "refreshed-access"
    assert simulator.refreshes == 1
    assert document["providers"]["xai"]["refresh_token"] == "keep-refresh"


def test_revoked_xai_refresh_token_is_cleared(
    monkeypatch,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "xai.auth.json"
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))
    save_oauth_credential(
        "xai",
        OAuthCredential(
            access_token="expired",
            refresh_token="revoked",
            expires_at=time.time() - 1,
        ),
    )

    def revoked_refresh(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "Refresh token has been revoked.",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(revoked_refresh)) as client:
        assert get_xai_access_token(client=client) is None

    document = json.loads(auth_path.read_text())
    assert "xai" not in document["providers"]
