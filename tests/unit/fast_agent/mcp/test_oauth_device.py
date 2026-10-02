"""Contract tests for the MCP OAuth device authorization grant (RFC 8628)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

import httpx2 as httpx
import keyring
import pytest
from keyring.backend import KeyringBackend

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.oauth_client import (
    build_oauth_provider,
    compute_server_identity,
    keyring_token_present,
)
from fast_agent.mcp.oauth_device import (
    DEVICE_CODE_GRANT_TYPE,
    MCPDeviceAuthorizationDeniedError,
    MCPDeviceAuthorizationExpiredError,
    MCPDeviceAuthorizationUnsupportedError,
    MCPDeviceCode,
    login_mcp_server_with_device_code,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

MCP_URL = "https://mcp.test/mcp"
AUTH_SERVER = "https://auth.test"


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


@pytest.fixture
def memory_keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryKeyring]:
    monkeypatch.setenv("FAST_AGENT_KEYRING_NOTICE", "false")
    original = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        keyring.set_keyring(original)


def _json(status: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status, json=payload)


@dataclass
class FakeAuthorizationServer:
    """A minimal RFC 9728 + RFC 8414 + RFC 7591 + RFC 8628 server."""

    token_responses: list[httpx.Response]
    device_endpoint: bool = True
    device_payload: dict[str, Any] = field(
        default_factory=lambda: {
            "device_code": "device-secret",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://auth.test/device",
            "verification_uri_complete": "https://auth.test/device?user_code=ABCD-EFGH",
            "expires_in": 300,
        }
    )
    registrations: list[dict[str, Any]] = field(default_factory=list)
    device_requests: list[dict[str, list[str]]] = field(default_factory=list)
    token_requests: list[dict[str, list[str]]] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://mcp.test/.well-known/oauth-protected-resource/mcp":
            return _json(
                200,
                {
                    "resource": MCP_URL,
                    "authorization_servers": [AUTH_SERVER],
                    "scopes_supported": ["openid", "read-mcp"],
                },
            )
        if url == f"{AUTH_SERVER}/.well-known/oauth-authorization-server":
            metadata: dict[str, Any] = {
                "issuer": AUTH_SERVER,
                "authorization_endpoint": f"{AUTH_SERVER}/authorize",
                "token_endpoint": f"{AUTH_SERVER}/token",
                "registration_endpoint": f"{AUTH_SERVER}/register",
                "grant_types_supported": [
                    "authorization_code",
                    "refresh_token",
                    DEVICE_CODE_GRANT_TYPE,
                ],
            }
            if self.device_endpoint:
                metadata["device_authorization_endpoint"] = f"{AUTH_SERVER}/device"
            return _json(200, metadata)
        if url == f"{AUTH_SERVER}/register":
            body = json.loads(request.content)
            self.registrations.append(body)
            return _json(201, {**body, "client_id": f"client-{len(self.registrations)}"})
        if url == f"{AUTH_SERVER}/device":
            self.device_requests.append(parse_qs(request.content.decode()))
            return _json(200, self.device_payload)
        if url == f"{AUTH_SERVER}/token":
            self.token_requests.append(parse_qs(request.content.decode()))
            return self.token_responses.pop(0)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


def _server_config() -> MCPServerSettings:
    return MCPServerSettings(name="docs", transport="http", url=MCP_URL)


def _pending() -> httpx.Response:
    return _json(400, {"error": "authorization_pending"})


def _token(access_token: str = "access-1") -> httpx.Response:
    return _json(
        200,
        {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "refresh_token": "refresh-1",
        },
    )


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        await asyncio.sleep(0)


async def _ignore_code(code: MCPDeviceCode) -> None:
    del code


async def _stored_token_is_used_by_mcp_provider(expected_token: str) -> bool:
    """A fresh MCP OAuth provider (as used by real connections) sends the stored token."""
    provider = build_oauth_provider(_server_config(), emit_console_output=False)
    assert provider is not None
    seen: list[str | None] = []

    def mcp_server(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200)

    async with httpx.AsyncClient(
        auth=provider, transport=httpx.MockTransport(mcp_server)
    ) as client:
        await client.post(MCP_URL)
    return seen == [f"Bearer {expected_token}"]


@pytest.mark.asyncio
async def test_pending_then_slow_down_then_success_persists_token_for_mcp_connections(
    memory_keyring: MemoryKeyring,
) -> None:
    del memory_keyring
    server = FakeAuthorizationServer(
        token_responses=[_pending(), _json(400, {"error": "slow_down"}), _token()]
    )
    shown: list[MCPDeviceCode] = []

    async def on_user_code(code: MCPDeviceCode) -> None:
        shown.append(code)

    sleep = RecordingSleep()
    async with server.client() as client:
        result = await login_mcp_server_with_device_code(
            _server_config(), on_user_code=on_user_code, http_client=client, sleep=sleep
        )

    # Poll at the advertised default interval, adding 5s after slow_down (RFC 8628 §3.5).
    assert sleep.calls == [5.0, 5.0, 10.0]
    assert [(code.user_code, code.verification_uri_complete) for code in shown] == [
        ("ABCD-EFGH", "https://auth.test/device?user_code=ABCD-EFGH")
    ]
    assert "ABCD-EFGH" not in repr(shown[0])

    # A dynamically registered public client declares the device grant.
    [registration] = server.registrations
    assert DEVICE_CODE_GRANT_TYPE in registration["grant_types"]
    assert registration["token_endpoint_auth_method"] == "none"

    # RFC 8707 resource and discovered scopes accompany the device request and polls.
    [device_request] = server.device_requests
    assert device_request["resource"] == [MCP_URL]
    assert device_request["scope"] == ["openid read-mcp"]
    assert all(r["resource"] == [MCP_URL] for r in server.token_requests)
    assert all(r["device_code"] == ["device-secret"] for r in server.token_requests)

    assert result.resource == compute_server_identity(_server_config())
    assert result.scope == "openid read-mcp"
    assert keyring_token_present(result.resource)
    assert await _stored_token_is_used_by_mcp_provider("access-1")


@pytest.mark.asyncio
async def test_second_login_reuses_stored_device_capable_client(
    memory_keyring: MemoryKeyring,
) -> None:
    del memory_keyring
    server = FakeAuthorizationServer(token_responses=[_token("access-1"), _token("access-2")])
    async with server.client() as client:
        for _ in range(2):
            await login_mcp_server_with_device_code(
                _server_config(),
                on_user_code=_ignore_code,
                http_client=client,
                sleep=RecordingSleep(),
            )

    assert len(server.registrations) == 1
    assert {r["client_id"][0] for r in server.device_requests} == {"client-1"}
    assert await _stored_token_is_used_by_mcp_provider("access-2")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("access_denied", MCPDeviceAuthorizationDeniedError),
        ("expired_token", MCPDeviceAuthorizationExpiredError),
    ],
)
async def test_terminal_poll_errors_store_nothing(
    memory_keyring: MemoryKeyring,
    error: str,
    expected: type[Exception],
) -> None:
    server = FakeAuthorizationServer(token_responses=[_pending(), _json(400, {"error": error})])
    async with server.client() as client:
        with pytest.raises(expected):
            await login_mcp_server_with_device_code(
                _server_config(),
                on_user_code=_ignore_code,
                http_client=client,
                sleep=RecordingSleep(),
            )

    assert memory_keyring.values == {}


@pytest.mark.asyncio
async def test_device_code_deadline_expires_while_pending(
    memory_keyring: MemoryKeyring,
) -> None:
    server = FakeAuthorizationServer(token_responses=[_pending() for _ in range(50)])
    server.device_payload = {**server.device_payload, "expires_in": 0.2, "interval": 0.05}
    async with server.client() as client:
        with pytest.raises(MCPDeviceAuthorizationExpiredError):
            await login_mcp_server_with_device_code(
                _server_config(), on_user_code=_ignore_code, http_client=client
            )

    assert server.token_requests
    assert memory_keyring.values == {}


@pytest.mark.asyncio
async def test_cancellation_while_polling_stores_nothing(
    memory_keyring: MemoryKeyring,
) -> None:
    server = FakeAuthorizationServer(token_responses=[_token()])
    polling = asyncio.Event()

    async def blocking_sleep(seconds: float) -> None:
        del seconds
        polling.set()
        await asyncio.Event().wait()

    async with server.client() as client:
        task = asyncio.create_task(
            login_mcp_server_with_device_code(
                _server_config(),
                on_user_code=_ignore_code,
                http_client=client,
                sleep=blocking_sleep,
            )
        )
        await polling.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert server.token_requests == []
    assert memory_keyring.values == {}


@pytest.mark.asyncio
async def test_missing_device_endpoint_fails_clearly_before_registering(
    memory_keyring: MemoryKeyring,
) -> None:
    del memory_keyring
    server = FakeAuthorizationServer(token_responses=[], device_endpoint=False)
    async with server.client() as client:
        with pytest.raises(
            MCPDeviceAuthorizationUnsupportedError, match="device_authorization_endpoint"
        ):
            await login_mcp_server_with_device_code(
                _server_config(), on_user_code=_ignore_code, http_client=client
            )

    assert server.registrations == []
    assert server.device_requests == []
