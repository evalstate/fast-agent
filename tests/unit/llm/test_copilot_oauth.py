"""Native device auth contracts; synthetic tokens and MockTransport only."""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import FrozenInstanceError, dataclass, field, replace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from urllib.parse import parse_qs

import httpx
import pytest

from fast_agent.auth import credentials
from fast_agent.auth.credentials import OAuthCredential
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.copilot import oauth

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

TOKEN = "synthetic-access-secret"
DEVICE_SECRET = "synthetic-device-secret"
USER_CODE = "ABCD-1234"
READ_OPERATIONS = (
    oauth.get_copilot_credential,
    oauth.get_copilot_access_token,
    oauth.get_copilot_token_status,
)
INVALID_TOKENS = (
    "",
    "   ",
    "invalid token",
    " padded-token ",
    "synthetic-token\n",
    "synthetic-token\r\nInjected:header",
    "synthetic-token\t",
    "synthetic-token\x00",
    "synthetic-token\x1b",
    "synthetic-token\x7f",
    "synthetic-tokené",
    "synthetic-token\u200b",
)


def device_response(**updates: object) -> httpx.Response:
    payload: dict[str, object] = {
        "device_code": DEVICE_SECRET,
        "user_code": USER_CODE,
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
    }
    payload.update(updates)
    return httpx.Response(200, json=payload)


def token_response(**updates: object) -> httpx.Response:
    payload: dict[str, object] = {
        "access_token": TOKEN,
        "token_type": "bearer",
        "scope": "read:user",
    }
    payload.update(updates)
    return httpx.Response(200, json=payload)


@dataclass
class Clock:
    now: float = 100
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return 1_000_000 + self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    # Replace only oauth's clock, not the event loop's monotonic clock.
    monkeypatch.setattr(oauth, "time", clock)
    monkeypatch.setattr(oauth.asyncio, "sleep", clock.sleep)
    return clock


@dataclass
class GitHub:
    responses: list[httpx.Response | httpx.RequestError] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    clients: list[httpx.AsyncClient] = field(default_factory=list)
    token_started: asyncio.Event = field(default_factory=asyncio.Event)
    block_token: bool = False

    async def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/login/oauth/access_token":
            self.token_started.set()
            if self.block_token:
                await asyncio.Event().wait()
        response = self.responses.pop(0)
        if isinstance(response, httpx.RequestError):
            raise response
        return response


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> GitHub:
    github = GitHub()
    client_type = httpx.AsyncClient

    def client_factory(
        *, timeout: float = 30, trust_env: bool = False, follow_redirects: bool = False
    ) -> httpx.AsyncClient:
        assert trust_env is False
        assert follow_redirects is False
        client = client_type(
            transport=httpx.MockTransport(github.respond),
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=follow_redirects,
        )
        github.clients.append(client)
        return client

    monkeypatch.setattr(oauth.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(oauth.console, "ensure_blocking_console", Mock())
    return github


@pytest.fixture(autouse=True)
def auth_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "auth.json"
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(path))
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    # These unrelated sources must never be consumed.
    monkeypatch.setenv("GH_TOKEN", "unrelated-gh-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "unrelated-github-secret")
    return path


def assert_safe(error: BaseException, *secrets: str) -> None:
    rendered = "".join(traceback.format_exception(error))
    for secret in (TOKEN, DEVICE_SECRET, *secrets):
        assert secret not in rendered
    assert isinstance(error, ProviderKeyError)


@pytest.mark.asyncio
async def test_request_contract_and_redacted_frozen_device(github: GitHub, clock: Clock) -> None:
    github.responses.append(device_response())
    async with httpx.AsyncClient() as client:
        device = await oauth.request_copilot_device_code(client)
    request = github.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://github.com/login/device/code"
    assert request.headers["Accept"] == "application/json"
    assert parse_qs(request.content.decode()) == {
        "client_id": ["Ov23li9BBH5sVoKopuI6"],
        "scope": ["read:user"],
    }
    assert device.device_code == DEVICE_SECRET
    assert device.user_code == USER_CODE
    assert device.deadline == clock.now + 900
    assert DEVICE_SECRET not in repr(device)
    with pytest.raises(FrozenInstanceError):
        device.__setattr__("device_code", "replacement")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        "http://github.com/login/device",
        "https://github.com.evil.invalid/login/device",
        "https://github.com@evil.invalid/login/device",
        "https://evil.invalid/login/device",
        "https://github.com/login/device?next=https://evil.invalid",
        "https://github.com/login/device#fragment",
        "https://github.com/login/device/",
        "https://github.com:443/login/device",
    ],
)
async def test_untrusted_verification_uri_is_rejected(github: GitHub, uri: str) -> None:
    github.responses.append(device_response(verification_uri=uri))
    async with httpx.AsyncClient() as client:
        with pytest.raises(oauth.CopilotAuthenticationError) as error:
            await oauth.request_copilot_device_code(client)
    assert_safe(error.value, uri)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"device_code": ""},
        {"device_code": 123},
        {"user_code": "[link=evil]code"},
        {"expires_in": 0},
        {"expires_in": True},
        {"expires_in": "900"},
        {"interval": -1},
        {"interval": "5"},
    ],
)
async def test_device_boundary_validation(github: GitHub, updates: dict[str, object]) -> None:
    github.responses.append(device_response(**updates))
    async with httpx.AsyncClient() as client:
        with pytest.raises(oauth.CopilotAuthenticationError) as error:
            await oauth.request_copilot_device_code(client)
    assert_safe(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 400])
async def test_pending_slowdown_and_success(github: GitHub, clock: Clock, status: int) -> None:
    github.responses.extend(
        [
            device_response(),
            httpx.Response(status, json={"error": "authorization_pending"}),
            httpx.Response(status, json={"error": "slow_down", "interval": 1}),
            httpx.Response(status, json={"error": "slow_down", "interval": 20}),
            token_response(),
        ]
    )
    async with httpx.AsyncClient() as client:
        device = await oauth.request_copilot_device_code(client)
        credential = await oauth.poll_copilot_device_code(client, device)
    assert clock.sleeps == [5, 5, 10, 20]
    assert credential.access_token == TOKEN
    assert credential.refresh_token is None
    assert credential.expires_at is None
    assert TOKEN not in repr(credential)
    for request in github.requests[1:]:
        assert str(request.url) == "https://github.com/login/oauth/access_token"
        assert request.headers["Accept"] == "application/json"
        assert parse_qs(request.content.decode()) == {
            "client_id": [oauth.COPILOT_CLIENT_ID],
            "device_code": [DEVICE_SECRET],
            "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
        }
    assert oauth.get_copilot_access_token() is None  # Polling alone never persists.


@pytest.mark.asyncio
async def test_network_timeout_backs_off(github: GitHub, clock: Clock) -> None:
    github.responses.extend(
        [device_response(), httpx.ReadTimeout(TOKEN), token_response(expires_in=60)]
    )
    async with httpx.AsyncClient() as client:
        device = await oauth.request_copilot_device_code(client)
        credential = await oauth.poll_copilot_device_code(client, device)
    assert clock.sleeps == [5, 10]
    assert credential.expires_at == clock.time() + 60


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("access_denied", "denied"),
        ("authorization_denied", "denied"),
        ("expired_token", "expired"),
        ("incorrect_device_code", "rejected"),
        (TOKEN, "rejected"),
    ],
)
@pytest.mark.parametrize("status", [200, 400])
async def test_oauth_errors_never_save(
    github: GitHub, clock: Clock, auth_file: Path, code: str, message: str, status: int
) -> None:
    github.responses.extend(
        [
            device_response(),
            httpx.Response(status, json={"error": code, "error_description": TOKEN}),
        ]
    )
    with pytest.raises(oauth.CopilotAuthenticationError, match=message) as error:
        await oauth.login_copilot_oauth_async()
    assert_safe(error.value)
    assert not auth_file.exists()
    assert github.clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["device", "token"])
@pytest.mark.parametrize("status", [200, 302, 401, 500])
async def test_malformed_and_http_errors_are_sanitized(
    github: GitHub, clock: Clock, stage: str, status: int, auth_file: Path
) -> None:
    if stage == "token":
        github.responses.append(device_response())
    github.responses.append(
        httpx.Response(status, text=TOKEN, headers={"Location": "https://evil.invalid"})
    )
    with pytest.raises(oauth.CopilotAuthenticationError) as error:
        await oauth.login_copilot_oauth_async()
    assert_safe(error.value)
    assert len(github.requests) == (2 if stage == "token" else 1)
    if status != 200:
        assert f"HTTP {status}" in str(error.value)
    assert not auth_file.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["device", "token"])
async def test_transport_errors_are_sanitized(github: GitHub, clock: Clock, stage: str) -> None:
    if stage == "token":
        github.responses.append(device_response())
    github.responses.append(httpx.ConnectError(TOKEN))
    with pytest.raises(oauth.CopilotAuthenticationError) as error:
        await oauth.login_copilot_oauth_async()
    assert_safe(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"access_token": TOKEN},
        {"access_token": "", "token_type": "bearer"},
        {"access_token": [TOKEN], "token_type": "bearer"},
        {"access_token": TOKEN, "token_type": "bearer", "expires_in": -1},
        {"access_token": TOKEN, "token_type": "not-bearer"},
        [],
        None,
    ],
)
async def test_invalid_token_response(github: GitHub, clock: Clock, payload: object) -> None:
    github.responses.extend([device_response(), httpx.Response(200, json=payload)])
    with pytest.raises(oauth.CopilotAuthenticationError) as error:
        await oauth.login_copilot_oauth_async()
    assert_safe(error.value)


@pytest.mark.asyncio
async def test_success_displays_only_user_instructions_and_saves(
    github: GitHub, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = Mock()
    monkeypatch.setattr(oauth.console.console, "print", output)
    github.responses.extend([device_response(), token_response()])
    credential = await oauth.login_copilot_oauth_async()
    assert isinstance(credential, OAuthCredential)
    assert oauth.get_copilot_access_token() == TOKEN
    stored = credentials.load_oauth_credential("copilot")
    assert stored is not None
    assert stored.credential.refresh_token is None
    rendered = str(output.call_args_list)
    assert "Waiting for GitHub approval. Ctrl+C to cancel." in rendered
    assert USER_CODE in rendered
    assert "https://github.com/login/device" in rendered
    assert TOKEN not in rendered
    assert DEVICE_SECRET not in rendered
    assert github.clients[0].is_closed


def test_sync_wrapper(github: GitHub, clock: Clock) -> None:
    github.responses.extend([device_response(), token_response()])
    assert oauth.login_copilot_oauth().access_token == TOKEN
    assert oauth.get_copilot_access_token() == TOKEN


@pytest.mark.asyncio
async def test_expiry_before_polling_and_during_sleep(github: GitHub, clock: Clock) -> None:
    github.responses.append(device_response(expires_in=3))
    async with httpx.AsyncClient() as client:
        device = await oauth.request_copilot_device_code(client)
        with pytest.raises(oauth.CopilotAuthenticationError, match="expired"):
            await oauth.poll_copilot_device_code(client, device)
        assert clock.sleeps == [3]
        with pytest.raises(oauth.CopilotAuthenticationError, match="expired"):
            await oauth.poll_copilot_device_code(client, device)
    assert len(github.requests) == 1
    assert clock.sleeps == [3]


@pytest.mark.asyncio
@pytest.mark.parametrize("during_network", [False, True])
async def test_real_deadline_bounds_sleep_and_network(github: GitHub, during_network: bool) -> None:
    github.responses.append(device_response())
    github.block_token = True
    async with httpx.AsyncClient() as client:
        device = await oauth.request_copilot_device_code(client)
        device = replace(
            device, deadline=oauth.time.monotonic() + 0.03, interval=0.001 if during_network else 5
        )
        with pytest.raises(oauth.CopilotAuthenticationError, match="expired"):
            async with asyncio.timeout(1):
                await oauth.poll_copilot_device_code(client, device)
    assert github.token_started.is_set() is during_network


@pytest.mark.asyncio
@pytest.mark.parametrize("during_network", [False, True])
async def test_cancel_never_saves(
    github: GitHub, auth_file: Path, during_network: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    github.responses.append(device_response(interval=0.001 if during_network else 5))
    github.block_token = True
    printed = asyncio.Event()
    monkeypatch.setattr(oauth.console.console, "print", lambda *args, **kwargs: printed.set())
    task = asyncio.create_task(oauth.login_copilot_oauth_async())
    async with asyncio.timeout(1):
        await (github.token_started if during_network else printed).wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not auth_file.exists()
    assert github.clients[0].is_closed


def test_store_status_expiry_and_clear(clock: Clock) -> None:
    assert oauth.get_copilot_credential() is None
    assert oauth.get_copilot_access_token() is None
    assert oauth.get_copilot_token_status() == {
        "present": False,
        "source": None,
        "expires_at": None,
        "expired": False,
    }
    assert not oauth.clear_copilot_tokens()
    expires_at = clock.time() + 10
    credentials.save_oauth_credential(
        "copilot", OAuthCredential(access_token=TOKEN, expires_at=expires_at)
    )
    assert oauth.get_copilot_access_token() == TOKEN
    credential = oauth.get_copilot_credential()
    assert credential is not None
    assert credential.access_token == TOKEN
    assert credential.expires_at == expires_at
    status = oauth.get_copilot_token_status()
    assert status == {"present": True, "source": "file", "expires_at": expires_at, "expired": False}
    assert TOKEN not in str(status)
    clock.now += 10
    assert oauth.get_copilot_token_status()["expired"] is True
    for operation in (oauth.get_copilot_credential, oauth.get_copilot_access_token):
        with pytest.raises(oauth.CopilotAuthenticationError, match="expired"):
            operation()
    assert oauth.clear_copilot_tokens()
    assert oauth.get_copilot_access_token() is None


def test_environment_precedence(monkeypatch: pytest.MonkeyPatch, auth_file: Path) -> None:
    # Even a corrupt store must not be consulted if the environment key is present.
    auth_file.write_text(TOKEN)
    token = "env-synthetic-token"
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", token)
    credential = oauth.get_copilot_credential()
    assert isinstance(credential, OAuthCredential)
    assert credential.access_token == token
    assert credential.expires_at is None
    assert token not in repr(credential)
    assert oauth.get_copilot_access_token() == token
    assert oauth.get_copilot_token_status() == {
        "present": True,
        "source": "environment",
        "expires_at": None,
        "expired": False,
    }


# The OS already rejects NUL bytes in environment values.
@pytest.mark.parametrize("token", [token for token in INVALID_TOKENS if "\x00" not in token])
@pytest.mark.parametrize("operation", READ_OPERATIONS)
def test_invalid_environment_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, token: str, operation: Callable[[], object]
) -> None:
    credentials.save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN))
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", token)
    loader = Mock(side_effect=AssertionError("Environment must take precedence"))
    monkeypatch.setattr(oauth, "load_oauth_credential", loader)
    with pytest.raises(ProviderKeyError, match="COPILOT_GITHUB_TOKEN") as error:
        operation()
    assert not isinstance(error.value, oauth.CopilotAuthenticationError)
    assert_safe(error.value, *([token] if token.strip() else []))
    loader.assert_not_called()


def test_clear_preserves_environment_and_other_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials.save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN))
    credentials.save_oauth_credential("other", OAuthCredential(access_token="other-synthetic"))
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "env-synthetic-token")
    assert oauth.clear_copilot_tokens()
    assert oauth.get_copilot_access_token() == "env-synthetic-token"
    assert credentials.load_oauth_credential("other") is not None
    assert not oauth.clear_copilot_tokens()


@pytest.mark.parametrize(
    "operation",
    [*READ_OPERATIONS, oauth.clear_copilot_tokens],
)
@pytest.mark.parametrize(
    "payload", [TOKEN, '{"providers":{"copilot":{"access_token":["' + TOKEN + '"]}}}']
)
def test_corrupt_store_errors_are_sanitized(
    auth_file: Path, operation: Callable[[], object], payload: str
) -> None:
    auth_file.write_text(payload)
    with pytest.raises(ProviderKeyError) as error:
        operation()
    assert not isinstance(error.value, oauth.CopilotAuthenticationError)
    assert_safe(error.value)


@pytest.mark.parametrize("operation", [*READ_OPERATIONS, oauth.clear_copilot_tokens])
@pytest.mark.parametrize("invalid_encoding", [False, True])
def test_unreadable_store_is_sanitized(
    auth_file: Path, operation: Callable[[], object], invalid_encoding: bool
) -> None:
    if invalid_encoding:
        auth_file.write_bytes(TOKEN.encode() + b"\xff")
    else:
        auth_file.mkdir()
    with pytest.raises(ProviderKeyError) as error:
        operation()
    assert not isinstance(error.value, oauth.CopilotAuthenticationError)
    assert_safe(error.value)


@pytest.mark.asyncio
async def test_save_error_is_sanitized(github: GitHub, clock: Clock, auth_file: Path) -> None:
    auth_file.write_text(TOKEN)
    github.responses.extend([device_response(), token_response()])
    with pytest.raises(ProviderKeyError) as error:
        await oauth.login_copilot_oauth_async()
    assert not isinstance(error.value, oauth.CopilotAuthenticationError)
    assert_safe(error.value)
    assert auth_file.read_text() == TOKEN


@pytest.mark.asyncio
async def test_default_poll_interval(github: GitHub, clock: Clock) -> None:
    github.responses.extend(
        [
            httpx.Response(
                200,
                json={
                    "device_code": DEVICE_SECRET,
                    "user_code": USER_CODE,
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                },
            ),
            httpx.Response(200, json={"error": "slow_down"}),
            token_response(),
        ]
    )
    await oauth.login_copilot_oauth_async()
    assert clock.sleeps == [5, 10]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["device", "token"])
async def test_response_after_deadline_is_rejected(
    github: GitHub, clock: Clock, auth_file: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    respond = github.respond

    async def delayed_response(request: httpx.Request) -> httpx.Response:
        path = "/login/device/code" if stage == "device" else "/login/oauth/access_token"
        if request.url.path == path:
            clock.now += 901
        return await respond(request)

    monkeypatch.setattr(github, "respond", delayed_response)
    github.responses.extend([device_response(), token_response()])
    with pytest.raises(oauth.CopilotAuthenticationError, match="expired"):
        await oauth.login_copilot_oauth_async()
    assert not auth_file.exists()


def test_keyring_source_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    stored = credentials.StoredCredential(OAuthCredential(access_token=TOKEN), "keyring")
    loader = Mock(return_value=stored)
    monkeypatch.setattr(oauth, "load_oauth_credential", loader)
    assert oauth.get_copilot_access_token() == TOKEN
    assert oauth.get_copilot_token_status() == {
        "present": True,
        "source": "keyring",
        "expires_at": None,
        "expired": False,
    }
    loader.assert_called_with("copilot")


@pytest.mark.parametrize("token", INVALID_TOKENS)
@pytest.mark.parametrize("operation", READ_OPERATIONS)
def test_invalid_saved_token_is_not_a_login_failure(
    token: str, operation: Callable[[], object]
) -> None:
    credentials.save_oauth_credential("copilot", OAuthCredential(access_token=token))
    with pytest.raises(ProviderKeyError, match="Invalid saved Copilot") as error:
        operation()
    assert not isinstance(error.value, oauth.CopilotAuthenticationError)
    assert_safe(error.value, *([token] if token.strip() else []))


@pytest.mark.parametrize("expires_at", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("operation", READ_OPERATIONS)
def test_nonfinite_stored_expiry_is_invalid(
    auth_file: Path, expires_at: float, operation: Callable[[], object]
) -> None:
    # Write the external payload directly: model JSON serialization normalizes NaN to null.
    auth_file.write_text(
        json.dumps({"providers": {"copilot": {"access_token": TOKEN, "expires_at": expires_at}}})
    )
    with pytest.raises(ProviderKeyError, match="Invalid saved Copilot") as error:
        operation()
    assert not isinstance(error.value, oauth.CopilotAuthenticationError)
    assert_safe(error.value)


@pytest.mark.parametrize("operation", READ_OPERATIONS)
@pytest.mark.parametrize(
    "credential",
    [
        OAuthCredential(access_token="synthetic-token\x00"),
        OAuthCredential(access_token=TOKEN, expires_at=float("nan")),
    ],
)
def test_invalid_keyring_credential_is_not_silently_signed_out(
    monkeypatch: pytest.MonkeyPatch, operation: Callable[[], object], credential: OAuthCredential
) -> None:
    loader = Mock(return_value=credentials.StoredCredential(credential, "keyring"))
    monkeypatch.setattr(oauth, "load_oauth_credential", loader)
    with pytest.raises(ProviderKeyError, match="Invalid saved Copilot") as error:
        operation()
    assert not isinstance(error.value, oauth.CopilotAuthenticationError)
    assert_safe(error.value)
    loader.assert_called_once_with("copilot")


@pytest.mark.parametrize("operation", READ_OPERATIONS)
def test_valid_credential_is_loaded_once(
    monkeypatch: pytest.MonkeyPatch, operation: Callable[[], object]
) -> None:
    credentials.save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN))
    loader = Mock(wraps=oauth.load_oauth_credential)
    monkeypatch.setattr(oauth, "load_oauth_credential", loader)
    assert operation() is not None
    loader.assert_called_once_with("copilot")


@pytest.mark.asyncio
@pytest.mark.parametrize("token", INVALID_TOKENS)
async def test_device_token_must_be_printable_nonspace_ascii(
    github: GitHub, clock: Clock, auth_file: Path, token: str
) -> None:
    github.responses.extend([device_response(), token_response(access_token=token)])
    with pytest.raises(oauth.CopilotAuthenticationError) as error:
        await oauth.login_copilot_oauth_async()
    assert_safe(error.value, *([token] if token.strip() else []))
    assert not auth_file.exists()
