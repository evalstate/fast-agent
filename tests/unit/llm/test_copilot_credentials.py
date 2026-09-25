"""Local credential and endpoint contracts, independent of gateway catalogs."""

import asyncio
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from secrets import token_urlsafe
from time import time
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import ValidationError

from fast_agent.auth.credentials import OAuthCredential, StoredCredential, save_oauth_credential
from fast_agent.config import CopilotSettings, Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError, ProviderKeyError
from fast_agent.llm.provider.copilot import broker as broker_module
from fast_agent.llm.provider.copilot import oauth
from fast_agent.llm.provider.copilot.broker import CopilotBroker
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.types import RequestParams

TOKEN = token_urlsafe()
ORIGIN = "https://copilot.invalid:8443"
CLAUDE = "claude-opus-5"
GPT = "gpt-6-astra"


@pytest.fixture(autouse=True)
def local_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", TOKEN)
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GITHUB_COPILOT_API_TOKEN"):
        monkeypatch.setenv(name, "unrelated-secret")
    forbidden = Mock(side_effect=AssertionError("Runtime must not create a network client or CLI"))
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", forbidden)


@pytest.fixture
def broker() -> CopilotBroker:
    return CopilotBroker(CopilotSettings(base_url=ORIGIN))


@pytest.mark.asyncio
async def test_native_endpoints_and_picker_credentials_need_no_network(
    broker: CopilotBroker,
) -> None:
    # This confirms local credentials only, not gateway acceptance or model entitlement.
    assert await broker.has_credentials(timeout=1.25)
    messages = await broker.resolve(CLAUDE, owner_id="messages-owner")
    responses = await broker.resolve(GPT, owner_id="responses-owner", transport="websocket")
    assert messages.model_id == CLAUDE
    assert messages.wire_api == "messages"
    assert messages.transport == "sse"
    assert messages.headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"
    assert responses.model_id == GPT
    assert responses.wire_api == "responses"
    assert responses.transport == "websocket"
    assert "anthropic-beta" not in responses.headers
    for endpoint, owner in ((messages, "messages-owner"), (responses, "responses-owner")):
        assert endpoint.base_url == ORIGIN
        assert endpoint.headers["authorization"] == f"Bearer {TOKEN}"
        assert endpoint.headers["x-interaction-id"] == owner
        assert endpoint.headers["user-agent"].startswith("fast-agent/")
        assert endpoint.headers["x-github-api-version"]
        assert endpoint.headers["openai-intent"] == "conversation-edits"
        assert endpoint.headers["copilot-integration-id"] == "copilot-sdk"
        assert "x-api-key" not in endpoint.headers
        assert all(secret not in repr(endpoint) for secret in (TOKEN, ORIGIN, owner))


@pytest.mark.asyncio
@pytest.mark.parametrize("catalog", [httpx.Response(404), httpx.Response(200, text="not JSON")])
async def test_missing_or_malformed_catalog_cannot_gate_runtime(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker, catalog: httpx.Response
) -> None:
    # Even a client that would return an unusable catalog must never be asked for it.
    get = AsyncMock(return_value=catalog)
    monkeypatch.setattr(httpx.AsyncClient, "get", get, raising=False)
    assert await broker.has_credentials()
    assert (await broker.resolve(CLAUDE, owner_id="owner")).model_id == CLAUDE
    assert (
        await broker.resolve(GPT, owner_id="owner", transport="websocket")
    ).transport == "websocket"
    get.assert_not_called()


@pytest.mark.asyncio
async def test_static_guards_reject_without_loading_credentials(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker
) -> None:
    load = Mock(side_effect=AssertionError("Invalid requests must fail before credential loading"))
    monkeypatch.setattr(oauth, "get_copilot_credential", load)
    with pytest.raises(ModelConfigError, match="Unknown Copilot model"):
        await broker.resolve("not-a-curated-model", owner_id="owner", transport="sse")
    with pytest.raises(ProviderKeyError, match="transport is not supported"):
        await broker.resolve(CLAUDE, owner_id="owner", transport="websocket")
    load.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["environment", "stored"])
async def test_rotation_and_removal_take_effect_on_every_request(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker, source: str
) -> None:
    if source == "stored":
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
        save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN))
    first = await broker.resolve(GPT, owner_id="first")
    rotated = token_urlsafe()
    if source == "stored":
        save_oauth_credential("copilot", OAuthCredential(access_token=rotated))
    else:
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", rotated)
    second = await broker.resolve(GPT, owner_id="second")
    assert first.headers["authorization"] == f"Bearer {TOKEN}"
    assert second.headers["authorization"] == f"Bearer {rotated}"
    assert first.headers["x-interaction-id"] == "first"
    assert second.headers["x-interaction-id"] == "second"
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    oauth.clear_copilot_tokens()
    assert not await broker.has_credentials()
    with pytest.raises(ProviderKeyError, match="credentials are missing"):
        await broker.resolve(GPT, owner_id="third")


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["", "   ", "bad token", "bad\ntoken", "non-ascii-é"])
async def test_invalid_override_raises_without_stored_fallback(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker, token: str
) -> None:
    save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN))
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", token)
    with pytest.raises(ProviderKeyError, match="COPILOT_GITHUB_TOKEN"):
        await broker.has_credentials()
    with pytest.raises(ProviderKeyError, match="COPILOT_GITHUB_TOKEN"):
        await broker.resolve(GPT, owner_id="owner")
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    assert await broker.has_credentials()


@pytest.mark.asyncio
async def test_valid_override_takes_precedence_over_saved_auth(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker
) -> None:
    save_oauth_credential("copilot", OAuthCredential(access_token="stored-secret"))
    endpoint = await broker.resolve(GPT, owner_id="owner")
    assert endpoint.headers["authorization"] == f"Bearer {TOKEN}"
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    endpoint = await broker.resolve(GPT, owner_id="owner")
    assert endpoint.headers["authorization"] == "Bearer stored-secret"


@pytest.mark.asyncio
async def test_stored_expiry_is_not_hidden_by_previous_resolution(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker
) -> None:
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN))
    await broker.resolve(GPT, owner_id="owner")
    save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN, expires_at=time() - 60))
    assert not await broker.has_credentials()
    with pytest.raises(ProviderKeyError, match="expired"):
        await broker.resolve(GPT, owner_id="owner")
    save_oauth_credential("copilot", OAuthCredential(access_token=TOKEN))
    assert await broker.has_credentials()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ["json", "token"])
async def test_malformed_store_surfaces_plain_provider_error_without_secret_disclosure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, broker: CopilotBroker, malformed: str
) -> None:
    await broker.resolve(GPT, owner_id="owner")
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    path = tmp_path / "auth.json"
    if malformed == "json":
        path.write_text(f"not JSON: {TOKEN}")
    else:
        save_oauth_credential("copilot", OAuthCredential(access_token=f"bad token {TOKEN}"))
    original = path.read_bytes()
    for operation in (broker.has_credentials(), broker.resolve(GPT, owner_id="owner")):
        with pytest.raises(ProviderKeyError) as error:
            await operation
        assert type(error.value) is ProviderKeyError
        assert TOKEN not in "".join(traceback.format_exception(error.value))
    assert path.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_blocked_store_read_is_read_only_and_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker, stop: str
) -> None:
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    returned = asyncio.Event()
    release = threading.Event()

    def blocked_load(provider: str) -> StoredCredential:
        assert provider == "copilot"
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(5)
            return StoredCredential(OAuthCredential(access_token=TOKEN), "keyring")
        finally:
            loop.call_soon_threadsafe(returned.set)

    monkeypatch.setattr(oauth, "load_oauth_credential", blocked_load)
    save = Mock(side_effect=AssertionError("Credential reads must not write auth"))
    monkeypatch.setattr(oauth, "save_oauth_credential", save)
    task = asyncio.create_task(broker.has_credentials(timeout=0.1 if stop == "timeout" else 30))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not returned.is_set()
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        else:
            with pytest.raises(ProviderKeyError, match="timed out"):
                await asyncio.wait_for(task, timeout=2)
        assert not returned.is_set()
    finally:
        release.set()
        await asyncio.wait_for(returned.wait(), timeout=2)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    save.assert_not_called()
    # A stopped read leaves the broker usable; the released store now answers immediately.
    assert await broker.has_credentials()


def test_native_runtime_imports_and_resolves_without_copilot_sdk_or_subprocess() -> None:
    code = """
import asyncio
import importlib.abc
import subprocess
import sys

class NoCopilotSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"copilot", "copilot_sdk"}:
            raise AssertionError("Copilot SDK import: " + fullname)

sys.meta_path.insert(0, NoCopilotSDK())

def forbidden(*args, **kwargs):
    raise AssertionError("No CLI or network allowed")

subprocess.Popen = forbidden
asyncio.create_subprocess_exec = forbidden
asyncio.create_subprocess_shell = forbidden
import httpx
httpx.AsyncClient = forbidden
httpx.Client = forbidden
from fast_agent.config import CopilotSettings
from fast_agent.llm.provider.copilot.broker import CopilotBroker

async def main():
    broker = CopilotBroker(CopilotSettings())
    assert await broker.has_credentials()
    endpoint = await broker.resolve("gpt-6-astra", owner_id="owner")
    assert endpoint.model_id == "gpt-6-astra"

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "url",
    [
        "http://copilot.invalid",
        "https://user:password@copilot.invalid",
        "https://copilot.invalid/v1",
        "https://copilot.invalid?",
        "https://copilot.invalid#fragment",
        "https://copilot.invalid:",
        "https://copilot.invalid:bad",
        "https://copilot.invalid\\evil",
        "https://copilot.invalid\n",
        "https://",
    ],
)
def test_base_url_requires_https_origin(url: str) -> None:
    with pytest.raises(ValidationError, match="HTTPS origin"):
        CopilotSettings(base_url=url)


def test_nested_environment_configures_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT__BASE_URL", f"{ORIGIN}/")
    settings = Settings()
    assert settings.copilot.base_url == ORIGIN


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_body", [False, True])
async def test_messages_adapter_uses_broker_auth_and_disables_eager_tool_streaming(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker, extra_body: bool
) -> None:
    monkeypatch.setattr(broker_module, "get_copilot_broker", Mock(return_value=broker))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-anthropic-secret")
    llm = CopilotMessagesLLM(context=Context(config=Settings()), model=CLAUDE)
    await llm._prepare_anthropic_client(CLAUDE)
    client = llm._initialize_anthropic_client()
    try:
        assert str(client.base_url).rstrip("/") == ORIGIN
        assert client.api_key == ""
        assert client.default_headers["authorization"] == f"Bearer {TOKEN}"
        assert "unrelated-anthropic-secret" not in str(client.default_headers)
        tool = {"name": "local", "input_schema": {"type": "object"}, "eager_input_streaming": True}
        base = {"model": CLAUDE, "messages": [{"role": "user", "content": "hello"}]}
        params = (
            RequestParams(metadata={"extra_body": {"tools": [tool]}})
            if extra_body
            else RequestParams()
        )
        if not extra_body:
            base["tools"] = [tool]
        arguments = llm.prepare_provider_arguments(base, params)
        tools = arguments["extra_body"]["tools"] if extra_body else arguments["tools"]
        assert tools == [{"name": "local", "input_schema": {"type": "object"}}]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_refresh_can_finish_after_broker_stops_waiting(
    monkeypatch: pytest.MonkeyPatch, broker: CopilotBroker, stop: str
) -> None:
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    save_oauth_credential(
        "copilot",
        OAuthCredential(access_token=TOKEN, refresh_token="synthetic-refresh", expires_at=1),
    )
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    saved = asyncio.Event()
    release = threading.Event()

    def post(*args: object, **kwargs: object) -> httpx.Response:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return httpx.Response(
            200,
            json={
                "access_token": "refreshed-token",
                "refresh_token": "rotated-token",
                "token_type": "bearer",
                "expires_in": 3600,
            },
        )

    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.post.side_effect = post
    monkeypatch.setattr(oauth.httpx, "Client", Mock(return_value=client))

    def save(provider: str, credential: OAuthCredential, *, source: str) -> None:
        assert source == "file"
        save_oauth_credential(provider, credential, source="file")
        loop.call_soon_threadsafe(saved.set)

    monkeypatch.setattr(oauth, "save_oauth_credential", save)
    task = asyncio.create_task(broker.has_credentials(timeout=0.1 if stop == "timeout" else 30))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ProviderKeyError, match="timed out"):
                await task
        assert not saved.is_set()
    finally:
        release.set()
        await asyncio.wait_for(saved.wait(), timeout=2)
        await asyncio.gather(task, return_exceptions=True)
    assert oauth.get_copilot_access_token() == "refreshed-token"
    client.post.assert_called_once()
