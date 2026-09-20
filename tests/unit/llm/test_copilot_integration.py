"""Native broker integration identity survives the real SDK wire boundaries."""

from __future__ import annotations

import socket
from importlib.metadata import version
from typing import TYPE_CHECKING, Literal, NoReturn

import httpx2
import pytest
import pytest_asyncio

from fast_agent.config import CopilotSettings, Settings
from fast_agent.constants import FAST_AGENT_AUTH_FILE
from fast_agent.context import Context
from fast_agent.llm.provider.copilot.broker import CopilotBroker
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize(
        "integration_id,expected",
        [(None, "copilot-sdk"), ("partner-agent", "partner-agent")],
        ids=["default", "custom"],
    ),
]

_TOKEN = "dummy-copilot-integration-test-token"


@pytest_asyncio.fixture
async def native_context(
    integration_id: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> AsyncIterator[Context]:
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", _TOKEN)
    for name in ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS"):
        monkeypatch.setenv(name, "Copilot-Integration-Id: ambient-integration")
    # Even a credential-loading regression must not consult the real store/keyring.
    monkeypatch.setenv(FAST_AGENT_AUTH_FILE, str(tmp_path / "auth.json"))

    def forbid_connect(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Live network access is forbidden in Copilot wire tests")

    monkeypatch.setattr(socket.socket, "connect", forbid_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_connect)
    settings = (
        CopilotSettings()
        if integration_id is None
        else CopilotSettings(integration_id=integration_id)
    )
    context = Context(config=Settings(copilot=settings))
    broker = CopilotBroker(settings)

    def get_broker(request_context: Context) -> CopilotBroker:
        assert request_context is context
        return broker

    # Only replace broker lookup, not resolution, credentials, or headers.
    monkeypatch.setattr("fast_agent.llm.provider.copilot.broker.get_copilot_broker", get_broker)
    yield context


def assert_native_identity(headers: httpx2.Headers, expected: str) -> None:
    assert headers.get_list("copilot-integration-id") == [expected]
    assert headers.get_list("authorization") == [f"Bearer {_TOKEN}"]
    assert headers.get_list("user-agent") == [f"fast-agent/{version('fast-agent-mcp')}"]
    assert "x-api-key" not in headers


@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_integration_identity_on_http_wire(
    wire: Literal["messages", "responses"],
    native_context: Context,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=b"")

    async with httpx2.MockTransport(respond) as transport:

        async def send(
            sdk_transport: httpx2.AsyncHTTPTransport, request: httpx2.Request
        ) -> httpx2.Response:
            return await transport.handle_async_request(request)

        monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", send)
        if wire == "messages":
            llm = CopilotMessagesLLM(context=native_context, model="claude-sonnet-5")
        else:
            llm = CopilotResponsesLLM(context=native_context, model="gpt-6-astra", transport="sse")
        try:
            if isinstance(llm, CopilotMessagesLLM):
                await llm._prepare_anthropic_client("claude-sonnet-5")
                arguments = llm.prepare_provider_arguments(
                    {
                        "model": "claude-sonnet-5",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    RequestParams(),
                    llm.ANTHROPIC_EXCLUDE_FIELDS,
                )
                async with llm._initialize_anthropic_client() as client:
                    async with client.messages.stream(**arguments) as stream:
                        async for _ in stream:
                            pass
            else:
                await llm._prepare_responses_client("gpt-6-astra", "sse")
                arguments = llm._build_response_args(
                    [{"role": "user", "content": "hello"}], RequestParams(), tools=None
                )
                async with llm._responses_client() as responses_client:
                    async with llm._response_sse_stream(
                        client=responses_client, arguments=arguments
                    ) as responses_stream:
                        async for _ in responses_stream:
                            pass
        finally:
            if isinstance(llm, CopilotResponsesLLM):
                await llm.close()

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith(f"/{wire}")
    assert_native_identity(request.headers, expected)


async def test_integration_identity_on_websocket_handshake(
    native_context: Context, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HandshakeCaptured(Exception):
        pass

    captured: list[tuple[str, httpx2.Headers]] = []

    async def connect(
        url: str, *, additional_headers: dict[str, str], **kwargs: object
    ) -> NoReturn:
        captured.append((url, httpx2.Headers(additional_headers)))
        raise HandshakeCaptured

    monkeypatch.setattr("openai.lib._websocket._WebSocketConnect", connect)
    llm = CopilotResponsesLLM(context=native_context, model="gpt-6-astra", transport="websocket")
    try:
        ws = await llm._responses_ws_context(
            input_items=[{"role": "user", "content": "hello"}],
            request_params=RequestParams(),
            tools=None,
            model_name="gpt-6-astra",
        )
        with pytest.raises(HandshakeCaptured):
            await llm._acquire_responses_ws_attempt(attempt=0, context=ws)
    finally:
        await llm.close()

    assert len(captured) == 1
    url, headers = captured[0]
    assert url.startswith("wss://")
    assert url.endswith("/responses")
    assert_native_identity(headers, expected)
