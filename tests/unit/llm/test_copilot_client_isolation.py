"""Exercise Copilot's real SDK clients, replacing only transport I/O."""

from __future__ import annotations

import os
from typing import Literal

import anthropic
import httpcore2
import httpx2
import openai
import pytest

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("status", [200, 307, 308])
async def test_sdk_wire_isolation(
    wire: Literal["messages", "responses"], status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient_headers = "\n".join(
        f"{name}: ambient-secret"
        for name in (
            "Authorization",
            "aUtHoRiZaTiOn",
            "X-Api-Key",
            "Cookie",
            "Host",
            "Proxy-Authorization",
            "OpenAI-Organization",
            "OpenAI-Project",
            "X-Unrecognized-Ambient-Header",
        )
    )
    for name in ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS"):
        monkeypatch.setenv(name, ambient_headers)
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_ADMIN_KEY",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
    ):
        monkeypatch.setenv(name, "ambient-secret")
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "http://ambient-proxy.invalid:8080")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    environment = dict(os.environ)

    model = "claude-sonnet-5" if wire == "messages" else "gpt-6-astra"
    endpoint: CopilotEndpoint

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            assert model_id == model
            assert transport == "sse"
            return endpoint

    def get_broker(context: Context) -> Broker:
        return Broker()

    monkeypatch.setattr("fast_agent.llm.provider.copilot.broker.get_copilot_broker", get_broker)
    context = Context(config=Settings())
    llm = (
        CopilotMessagesLLM(context=context, model=model)
        if wire == "messages"
        else CopilotResponsesLLM(context=context, model=model)
    )
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if status != 200:
            return httpx2.Response(status, headers={"location": "https://redirect.invalid/stolen"})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=b"")

    mock_transport = httpx2.MockTransport(respond)
    http_client: httpx2.AsyncClient

    async def send(
        transport: httpx2.AsyncHTTPTransport, request: httpx2.Request
    ) -> httpx2.Response:
        # Keep SDK-created transports/mounts intact: selecting a proxy must fail.
        # Mock *all* HTTP transports so even a regression cannot use the network.
        assert transport is http_client._transport
        return await mock_transport.handle_async_request(request)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", send)
    sdk = anthropic if wire == "messages" else openai
    # Rebind one adapter across broker generations to catch client/header carryover.
    for generation in (1, 2):
        endpoint = CopilotEndpoint(
            model_id=model,
            wire_api=wire,
            transport="sse",
            base_url="https://copilot.example/v1",
            headers={
                "X-Broker-Header": "trusted",
                "authorization": f"Bearer broker-token-{generation}",
            },
        )
        if isinstance(llm, CopilotMessagesLLM):
            await llm._prepare_anthropic_client(model)
            client = llm._initialize_anthropic_client()
        else:
            await llm._prepare_responses_client(model, "sse")
            client = llm._responses_client()
        http_client = client._client
        assert not http_client.trust_env
        assert not http_client.follow_redirects
        assert http_client.timeout == sdk.DEFAULT_TIMEOUT
        transport = http_client._transport
        assert isinstance(transport, httpx2.AsyncHTTPTransport)
        assert isinstance(transport._pool, httpcore2.AsyncConnectionPool)
        assert transport._pool._max_connections == sdk.DEFAULT_CONNECTION_LIMITS.max_connections
        assert (
            transport._pool._max_keepalive_connections
            == sdk.DEFAULT_CONNECTION_LIMITS.max_keepalive_connections
        )

        async def request_stream() -> None:
            if isinstance(client, anthropic.AsyncAnthropic):
                async with client.messages.stream(
                    model=model, max_tokens=16, messages=[{"role": "user", "content": "hello"}]
                ) as stream:
                    async for _ in stream:
                        pass
            else:
                assert isinstance(llm, CopilotResponsesLLM)
                async with llm._response_sse_stream(
                    client=client, arguments={"model": model, "input": "hello"}
                ) as stream:
                    async for _ in stream:
                        pass

        requests.clear()
        async with client:
            if status == 200:
                await request_stream()
            else:
                with pytest.raises(sdk.APIStatusError) as error:
                    await request_stream()
                assert error.value.status_code == status
        assert http_client.is_closed
        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == f"https://copilot.example/v1/{wire}"
        assert request.headers["host"] == "copilot.example"
        assert request.headers["x-broker-header"] == "trusted"
        assert all("ambient-secret" not in value for value in request.headers.values())
        for name in ("cookie", "proxy-authorization", "x-unrecognized-ambient-header"):
            assert name not in request.headers
        for name in ("openai-organization", "openai-project"):
            assert not request.headers.get(name)
        assert request.headers.get_list("authorization") == [f"Bearer broker-token-{generation}"]
        assert "x-api-key" not in request.headers
        assert dict(os.environ) == environment


@pytest.mark.asyncio
async def test_sdk_private_attribute_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the private SDK attributes the adapters rely on so an SDK bump fails loudly."""
    ambient = "X-Ambient: ambient-secret\nAuthorization: ambient-secret"
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", ambient)
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", ambient)
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient-proxy.invalid:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    context = Context(config=Settings())

    def endpoint(model: str, wire: Literal["messages", "responses"]) -> CopilotEndpoint:
        return CopilotEndpoint(
            model_id=model,
            wire_api=wire,
            transport="sse",
            base_url="https://copilot.example",
            headers={"authorization": "Bearer broker-token", "x-broker": "trusted"},
        )

    messages = CopilotMessagesLLM(context=context, model="claude-sonnet-5")
    messages._copilot_endpoint.set(endpoint("claude-sonnet-5", "messages"))
    anthropic_client = messages._initialize_anthropic_client()
    responses = CopilotResponsesLLM(context=context, model="gpt-6-astra")
    responses._copilot_endpoint.set(endpoint("gpt-6-astra", "responses"))
    openai_client = responses._responses_client()
    async with anthropic_client, openai_client:
        assert anthropic_client._custom_headers == {
            "authorization": "Bearer broker-token",
            "x-broker": "trusted",
            "X-Api-Key": anthropic.omit,
        }
        assert openai_client._custom_headers == {
            "authorization": "Bearer broker-token",
            "x-broker": "trusted",
        }
        for client in (anthropic_client, openai_client):
            effective = client.default_headers
            assert effective["authorization"] == "Bearer broker-token"
            assert "x-ambient" not in {name.lower() for name in effective}
            assert "ambient-secret" not in str(effective)
        assert anthropic_client._client._mounts == {}
        # AsyncOpenAI must keep honouring the private credential-enforcement switch:
        # the adapter's empty key is only accepted because it opts out.
        assert openai_client.api_key == ""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
        with pytest.raises(openai.OpenAIError, match="Missing credentials"):
            openai.AsyncOpenAI(api_key="", base_url="https://copilot.example")
        openai.AsyncOpenAI(
            api_key="", base_url="https://copilot.example", _enforce_credentials=False
        )
