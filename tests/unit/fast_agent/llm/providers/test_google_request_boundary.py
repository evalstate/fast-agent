"""Exercise Google's real SDK serialization and transport boundary without credentials/network."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import httpx
import pytest
import pytest_asyncio
from google import genai
from google.genai import types as google_types
from google.oauth2.credentials import Credentials
from mcp import Tool

from fast_agent.config import GoogleSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest_asyncio.fixture
async def google_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[list[httpx.Request]]:
    requests: list[httpx.Request] = []
    injected_clients: list[httpx.AsyncClient] = []
    real_client = genai.Client

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "weather", "args": {"city": "Paris"}},
                                "thoughtSignature": "c2ln",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        if request.url.params.get("alt") == "sse":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=f"data: {json.dumps(response)}\n\n",
            )
        return httpx.Response(200, json=response)

    def create_client(
        *,
        api_key: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        http_options: google_types.HttpOptions | None = None,
    ) -> genai.Client:
        options = http_options or google_types.HttpOptions()
        # Inject only the transport; keep the SDK's real API-version resolution.
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        injected_clients.append(async_client)
        options.httpx_async_client = async_client
        options.retry_options = google_types.HttpRetryOptions(attempts=1)
        return real_client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=Credentials(token="test-token") if vertexai else None,
            http_options=options,
        )

    monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_HTTP_CLIENT", raising=False)
    monkeypatch.setattr(
        "fast_agent.llm.provider.google.llm_google_native.genai.Client", create_client
    )
    try:
        yield requests
    finally:
        # The SDK deliberately leaves externally supplied async clients open.
        for async_client in injected_clients:
            await async_client.aclose()
            assert async_client.is_closed


@pytest.mark.parametrize("model", ["gemini-3-flash-preview", "gemini-flash-latest"])
@pytest.mark.parametrize("structured", [False, True], ids=["stream", "structured"])
@pytest.mark.asyncio
async def test_developer_requests_preserve_beta_endpoint(
    google_transport: list[httpx.Request], model: str, structured: bool
) -> None:
    llm = GoogleNativeLLM(
        context=Context(config=Settings(google=GoogleSettings(api_key="test-key"))),
        model=model,
    )
    config = llm._google_generate_content_config(
        llm.get_request_params(),
        tools=None,
        available_tools=[],
        response_mime_type="application/json" if structured else None,
        response_schema=None,
        suppress_tools=False,
    )
    client = llm._initialize_google_client()
    try:
        await llm._call_google_generate_content(
            client=client,
            model_name=model,
            conversation_history=[google_types.Content(parts=[google_types.Part(text="hello")])],
            generate_content_config=config,
            response_mime_type=config.response_mime_type,
            response_schema=None,
            streaming_timeout=None,
        )
    finally:
        await client.aio.aclose()
        client.close()

    assert len(google_transport) == 1
    method = "generateContent" if structured else "streamGenerateContent"
    assert google_transport[0].url.path == f"/v1beta/models/{model}:{method}"


@pytest.mark.parametrize("vertex", [False, True], ids=["developer", "vertex"])
@pytest.mark.parametrize("tier", [None, "flex"], ids=["standard", "flex"])
@pytest.mark.asyncio
async def test_google_stream_preserves_endpoint_thinking_and_tool_signature(
    google_transport: list[httpx.Request],
    vertex: bool,
    tier: Literal["flex"] | None,
) -> None:
    settings = Settings.model_validate(
        {
            "google": {
                "api_key": "test-key",
                "base_url": "https://proxy.example",
                "default_headers": {"X-Proxy": "preserved"},
                "vertex_ai": {"enabled": vertex, "project_id": "proj", "location": "global"},
            },
        }
    )
    llm = GoogleNativeLLM(
        context=Context(config=settings), model="gemini-3.7-flash", reasoning_effort="high"
    )
    tools = (
        [
            Tool(
                name="weather",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                },
            )
        ]
        if not vertex
        else []
    )
    config = llm._google_generate_content_config(
        llm.get_request_params(),
        tools=tools,
        available_tools=llm._google_available_tools(tools, suppress_tools=False),
        response_mime_type=None,
        response_schema=None,
        suppress_tools=False,
        service_tier=tier,
        vertex_enabled=vertex,
    )
    client = llm._initialize_google_client(service_tier=tier)
    try:
        response = await llm._call_google_generate_content(
            client=client,
            model_name=llm._resolve_model_name("gemini-3.7-flash"),
            conversation_history=[google_types.Content(parts=[google_types.Part(text="weather")])],
            generate_content_config=config,
            response_mime_type=None,
            response_schema=None,
            streaming_timeout=None,
        )
    finally:
        await client.aio.aclose()
        client.close()

    assert len(google_transport) == 1
    request = google_transport[0]
    version = ("v1" if tier == "flex" else "v1beta1") if vertex else "v1beta"
    assert request.url.path.startswith(f"/{version}/")
    assert request.url.host == "proxy.example"
    assert request.headers["X-Proxy"] == "preserved"
    payload = json.loads(request.content)
    assert payload["generationConfig"]["thinkingConfig"] == {
        "include_thoughts": True,
        "thinking_level": "HIGH",
    }
    if tools:
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "weather"
        assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert response.candidates is not None
    content = response.candidates[0].content
    assert content is not None and content.parts is not None
    assert content.parts[0].thought_signature == b"sig"
    assert content.parts[0].function_call is not None
    assert content.parts[0].function_call.args == {"city": "Paris"}


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
@pytest.mark.parametrize("structured", [False, True], ids=["stream", "structured"])
@pytest.mark.asyncio
async def test_httpx_transport_errors_propagate_from_google_requests(
    google_transport: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[httpx.TransportError],
    structured: bool,
) -> None:
    attempts: list[httpx.Request] = []

    async def fail(self: httpx.MockTransport, request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise error_type("transport failed", request=request)

    monkeypatch.setattr(httpx.MockTransport, "handle_async_request", fail)
    llm = GoogleNativeLLM(
        context=Context(config=Settings(google=GoogleSettings(api_key="test-key"))),
        model="gemini-3-flash-preview",
    )
    client = llm._initialize_google_client()
    try:
        with pytest.raises(error_type, match="transport failed"):
            await llm._call_google_generate_content(
                client=client,
                model_name="gemini-3-flash-preview",
                conversation_history=[
                    google_types.Content(parts=[google_types.Part(text="hello")])
                ],
                generate_content_config=google_types.GenerateContentConfig(),
                response_mime_type="application/json" if structured else None,
                response_schema=None,
                streaming_timeout=None,
            )
    finally:
        await client.aio.aclose()
        client.close()

    assert len(attempts) == 1
    method = "generateContent" if structured else "streamGenerateContent"
    assert attempts[0].url.path.endswith(f":{method}")
