from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
from fastmcp.server.auth import AccessToken
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fast_agent.core.harness_app import AppOpenRequest
from fast_agent.mcp.auth.huggingface import HuggingFaceOAuthOrHubTokenVerifier
from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.server.harness_app_server import (
    HarnessMCPAppServer,
    HarnessMCPAppServerOptions,
    ManagedAgentToolSpec,
)
from fast_agent.types import AgentRequest, AgentResponse


@contextmanager
def _temporary_env(**env_vars: str) -> Iterator[None]:
    import os

    originals = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            os.environ[key] = value
        yield
    finally:
        for key, original in originals.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


class _TokenEchoHarnessSession:
    @property
    def agent_app(self) -> object:
        return object()

    @property
    def env(self) -> object:
        return object()

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        token = request.auth.token if request.auth is not None else None
        return AgentResponse.text(token or "missing")


class _TokenEchoHarnessApp:
    @asynccontextmanager
    async def open(
        self,
        request: AppOpenRequest | None = None,
    ) -> AsyncIterator[_TokenEchoHarnessSession]:
        del request
        yield _TokenEchoHarnessSession()


@pytest.fixture(autouse=True)
def _mock_hf_token_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    async def verify_token(self: object, token: str) -> AccessToken | None:
        del self
        if token == "invalid-token":
            return None
        return AccessToken(token=token, client_id="test-client", scopes=["access"])

    monkeypatch.setattr(HuggingFaceOAuthOrHubTokenVerifier, "verify_token", verify_token)


def _build_server() -> HarnessMCPAppServer:
    return HarnessMCPAppServer(
        _TokenEchoHarnessApp(),
        HarnessMCPAppServerOptions(
            server_name="auth-test",
            default_agent="worker",
            managed_agent_tools=(
                ManagedAgentToolSpec(
                    name="worker",
                    agent="worker",
                    description="Echo the request token.",
                ),
            ),
        ),
    )


async def _call_send_tool(
    headers: dict[str, str],
    *,
    wrap_hf_auth_headers: bool = False,
) -> str:
    with _temporary_env(
        FAST_AGENT_SERVE_OAUTH="huggingface",
        FAST_AGENT_OAUTH_RESOURCE_URL="http://testserver",
    ):
        server = _build_server()
        starlette_app = server.mcp_server.http_app(
            transport="http",
            middleware=server._http_middleware(),
        )
        transport_app = (
            HFAuthHeaderMiddleware(starlette_app) if wrap_hf_auth_headers else starlette_app
        )

        async with starlette_app.router.lifespan_context(starlette_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=transport_app),
                base_url="http://testserver",
                headers=headers,
            ) as client:
                async with streamable_http_client(
                    "http://testserver/mcp",
                    http_client=client,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.call_tool("worker", {"message": "hello"})

        assert result.content
        return get_text(result.content[0]) or ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streamable_http_authorization_header_token_reaches_harness_request() -> None:
    response_text = await _call_send_tool({"Authorization": "Bearer integration-token"})

    assert response_text == "integration-token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streamable_http_hf_header_is_normalized_and_reaches_harness_request() -> None:
    response_text = await _call_send_tool(
        {"X-HF-Authorization": "Bearer hf-space-token"},
        wrap_hf_auth_headers=True,
    )

    assert response_text == "hf-space-token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streamable_http_rejects_invalid_hf_bearer_token() -> None:
    with pytest.raises(Exception):
        await _call_send_tool({"Authorization": "Bearer invalid-token"})
