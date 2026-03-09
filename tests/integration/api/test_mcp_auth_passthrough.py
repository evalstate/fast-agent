from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.server.agent_server import AgentMCPServer

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


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


class _TokenEchoAgent:
    def __init__(self) -> None:
        self.config = SimpleNamespace(default_request_params=None, description=None)

    async def send(self, message: str, request_params=None) -> str:
        del message, request_params
        return request_bearer_token.get() or "missing"

    async def shutdown(self) -> None:
        return None


async def _build_server() -> AgentMCPServer:
    agent = _TokenEchoAgent()

    async def create_instance() -> AgentInstance:
        wrapped = cast("AgentProtocol", agent)
        app = AgentApp({"worker": wrapped})
        return AgentInstance(app=app, agents={"worker": wrapped})

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    primary = await create_instance()
    return AgentMCPServer(
        primary_instance=primary,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        host="testserver",
    )


async def _call_worker_tool(
    headers: dict[str, str],
    *,
    wrap_hf_auth_headers: bool = False,
) -> str:
    with _temporary_env(
        FAST_AGENT_SERVE_OAUTH="huggingface",
        FAST_AGENT_OAUTH_RESOURCE_URL="http://testserver",
    ):
        server = await _build_server()
        starlette_app = server.mcp_server.streamable_http_app()
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
async def test_streamable_http_authorization_header_token_reaches_agent_context() -> None:
    response_text = await _call_worker_tool({"Authorization": "Bearer integration-token"})

    assert response_text == "integration-token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streamable_http_hf_header_is_normalized_and_reaches_agent_context() -> None:
    response_text = await _call_worker_tool(
        {"X-HF-Authorization": "Bearer hf-space-token"},
        wrap_hf_auth_headers=True,
    )

    assert response_text == "hf-space-token"
