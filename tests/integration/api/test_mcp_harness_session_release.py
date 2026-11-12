from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.harness_app import AgentRuntimeEnvironment, AppOpenRequest
from fast_agent.mcp.server.harness_app_server import (
    HarnessMCPAppServer,
    HarnessMCPAppServerOptions,
)
from fast_agent.types import AgentRequest, AgentResponse


class _EchoSession:
    @property
    def agent_app(self) -> AgentApp:
        return AgentApp({})

    @property
    def env(self) -> AgentRuntimeEnvironment:
        raise AssertionError("not used")

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse.text(request.session_id or "none")


class _EchoApp:
    def __init__(self) -> None:
        self.opened: list[str | None] = []

    @asynccontextmanager
    async def open(self, request: AppOpenRequest | None = None) -> AsyncGenerator[_EchoSession]:
        self.opened.append(request.session_id if request else None)
        yield _EchoSession()


async def _run_mcp_session(arguments: dict[str, str]) -> tuple[list[str], list[str | None]]:
    app_under_test = _EchoApp()
    released: list[str] = []

    async def release(session_id: str) -> None:
        released.append(session_id)

    server = HarnessMCPAppServer(
        app_under_test,
        HarnessMCPAppServerOptions(
            server_name="release-test",
            default_agent="worker",
            release_session=release,
        ),
    )
    server.adapter.register_default_tools(server.mcp_server)
    app = server.mcp_server.http_app(transport="http")
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async with streamable_http_client("http://testserver/mcp", http_client=client) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    await session.call_tool("send", arguments)
                    await session.call_tool("send", arguments)
                    assert released == []
        # Client exit sends DELETE; the server unwinds the connection asynchronously.
        with anyio.fail_after(5):
            while not released and not arguments.get("session_id"):
                await anyio.sleep(0.01)
    return released, app_under_test.opened


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ending_mcp_session_releases_its_harness_session_once() -> None:
    released, opened = await _run_mcp_session({"message": "hello"})

    assert opened[0] is not None
    assert opened == [opened[0], opened[0]]
    assert released == [opened[0]]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ending_mcp_session_keeps_explicitly_requested_harness_session() -> None:
    released, opened = await _run_mcp_session({"message": "hello", "session_id": "client-owned"})

    assert opened == ["client-owned", "client-owned"]
    assert released == []
