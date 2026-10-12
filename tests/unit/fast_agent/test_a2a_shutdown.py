from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, create_autospec

import httpx
import pytest
from a2a.client import Client
from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "close_error", [None, RuntimeError("transport close failed"), asyncio.CancelledError()]
)
async def test_shutdown_releases_http_client_and_finalizes_agent(monkeypatch, close_error) -> None:
    http_client = httpx.AsyncClient()
    client = create_autospec(Client, instance=True)
    client.close.side_effect = close_error
    card = AgentCard(
        name="remote",
        description="Remote agent",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        supported_interfaces=[
            AgentInterface(url="http://example.test", protocol_binding="HTTP+JSON")
        ],
    )
    monkeypatch.setattr(
        "fast_agent.a2a.remote_agent.httpx.AsyncClient", lambda **kwargs: http_client
    )
    monkeypatch.setattr(
        "fast_agent.a2a.remote_agent.A2ACardResolver.get_agent_card", AsyncMock(return_value=card)
    )
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", AsyncMock(return_value=client))
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A),
        a2a_config=A2AAgentConfig(url="http://example.test"),
    )
    await agent.initialize()

    try:
        if close_error is None:
            await agent.shutdown()
        else:
            with pytest.raises(type(close_error)) as raised:
                await agent.shutdown()
            assert raised.value is close_error

        assert http_client.is_closed
        assert not agent.initialized
        # Shutdown remains safe after a failed close and does not retry the dead transport.
        await agent.shutdown()
        client.close.assert_awaited_once()
    finally:
        await http_client.aclose()
