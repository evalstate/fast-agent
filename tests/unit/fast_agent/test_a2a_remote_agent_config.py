from __future__ import annotations

from typing import Any

import pytest
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import SUPPORTED_A2A_HTTP_TRANSPORTS, A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType


@pytest.mark.asyncio
async def test_a2a_remote_agent_defaults_to_supported_http_transports(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return AgentCard(
                name="http-only",
                description="HTTP+JSON only",
                provider=AgentProvider(organization="test", url="https://example.com"),
                version="1.0.0",
                capabilities=AgentCapabilities(streaming=True, push_notifications=False),
                default_input_modes=["text"],
                default_output_modes=["text"],
                skills=[
                    AgentSkill(
                        id="echo",
                        name="Echo",
                        description="Echo input",
                        tags=["test"],
                        examples=["hello"],
                        input_modes=["text"],
                        output_modes=["text"],
                    )
                ],
                supported_interfaces=[
                    AgentInterface(
                        protocol_binding="HTTP+JSON",
                        protocol_version="1.0",
                        url="http://127.0.0.1:41242/a2a/rest",
                    )
                ],
            )

    class FakeClient:
        async def close(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["agent_card"] = agent_card
        captured["client_config"] = client_config
        return FakeClient()

    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="http://127.0.0.1:41242"),
    )
    await agent.initialize()
    try:
        client_config = captured["client_config"]
        assert client_config.supported_protocol_bindings == SUPPORTED_A2A_HTTP_TRANSPORTS
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_a2a_remote_agent_honors_explicit_transport(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return AgentCard(
                name="jsonrpc",
                description="JSON-RPC",
                provider=AgentProvider(organization="test", url="https://example.com"),
                version="1.0.0",
                capabilities=AgentCapabilities(streaming=True, push_notifications=False),
                default_input_modes=["text"],
                default_output_modes=["text"],
                skills=[
                    AgentSkill(
                        id="echo",
                        name="Echo",
                        description="Echo input",
                        tags=["test"],
                        examples=["hello"],
                        input_modes=["text"],
                        output_modes=["text"],
                    )
                ],
                supported_interfaces=[
                    AgentInterface(
                        protocol_binding="JSONRPC",
                        protocol_version="1.0",
                        url="http://127.0.0.1:41242/a2a/jsonrpc",
                    )
                ],
            )

    class FakeClient:
        async def close(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["agent_card"] = agent_card
        captured["client_config"] = client_config
        return FakeClient()

    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="http://127.0.0.1:41242", transport="JSONRPC"),
    )
    await agent.initialize()
    try:
        client_config = captured["client_config"]
        assert client_config.supported_protocol_bindings == ["JSONRPC"]
    finally:
        await agent.shutdown()
