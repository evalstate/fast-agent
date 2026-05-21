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

    class FakeAsyncClient:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["httpx_timeout"] = kwargs.get("timeout")
            captured["httpx_headers"] = kwargs.get("headers")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.httpx.AsyncClient", FakeAsyncClient)

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="http://127.0.0.1:41242"),
    )
    await agent.initialize()
    try:
        client_config = captured["client_config"]
        assert client_config.supported_protocol_bindings == SUPPORTED_A2A_HTTP_TRANSPORTS
        assert captured["httpx_timeout"] == 120.0
        assert captured["httpx_headers"] is None
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_a2a_remote_agent_adds_hf_auth_headers_for_hf_space(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return AgentCard(
                name="hf-space",
                description="HF Space",
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
                        url="https://demo.hf.space/a2a/jsonrpc",
                    )
                ],
            )

    class FakeClient:
        async def close(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["httpx_headers"] = kwargs.get("headers")

        async def aclose(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["agent_card"] = agent_card
        captured["client_config"] = client_config
        return FakeClient()

    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.httpx.AsyncClient", FakeAsyncClient)

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="https://demo.hf.space"),
    )
    await agent.initialize()
    try:
        assert captured["httpx_headers"] == {
            "Authorization": "Bearer hf-test-token",
            "X-HF-Authorization": "Bearer hf-test-token",
        }
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_a2a_remote_agent_preserves_explicit_auth_headers_for_hf_space(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return AgentCard(
                name="hf-space",
                description="HF Space",
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
                        url="https://demo.hf.space/a2a/jsonrpc",
                    )
                ],
            )

    class FakeClient:
        async def close(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["httpx_headers"] = kwargs.get("headers")

        async def aclose(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["agent_card"] = agent_card
        captured["client_config"] = client_config
        return FakeClient()

    monkeypatch.setenv("HF_TOKEN", "hf-env-token")
    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.httpx.AsyncClient", FakeAsyncClient)

    explicit_headers = {"Authorization": "Bearer explicit-token"}
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="https://demo.hf.space", headers=explicit_headers),
    )
    await agent.initialize()
    try:
        assert captured["httpx_headers"] == explicit_headers
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_a2a_remote_agent_uses_configured_request_timeout(monkeypatch) -> None:
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

    class FakeAsyncClient:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["httpx_timeout"] = kwargs.get("timeout")

        async def aclose(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["agent_card"] = agent_card
        captured["client_config"] = client_config
        return FakeClient()

    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.httpx.AsyncClient", FakeAsyncClient)

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(
            url="http://127.0.0.1:41242",
            request_timeout_seconds=30.0,
        ),
    )
    await agent.initialize()
    try:
        assert captured["httpx_timeout"] == 30.0
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
