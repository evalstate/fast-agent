from __future__ import annotations

from typing import Any

import pytest
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    AuthorizationCodeOAuthFlow,
    HTTPAuthSecurityScheme,
    OAuth2SecurityScheme,
    OAuthFlows,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import SUPPORTED_A2A_HTTP_TRANSPORTS, A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.config import MCPServerAuthSettings


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
            "X-HF-Authorization": "Bearer hf-test-token",
        }
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_a2a_remote_agent_switches_hf_space_bearer_card_to_endpoint_auth(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {"httpx_headers": []}

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return _hf_bearer_agent_card()

    class FakeClient:
        async def close(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["httpx_headers"].append(kwargs.get("headers"))

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
        assert captured["httpx_headers"] == [
            {"X-HF-Authorization": "Bearer hf-test-token"},
            {"Authorization": "Bearer hf-test-token"},
        ]
        assert captured["client_config"].httpx_client is agent._httpx_client
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_a2a_remote_agent_uses_oauth_for_hf_bearer_card_without_token(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {"httpx_auth": []}
    oauth_provider = object()

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return _hf_bearer_agent_card()

    class FakeClient:
        async def close(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["httpx_auth"].append(kwargs.get("auth"))
            captured["httpx_headers"] = kwargs.get("headers")

        async def aclose(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["agent_card"] = agent_card
        captured["client_config"] = client_config
        return FakeClient()

    def fake_build_oauth_provider(server_config: Any) -> object:
        captured["oauth_server"] = server_config
        return oauth_provider

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.get_hf_token_from_env", lambda: None)
    monkeypatch.setattr("fast_agent.mcp.hf_auth.get_hf_token_from_env", lambda *_args: None)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        "fast_agent.a2a.remote_agent.build_oauth_provider",
        fake_build_oauth_provider,
    )

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="https://demo.hf.space"),
    )
    await agent.initialize()
    try:
        assert captured["httpx_auth"] == [None, None, oauth_provider]
        assert captured["oauth_server"].url == "https://demo.hf.space"
        assert captured["client_config"].httpx_client is agent._httpx_client
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


@pytest.mark.asyncio
async def test_a2a_remote_agent_enables_oauth_for_oauth_agent_card(monkeypatch) -> None:
    captured: dict[str, Any] = {"httpx_auth": []}
    oauth_provider = object()

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return _oauth_agent_card()

    class FakeClient:
        async def close(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["httpx_auth"].append(kwargs.get("auth"))

        async def aclose(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["agent_card"] = agent_card
        captured["client_config"] = client_config
        return FakeClient()

    def fake_build_oauth_provider(server_config: Any) -> object:
        captured["oauth_server"] = server_config
        return oauth_provider

    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        "fast_agent.a2a.remote_agent.build_oauth_provider",
        fake_build_oauth_provider,
    )

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="https://agent.example.com"),
    )
    await agent.initialize()
    try:
        assert captured["httpx_auth"] == [None, oauth_provider]
        assert captured["client_config"].httpx_client is agent._httpx_client
        assert captured["oauth_server"].transport == "http"
        assert captured["oauth_server"].url == "https://agent.example.com"
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_a2a_remote_agent_no_oauth_disables_advertised_oauth(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResolver:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def get_agent_card(self) -> AgentCard:
            return _oauth_agent_card()

    class FakeClient:
        async def close(self) -> None:
            return None

    async def fake_create_client(agent_card: AgentCard, *, client_config: Any) -> FakeClient:
        captured["client_config"] = client_config
        return FakeClient()

    def fail_build_oauth_provider(_server_config: Any) -> object:
        raise AssertionError("OAuth provider should not be built")

    monkeypatch.setattr("fast_agent.a2a.remote_agent.A2ACardResolver", FakeResolver)
    monkeypatch.setattr("fast_agent.a2a.remote_agent.create_client", fake_create_client)
    monkeypatch.setattr(
        "fast_agent.a2a.remote_agent.build_oauth_provider",
        fail_build_oauth_provider,
    )

    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(
            url="https://agent.example.com",
            auth=MCPServerAuthSettings(oauth=False),
        ),
    )
    await agent.initialize()
    try:
        assert captured["client_config"].httpx_client is agent._httpx_client
    finally:
        await agent.shutdown()


def _oauth_agent_card() -> AgentCard:
    return AgentCard(
        name="oauth-agent",
        description="OAuth Agent",
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
                url="https://agent.example.com/a2a/jsonrpc",
            )
        ],
        security_requirements=[
            SecurityRequirement(schemes={"oauth": StringList(list=["openid"])})
        ],
        security_schemes={
            "oauth": SecurityScheme(
                oauth2_security_scheme=OAuth2SecurityScheme(
                    flows=OAuthFlows(
                        authorization_code=AuthorizationCodeOAuthFlow(
                            authorization_url="https://auth.example.com/authorize",
                            token_url="https://auth.example.com/token",
                        )
                    )
                )
            )
        },
    )


def _hf_bearer_agent_card() -> AgentCard:
    return AgentCard(
        name="hf-bearer-agent",
        description="HF bearer protected agent",
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
                security_requirements=[
                    SecurityRequirement(schemes={"hf_bearer": StringList(list=[])})
                ],
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url="https://demo.hf.space/a2a/jsonrpc",
            )
        ],
        security_requirements=[
            SecurityRequirement(schemes={"hf_bearer": StringList(list=[])})
        ],
        security_schemes={
            "hf_bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="HF_TOKEN",
                    description="Hugging Face bearer token",
                )
            )
        },
    )
