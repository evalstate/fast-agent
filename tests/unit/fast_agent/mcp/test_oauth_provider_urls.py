import pytest
from mcp.shared.auth import ProtectedResourceMetadata
from pydantic import AnyHttpUrl

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
from fast_agent.mcp.oauth_client import build_oauth_provider, compute_server_identity


@pytest.mark.asyncio
async def test_build_oauth_provider_preserves_http_endpoint_for_resource_validation() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://example.com/api/mcp?login#fragment",
        auth=MCPServerAuthSettings(persist="memory"),
    )

    provider = build_oauth_provider(config, emit_console_output=False)

    assert provider is not None
    assert provider.context.server_url == "https://example.com/api/mcp"
    assert compute_server_identity(config) == "https://example.com/api"

    prm = ProtectedResourceMetadata(
        resource=AnyHttpUrl("https://example.com/api/mcp"),
        authorization_servers=[AnyHttpUrl("https://auth.example.com")],
    )
    await provider._validate_resource_match(prm)


@pytest.mark.asyncio
async def test_build_oauth_provider_preserves_sse_endpoint_for_resource_validation() -> None:
    config = MCPServerSettings(
        name="test",
        transport="sse",
        url="https://example.com/nested/sse/?login#fragment",
        auth=MCPServerAuthSettings(persist="memory"),
    )

    provider = build_oauth_provider(config, emit_console_output=False)

    assert provider is not None
    assert provider.context.server_url == "https://example.com/nested/sse"
    assert compute_server_identity(config) == "https://example.com/nested"

    prm = ProtectedResourceMetadata(
        resource=AnyHttpUrl("https://example.com/nested/sse"),
        authorization_servers=[AnyHttpUrl("https://auth.example.com")],
    )
    await provider._validate_resource_match(prm)
