import httpx2 as httpx
import pytest
from mcp.client.auth import OAuthFlowError, OAuthRegistrationError
from mcp.shared.auth import OAuthClientInformationFull, ProtectedResourceMetadata
from pydantic import AnyHttpUrl

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
from fast_agent.mcp.oauth_client import build_oauth_provider, compute_server_identity


def test_server_identity_canonicalizes_host_case_and_default_ports() -> None:
    lower = MCPServerSettings(
        name="lower",
        transport="http",
        url="https://example.com/api/mcp",
    )
    upper = MCPServerSettings(
        name="upper",
        transport="http",
        url="HTTPS://EXAMPLE.COM/api/mcp",
    )
    default_port = MCPServerSettings(
        name="port",
        transport="http",
        url="https://example.com:443/api/mcp",
    )

    assert {
        compute_server_identity(lower),
        compute_server_identity(upper),
        compute_server_identity(default_port),
    } == {"https://example.com/api"}


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


@pytest.mark.asyncio
async def test_build_oauth_provider_checks_endpoint_parent_and_root_prm_urls() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://example.com/api/mcp?login#fragment",
        auth=MCPServerAuthSettings(persist="memory"),
    )

    provider = build_oauth_provider(config, emit_console_output=False)

    assert provider is not None
    provider._initialized = True

    request = httpx.Request("GET", "https://example.com/api/mcp")
    flow = provider.async_auth_flow(request)

    first_request = await flow.__anext__()
    assert first_request is request

    endpoint_discovery_request = await flow.asend(httpx.Response(401, request=request))
    assert str(endpoint_discovery_request.url) == (
        "https://example.com/.well-known/oauth-protected-resource/api/mcp"
    )

    parent_discovery_request = await flow.asend(
        httpx.Response(404, request=endpoint_discovery_request)
    )
    assert str(parent_discovery_request.url) == (
        "https://example.com/.well-known/oauth-protected-resource/api"
    )

    root_discovery_request = await flow.asend(httpx.Response(404, request=parent_discovery_request))
    assert (
        str(root_discovery_request.url)
        == "https://example.com/.well-known/oauth-protected-resource"
    )

    await flow.aclose()


@pytest.mark.asyncio
async def test_oauth_registration_rejects_credentials_the_client_cannot_use() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://resource.example/api/mcp",
        auth=MCPServerAuthSettings(persist="memory", client_metadata_url=None),
    )
    provider = build_oauth_provider(config, emit_console_output=False)
    assert provider is not None
    provider._initialized = True
    assert config.url is not None

    request = httpx.Request("GET", config.url)
    flow = provider.async_auth_flow(request)
    assert await anext(flow) is request

    prm_request = await flow.asend(httpx.Response(401, request=request))
    metadata_request = await flow.asend(
        httpx.Response(
            200,
            request=prm_request,
            json={
                "resource": config.url,
                "authorization_servers": ["https://auth.example.com"],
                "scopes_supported": ["read"],
            },
        )
    )
    registration_request = await flow.asend(
        httpx.Response(
            200,
            request=metadata_request,
            json={
                "issuer": "https://auth.example.com",
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
                "registration_endpoint": "https://auth.example.com/register",
                "scopes_supported": ["read", "offline_access"],
            },
        )
    )
    assert provider.context.client_metadata.scope == "read offline_access"

    with pytest.raises(OAuthRegistrationError, match="private_key_jwt"):
        await flow.asend(
            httpx.Response(
                201,
                request=registration_request,
                json={
                    "client_id": "unusable-client",
                    "token_endpoint_auth_method": "private_key_jwt",
                },
            )
        )
    assert await provider.context.storage.get_client_info() is None


@pytest.mark.asyncio
async def test_oauth_metadata_issuer_must_match_protected_resource_metadata() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://resource.example/api/mcp",
        auth=MCPServerAuthSettings(persist="memory"),
    )
    provider = build_oauth_provider(config, emit_console_output=False)
    assert provider is not None
    provider._initialized = True
    assert config.url is not None

    request = httpx.Request("GET", config.url)
    flow = provider.async_auth_flow(request)
    assert await anext(flow) is request

    prm_request = await flow.asend(httpx.Response(401, request=request))
    metadata_request = await flow.asend(
        httpx.Response(
            200,
            request=prm_request,
            json={
                "resource": config.url,
                "authorization_servers": ["https://auth.example.com"],
            },
        )
    )

    with pytest.raises(OAuthFlowError, match="issuer mismatch"):
        await flow.asend(
            httpx.Response(
                200,
                request=metadata_request,
                json={
                    "issuer": "https://evil.example.com",
                    "authorization_endpoint": "https://evil.example.com/authorize",
                    "token_endpoint": "https://evil.example.com/token",
                },
            )
        )


@pytest.mark.asyncio
async def test_legacy_oauth_discovery_discards_credentials_bound_to_another_issuer() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://resource.example/api/mcp",
        auth=MCPServerAuthSettings(persist="memory", client_metadata_url=None),
    )
    provider = build_oauth_provider(config, emit_console_output=False)
    assert provider is not None
    provider._initialized = True
    provider.context.client_info = OAuthClientInformationFull(
        client_id="old-client",
        issuer="https://old-auth.example.com",
    )
    assert config.url is not None

    request = httpx.Request("GET", config.url)
    flow = provider.async_auth_flow(request)
    assert await anext(flow) is request

    discovery_request = await flow.asend(httpx.Response(401, request=request))
    for _ in range(3):
        next_request = await flow.asend(httpx.Response(404, request=discovery_request))
        discovery_request = next_request

    registration_request = await flow.asend(
        httpx.Response(
            200,
            request=discovery_request,
            json={
                "issuer": "https://new-auth.example.com",
                "authorization_endpoint": "https://new-auth.example.com/authorize",
                "token_endpoint": "https://new-auth.example.com/token",
                "registration_endpoint": "https://new-auth.example.com/register",
            },
        )
    )

    assert str(registration_request.url) == "https://new-auth.example.com/register"
    assert provider.context.client_info is None
    await flow.aclose()
