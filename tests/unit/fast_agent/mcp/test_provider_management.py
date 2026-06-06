from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import MCPServerSettings
from fast_agent.mcp.provider_management import (
    build_anthropic_provider_managed_mcp_payload,
    build_openai_provider_managed_mcp_tools,
    build_provider_managed_mcp_state,
    get_openai_connector_ids,
    has_authorization_header,
    normalize_access_token,
    normalize_client_managed_url_server,
    normalize_provider_managed_url_server,
    provider_managed_base_url,
    split_managed_server_names,
    validate_provider_managed_server_settings,
)


def test_build_provider_managed_mcp_state_reuses_exact_tool_allowlist() -> None:
    config = AgentConfig(
        name="billing",
        instruction="Use Stripe.",
        servers=["stripe"],
        tools={"stripe": ["create_payment_link", "list_products"]},
    )
    settings = {
        "stripe": MCPServerSettings(
            name="stripe",
            management="provider",
            transport="http",
            url="https://mcp.stripe.com",
            access_token="token-123",
            description="Stripe official MCP",
        )
    }

    state = build_provider_managed_mcp_state(
        agent_config=config,
        server_settings_by_name=settings,
    )

    assert [attachment.server_name for attachment in state.attachments] == ["stripe"]
    assert state.tool_allowlists["stripe"] == ("create_payment_link", "list_products")


def test_build_provider_managed_mcp_state_rejects_wildcard_tool_filters() -> None:
    config = AgentConfig(
        name="billing",
        instruction="Use Stripe.",
        servers=["stripe"],
        tools={"stripe": ["create_*"]},
    )
    settings = {
        "stripe": MCPServerSettings(
            name="stripe",
            management="provider",
            transport="http",
            url="https://mcp.stripe.com",
        )
    }

    with pytest.raises(ValueError, match="exact tool names"):
        build_provider_managed_mcp_state(
            agent_config=config,
            server_settings_by_name=settings,
        )


def test_build_provider_managed_mcp_state_rejects_prompt_filters() -> None:
    config = AgentConfig(
        name="billing",
        instruction="Use Stripe.",
        servers=["stripe"],
        prompts={"stripe": ["billing_prompt"]},
    )
    settings = {
        "stripe": MCPServerSettings(
            name="stripe",
            management="provider",
            transport="http",
            url="https://mcp.stripe.com",
        )
    }

    with pytest.raises(ValueError, match="prompt filters"):
        build_provider_managed_mcp_state(
            agent_config=config,
            server_settings_by_name=settings,
        )


def test_provider_managed_base_url_strips_endpoint_suffixes() -> None:
    assert provider_managed_base_url("https://example.com/mcp") == "https://example.com"
    assert provider_managed_base_url("https://example.com/api/mcp") == "https://example.com/api"
    assert provider_managed_base_url("https://example.com/sse") == "https://example.com"


def test_normalize_access_token_strips_bearer_prefix_case_insensitively() -> None:
    assert normalize_access_token("  BEARER token-123  ") == "token-123"


def test_has_authorization_header_normalizes_header_names() -> None:
    assert has_authorization_header({" Authorization ": "Bearer user-token"})


def test_split_managed_server_names_returns_named_groups() -> None:
    split = split_managed_server_names(
        ("local", "stripe", "missing"),
        {
            "local": MCPServerSettings(
                name="local",
                command="python",
                args=["server.py"],
            ),
            "stripe": MCPServerSettings(
                name="stripe",
                management="provider",
                transport="http",
                url="https://mcp.stripe.com",
                access_token="token-123",
            ),
        },
    )

    assert split.client_managed == ["local", "missing"]
    assert split.provider_managed == ["stripe"]


def test_validate_provider_managed_server_settings_reports_shared_constraints() -> None:
    settings = MCPServerSettings.model_construct(
        name="gmail",
        management="provider",
        connector_id="connector_gmail",
        command="python",
        args=["server.py"],
        env={"TOKEN": "secret"},
        transport="http",
        _fields_set={"connector_id", "command", "args", "env", "transport"},
    )

    validation = validate_provider_managed_server_settings(settings)

    assert validation.has_exactly_one_source()
    assert validation.invalid_fields == ("args", "command", "env", "transport")
    assert validation.missing_connector_access_token is True


def test_normalize_client_managed_url_server_returns_named_result() -> None:
    normalized = normalize_client_managed_url_server(
        transport="http",
        url="https://example.com/api",
        headers={"X-Trace": "trace-1"},
        access_token="token-123",
    )

    assert normalized.url == "https://example.com/api/mcp"
    assert normalized.headers == {
        "X-Trace": "trace-1",
        "Authorization": "Bearer token-123",
    }


@pytest.mark.parametrize(
    "helper",
    (
        lambda value: normalize_client_managed_url_server(
            transport="http",
            url=value,
            headers=None,
            access_token=None,
        ),
        lambda value: normalize_provider_managed_url_server(
            transport="http",
            url=value,
        ),
        provider_managed_base_url,
    ),
)
def test_provider_management_url_helpers_reject_non_string_urls(
    helper: "Callable[[Any], object]",
) -> None:
    with pytest.raises(TypeError, match="url must be a string"):
        helper(cast("Any", 123))


def test_build_anthropic_provider_mcp_payload() -> None:
    config = AgentConfig(
        name="billing",
        instruction="Use Stripe.",
        servers=["stripe"],
        tools={"stripe": ["create_payment_link"]},
    )
    settings = {
        "stripe": MCPServerSettings(
            name="stripe",
            management="provider",
            transport="http",
            url="https://mcp.stripe.com",
            access_token="token-123",
        )
    }

    state = build_provider_managed_mcp_state(
        agent_config=config,
        server_settings_by_name=settings,
    )
    payload = build_anthropic_provider_managed_mcp_payload(state)

    assert payload.servers == [
        {
            "type": "url",
            "name": "stripe",
            "url": "https://mcp.stripe.com/mcp",
            "authorization_token": "token-123",
        }
    ]
    assert payload.tools == [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "stripe",
            "default_config": {"enabled": False},
            "configs": {"create_payment_link": {"enabled": True}},
        }
    ]


def test_build_openai_provider_mcp_tools() -> None:
    config = AgentConfig(
        name="billing",
        instruction="Use Stripe.",
        servers=["stripe"],
        tools={"stripe": ["create_payment_link"]},
    )
    settings = {
        "stripe": MCPServerSettings(
            name="stripe",
            management="provider",
            transport="http",
            url="https://mcp.stripe.com",
            access_token="token-123",
            description="Stripe official MCP",
            defer_loading=True,
        )
    }

    state = build_provider_managed_mcp_state(
        agent_config=config,
        server_settings_by_name=settings,
    )
    tools = build_openai_provider_managed_mcp_tools(state)

    assert tools == [
        {
            "type": "mcp",
            "server_label": "stripe",
            "server_url": "https://mcp.stripe.com/mcp",
            "require_approval": "never",
            "server_description": "Stripe official MCP",
            "authorization": "token-123",
            "allowed_tools": ["create_payment_link"],
            "defer_loading": True,
        }
    ]


def test_build_provider_managed_mcp_state_preserves_provider_endpoint_url() -> None:
    config = AgentConfig(
        name="billing",
        instruction="Use provider-managed MCP.",
        servers=["stripe"],
    )
    settings = {
        "stripe": MCPServerSettings(
            name="stripe",
            management="provider",
            transport="http",
            url="https://example.com/api/mcp",
        )
    }

    state = build_provider_managed_mcp_state(
        agent_config=config,
        server_settings_by_name=settings,
    )

    assert state.attachments[0].server_url == "https://example.com/api/mcp"


def test_get_openai_connector_ids_reads_sdk_literals() -> None:
    assert "connector_dropbox" in get_openai_connector_ids()


def test_build_openai_provider_mcp_tools_supports_connectors() -> None:
    config = AgentConfig(
        name="mail",
        instruction="Use Gmail.",
        servers=["gmail"],
        tools={"gmail": ["search_gmail"]},
    )
    settings = {
        "gmail": MCPServerSettings(
            name="gmail",
            management="provider",
            connector_id="connector_gmail",
            access_token="token-123",
            description="Gmail connector",
            defer_loading=True,
        )
    }

    state = build_provider_managed_mcp_state(
        agent_config=config,
        server_settings_by_name=settings,
    )
    tools = build_openai_provider_managed_mcp_tools(state)

    assert tools == [
        {
            "type": "mcp",
            "server_label": "gmail",
            "connector_id": "connector_gmail",
            "require_approval": "never",
            "server_description": "Gmail connector",
            "authorization": "token-123",
            "allowed_tools": ["search_gmail"],
            "defer_loading": True,
        }
    ]


def test_build_anthropic_provider_mcp_payload_rejects_connectors() -> None:
    config = AgentConfig(
        name="mail",
        instruction="Use Gmail.",
        servers=["gmail"],
    )
    settings = {
        "gmail": MCPServerSettings(
            name="gmail",
            management="provider",
            connector_id="connector_gmail",
            access_token="token-123",
        )
    }

    state = build_provider_managed_mcp_state(
        agent_config=config,
        server_settings_by_name=settings,
    )

    with pytest.raises(ValueError, match="only supported for the OpenAI Responses provider"):
        build_anthropic_provider_managed_mcp_payload(state)
