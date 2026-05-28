from __future__ import annotations

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import MCPServerSettings
from fast_agent.mcp.provider_management import (
    build_anthropic_provider_managed_mcp_payload,
    build_openai_provider_managed_mcp_tools,
    build_provider_managed_mcp_state,
    get_openai_connector_ids,
    provider_managed_base_url,
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
    mcp_servers, tools = build_anthropic_provider_managed_mcp_payload(state)

    assert mcp_servers == [
        {
            "type": "url",
            "name": "stripe",
            "url": "https://mcp.stripe.com/mcp",
            "authorization_token": "token-123",
        }
    ]
    assert tools == [
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
