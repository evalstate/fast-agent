import pytest

from fast_agent.mcp.app_integrations import AppIntegrationKind
from fast_agent.mcp.app_integrations.mcp_apps import MCP_APPS_MIME_TYPE
from fast_agent.mcp.app_integrations.openai_apps_sdk import OPENAI_APPS_SDK_MIME_TYPE


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_apps_sdk_valid_tool_and_resource(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="openai_apps_sdk_valid_agent",
        instruction="Exercise OpenAI Apps SDK detection for valid resources.",
        model="passthrough",
        servers=["openai_apps_sdk_valid"],
    )
    async def agent_case():
        async with fast.run() as app:
            agent = app.openai_apps_sdk_valid_agent
            await agent.list_mcp_tools()
            aggregator = agent._aggregator
            configs = await aggregator.get_app_integration_configs()
            config = configs["openai_apps_sdk_valid"]

            assert config.supports_resources is True
            assert config.enabled is True
            assert not config.warnings
            assert len(config.resources) == 1
            resource = config.resources[0]
            assert resource.kind is AppIntegrationKind.OPENAI_APPS_SDK
            assert resource.mime_type == OPENAI_APPS_SDK_MIME_TYPE

            assert len(config.tools) == 1
            tool = config.tools[0]
            assert tool.is_valid is True
            assert tool.linked_resource_uri == resource.uri

    await agent_case()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_apps_sdk_invalid_mime_generates_warning(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="openai_apps_sdk_invalid_mime_agent",
        instruction="OpenAI Apps SDK detection with invalid MIME type.",
        model="passthrough",
        servers=["openai_apps_sdk_invalid_mime"],
    )
    async def agent_case():
        async with fast.run() as app:
            agent = app.openai_apps_sdk_invalid_mime_agent
            await agent.list_mcp_tools()
            aggregator = agent._aggregator
            configs = await aggregator.get_app_integration_configs()
            config = configs["openai_apps_sdk_invalid_mime"]

            assert config.supports_resources is True
            assert config.enabled is False
            assert config.resources, "Expected to discover the ui:// resource"
            resource = config.resources[0]
            assert resource.kind is None
            assert resource.warning == "served as 'text/html' instead of 'text/html+skybridge'"

            assert config.tools, "Expected to capture the tool metadata"
            tool = config.tools[0]
            assert tool.is_valid is False
            assert tool.warning is not None
            assert "served as 'text/html' instead of 'text/html+skybridge'" in tool.warning
            assert any(
                "served as 'text/html' instead of 'text/html+skybridge'" in warning
                for warning in config.warnings
            )

    await agent_case()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_apps_sdk_missing_resource_warns_and_flags_tools(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="openai_apps_sdk_missing_resource_agent",
        instruction="OpenAI Apps SDK detection with missing resource linkage.",
        model="passthrough",
        servers=["openai_apps_sdk_missing_resource"],
    )
    async def agent_case():
        async with fast.run() as app:
            agent = app.openai_apps_sdk_missing_resource_agent
            await agent.list_mcp_tools()
            aggregator = agent._aggregator
            configs = await aggregator.get_app_integration_configs()
            config = configs["openai_apps_sdk_missing_resource"]

            assert config.enabled is True, "Valid resource should mark server as enabled"
            assert config.resources, "Expected at least one OpenAI Apps SDK resource"
            assert any(
                "references missing OpenAI Apps SDK resource" in warning
                for warning in config.warnings
            )
            assert any("no tools expose them" in warning.lower() for warning in config.warnings)

            assert config.tools, "Expected to capture tool metadata"
            tool = config.tools[0]
            assert tool.is_valid is False
            assert tool.warning is not None
            assert "references missing OpenAI Apps SDK resource" in tool.warning

    await agent_case()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_apps_valid_tool_resource_and_visibility(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="mcp_apps_valid_agent",
        instruction="Exercise MCP Apps discovery for nested metadata.",
        model="passthrough",
        servers=["mcp_apps_valid"],
    )
    async def agent_case():
        async with fast.run() as app:
            agent = app.mcp_apps_valid_agent
            await agent.list_mcp_tools()
            config = await agent.aggregator.get_app_integration_config("mcp_apps_valid")

            assert config is not None
            assert config.enabled is True
            assert not config.warnings
            assert len(config.resources) == 1
            assert config.resources[0].kind is AppIntegrationKind.MCP_APPS
            assert config.resources[0].mime_type == MCP_APPS_MIME_TYPE

            assert len(config.tools) == 1
            tool = config.tools[0]
            assert tool.is_valid is True
            assert tool.kind is AppIntegrationKind.MCP_APPS
            assert tool.visibility == ["model", "app"]
            assert tool.linked_resource_uri == config.resources[0].uri

    await agent_case()
