import pytest

from fast_agent.mcp.skybridge import SKYBRIDGE_MIME_TYPE


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skybridge_valid_tool_and_resource(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="skybridge_valid_agent",
        instruction="Exercise Skybridge detection for valid resources.",
        model="passthrough",
        servers=["skybridge_valid"],
    )
    async def agent_case():
        async with fast.run() as app:
            agent = app.skybridge_valid_agent
            await agent.list_mcp_tools()
            aggregator = agent._aggregator
            configs = await aggregator.get_skybridge_configs()
            config = configs["skybridge_valid"]

            assert config.supports_resources is True
            assert config.enabled is True
            assert not config.warnings
            assert len(config.ui_resources) == 1
            resource = config.ui_resources[0]
            assert resource.is_skybridge is True
            assert resource.mime_type == SKYBRIDGE_MIME_TYPE

            assert len(config.tools) == 1
            tool = config.tools[0]
            assert tool.is_valid is True
            assert tool.resource_uri == resource.uri

    await agent_case()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skybridge_invalid_mime_generates_warning(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="skybridge_invalid_mime_agent",
        instruction="Skybridge detection with invalid MIME type.",
        model="passthrough",
        servers=["skybridge_invalid_mime"],
    )
    async def agent_case():
        async with fast.run() as app:
            agent = app.skybridge_invalid_mime_agent
            await agent.list_mcp_tools()
            aggregator = agent._aggregator
            configs = await aggregator.get_skybridge_configs()
            config = configs["skybridge_invalid_mime"]

            assert config.supports_resources is True
            assert config.enabled is False
            assert config.ui_resources, "Expected to discover the ui:// resource"
            resource = config.ui_resources[0]
            assert resource.is_skybridge is False
            assert (
                resource.warning
                == "served as 'text/html' instead of 'text/html+skybridge'"
            )

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
async def test_skybridge_missing_resource_warns_and_flags_tools(fast_agent):
    fast = fast_agent

    @fast.agent(
        name="skybridge_missing_resource_agent",
        instruction="Skybridge detection with missing resource linkage.",
        model="passthrough",
        servers=["skybridge_missing_resource"],
    )
    async def agent_case():
        async with fast.run() as app:
            agent = app.skybridge_missing_resource_agent
            await agent.list_mcp_tools()
            aggregator = agent._aggregator
            configs = await aggregator.get_skybridge_configs()
            config = configs["skybridge_missing_resource"]

            assert config.enabled is True, "Valid resource should mark server as enabled"
            assert config.ui_resources, "Expected at least one Skybridge resource"
            assert any(
                "references missing Skybridge resource" in warning for warning in config.warnings
            )
            assert any(
                "no tools expose them" in warning.lower() for warning in config.warnings
            )

            assert config.tools, "Expected to capture tool metadata"
            tool = config.tools[0]
            assert tool.is_valid is False
            assert tool.warning is not None
            assert "references missing Skybridge resource" in tool.warning

    await agent_case()
