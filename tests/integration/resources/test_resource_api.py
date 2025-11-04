"""
Integration tests for the enhanced resource API features.
"""

import pytest
from mcp.shared.exceptions import McpError

from fast_agent.mcp.prompts.prompt_helpers import get_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_resource_with_explicit_server(fast_agent):
    """Test get_resource with explicit server parameter."""
    fast = fast_agent

    @fast.agent(name="test", servers=["resource_server_one", "resource_server_two"])
    async def agent_function():
        async with fast.run() as agent:
            # Test get_resource with explicit server parameter
            resource = await agent.test.get_resource(
                "resource://fast-agent/r1file1.txt", "resource_server_one"
            )
            assert "test 1" == get_text(resource.contents[0])

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_resource_with_auto_server(fast_agent):
    """Test get_resource with automatic server selection."""
    fast = fast_agent

    @fast.agent(name="test", servers=["resource_server_one", "resource_server_two"])
    async def agent_function():
        async with fast.run() as agent:
            # Test get_resource with auto server selection (should use first server)
            resource = await agent.test.get_resource("resource://fast-agent/r2file1.txt")
            assert "test 3" == get_text(resource.contents[0])

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_resources(fast_agent):
    """Test list_resources API functionality."""
    fast = fast_agent

    @fast.agent(name="test", servers=["resource_server_one", "resource_server_two"])
    async def agent_function():
        async with fast.run() as agent:
            # Test list_resources with explicit server
            resources = await agent.test.list_resources("resource_server_one")

            assert "resource_server_one" in resources

            # Verify some test files are in the list
            resource_list = resources["resource_server_one"]
            assert any("resource://fast-agent/r1file1.txt" in r for r in resource_list)
            assert any("resource://fast-agent/r1file2.txt" in r for r in resource_list)

            # Test list_resources without server parameter
            all_resources = await agent.test.list_resources()
            assert all_resources is not None
            assert "resource_server_one" in all_resources
            assert "resource_server_two" in all_resources

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling(fast_agent):
    """Test error handling for nonexistent resources and servers."""
    fast = fast_agent

    @fast.agent(name="test", servers=["resource_server_one"])
    async def agent_function():
        async with fast.run() as agent:
            # Test nonexistent resource
            with pytest.raises(McpError) as exc_info:
                await agent.test.get_resource(
                    "resource://fast-agent/nonexistent.txt", "resource_server_one"
                )
                assert True

            # Test nonexistent server
            with pytest.raises(ValueError) as exc_info:
                await agent.test.get_resource(
                    "resource://fast-agent/r1file1.txt", "nonexistent_server"
                )

            assert (
                "server" in str(exc_info.value).lower()
                and "not found" in str(exc_info.value).lower()
            )

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_resource_api(fast_agent):
    """Test with_resource API with new parameter ordering."""
    fast = fast_agent

    @fast.agent(name="test", servers=["resource_server_one"], model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            # Test with explicit server parameter
            response = await agent.test.with_resource(
                "Reading resource content:",
                "resource://fast-agent/r1file1.txt",
                "resource_server_one",
            )
            assert response is not None

            # Test with another resource
            response = await agent.test.with_resource(
                "Reading resource content:",
                "resource://fast-agent/r1file2.txt",
                "resource_server_one",
            )
            assert response is not None

            # Test with auto server selection
            response = await agent.test.with_resource(
                "Reading resource content:", "resource://fast-agent/r1file1.txt"
            )
            assert response is not None

    await agent_function()
