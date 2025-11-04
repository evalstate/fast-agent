"""
Integration tests for the enhanced resource API features.
"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resource_links_from_tools(fast_agent):
    """Test get_resource with explicit server parameter."""
    fast = fast_agent

    @fast.agent(name="test", servers=["linked_resource_server"])
    async def agent_function():
        async with fast.run() as agent:
            result: str = await agent.test.send("***CALL_TOOL getlink")
            # Test get_resource with explicit server parameter
            assert "A description, perhaps for the LLM" in result

    await agent_function()
