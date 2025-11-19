import pytest

from fast_agent.mcp import SEP


@pytest.mark.integration
@pytest.mark.asyncio
async def test_roots_returned(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="foo", instruction="bar", servers=["roots_test"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent.foo.send(f"***CALL_TOOL roots_test{SEP}show_roots {{}}")
            assert "file:///mnt/data/" in result  # alias
            assert "test_data" in result
            assert "file://no/alias" in result  # no alias.

    await agent_function()
