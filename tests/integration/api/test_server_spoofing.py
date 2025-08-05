import pytest


@pytest.mark.integration
@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_spoofing(fast_agent):
    @fast_agent.agent(instruction="You are a helpful AI Agent", servers=["spoof_test"])
    async def agent_function():
        async with fast_agent.run() as agent:
            implementation = await agent.send("***CALL_TOOL implementation {}")
            assert "spoof" in implementation
            assert "9.9.9" in implementation

    await agent_function()
