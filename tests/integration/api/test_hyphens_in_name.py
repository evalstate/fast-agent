import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hyphenated_server_name(fast_agent):
    fast = fast_agent

    @fast.agent(name="test", instruction="here are you instructions", servers=["hyphen-test"])
    async def agent_function():
        async with fast.run() as app:
            result = await app.test.send('***CALL_TOOL check_weather {"location": "New York"}')
            assert "sunny" in result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hyphenated_tool_name(fast_agent):
    fast = fast_agent

    @fast.agent(name="test", instruction="here are you instructions", servers=["hyphen-test"])
    async def agent_function():
        async with fast.run() as app:
            result = await app.test.send("***CALL_TOOL shirt-colour {}")
            assert "polka" in result

            assert 1 == app.test.llm.usage_accumulator.cumulative_tool_calls

    await agent_function()
