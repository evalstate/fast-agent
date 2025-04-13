# integration_tests/mcp_agent/test_agent_with_image.py
import os

import pytest


@pytest.mark.integration
@pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "haiku35",
    ],
)
async def test_router_can_use_tools(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "sunny",
        instruction="You dispense advice on clothing and activities for clement weather. Always mention 'beachball'",
        model=model_name,
    )
    @fast.agent(
        "stormy",
        instruction="You dispense advice on clothing and activities for stormy  weather. Always mention 'umbrella'",
        model=model_name,
    )
    @fast.router(
        "weather",
        instruction="Route to the most appropriate agent for the weather in the requested location",
        agents=["sunny", "stormy"],
        servers=["test_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            if os.path.exists("weather_location.txt"):
                os.remove("weather_location.txt")
            assert not os.path.exists("weather_location.txt")

            response = await agent.weather.send("advise me on what i need to visit London")
            assert os.path.exists("weather_location.txt"), (
                "File should exist after response (created by tool call)"
            )

            assert "umbrella" in response.lower()
            assert 0 == len(agent.sunny._llm.message_history)
            assert 2 == len(agent.stormy._llm.message_history)

    await agent_function()
