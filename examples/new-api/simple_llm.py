import asyncio

from mcp.server.fastmcp import FastMCP

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory

# Initialize FastMCP instance for decorator-based tools
# Set log_level to WARNING or ERROR to avoid httpx INFO logs
mcp = FastMCP("Weather Bot", log_level="WARNING")


# Option 1: Using @mcp.tool decorator
@mcp.tool()
async def check_weather(city: str) -> str:
    """Check the weather in a given city.

    Args:
        city: The city to check the weather for

    Returns:
        Weather information for the city
    """
    return f"The weather in {city} is sunny."


# Option 2: Simple function-based tool (without decorator)
async def check_weather_function(city: str) -> str:
    """Check the weather in a given city (function version).

    Args:
        city: The city to check the weather for

    Returns:
        Weather information for the city
    """
    return f"The weather in {city} is sunny."


# Alternative: Regular (non-async) function also works
def get_temperature(city: str) -> int:
    """Get the temperature in a city.

    Args:
        city: The city to get temperature for

    Returns:
        Temperature in degrees Celsius
    """
    return 22


async def main():
    core: Core = Core()
    await core.initialize()

    # Create agent configuration
    config = AgentConfig(name="weather_bot", model="haiku")

    tool_agent = ToolAgent(
        config,
        tools=[
            check_weather,
            get_temperature,
        ],
        context=core.context,
    )

    # Attach the LLM
    await tool_agent.attach_llm(ModelFactory.create_factory("haiku"))

    # Test the agent
    await tool_agent.send("What's the weather like in San Francisco and what's the temperature?")
    await core.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
