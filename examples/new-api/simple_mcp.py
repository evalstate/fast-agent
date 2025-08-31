import asyncio

from fast_agent.core import Core


async def main():
    core: Core = Core()
    await core.initialize()

    # # Create agent configuration
    # config = AgentConfig(name="weather_bot", model="haiku")

    # tool_agent = McpAgent(
    #     config,
    #     context=core.context,
    # )

    # # Attach the LLM
    # await tool_agent.attach_llm(ModelFactory.create_factory("haiku"))

    # # Test the agent
    # await tool_agent.send("What's the weather like in San Francisco and what's the temperature?")
    # await core.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
