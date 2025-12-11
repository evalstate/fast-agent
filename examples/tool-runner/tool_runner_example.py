"""
Example: Iterable Tool Runner

This example demonstrates how to use the ToolRunner for fine-grained control
over the tool call loop. The ToolRunner allows you to:

1. Iterate over each LLM response in the tool call loop
2. Inspect intermediate messages and tool calls
3. Modify behavior between iterations
4. Break early from the loop

This pattern is similar to Anthropic's SDK tool_runner.
"""

import asyncio

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory


# Define some simple tools
def get_weather(city: str) -> str:
    """Get the current weather in a city.

    Args:
        city: The city to check weather for

    Returns:
        Weather information for the city
    """
    # Simulated weather data
    weather_data = {
        "paris": "Partly cloudy, 18°C",
        "london": "Rainy, 12°C",
        "tokyo": "Sunny, 24°C",
        "new york": "Clear skies, 22°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


async def example_basic_iteration():
    """
    Basic example: Iterate over each message in the tool loop.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Iteration")
    print("=" * 60 + "\n")

    core = Core()
    await core.initialize()

    config = AgentConfig(name="weather_bot", model="haiku")
    agent = ToolAgent(config, tools=[get_weather, calculate_sum], context=core.context)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))

    # Create a tool runner
    runner = await agent.tool_runner(
        "What's the weather in Paris? Also, what's 15 + 27?"
    )

    # Iterate over each LLM response
    iteration = 0
    async for message in runner:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Stop reason: {message.stop_reason}")

        # Show the text content
        text = message.first_text()
        if text:
            print(f"Response: {text[:200]}...")  # Truncate for display

        # Check if there are tool calls
        if message.tool_calls:
            print(f"Tool calls: {list(message.tool_calls.keys())}")
            for tc_id, tc in message.tool_calls.items():
                print(f"  - {tc.params.name}({tc.params.arguments})")

        # Check if there's a pending tool response
        tool_response = runner.generate_tool_call_response()
        if tool_response:
            print("Tool response pending for next iteration")

    print(f"\nLoop completed after {runner.iteration} iterations")
    await core.cleanup()


async def example_until_done():
    """
    Simple example: Just run to completion with until_done().
    """
    print("\n" + "=" * 60)
    print("Example 2: Using until_done()")
    print("=" * 60 + "\n")

    core = Core()
    await core.initialize()

    config = AgentConfig(name="weather_bot", model="haiku")
    agent = ToolAgent(config, tools=[get_weather], context=core.context)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))

    # Create a runner and run to completion
    runner = await agent.tool_runner("What's the weather in Tokyo?")
    final_message = await runner.until_done()

    print(f"Final response: {final_message.first_text()}")
    print(f"Total iterations: {runner.iteration}")

    await core.cleanup()


async def example_early_break():
    """
    Example: Breaking early from the loop.
    """
    print("\n" + "=" * 60)
    print("Example 3: Breaking Early")
    print("=" * 60 + "\n")

    core = Core()
    await core.initialize()

    config = AgentConfig(name="weather_bot", model="haiku")
    agent = ToolAgent(config, tools=[get_weather], context=core.context)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))

    runner = await agent.tool_runner(
        "Check the weather in Paris, London, Tokyo, and New York"
    )

    cities_checked = 0
    async for message in runner:
        # Count tool calls in this iteration
        if message.tool_calls:
            cities_checked += len(message.tool_calls)
            print(f"Checked {cities_checked} cities so far...")

        # Break early after checking 2 cities
        if cities_checked >= 2:
            print("Breaking early - we have enough weather data!")
            break

    print(f"Stopped after {runner.iteration} iterations")
    print(f"Is done: {runner.is_done}")

    await core.cleanup()


async def example_modify_between_iterations():
    """
    Advanced example: Modifying the conversation between iterations.
    """
    print("\n" + "=" * 60)
    print("Example 4: Modifying Between Iterations")
    print("=" * 60 + "\n")

    core = Core()
    await core.initialize()

    config = AgentConfig(name="weather_bot", model="haiku")
    agent = ToolAgent(config, tools=[get_weather], context=core.context)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))

    runner = await agent.tool_runner("What's the weather in London?")

    async for message in runner:
        print(f"Iteration {runner.iteration}: {message.stop_reason}")

        # After the first tool call, add a follow-up instruction
        if runner.has_pending_tool_response and runner.iteration == 1:
            print("Adding follow-up instruction...")
            runner.append_messages(
                "After providing the weather, please also recommend what to wear."
            )

    print(f"\nFinal response: {runner.last_message.first_text()}")

    await core.cleanup()


async def example_inspect_tool_responses():
    """
    Example: Inspecting tool responses before they're sent to the LLM.
    """
    print("\n" + "=" * 60)
    print("Example 5: Inspecting Tool Responses")
    print("=" * 60 + "\n")

    core = Core()
    await core.initialize()

    config = AgentConfig(name="math_bot", model="haiku")
    agent = ToolAgent(config, tools=[calculate_sum], context=core.context)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))

    runner = await agent.tool_runner("What is 100 + 200?")

    async for message in runner:
        tool_response = runner.generate_tool_call_response()
        if tool_response and tool_response.tool_results:
            print("Tool results that will be sent to the LLM:")
            for tool_id, result in tool_response.tool_results.items():
                for content in result.content:
                    if hasattr(content, "text"):
                        print(f"  Tool {tool_id}: {content.text}")

    print(f"\nFinal answer: {runner.last_message.first_text()}")

    await core.cleanup()


async def main():
    """Run all examples."""
    print("Tool Runner Examples")
    print("=" * 60)

    await example_basic_iteration()
    await example_until_done()
    await example_early_break()
    await example_modify_between_iterations()
    await example_inspect_tool_responses()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
