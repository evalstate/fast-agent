"""
Tool Runner Example - Demonstrating fine-grained control over the tool loop.

This example shows how to use the iterable tool_runner() API to:
1. Iterate over each message from Claude, including intermediate tool_use messages
2. Inspect tool calls before and after execution
3. Modify request parameters between iterations
4. Use until_done() for simple cases

Run with: uv run python examples/tool-runner/tool_runner_example.py
"""

import asyncio

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.context import Context


# Define some simple tools
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Partly cloudy, 20°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Simple and safe evaluation for basic math
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations are allowed"
        result = eval(expression)  # Safe for numeric expressions only
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a timezone."""
    from datetime import datetime

    # Simplified - just returns UTC time
    return f"Current time ({timezone}): {datetime.utcnow().strftime('%H:%M:%S')}"


class WeatherCalculatorAgent(ToolAgent):
    """An agent with weather and calculator tools."""

    def __init__(self, config: AgentConfig, context: Context | None = None):
        tools = [get_weather, calculate, get_time]
        super().__init__(config, tools, context)


fast = FastAgent("Tool Runner Example")


@fast.custom(WeatherCalculatorAgent)
async def main() -> None:
    async with fast.run() as app:
        agent = app.default

        print("=" * 60)
        print("Example 1: Iterating over tool loop messages")
        print("=" * 60)

        # Create a tool runner and iterate over messages
        runner = agent.tool_runner(
            "What's the weather in Paris? Also, what's 15 + 27?"
        )

        iteration = 0
        async for message in runner:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            print(f"Stop reason: {message.stop_reason}")

            # Show any text content
            text = message.first_text()
            if text and text != "<no text>":
                print(f"Claude says: {text[:200]}...")

            # Show tool calls if any
            if message.tool_calls:
                print(f"Tool calls requested:")
                for tool_id, call in message.tool_calls.items():
                    print(f"  - {call.params.name}({call.params.arguments})")

                # We can inspect the pending tool response
                pending = runner.get_pending_tool_response()
                if pending and pending.tool_results:
                    print("Tool results (will be sent next iteration):")
                    for tool_id, result in pending.tool_results.items():
                        result_text = result.content[0].text if result.content else "N/A"
                        print(f"  - {tool_id}: {result_text[:100]}")

        print(f"\nFinal message: {runner.current_message.first_text()[:200]}...")
        print(f"Total iterations: {runner.iterations}")

        # Clear history for next example
        agent.clear()

        print("\n" + "=" * 60)
        print("Example 2: Using until_done() for simple cases")
        print("=" * 60)

        # For simple cases, just use until_done()
        runner = agent.tool_runner("What time is it?")
        final = await runner.until_done()
        print(f"Final response: {final.first_text()}")

        # Clear history for next example
        agent.clear()

        print("\n" + "=" * 60)
        print("Example 3: Breaking out of the loop early")
        print("=" * 60)

        runner = agent.tool_runner(
            "Get weather for Paris, London, and Tokyo, then calculate their average temperature."
        )

        async for message in runner:
            print(f"Got message with stop_reason: {message.stop_reason}")

            # Break early after first tool use
            if message.tool_calls:
                print("Breaking after first tool call...")
                break

        print(f"Stopped early. Is runner done? {runner.is_done}")


if __name__ == "__main__":
    asyncio.run(main())
