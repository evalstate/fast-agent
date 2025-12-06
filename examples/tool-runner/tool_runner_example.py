"""
Example demonstrating the ToolRunner for fine-grained control over tool loops.

The ToolRunner is an async iterator that yields each message from Claude
during a tool-calling conversation, allowing you to:
- Observe intermediate messages as they're generated
- Break out of the loop early
- Customize parameters between iterations
- Manually handle tool responses before they're sent back
"""

import asyncio

from fast_agent import FastAgent

# Create the FastAgent app
fast = FastAgent("Tool Runner Example")


# Define some simple tools
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city to check weather for

    Returns:
        Weather information
    """
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Partly cloudy, 20°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate

    Returns:
        The result of the calculation
    """
    try:
        # Simple eval for demo - in production use a proper math parser
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@fast.agent(
    name="assistant",
    instruction="You are a helpful assistant with access to weather and calculator tools.",
    tools=[get_weather, calculate],
)
async def main():
    async with fast.run() as app:
        print("=" * 60)
        print("Example 1: Basic iteration - observe each step")
        print("=" * 60)

        runner = app.tool_runner(
            "What's the weather in Paris and London? Also, what's 15 * 7?"
        )

        iteration = 0
        async for message in runner:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            print(f"Stop reason: {message.stop_reason}")

            # Show text content if any
            text = message.last_text()
            if text:
                print(f"Response: {text[:200]}..." if len(text) > 200 else f"Response: {text}")

            # Show tool calls if any
            if message.tool_calls:
                print(f"Tool calls: {list(message.tool_calls.keys())}")
                for call_id, call in message.tool_calls.items():
                    print(f"  - {call.params.name}({call.params.arguments})")

        print(f"\nCompleted in {runner.iterations} iterations")

        print("\n" + "=" * 60)
        print("Example 2: Using until_done() for simple cases")
        print("=" * 60)

        # Reset by creating a new runner
        runner2 = app.tool_runner("What's 100 + 200?")
        final = await runner2.until_done()
        print(f"Final answer: {final.last_text()}")

        print("\n" + "=" * 60)
        print("Example 3: Early exit from the loop")
        print("=" * 60)

        runner3 = app.tool_runner(
            "Check weather in Paris, London, Tokyo, and New York, then summarize."
        )

        async for message in runner3:
            print(f"Got message with stop_reason: {message.stop_reason}")

            # Exit after first tool call round
            if runner3.iterations >= 1 and message.tool_calls:
                print("Exiting early after first tool call...")
                break

        print(f"Exited after {runner3.iterations} iteration(s)")


if __name__ == "__main__":
    asyncio.run(main())
