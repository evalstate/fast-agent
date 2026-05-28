"""
Basic @fast.tool example.

Register Python functions as tools using the @fast.tool decorator.
Tools are automatically available to all agents.

Run with: uv run examples/function-tools/basic.py
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Function Tools Example")


@fast.tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Currently sunny and 22°C in {city}"


@fast.tool(name="add", description="Add two numbers together")
def add_numbers(a: int, b: int) -> int:
    return a + b


@fast.agent(instruction="You are a helpful assistant with access to tools.")
async def main() -> None:
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
