"""
Simple test agent for integration testing.
"""

import asyncio
import sys

from fast_agent import FastAgent

# Create the application
fast = FastAgent("Integration Test Agent")


# Define a simple agent
@fast.agent(
    name="test",  # Important: This name matches what we use in the CLI test
    instruction="You are a test agent that simply echoes back any input received.",
)
async def main() -> None:
    async with fast.run() as agent:
        # This executes only for interactive mode, not needed for command-line testing
        if sys.stdin.isatty():  # Only run interactive mode if attached to a terminal
            user_input = input("Enter a message: ")
            response = await agent.send(user_input)
            print(f"Agent response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
