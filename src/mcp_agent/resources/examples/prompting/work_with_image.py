import asyncio
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

# Create the application
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent("agent", instruction="You are a helpful AI Agent", servers=["prompts"])
async def main() -> None:
    async with fast.run() as agent:
        await agent.agent.generate([Prompt.user("What's in this image?", Path("image.png"))])


if __name__ == "__main__":
    asyncio.run(main())
