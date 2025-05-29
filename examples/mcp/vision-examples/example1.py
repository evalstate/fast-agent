import asyncio
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

# Create the application
fast = FastAgent("fast-agent example")


# Define the agent
@fast.agent(instruction="You are a helpful AI Agent", servers=["filesystem"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.default.generate(
            [
                Prompt.user(
                    Path("cat.png"), "Write a report on the content of the image to 'report.md'"
                )
            ]
        )
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
