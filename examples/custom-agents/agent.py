import asyncio

from fast_agent import FastAgent
from fast_agent.agents import McpAgent

# Create the application
fast = FastAgent("fast-agent example")


class MyAgent(McpAgent):
    async def initialize(self):
        await super().initialize()
        print("it's a-me!...Mario!")


# Define the agent
@fast.custom(MyAgent, instruction="You are a helpful AI Agent")
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
