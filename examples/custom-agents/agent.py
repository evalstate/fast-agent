import asyncio

from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("fast-agent example")


class MyAgent(BaseAgent):
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
