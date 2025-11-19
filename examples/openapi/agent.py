import asyncio

from fast_agent import FastAgent
from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION

# Create the application
fast = FastAgent("fast-agent example")


default_instruction = DEFAULT_AGENT_INSTRUCTION


# Define the agent
@fast.agent(instruction=default_instruction,servers=["petstore"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
