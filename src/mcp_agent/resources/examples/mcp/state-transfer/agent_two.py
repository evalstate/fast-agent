import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("fast-agent agent_two (mcp host)")


# Define the agent
@fast.agent(name="agent_two", instruction="You are a helpful AI Agent.", servers=["agent_one"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
