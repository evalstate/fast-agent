import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent(servers=["category", "mcp_hfspace","mcp_webcam"])
#@fast.agent(name="test")
async def main() -> None:
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
#        await agent.prompt(agent_name="test")
        await agent()


if __name__ == "__main__":
    asyncio.run(main())
