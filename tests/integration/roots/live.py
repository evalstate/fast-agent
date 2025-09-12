import asyncio

from fast_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("FastAgent Example")


# Define the agent
@fast.agent(servers=["roots_test"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.send("***CALL_TOOL roots_test-show_roots {}")


if __name__ == "__main__":
    asyncio.run(main())
