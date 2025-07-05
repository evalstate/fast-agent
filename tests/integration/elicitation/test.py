import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application with specified model
fast = FastAgent("FastAgent Elicitation Example")


# Define the agent
@fast.agent(
    "elicit-me",
    servers=[
        "elicitation_test",
    ],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:

        await agent.send("foo")
        result = await agent.get_resource("elicitation://generate")
        print(f"RESULT: {result}")


if __name__ == "__main__":
    asyncio.run(main())
