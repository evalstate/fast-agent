import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application with specified model
fast = FastAgent("FastAgent Advanced Elicitation Example")


# Define the agent
@fast.agent(
    "elicit-advanced",
    servers=[
        "elicit_advanced",
    ],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        #        print("\n=== Testing Simple Rating ===")
        #        result = await agent.get_resource("elicitation://simple-rating")
        #        print(f"Result: {result}")
        await agent.send("Hello, World!")
        result = await agent.get_resource("elicitation://user-profile")
        await agent.send(result.contents[0].text)


# result = await agent.get_resource("elicitation://preferences")

# result = await agent.get_resource("elicitation://feedback")


if __name__ == "__main__":
    asyncio.run(main())
