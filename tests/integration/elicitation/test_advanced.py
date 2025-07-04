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
        
        print("\n=== Testing User Profile ===")
        result = await agent.get_resource("elicitation://user-profile")
        print(f"Result: {result}")
        
        print("\n=== Testing Preferences ===")
        result = await agent.get_resource("elicitation://preferences")
        print(f"Result: {result}")
        
        print("\n=== Testing Feedback ===")
        result = await agent.get_resource("elicitation://feedback")
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())