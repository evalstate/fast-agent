import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application with specified model
fast = FastAgent("fast-agent elicitation example")


# Define the agent
@fast.agent(
    "elicit-advanced",
    servers=[
        "resource_forms",
    ],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.send("Hello, World!")
        result = await agent.get_resource("elicitation://user-profile")
        await agent.send(result.contents[0].text)

        result = await agent.get_resource("elicitation://preferences")
        await agent.send(result.contents[0].text)
        
        result = await agent.get_resource("elicitation://simple-rating")
        await agent.send(result.contents[0].text)
        
        result = await agent.get_resource("elicitation://feedback")
        await agent.send(result.contents[0].text)


if __name__ == "__main__":
    asyncio.run(main())
