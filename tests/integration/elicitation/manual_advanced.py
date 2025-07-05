import asyncio

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.helpers.content_helpers import get_resource_text

# Create the application with specified model
fast = FastAgent("fast-agent elicitation example")


# Define the agent
@fast.agent(
    "elicit-advanced",
    servers=[
        "elicitation_forms_mode",
    ],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.send("Hello, World!")
        result = await agent.get_resource("elicitation://user-profile")
        await agent.send(get_resource_text(result) or "<no result>")

        result = await agent.get_resource("elicitation://preferences")
        await agent.send(get_resource_text(result) or "<no result>")

        result = await agent.get_resource("elicitation://simple-rating")
        await agent.send(get_resource_text(result) or "<no result>")

        result = await agent.get_resource("elicitation://feedback")
        await agent.send(get_resource_text(result) or "<no result>")


if __name__ == "__main__":
    asyncio.run(main())
