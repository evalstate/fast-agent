import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("fast-agent example")


# Define the agent
@fast.agent(instruction="You are a helpful AI Agent", servers=["webcam", "hfspace"])
async def main():
    async with fast.run() as agent:
        await agent.interactive(
            default_prompt="take an image with the webcam, describe it to flux to "
            "reproduce it and then judge the quality of the result"
        )


if __name__ == "__main__":
    asyncio.run(main())
