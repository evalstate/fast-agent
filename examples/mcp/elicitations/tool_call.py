import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("fast-agent example")


# Define the agent
@fast.agent(
    instruction="You are a helpful AI Agent",
    servers=["elicitation_account_server"],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.send('***CALL_TOOL create_user_account {"service_name": "fast-agent"}')


if __name__ == "__main__":
    asyncio.run(main())
