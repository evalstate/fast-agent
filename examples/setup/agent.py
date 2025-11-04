import asyncio

from fast_agent import FastAgent

# Create the application
fast = FastAgent("fast-agent example")


default_instruction = """You are a helpful AI Agent.

{{serverInstructions}}

{{agentSkills}}

The current date is {{currentDate}}."""


# Define the agent
@fast.agent(instruction=default_instruction)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
