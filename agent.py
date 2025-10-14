import asyncio

from fast_agent import FastAgent

# Create the application
fast = FastAgent("fast-agent example")


default_instruction = """You are a helpful AI Agent.

{{serverInstructions}}

The current date is {{currentDate}}."""


# Define the agent
@fast.agent(instruction=default_instruction)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.send("tabulate the top 50 airports and include a small fact about the city it is closest to")
        await agent.interactive()
        await agent.send("write 10 demonstration typescript programs of around  50 lines each demonstrating different transport features")
        await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())
