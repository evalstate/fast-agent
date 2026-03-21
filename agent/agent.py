import asyncio
import subprocess

from fast_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("fast-agent example",config_path="my-special-place.yaml")


# Define the agent
@fast.agent(
    instruction="You are a documentation production assistant", servers=["filesystem"]
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        # Execute shell command
        result = subprocess.run(
            ["repomix", "../fast-agent/", "repo.xml"], capture_output=True, text=True
        )
        result = result.stdout  # Or use result.stdout + result.stderr if you want both
        # You can print or process the result if needed
        print(f"Command output: {result}")

        # Continue with agent interaction if needed
        await agent()


if __name__ == "__main__":
    asyncio.run(main())
