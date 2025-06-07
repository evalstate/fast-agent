from dotenv import load_dotenv, find_dotenv
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

_ = load_dotenv(find_dotenv())

fast = FastAgent("deepinfra example")


# Define the agent
@fast.agent(instruction="You are a helpful AI Agent", servers=["filesystem", "fetch"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.default.generate(
            [
                Prompt.user(
                    "Write a beautiful poem about the ocean",
                )
            ]
        )
