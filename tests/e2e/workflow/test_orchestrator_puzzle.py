# integration_tests/mcp_agent/test_agent_with_image.py

import asyncio

import pytest

from mcp_agent.core.fastagent import FastAgent


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1",
    ],
)
async def test_iterative_orchestration(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "first_two",
        instruction="You can provide the first 2 digits of the secret code.",
        model="gpt-4.1-mini",
        servers=["puzzle_1"],
    )
    @fast.agent(
        "last_two",
        instruction="You can provide the last 2 digits of the secret code.",
        model="gpt-4.1-mini",
        servers=["puzzle_2"],
    )
    @fast.agent(
        "validator",
        instruction="You can validate the 4 digit secret code.",
        model="gpt-4.1-mini",
        servers=["puzzle_validator"],
    )
    @fast.iterative_planner(
        "orchestrator",
        agents=["first_two", "last_two", "validator"],
        model="gpt-4.1",
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.orchestrator.send("find the secret code")
            assert "4712" in response

    await agent_function()


async def main():
    """Main function to run the orchestrator test from command line."""
    # Create a FastAgent instance
    fast = FastAgent("puzzler")

    # Define the agents
    @fast.agent(
        "first_two",
        instruction="You can provide the first 2 digits of the secret code.",
        model="gpt-5-mini.low",
        servers=["puzzle_1"],
    )
    @fast.agent(
        "last_two",
        instruction="You can provide the last 2 digits of the secret code.",
        model="gpt-5-mini.low",
        servers=["puzzle_2"],
    )
    @fast.agent(
        "validator",
        instruction="You can validate the 4 digit secret code.",
        model="gpt-5-mini.low",
        servers=["puzzle_validator"],
    )
    @fast.iterative_planner(
        "orchestrator",
        agents=["first_two", "last_two", "validator"],
        model="gpt-5-mini.low",
        # model="sonnet",
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.interactive()
            response = await agent.orchestrator.send("find the secret code")

            print()
            print()
            if "4712" in response:
                print("✓ Secret code found successfully!")
            else:
                print("✗ Secret code not found in response")

    await agent_function()


if __name__ == "__main__":
    asyncio.run(main())
