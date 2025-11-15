#!/usr/bin/env python3
"""Multi-agent test application for ACP modes integration tests."""

import asyncio

from fast_agent import FastAgent

# Create FastAgent app
app = FastAgent(
    name="multi-agent-test",
    parse_cli_args=True,
    quiet=True,
)


@app.agent(
    name="agent_one",
    instruction="This is the first agent for testing ACP modes.",
)
async def agent_one():
    """First test agent."""
    pass


@app.agent(
    name="agent_two",
    instruction="This is the second agent for testing ACP modes.",
)
async def agent_two():
    """Second test agent."""
    pass


async def main():
    """Run the multi-agent application."""
    async with app.run():
        # In server mode, this won't be reached due to SystemExit
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
