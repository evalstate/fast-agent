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
@app.agent(
    name="agent_two",
    instruction="This is the second agent for testing ACP modes.",
)
async def main():
    """Entry point for the multi-agent application."""
    async with app.run():
        # When run as ACP server, this context manager handles the server lifecycle
        # The server will run until the connection closes
        pass


if __name__ == "__main__":
    asyncio.run(main())
