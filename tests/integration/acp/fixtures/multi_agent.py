"""
Multi-agent test fixture for ACP mode switching tests.

This defines multiple agents to test mode switching functionality.
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent(
    "multi-agent-test",
    parse_cli_args=False,
    quiet=True,
)


@fast.agent(
    name="code_expert",
    instruction="You are an expert in code analysis and software engineering.\nYou help with code reviews, refactoring, and best practices.",
    model="passthrough",
)
async def code_expert_main():
    """Code expert agent for testing mode switching."""
    pass


@fast.agent(
    name="general_assistant",
    instruction="You are a knowledgeable general assistant.\nYou help with a wide variety of tasks and questions.",
    model="passthrough",
    default=True,
)
async def general_assistant_main():
    """General assistant agent for testing mode switching."""
    pass


async def main():
    """Run the multi-agent test server."""
    async with fast.run():
        # Server mode is activated via args, not here
        pass


if __name__ == "__main__":
    asyncio.run(main())
