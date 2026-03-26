"""
@fast.tool scoping example.

Demonstrates how global @fast.tool tools are available to all agents,
but agents with an explicit function_tools= list only see those tools.

Run with: uv run examples/function-tools/scoping.py
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Tool Scoping Example")


# Global tools -- available to any agent without explicit function_tools
@fast.tool
def translate(text: str, language: str) -> str:
    """Translate text to the given language."""
    return f"[{language}] {text}"


@fast.tool
def summarize(text: str) -> str:
    """Produce a one-line summary."""
    return f"Summary: {text[:80]}..."


# A standalone function used as an explicit function_tool
def word_count(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())


@fast.agent(
    name="writer",
    instruction="You are a writing assistant with translation and summarization tools.",
    default=True,
)
@fast.agent(
    name="analyst",
    instruction="You analyse text. You can only count words.",
    function_tools=[word_count],
)
async def main() -> None:
    async with fast.run() as agent:
        # "writer" sees translate and summarize (global tools)
        # "analyst" sees only word_count (explicit list overrides globals)
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
