"""
@agent.tool scoping example.

Demonstrates how tools can be scoped to individual agents using
@agent_func.tool, and how @fast.tool broadcasts globally.

Run with: uv run examples/function-tools/scoping.py
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Tool Scoping Example")


@fast.agent(
    name="writer",
    instruction="You are a writing assistant with translation and summarization tools.",
    default=True,
)
async def writer() -> None:
    pass


@fast.agent(
    name="analyst",
    instruction="You analyse text. You can only count words.",
)
async def analyst() -> None:
    pass


@writer.tool
def translate(text: str, language: str) -> str:
    """Translate text to the given language."""
    return f"[{language}] {text}"


@writer.tool
def summarize(text: str) -> str:
    """Produce a one-line summary."""
    return f"Summary: {text[:80]}..."


@analyst.tool(name="word_count", description="Count words in text")
def count_words(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())


async def main() -> None:
    async with fast.run() as agent:
        # "writer" sees translate and summarize (its own @writer.tool tools)
        # "analyst" sees only word_count (its own @analyst.tool tool)
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
