"""Manual demo for structuredContent preview behavior.

Run from this directory:
    uv run example.py
"""

from __future__ import annotations

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Structured content preview demo")


@fast.agent(
    instruction=(
        "Use the passthrough model to invoke the requested tool exactly as provided."
    ),
    servers=["structured_preview"],
)
async def main() -> None:
    async with fast.run() as agent:
        print("\n=== matching text + structuredContent ===")
        matching = await agent.send("***CALL_TOOL structured_content_match {}")
        print(matching)

        print("\n=== mismatched text + structuredContent ===")
        mismatched = await agent.send("***CALL_TOOL structured_content_mismatch {}")
        print(mismatched)


if __name__ == "__main__":
    asyncio.run(main())
