"""Manual demo for Skills-over-MCP discovery.

Run from this directory:
    uv run example.py

For the full interactive flow, run `uv run fast-agent --env . go` and try the slash
commands listed in README.md.
"""

from __future__ import annotations

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Skills-over-MCP demo", config_path="fast-agent.yaml")


@fast.agent(
    instruction=(
        "You are a passthrough demo agent. Available skills, if any, are "
        "listed in the system prompt."
    ),
    servers=["skill_demo"],
)
async def main() -> None:
    async with fast.run() as agent:
        print("\n=== read MCP-served skill ===")
        result = await agent.send('***CALL_TOOL read_skill {"path":"skill://pirate/SKILL.md"}')
        print(result)

        print("\n=== read MCP-served skill reference ===")
        ref = await agent.send(
            '***CALL_TOOL read_skill {"path":"skill://pirate/references/example.md"}'
        )
        print(ref)


if __name__ == "__main__":
    asyncio.run(main())
