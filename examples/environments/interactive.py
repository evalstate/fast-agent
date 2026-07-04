"""Interactive fast-agent prompt using a configured execution environment."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fast_agent import FastAgent

HERE = Path(__file__).parent

fast = FastAgent(
    "Named environment interactive demo",
    parse_cli_args=False,
    home=HERE / ".fast-agent",
)


@fast.agent(
    name="workspace",
    instruction=(
        "You are a concise workspace assistant. Shell commands run in the "
        "configured execution environment selected at startup."
    ),
    shell=True,
    default=True,
    model="codexplan"
)
async def workspace_agent() -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", "-E", default="local")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    async with fast.run(environment=args.environment) as agent_app:
        await agent_app.interactive()


if __name__ == "__main__":
    asyncio.run(main())
