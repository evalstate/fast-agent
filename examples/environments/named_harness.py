"""Use a named execution environment from .fast-agent/fast-agent.yaml."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fast_agent import FastAgent

HERE = Path(__file__).parent

fast = FastAgent(
    "Named environment harness demo",
    parse_cli_args=False,
    quiet=True,
    home=HERE / ".fast-agent",
)


@fast.agent(
    name="demo",
    instruction="This agent exists so the harness runtime can start.",
    default=True,
)
async def demo_agent() -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        "-E",
        default=None,
        help="Configured environment name. Defaults to default_environment.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    async with fast.harness(environment=args.environment) as harness:
        info = harness.environment.runtime_info()
        result = await harness.shell("pwd")
        print(f"environment: {info.environment_name or '<instance>'} ({info.kind})")
        print(f"cwd: {result.stdout.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
