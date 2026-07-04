"""Copy files through harness.local and the active execution environment."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fast_agent import FastAgent
from fast_agent.tools.environment_transfer import copy_tree

HERE = Path(__file__).parent
INPUTS = HERE / "inputs"
OUTPUTS = HERE / "outputs"

fast = FastAgent(
    "Environment artifact round-trip demo",
    parse_cli_args=False,
    quiet=True,
    home=HERE / ".fast-agent",
)


@fast.agent(
    name="worker",
    instruction="This agent exists so the harness runtime can start.",
    default=True,
)
async def worker_agent() -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", "-E", default="local")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    INPUTS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    (INPUTS / "message.txt").write_text("hello from the host\n", encoding="utf-8")

    async with fast.harness(environment=args.environment) as harness:
        await copy_tree(harness.local, "inputs", harness.environment, "work/input")
        await harness.shell("mkdir -p work/output && tr a-z A-Z < work/input/message.txt > work/output/message.txt")
        await copy_tree(harness.environment, "work/output", harness.local, "outputs")

    print((OUTPUTS / "message.txt").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    asyncio.run(main())
