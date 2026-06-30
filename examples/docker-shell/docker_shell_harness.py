"""Manual smoke test for Docker-backed harness shell execution.

Run from the repository root:

    uv run python examples/docker-shell/docker_shell_harness.py
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from fast_agent import FastAgent
from fast_agent.tools.docker_shell_environment import (
    DockerManagedShellEnvironment,
    DockerMount,
)

HERE = Path(__file__).parent

fast = FastAgent(
    "Docker shell environment demo",
    parse_cli_args=False,
    quiet=True,
    environment_dir=HERE / ".fast-agent",
)


@fast.agent(
    name="noop",
    instruction="This agent exists so the harness runtime can start; the demo uses shell only.",
    default=True,
)
async def noop_agent() -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="ubuntu:24.04", help="Docker image to run")
    parser.add_argument("--shell", default="bash", help="Shell binary inside the container")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    with TemporaryDirectory(prefix="fast-agent-docker-shell-") as workspace:
        workspace_path = Path(workspace)
        (workspace_path / "host-input.txt").write_text("hello from the host\n")

        environment = DockerManagedShellEnvironment(
            image=args.image,
            shell=args.shell,
            cwd="/workspace",
            mounts=[DockerMount(source=workspace_path, target="/workspace", mode="rw")],
        )

        async with fast.harness(environment=environment) as harness:
            info = harness.environment.runtime_info()
            print(f"runtime: {info.kind}/{info.name} provider={info.provider}")

            pwd = await harness.shell("pwd")
            print(f"pwd: {pwd.stdout.strip()} (exit {pwd.exit_code})")

            read_result = await harness.shell("cat host-input.txt")
            print(read_result.stdout.strip())

            write_result = await harness.shell(
                "printf 'hello from the container\\n' > container-output.txt"
            )
            print(f"write exit: {write_result.exit_code}")

        output = (workspace_path / "container-output.txt").read_text().strip()
        print(f"host saw: {output}")


if __name__ == "__main__":
    asyncio.run(main())
