"""Interactive fast-agent session with the execute tool running in Docker.

Run from the repository root:

    uv run python examples/docker-shell/interactive.py
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from fast_agent import FastAgent
from fast_agent.tools.docker_shell_environment import DockerMountedEnvironment

HERE = Path(__file__).parent

fast = FastAgent(
    "Docker interactive shell demo",
    parse_cli_args=False,
    home=HERE / ".fast-agent",
)


@fast.agent(
    name="docker",
    instruction=(
        "You are a concise assistant. When you use the execute tool, commands run "
        "inside a Docker container mounted at /workspace, not on the host shell."
    ),
    shell=True,
    default=True,
)
async def docker_agent() -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="ubuntu:24.04", help="Docker image to run")
    parser.add_argument("--shell", default="bash", help="Shell binary inside the container")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Host directory to mount at /workspace. Defaults to a temporary directory.",
    )
    return parser.parse_args()


async def run_interactive(workspace: Path, *, image: str, shell: str) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "README.txt").write_text(
        "This file is mounted from the host into the Docker shell environment.\n"
    )

    environment = DockerMountedEnvironment(
        image=image,
        shell=shell,
        workspace=workspace,
        target="/workspace",
    )

    async with fast.run(environment=environment) as agent_app:
        await agent_app.interactive()


async def main() -> None:
    args = parse_args()
    if args.workspace is not None:
        await run_interactive(args.workspace.resolve(), image=args.image, shell=args.shell)
        return

    with TemporaryDirectory(prefix="fast-agent-docker-interactive-") as workspace:
        await run_interactive(Path(workspace), image=args.image, shell=args.shell)


if __name__ == "__main__":
    asyncio.run(main())
