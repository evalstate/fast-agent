"""Interactive fast-agent session backed by a Hugging Face Sandbox.

Run from the repository root:

    uv run python examples/huggingface-sandbox/interactive.py
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fast_agent import FastAgent
from fast_agent.tools.huggingface_sandbox_environment import (
    HuggingFaceBucketMount,
    HuggingFaceSandboxEnvironment,
)

HERE = Path(__file__).parent

fast = FastAgent(
    "Hugging Face sandbox demo",
    parse_cli_args=False,
    environment_dir=HERE / ".fast-agent",
)


@fast.agent(
    name="hf_sandbox",
    instruction=(
        "You are a concise assistant. Shell commands and filesystem tools run "
        "inside the active Hugging Face Sandbox."
    ),
    shell=True,
    default=True,
)
async def hf_sandbox_agent() -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.12", help="Docker image for the sandbox")
    parser.add_argument("--flavor", default="cpu-basic", help="Hugging Face Jobs hardware flavor")
    parser.add_argument("--cwd", default="/workspace", help="Sandbox working directory")
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        metavar="SOURCE:MOUNT_PATH[:ro|rw]",
        help="Mount a Hugging Face Storage Bucket. Example: username/my-bucket:/workspace:rw",
    )
    parser.add_argument("--namespace", default=None, help="User or organization namespace")
    parser.add_argument(
        "--forward-hf-token",
        action="store_true",
        help="Forward your HF token into the sandbox as HF_TOKEN",
    )
    return parser.parse_args()


def parse_bucket_mount(raw: str) -> HuggingFaceBucketMount:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            "Bucket mount must be SOURCE:MOUNT_PATH or SOURCE:MOUNT_PATH:ro|rw"
        )
    source, mount_path = parts[0], parts[1]
    mode = parts[2] if len(parts) == 3 else "rw"
    if mode not in {"ro", "rw"}:
        raise argparse.ArgumentTypeError("Bucket mount mode must be 'ro' or 'rw'")
    return HuggingFaceBucketMount(source=source, mount_path=mount_path, read_only=mode == "ro")


async def main() -> None:
    args = parse_args()
    environment = HuggingFaceSandboxEnvironment(
        image=args.image,
        flavor=args.flavor,
        cwd=args.cwd,
        bucket_mounts=tuple(parse_bucket_mount(raw) for raw in args.bucket),
        namespace=args.namespace,
        forward_hf_token=args.forward_hf_token,
    )

    async with fast.run(environment=environment) as agent_app:
        await agent_app.interactive()


if __name__ == "__main__":
    asyncio.run(main())
