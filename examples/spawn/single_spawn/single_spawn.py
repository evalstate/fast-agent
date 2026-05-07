"""Single Agent Spawn — minimal example.

Demonstrates spawning an isolated sub-agent in a subprocess.
"""

import asyncio

from fast_agent.spawn.isolated_spawner import run_isolated_agent


async def main() -> None:
    result = await run_isolated_agent(
        task="What is the capital of France? Reply in one sentence.",
        project_dir=".",
        instruction="You are a geography expert.",
        timeout_seconds=60,
        role="geo-agent",
    )

    print(f"Status: {result['status']}")
    print(f"Result: {result['result']}")


if __name__ == "__main__":
    asyncio.run(main())
