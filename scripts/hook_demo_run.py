"""Run the hook demo agent to exercise hook messaging and failures."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fast_agent import FastAgent


async def main() -> None:
    fast = FastAgent(
        name="hook-demo",
        config_path="fastagent.config.yaml",
        parse_cli_args=False,
        environment_dir=Path(".dev"),
    )

    async with fast.run() as app:
        print("\n--- Hook demo: normal call ---")
        response = await app.send(
            "Call the echo_text tool with text 'hook demo', then reply with its output.",
            agent_name="hook-kimi",
        )
        print(response)

        print("\n--- Hook demo: trigger hook failure ---")
        try:
            response = await app.send(
                "Call the echo_text tool with text 'hook-fail', then reply with its output.",
                agent_name="hook-kimi",
            )
            print(response)
        except Exception as exc:  # noqa: BLE001
            print(f"Hook failure propagated to caller: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
