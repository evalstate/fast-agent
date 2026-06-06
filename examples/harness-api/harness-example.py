import asyncio
from pathlib import Path

from fast_agent import FastAgent

HERE = Path(__file__).parent
ENV_DIR = HERE / "harness-cards"

fast = FastAgent("Support Bot", parse_cli_args=False, quiet=True, environment_dir=ENV_DIR)


async def main() -> None:
    async with fast.harness() as harness:
        session = await harness.session("customer-123", agent_name="support")
        result = await session.generate(
            "Help this customer reset their password. Report how many times this has been asked"
        )
        print(result.last_text())
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
