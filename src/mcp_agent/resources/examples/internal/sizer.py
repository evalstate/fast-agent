import asyncio

from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("Sizer Prompt Test")


@fast.agent(
    "sizer",
    "given an object return its size",
    servers=["sizer", "category"],
    use_history=True,
)
async def main() -> None:
    async with fast.run() as agent:
        await agent()


if __name__ == "__main__":
    asyncio.run(main())
