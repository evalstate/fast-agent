import asyncio
import os

from fast_agent import FastAgent

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8001"))
MODEL = os.getenv("FAST_AGENT_MODEL", os.getenv("MODEL", "gemini25"))

fast = FastAgent(
    "fast-agent facts A2A server",
    parse_cli_args=False,
    quiet=True,
)


@fast.agent(
    name="facts_agent",
    model=MODEL,
    instruction="You are a helpful agent who can provide interesting facts.",
    default=True,
)
async def facts_agent() -> None:
    """Default A2A facts agent."""
    pass


async def main() -> None:
    await fast.start_server(
        transport="a2a",
        host=HOST,
        port=PORT,
        server_name="facts_agent",
        server_description="Agent to give interesting facts.",
        instance_scope="connection",
    )


if __name__ == "__main__":
    asyncio.run(main())
