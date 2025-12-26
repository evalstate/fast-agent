import asyncio

from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.types import PromptMessageExtended


def lookup_order_status(order_id: str) -> str:
    return f"Order {order_id} is packed and ready to ship."


async def main() -> None:
    core: Core = Core()
    await core.initialize()

    config = AgentConfig(name="order_bot")
    agent = ToolAgent(config, tools=[lookup_order_status], context=core.context)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))

    messages = [
        PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Check order 12345, then summarize in one line.")
            ],
        )
    ]

    runner = ToolRunner(
        agent=agent,
        messages=messages,
    )

    async for assistant_message in runner:
        text = assistant_message.last_text() or "<no text>"
        print(f"[assistant] {text}")

    await core.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
