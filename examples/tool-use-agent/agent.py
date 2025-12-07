"""Minimal tool-call example using the ToolRunner-enabled ToolAgent.

This shows the latest pattern: defining tools with the `@tool` decorator and
letting ToolAgent run the tool loop (now parallel-capable) automatically.
"""

import asyncio

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.context import Context
from fast_agent.tools.tool_runner import tool


@tool(description="Return the transcript for a video call by ID")
def get_video_call_transcript(video_id: str) -> str:
    return (
        "Assistant: Hi, how can I assist you today?\n\n"
        "Customer: Hi, I wanted to ask you about last invoice I received..."
    )


class CustomToolAgent(ToolAgent):
    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
    ):
        tools = [get_video_call_transcript]
        super().__init__(config, tools, context)


fast = FastAgent("Example Tool Use Application")


@fast.custom(CustomToolAgent)
async def main() -> None:
    async with fast.run() as agent:
        await agent.default.generate("What is the topic of the video call no.1234?")


if __name__ == "__main__":
    asyncio.run(main())
