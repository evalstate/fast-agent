import asyncio

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.request_params import RequestParams


async def main():
    core: Core = Core()
    await core.initialize()
    test: AgentConfig = AgentConfig("hello", model="kimi")
    agent: LlmAgent = LlmAgent(test, context=core.context)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))
    await agent.send("hello world, render some xml tags both inside and outside of code fences")
    await agent.generate("write a 200 word story", RequestParams(maxTokens=50))
    await agent.generate(
        "repeat after me: `one, two, three, four`",
        RequestParams(stopSequences=[" two,"]),
    )


if __name__ == "__main__":
    asyncio.run(main())
