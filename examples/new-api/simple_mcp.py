"""
Demonstration of changing MCP settings dynamically.

Technically applicable to any setting, but MCP I found to be the most difficult due to MCP Aggregation at the startup.
"""

import asyncio

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.core.core_app import Core
from fast_agent.llm.model_factory import ModelFactory


async def main():
    core: Core = Core()
    await core.initialize()

    # Create agent configuration
    config = AgentConfig(
        name="dynamic_bot",
        model="gpt-4o-mini",
        servers=[  # Do not forget to add the servers here
            "fetch",
        ],
    )

    agent = McpAgent(
        config,
        connection_persistence=True,
        context=core.context,
    )

    # Attach the LLM
    await agent.attach_llm(ModelFactory.create_factory("gpt-4o-mini"))
    await agent.initialize()  # MCP Agents need to be initialized before use

    # Test the agent
    result = await agent.send(
        "Tell me about fast-agent framework. Find info at https://fast-agent.ai/ and summarize it."
    )
    print(result)
    await core.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
