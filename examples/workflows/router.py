"""
Example MCP Agent application showing router workflow with decorator syntax.
Demonstrates router's ability to either:
1. Use tools directly to handle requests
2. Delegate requests to specialized agents
"""

import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent(
    "Router Workflow",
)

# Sample requests demonstrating direct tool use vs agent delegation
SAMPLE_REQUESTS = [
    "Download and summarize https://llmindset.co.uk/posts/2024/12/mcp-build-notes/",  # Router handles directly with fetch
    "Analyze the quality of the Python codebase in the current working directory",  # Delegated to code expert
    "What are the key principles of effective beekeeping?",  # Delegated to general assistant
]


@fast.agent(
    name="fetcher",
    instruction="""You are an agent, with a tool enabling you to fetch URLs.""",
    servers=["fetch"],
)
@fast.agent(
    name="code_expert",
    instruction="""You are an expert in code analysis and software engineering.
    When asked about code, architecture, or development practices,
    you provide thorough and practical insights.""",
    servers=["filesystem"],
)
@fast.agent(
    name="general_assistant",
    instruction="""You are a knowledgeable assistant that provides clear,
    well-reasoned responses about general topics, concepts, and principles.""",
)
@fast.router(
    name="route",
    model="sonnet",
    agents=["code_expert", "general_assistant", "fetcher"],
)
async def main() -> None:
    async with fast.run() as agent:
        for request in SAMPLE_REQUESTS:
            await agent.route(request)


if __name__ == "__main__":
    asyncio.run(main())
