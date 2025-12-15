"""Agents-as-Tools PMO example.

PMO-orchestrator calls NY/London agents as tools; each uses the `time` MCP server.

Illustrates: routing, parallelization, and orchestrator-workers from Anthropic’s
[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents).

This pattern and sample are inspired by the OpenAI Agents SDK
[Agents as tools](https://openai.github.io/openai-agents-python/tools/#agents-as-tools) feature.
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Agents-as-Tools simple demo")


@fast.agent(
    name="NY-Project-Manager",
    instruction="Return NY time + timezone, plus a one-line project status.",
    servers=["time"],
)
@fast.agent(
    name="London-Project-Manager",
    instruction="Return London time + timezone, plus a one-line news update.",
    servers=["time"],
)
@fast.agent(
    name="PMO-orchestrator",
    instruction=(
        "Get reports. Always use one tool call per project/news. "  # parallelization
        "Responsibilities: NY projects: [OpenAI, Fast-Agent, Anthropic]. London news: [Economics, Art, Culture]. "  # routing
        "Aggregate results and add a one-line PMO summary."
    ),
    default=True,
    agents=["NY-Project-Manager", "London-Project-Manager"],  # orchestrator-workers
)
async def main() -> None:
    async with fast.run() as agent:
        await agent("Get PMO report. Projects: all. News: Art, Culture")


if __name__ == "__main__":
    asyncio.run(main())
