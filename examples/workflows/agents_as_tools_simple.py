"""Simple Agents-as-Tools PMO example.

Parent agent ("PMO-orchestrator") calls two child agents ("NY-Project-Manager"
and "London-Project-Manager") as tools. Each child uses the ``time`` MCP
server to include local time in a brief report.

Defaults: clones fork parent history (no merge-back), no timeout, no parallel cap,
and collapses progress display after the first 20 instances.
If you want merge-back or other limits, pass decorator args:
`history_mode=HistoryMode.FORK_AND_MERGE`, `child_timeout_sec=600`,
`max_parallel=8`, `max_display_instances=10`
(HistoryMode import: fast_agent.agents.workflow.agents_as_tools_agent).
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Agents-as-Tools simple demo")


@fast.agent(
    name="NY-Project-Manager",
    instruction="Return current time and project status.",
    servers=["time"],  # MCP server 'time' configured in fastagent.config.yaml
)
@fast.agent(
    name="London-Project-Manager",
    instruction="Return current time and news.",
    servers=["time"],
)
@fast.agent(
    name="PMO-orchestrator",
    instruction="Get reports. Separate call per topic. NY: {OpenAI, Fast-Agent, Anthropic}, London: Economics",
    default=True,
    agents=[
        "NY-Project-Manager",
        "London-Project-Manager",
    ],  # children are exposed as tools: agent__NY-Project-Manager, agent__London-Project-Manager
    # optional: history_mode="fork_and_merge", child_timeout_sec=600, max_parallel=8, max_display_instances=10
)
async def main() -> None:
    async with fast.run() as agent:
        result = await agent("Get PMO report")
        await agent.interactive()
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
