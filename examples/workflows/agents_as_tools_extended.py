"""Agents-as-Tools example: project managers for NY and London.

Children are exposed as tools: agent__NY-Project-Manager, agent__London-Project-Manager.

Parent agent ("PMO-orchestrator") calls two child agents
("NY-Project-Manager" and "London-Project-Manager") as tools. 

Each child uses the `time` MCP server for local time and the `fetch` MCP server for a short
news-based update on the given topics.

Common usage patterns may combine:

- Routing: choose the right specialist tool(s) based on the user prompt.
- Parallelization: fan out over independent items/projects, then aggregate.
- Orchestrator-workers: break a task into scoped subtasks (often via a simple JSON plan), then coordinate execution.

How it works:

- You declare a parent agent with `agents=[...]`.
- At runtime it is instantiated as an internal `AgentsAsToolsAgent`, which:
  - Inherits from `McpAgent` (keeps its own MCP servers/tools).
  - Exposes each child agent as a tool (`agent__ChildName`).
  - Merges MCP tools and agent-tools in a single `list_tools()` surface.
  - Supports history/parallel controls:
    - `history_mode` (controls child history):
      - `scratch`: clones start with empty history
      - `**fork**` (default): clones start from the template child history; no merge-back
      - `fork_and_merge`: merge clone history back into the template child agent
    - `max_parallel` (default unlimited)
    - `child_timeout_sec` (default none)
    - `max_display_instances` (default 20; collapse progress after top-N)

"""

import asyncio

from fast_agent import FastAgent
from fast_agent.agents.workflow.agents_as_tools_agent import HistoryMode

# Create the application
fast = FastAgent("Agents-as-Tools demo")


@fast.agent(
    name="NY-Project-Manager",
    instruction=(
        "You are a New York project manager. For each given topic, get the "
        "current local time in New York and a brief, project-relevant news "
        "summary using the 'time' and 'fetch' MCP servers. If a source returns "
        "HTTP 403 or is blocked by robots.txt, try up to five alternative "
        "public sources before giving up and clearly state any remaining "
        "access limits. Hint: Fast-Agent site: https://fast-agent.ai"
    ),
    servers=[
        "time",
        "fetch",
    ],  # MCP servers 'time' and 'fetch' configured in fastagent.config.yaml
)
@fast.agent(
    name="London-Project-Manager",
    instruction=(
        "You are a London project manager. For each given topic, get the "
        "current local time in London and a brief, project-relevant news "
        "summary using the 'time' and 'fetch' MCP servers. If a source returns "
        "HTTP 403 or is blocked by robots.txt, try up to five alternative "
        "public sources before giving up and clearly state any remaining "
        "access limits. Hint: BBC: https://www.bbc.com/ and FT: https://www.ft.com/"
    ),
    servers=["time", "fetch"],
)
@fast.agent(
    name="PMO-orchestrator",
    instruction=(
        "Get project updates from the New York and London project managers. "
        "Ask NY-Project-Manager three times about different projects: Anthropic, "
        "evalstate/fast-agent, and OpenAI, and London-Project-Manager for economics review. "
        "Return a brief, concise combined summary with clear city/time/topic labels."
    ),
    default=True,
    agents=[
        "NY-Project-Manager",
        "London-Project-Manager",
    ],  
    # Defaults: clones fork parent history (no merge-back), no timeout, no parallel cap,
    # and collapses progress display after the first 20 instances.
    # To change behavior, pass decorator args
    history_mode=HistoryMode.SCRATCH,
    max_parallel=128,
    child_timeout_sec=120,
    max_display_instances=20
)
async def main() -> None:
    async with fast.run() as agent:
        result = await agent("pls send me daily review.")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
