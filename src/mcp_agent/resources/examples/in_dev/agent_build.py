"""
This demonstrates creating multiple agents and an orchestrator to coordinate them.
"""

import asyncio

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.llm.augmented_llm import RequestParams

# Create the application
fast = FastAgent("Agent Builder")


@fast.agent(
    "agent_expert",
    instruction="""
You design agent workflows, adhering to 'Building Effective Agents' (details to follow). 

You provide concise specific guidance on design and composition. Prefer simple solutions, 
and don't nest workflows more than one level deep. 

Your objective is to produce a single '.py' agent in the style of the examples.

Keep the application simple, concentrationg on defining Agent instructions, MCP Servers and
appropriate use of Workflows. 

The style of the program should be like the examples you have been shown, with a minimum of
additional code, using only very simple Python where absolutely necessary.

Concentrate on the quality of the Agent instructions and "warmup" prompts given to them.

Keep requirements minimal: focus on building the prompts and the best workflow. The program
is expected to be adjusted and refined later.

If you are unsure about how to proceed, request input from the Human.

Use the filesystem tools to save your completed fastagent program, in an appropriately named '.py' file. 

 """,
    servers=["filesystem", "fetch"],
    request_params=RequestParams(maxTokens=8192),
)
# Define worker agents
@fast.agent(
    "requirements_capture",
    instruction="""
You help the Human define their requirements for building Agent based systems.

Keep questions short, simple and minimal, always offering to complete the questioning
if desired. If uncertain about something, respond asking the 'agent_expert' for guidance.

Do not interrogate the Human, prefer to move the process on, as more details can be requested later 
if needed. Remind the Human of this.
 """,
    human_input=True,
)
# Define the orchestrator to coordinate the other agents
@fast.orchestrator(
    name="agent_builder",
    agents=["agent_expert", "requirements_capture"],
    model="sonnet",
    plan_type="iterative",
    request_params=RequestParams(maxTokens=8192),
    plan_iterations=5,
)
async def main() -> None:
    async with fast.run() as agent:
        CODER_WARMUP = """
- Read this paper: https://www.anthropic.com/research/building-effective-agents" to understand how 
and when to use different types of Agents and Workflow types.

- Read this README https://raw.githubusercontent.com/evalstate/fast-agent/refs/heads/main/README.md file
to see how to use "fast-agent" framework.

- Look at the 'fastagent.config.yaml' file to see the available and configured MCP Servers.

"""
        await agent.agent_expert(CODER_WARMUP)

        await agent.agent_builder()


if __name__ == "__main__":
    asyncio.run(main())
