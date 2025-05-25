"""
This demonstrates creating multiple agents and an orchestrator to coordinate them.
"""

import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Orchestrator-Workers")


@fast.agent(
    "author",
    instruction="""You are to role play a poorly skilled writer, 
    who makes frequent grammar, punctuations and spelling errors. You enjoy
    writing short stories, but the narrative doesn't always make sense""",
    servers=["filesystem"],
)
# Define worker agents
@fast.agent(
    name="finder",
    instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
    servers=["fetch", "filesystem"],
    model="gpt-4o",
)
@fast.agent(
    name="writer",
    instruction="""You are an agent that can write to the filesystem.
            You are tasked with taking the user's input, addressing it, and 
            writing the result to disk in the appropriate location.""",
    servers=["filesystem"],
)
@fast.agent(
    name="proofreader",
    instruction=""""Review the short story for grammar, spelling, and punctuation errors.
            Identify any awkward phrasing or structural issues that could improve clarity. 
            Provide detailed feedback on corrections.""",
    servers=["fetch"],
    model="gpt-4o",
)
# Define the orchestrator to coordinate the other agents
@fast.orchestrator(
    name="orchestrate",
    agents=["finder", "writer", "proofreader"],
    plan_type="full",
)
async def main() -> None:
    async with fast.run() as agent:
        await agent.author(
            "write a 250 word short story about kittens discovering a castle, and save it to short_story.md"
        )

        # The orchestrator can be used just like any other agent
        task = """Load the student's short story from short_story.md, 
        and generate a report with feedback across proofreading, 
        factuality/logical consistency and style adherence. Use the style rules from 
        https://apastyle.apa.org/learn/quick-guide-on-formatting and 
        https://apastyle.apa.org/learn/quick-guide-on-references.
        Write the graded report to graded_report.md in the same directory as short_story.md"""

        await agent.orchestrate(task)


if __name__ == "__main__":
    asyncio.run(main())
