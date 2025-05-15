"""
Parallel Workflow showing Fan Out and Fan In agents, using different models
"""

import asyncio
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

# Create the application
fast = FastAgent(
    "Parallel Workflow",
)


@fast.agent(
    name="proofreader",
    instruction=""""Review the short story for grammar, spelling, and punctuation errors.
    Identify any awkward phrasing or structural issues that could improve clarity. 
    Provide detailed feedback on corrections.""",
)
@fast.agent(
    name="fact_checker",
    instruction="""Verify the factual consistency within the story. Identify any contradictions,
    logical inconsistencies, or inaccuracies in the plot, character actions, or setting. 
    Highlight potential issues with reasoning or coherence.""",
    model="gpt-4.1-mini",
)
@fast.agent(
    name="style_enforcer",
    instruction="""Analyze the story for adherence to style guidelines.
    Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
    enhance storytelling, readability, and engagement.""",
    model="sonnet",
)
@fast.agent(
    name="grader",
    instruction="""Compile the feedback from the Proofreader, Fact Checker, and Style Enforcer
    into a structured report. Summarize key issues and categorize them by type. 
    Provide actionable recommendations for improving the story, 
    and give an overall grade based on the feedback.""",
    model="o3-mini.low",
)
@fast.parallel(
    fan_out=["proofreader", "fact_checker", "style_enforcer"],
    fan_in="grader",
    name="parallel",
)
async def main() -> None:
    async with fast.run() as agent:
        await agent.parallel.send(
            Prompt.user("Student short story submission", Path("short_story.txt"))
        )


if __name__ == "__main__":
    asyncio.run(main())  # type: ignore
