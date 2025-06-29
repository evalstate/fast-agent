import asyncio

from mcp_agent.core.fastagent import FastAgent

agents = FastAgent(name="Researcher Agent (EO)")


@agents.agent(
    name="Researcher",
    instruction="""
You are a research assistant, with access to internet search (via Brave),
website fetch, a python interpreter (you can install packages with uv) and a filesystem.
Use the current working directory to save and create files with both the Interpreter and Filesystem tools.
The interpreter has numpy, pandas, matplotlib and seaborn already installed.

You must always provide a summary of the specific sources you have used in your research.
    """,
    servers=["brave", "interpreter", "filesystem", "fetch"],
)
@agents.agent(
    name="Evaluator",
    model="sonnet",
    instruction="""
Evaluate the response from the researcher based on the criteria:
 - Sources cited. Has the researcher provided a summary of the specific sources used in the research?
 - Validity. Has the researcher cross-checked and validated data and assumptions.
 - Alignment. Has the researher acted and addressed feedback from any previous assessments?
 
For each criterion:
- Provide a rating (EXCELLENT, GOOD, FAIR, or POOR).
- Offer specific feedback or suggestions for improvement.

Summarize your evaluation as a structured response with:
- Overall quality rating.
- Specific feedback and areas for improvement.""",
)
@agents.evaluator_optimizer(
    generator="Researcher",
    evaluator="Evaluator",
    max_refinements=5,
    min_rating="EXCELLENT",
    name="Researcher_Evaluator",
)
async def main() -> None:
    async with agents.run() as agent:
        await agent.prompt("Researcher_Evaluator")

        print("Ask follow up quesions to the Researcher?")
        await agent.prompt("Researcher", default_prompt="STOP")


if __name__ == "__main__":
    asyncio.run(main())
