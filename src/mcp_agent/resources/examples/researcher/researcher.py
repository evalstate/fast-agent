import asyncio

from mcp_agent.core.fastagent import FastAgent

# from rich import print

agents = FastAgent(name="Researcher Agent")


@agents.agent(
    "Researcher",
    instruction="""
You are a research assistant, with access to internet search (via Brave),
website fetch, a python interpreter (you can install packages with uv) and a filesystem.
Use the current working directory to save and create files with both the Interpreter and Filesystem tools.
The interpreter has numpy, pandas, matplotlib and seaborn already installed
    """,
    servers=["brave", "interpreter", "filesystem", "fetch"],
)
async def main() -> None:
    research_prompt = """
Produce an investment report for the company Eutelsat. The final report should be saved in the filesystem in markdown format, and
contain at least the following: 
1 - A brief description of the company
2 - Current financial position (find data, create and incorporate charts)
3 - A PESTLE analysis
4 - An investment thesis for the next 3 years. Include both 'buy side' and 'sell side' arguments, and a final 
summary and recommendation.
Todays date is 15 February 2025. Include the main data sources consulted in presenting the report."""  # noqa: F841

    async with agents.run() as agent:
        await agent.prompt()


if __name__ == "__main__":
    asyncio.run(main())
