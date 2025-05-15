import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.llm.augmented_llm import RequestParams
from mcp_agent.mcp.prompts.prompt_load import load_prompt_multipart

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Create the application
fast = FastAgent("Data Analysis (Roots)")


# The sample data is under Database Contents License (DbCL) v1.0.

# Available here : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# The CSS files are distributed under the MIT License from the excellent
# marpstyle project : https://github.com/cunhapaulo/marpstyle


@fast.agent(
    name="slides",
    servers=["filesystem"],
    instruction="""
You produce compelling slide decks for impactful presentations. You usually try and keep the pack to between
8-12 slides, with key insights at the start, backed up with data, diagrams and analysis as available. You 
are able to help direct colour, style and and questions for enhancing the presentation. Produced charts and
visualisations will be in the ./mount-point/ directory. You output MARP markdown files.
""",
    request_params=RequestParams(maxTokens=8192),
)
@fast.agent(
    name="data_analysis",
    instruction="""
You have access to a Python 3.12 interpreter and you can use this to analyse and process data. 
Common analysis packages such as Pandas, Seaborn and Matplotlib are already installed. 
You can add further packages if needed.
Data files are accessible from the /mnt/data/ directory (this is the current working directory).
Visualisations should be saved as .png files in the current working directory.
""",
    servers=["interpreter"],
    request_params=RequestParams(maxTokens=8192),
)
@fast.orchestrator(
    name="orchestrator",
    plan_type="iterative",
    agents=["slides", "data_analysis"],
)
async def main() -> None:
    # Use the app's context manager
    async with fast.run() as agent:
        prompts: list[PromptMessageMultipart] = load_prompt_multipart(Path("slides.md"))
        await agent.slides.apply_prompt_messages(prompts)

        await agent.orchestrator.send(
            "Produce a compelling presentation for the CSV data file in the /mnt/data/ directory."
            "The slides agent will produce a presentation, make sure to consult it first for "
            "colour scheme and formatting guidance. Make sure that any 'call-outs' have a distinct"
            "background to ensure they stand out."
            "Make sure the presentation is impactful, concise and visualises key insights in to the data"
            " in a compelling way."
            "The presentation is by the 'llmindset team' and produced in 'march 2025'."
            "Produce it step-by-step; long responses without checking in are likely to exceed"
            "maximum output token limits."
        )
        # colours: str = await agent.slides.send("Tell the Data Analysis agent what colour schemes and chart sizes you prefer for the presentation")

        # analysis: str = await agent.data_analysis.send(
        #     "Examine the CSV file in /mnt/data, produce a detailed analysis of the data,"
        #     "and any patterns it contains. Visualise some of the key points, saving .png files to"
        #     "your current workig folder (/mnt/data). Respond with a summary of your findings, and a list"
        #     "of visualiations and their filenames ready to incorporate in to a slide deck. The presentation agent has"
        #     f"specified the following style guidelines for generated charts:\n {colours}"
        # )
        # await agent.slides.send(
        #     "Produce a MARP Presentation for the this analysis. You will find the visualisations in "
        #     f"in the ./mount-point/ folder. The analysis follows....\n{analysis}"
        # )

        await agent()


if __name__ == "__main__":
    asyncio.run(main())


############################################################################################################
# Example of evaluator/optimizer flow
############################################################################################################
# @fast.agent(
#     "evaluator",
#     """You are collaborating with a Data Analysis tool that has the capability to analyse data and produce visualisations.
#     You must make sure that the tool has:
#      - Considered the best way for a Human to interpret the data
#      - Produced insightful visualasions.
#      - Provided a high level summary report for the Human.
#      - Has had its findings challenged, and justified
#     """,
#     request_params=RequestParams(maxTokens=8192),
# )
# @fast.evaluator_optimizer(
#     "analysis_tool",
#     generator="data_analysis",
#     evaluator="evaluator",
#     max_refinements=3,
#     min_rating="EXCELLENT",
# )
