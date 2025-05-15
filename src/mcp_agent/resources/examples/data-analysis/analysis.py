import asyncio

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.llm.augmented_llm import RequestParams

# Create the application
fast = FastAgent("Data Analysis (Roots)")


# The sample data is under Database Contents License (DbCL) v1.0.
# Available here : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset


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
async def main() -> None:
    # Use the app's context manager
    async with fast.run() as agent:
        await agent(
            "There is a csv file in the current directory. "
            "Analyse the file, produce a detailed description of the data, and any patterns it contains.",
        )
        await agent(
            "Consider the data, and how to usefully group it for presentation to a Human. Find insights, using the Python Interpreter as needed.\n"
            "Use MatPlotLib to produce insightful visualisations. Save them as '.png' files in the current directory. Be sure to run the code and save the files.\n"
            "Produce a summary with major insights to the data",
        )
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
