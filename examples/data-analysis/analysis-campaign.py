import asyncio

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.llm.augmented_llm import RequestParams

# Create the application
fast = FastAgent("Data Analysis & Campaign Generator")


# Original data analysis components
@fast.agent(
    name="data_analysis",
    instruction="""
You have access to a Python 3.12 interpreter and you can use this to analyse and process data. 
Common analysis packages such as Pandas, Seaborn and Matplotlib are already installed. 
You can add further packages if needed.
Data files are accessible from the /mnt/data/ directory (this is the current working directory).
Visualisations should be saved as .png files in the current working directory.
Extract key insights that would be compelling for a social media campaign.
""",
    servers=["interpreter"],
    request_params=RequestParams(maxTokens=8192),
    model="sonnet",
)
@fast.agent(
    "evaluator",
    """You are collaborating with a Data Analysis tool that has the capability to analyse data and produce visualisations.
    You must make sure that the tool has:
     - Considered the best way for a Human to interpret the data
     - Produced insightful visualisations.
     - Provided a high level summary report for the Human.
     - Has had its findings challenged, and justified
     - Extracted compelling insights suitable for social media promotion
    """,
    request_params=RequestParams(maxTokens=8192),
    model="gpt-4.1",
)
@fast.evaluator_optimizer(
    "analysis_tool",
    generator="data_analysis",
    evaluator="evaluator",
    max_refinements=3,
    min_rating="EXCELLENT",
)
# Research component using Brave search
@fast.agent(
    "context_researcher",
    """You are a research specialist who provides cultural context for different regions.
    For any given data insight and target language/region, research:
    1. Cultural sensitivities related to presenting this type of data
    2. Local social media trends and preferences
    3. Region-specific considerations for marketing campaigns
    
    Always provide actionable recommendations for adapting content to each culture.
    """,
    servers=["fetch", "brave"],  # Using the fetch MCP server for Brave search
    request_params=RequestParams(temperature=0.3),
    model="gpt-4.1",
)
# Social media content generator
@fast.agent(
    "campaign_generator",
    """Generate engaging social media content based on data insights.
    Create compelling, shareable content that:
    - Highlights key research findings in an accessible way
    - Uses appropriate tone for the platform (Twitter/X, LinkedIn, Instagram, etc.)
    - Is concise and impactful
    - Includes suggested hashtags and posting schedule
    
    Format your response with clear sections for each platform.
    Save different campaign elements as separate files in the current directory.
    """,
    servers=["filesystem"],  # Using filesystem MCP server to save files
    request_params=RequestParams(temperature=0.7),
    model="sonnet",
    use_history=False,
)
# Translation agents with cultural adaptation
@fast.agent(
    "translate_fr",
    """Translate social media content to French with cultural adaptation.
    Consider French cultural norms, expressions, and social media preferences.
    Ensure the translation maintains the impact of the original while being culturally appropriate.
    Save the translated content to a file with appropriate naming.
    """,
    model="haiku",
    use_history=False,
    servers=["filesystem"],
)
@fast.agent(
    "translate_es",
    """Translate social media content to Spanish with cultural adaptation.
    Consider Spanish-speaking cultural contexts, expressions, and social media preferences.
    Ensure the translation maintains the impact of the original while being culturally appropriate.
    Save the translated content to a file with appropriate naming.
    """,
    model="haiku",
    use_history=False,
    servers=["filesystem"],
)
@fast.agent(
    "translate_de",
    """Translate social media content to German with cultural adaptation.
    Consider German cultural norms, expressions, and social media preferences.
    Ensure the translation maintains the impact of the original while being culturally appropriate.
    Save the translated content to a file with appropriate naming.
    """,
    model="haiku",
    use_history=False,
    servers=["filesystem"],
)
@fast.agent(
    "translate_ja",
    """Translate social media content to Japanese with cultural adaptation.
    Consider Japanese cultural norms, expressions, and social media preferences.
    Ensure the translation maintains the impact of the original while being culturally appropriate.
    Save the translated content to a file with appropriate naming.
    """,
    model="haiku",
    use_history=False,
    servers=["filesystem"],
)
# Parallel workflow for translations
@fast.parallel(
    "translate_campaign",
    instruction="Translates content to French, Spanish, German and Japanese. Supply the content to translate, translations will be saved to the filesystem.",
    fan_out=["translate_fr", "translate_es", "translate_de", "translate_ja"],
    include_request=True,
)
# Cultural sensitivity review agent
@fast.agent(
    "cultural_reviewer",
    """Review all translated content for cultural sensitivity and appropriateness.
    For each language version, evaluate:
    - Cultural appropriateness
    - Potential misunderstandings or sensitivities
    - Effectiveness for the target culture
    
    Provide specific recommendations for any needed adjustments and save a review report.
    """,
    servers=["filesystem"],
    request_params=RequestParams(temperature=0.2),
)
# Campaign optimization workflow
@fast.evaluator_optimizer(
    "campaign_optimizer",
    generator="campaign_generator",
    evaluator="cultural_reviewer",
    max_refinements=2,
    min_rating="EXCELLENT",
)
# Main workflow orchestration
@fast.orchestrator(
    "research_campaign_creator",
    instruction="""
    Create a complete multi-lingual social media campaign based on data analysis results.
    The workflow will:
    1. Analyze the provided data and extract key insights
    2. Research cultural contexts for target languages
    3. Generate appropriate social media content
    4. Translate and culturally adapt the content
    5. Review and optimize all materials
    6. Save all campaign elements to files
    """,
    agents=[
        "analysis_tool",
        "context_researcher",
        "campaign_optimizer",
        "translate_campaign",
    ],
    model="sonnet",  # Using a more capable model for orchestration
    request_params=RequestParams(maxTokens=8192),
    plan_type="full",
)
async def main() -> None:
    # Use the app's context manager
    print(
        "WARNING: This workflow will likely run for >10 minutes and consume a lot of tokens. Press Enter to accept the default prompt and proceed"
    )

    async with fast.run() as agent:
        await agent.research_campaign_creator.prompt(
            default_prompt="Analyze the CSV file in the current directory and create a comprehensive multi-lingual social media campaign based on the findings. Save all campaign elements as separate files."
        )


if __name__ == "__main__":
    asyncio.run(main())
