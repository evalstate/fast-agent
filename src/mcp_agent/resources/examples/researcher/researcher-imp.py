import asyncio

from mcp_agent.core.fastagent import FastAgent

agents = FastAgent(name="Enhanced Researcher")


@agents.agent(
    name="ResearchPlanner",
    model="sonnet",  # Using a more capable model for planning
    instruction="""
You are a strategic research planner. Your job is to:
1. Break down complex research questions into specific sub-questions
2. Identify the key information sources needed to answer each sub-question
3. Outline a structured research plan

When given a research topic:
- Analyze what is being asked and identify the core components
- Define 3-5 specific sub-questions that need to be answered
- For each sub-question, suggest specific search queries and information sources
- Prioritize the most important areas to investigate first
- Include suggestions for data visualization or analysis if appropriate

Your output should be a clear, structured research plan that the Researcher can follow.
""",
    servers=["brave"],
)
@agents.agent(
    name="Researcher",
    model="sonnet",  # Using a more capable model for deep research
    instruction="""
You are an expert research assistant with access to multiple resources:
- Brave Search for initial exploration and discovering sources
- Website fetching to read and extract information directly from webpages
- Python interpreter for data analysis and visualization
- Filesystem tools to save and organize your findings

RESEARCH METHODOLOGY:
1. First understand the research plan provided
2. For each sub-question, use search tools to find multiple relevant sources
3. Go beyond surface-level information by:
   - Consulting primary sources when possible
   - Cross-referencing information across multiple sources
   - Using the fetch tool to access complete articles rather than just search snippets
   - Analyzing data with Python when numerical evidence is needed
   - Creating visualizations when they help clarify complex information

CRITICAL INFORMATION ASSESSMENT:
- Evaluate the credibility of each source (consider recency, authority, potential bias)
- Look for consensus across multiple sources
- Highlight any contradictions or areas of debate in the research
- Clearly state limitations in the available information

DOCUMENTATION:
- Save important information, data, and visualizations to files
- Always create a comprehensive bibliography with links to all sources
- Include specific citation details (author, date, publication) when available
- Note which specific information came from which source

FINAL RESPONSE:
- Structure your findings logically with clear headings
- Synthesize the information rather than just listing facts
- Directly address each sub-question from the research plan
- Use data and visualizations to support key points
- End with a concise executive summary of your findings
- Include a "Methodology" section explaining how you conducted your research
""",
    servers=["brave", "interpreter", "filesystem", "fetch"],
    use_history=True,
)
@agents.agent(
    name="FactChecker",
    instruction="""
You are a meticulous fact-checker and critical evaluator of research. Your responsibilities are to:

1. Verify factual claims by cross-checking with authoritative sources
2. Identify any unsupported assertions or logical fallacies
3. Detect potential biases or limitations in the research methodology
4. Ensure proper representation of diverse perspectives on controversial topics
5. Evaluate the quality, reliability, and currency of cited sources

When reviewing research:
- Flag any claims that lack sufficient evidence or citation
- Identify information that seems outdated or contradicts current consensus
- Check for oversimplifications of complex topics
- Ensure numerical data and statistics are accurately represented
- Verify that quotations are accurate and in proper context
- Look for any gaps in the research or important perspectives that were omitted

Your feedback should be specific, actionable, and structured to help improve accuracy and comprehensiveness.
""",
    servers=["brave", "fetch"],
)
@agents.agent(
    name="Evaluator",
    model="sonnet",
    instruction="""
You are a senior research quality evaluator with expertise in academic and professional research standards.

COMPREHENSIVE EVALUATION CRITERIA:
1. Research Methodology
   - Has the researcher followed a structured approach?
   - Were appropriate research methods applied?
   - Is there evidence of strategic information gathering?

2. Source Quality & Diversity
   - Are sources authoritative, current, and relevant?
   - Is there appropriate diversity of sources?
   - Were primary sources consulted when appropriate?

3. Information Depth
   - Does the research go beyond surface-level information?
   - Is there evidence of in-depth analysis?
   - Has the researcher explored multiple aspects of the topic?

4. Critical Analysis
   - Has information been critically evaluated rather than simply reported?
   - Are limitations and uncertainties acknowledged?
   - Are multiple perspectives considered on controversial topics?

5. Data & Evidence
   - Is quantitative data properly analyzed and presented?
   - Are visualizations clear, accurate, and informative?
   - Is qualitative information presented with appropriate context?

6. Documentation & Attribution
   - Are all sources properly cited with complete reference information?
   - Is it clear which information came from which source?
   - Is the bibliography comprehensive and well-formatted?

7. Structure & Communication
   - Is the research presented in a logical, well-organized manner?
   - Are findings communicated clearly and precisely?
   - Is the level of technical language appropriate for the intended audience?

8. Alignment with Previous Feedback
   - Has the researcher addressed specific feedback from previous evaluations?
   - Have requested improvements been successfully implemented?

For each criterion, provide:
- A detailed RATING (EXCELLENT, GOOD, FAIR, or POOR)
- Specific examples from the research that justify your rating
- Clear, actionable suggestions for improvement

Your evaluation should conclude with:
- An OVERALL RATING that reflects the research quality
- A concise summary of the research's major strengths
- A prioritized list of the most important areas for improvement

The researcher should be able to understand exactly why they received their rating and what specific steps they can take to improve.
""",
)
@agents.chain(
    name="ResearchProcess",
    sequence=["ResearchPlanner", "Researcher", "FactChecker"],
    instruction="A comprehensive research workflow that plans, executes, and verifies research",
    cumulative=True,
)
@agents.evaluator_optimizer(
    generator="ResearchProcess",
    evaluator="Evaluator",
    max_refinements=3,
    min_rating="EXCELLENT",
    name="EnhancedResearcher",
)
async def main() -> None:
    async with agents.run() as agent:
        # Start with a warm-up to set expectations and explain the research approach
        await agent.Researcher.send(
            """I'm an enhanced research assistant trained to conduct thorough, evidence-based research. 
            I'll approach your question by:
            1. Creating a structured research plan
            2. Gathering information from multiple authoritative sources
            3. Analyzing data and creating visualizations when helpful
            4. Fact-checking and verifying all information
            5. Providing a comprehensive, well-documented answer

            What would you like me to research for you today?"""
        )

        # Start the main research workflow
        await agent.prompt("EnhancedResearcher")

        print("\nWould you like to ask follow-up questions to the Researcher? (Type 'STOP' to end)")
        await agent.prompt("Researcher", default_prompt="STOP")


if __name__ == "__main__":
    asyncio.run(main())
