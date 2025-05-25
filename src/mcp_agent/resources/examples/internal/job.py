"""
PMO Job Description Generator Agent
  Purpose: Generate comprehensive PMO job descriptions using a multi-stage approach
  for clarity, consistency and quality control
"""

import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("PMO Job Description Generator")


@fast.agent(
    name="content_generator",
    instruction="""You are a PMO job description expert. Generate job descriptions for PMO roles         
    following these guidelines:                                                                          
    - Focus on modern lean/agile and product-based approaches                                            
    - Emphasize practical experience and demonstrated results over formal requirements                   
    - Ensure clear role differentiation with minimal overlap                                             
    - Format output in Markdown                                                                          
    - Context: Telecommunications industry in open organization valuing practical experience             
                                                                                                        
    Structure each job description with:                                                                 
    1. Role Title                                                                                        
    2. Position Summary                                                                                  
    3. Key Responsibilities                                                                              
    4. Required Experience                                                                               
    5. Desired Capabilities                                                                              
    """,
    model="anthropic.claude-3-5-haiku-latest",
)
@fast.agent(
    name="consistency_checker",
    instruction="""Review PMO job descriptions for:                                                      
    1. Alignment with lean/agile principles                                                              
    2. Clear role differentiation                                                                        
    3. Progressive responsibility levels                                                                 
    4. Consistent formatting and structure                                                               
    5. Telecommunications industry relevance                                                             
    6. Emphasis on practical experience over formal requirements                                         
                                                                                                        
    Provide specific feedback for improvements.""",
    model="gpt-4.1",
)
@fast.agent(
    name="file_handler",
    instruction="""Save the finalized job descriptions as individual Markdown files.                     
    Use consistent naming like 'pmo_director.md', 'pmo_manager.md' etc.""",
    servers=["filesystem"],
    use_history=False,
)
@fast.evaluator_optimizer(
    name="job_description_writer",
    generator="content_generator",
    evaluator="consistency_checker",
    min_rating="EXCELLENT",
    max_refinements=2,
)
async def main() -> None:
    async with fast.run() as agent:
        roles = [
            "PMO Director",
            "Portfolio Manager",
            "Senior Program Manager",
            "Project Manager",
            "PMO Analyst",
            "Project Coordinator",
        ]

        # Pre-initialize the file_handler to establish a persistent connection
        await agent.file_handler("Test connection to filesystem")

        for role in roles:
            # Generate and refine job description
            description = await agent.job_description_writer(
                f"Create job description for {role} role"
            )
            await agent.file_handler(f"Save this job description: {description}")


if __name__ == "__main__":
    asyncio.run(main())
