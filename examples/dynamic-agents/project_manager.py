"""
Project Manager Dynamic Agents Example

This example demonstrates a project manager that creates specialized development teams
to handle complex software projects. The manager analyzes requirements and creates
appropriate specialists on-the-fly.
"""

import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Project Manager Demo")


@fast.agent(
    name="project_manager",
    instruction="""You are a project manager that creates and manages specialized development teams.

When given a project, you should:
1. Analyze what specialists are needed
2. Create appropriate dynamic agents with specific roles
3. Delegate tasks to the specialists
4. Coordinate their work to complete the project

You have access to dynamic agent tools:
- dynamic_agent_create: Create new specialized agents
- dynamic_agent_send: Send tasks to specific agents  
- dynamic_agent_broadcast: Send tasks to multiple agents in parallel
- dynamic_agent_list: See all your active agents
- dynamic_agent_terminate: Clean up agents when done

Available MCP servers for your specialists:
- filesystem: For reading/writing files
- fetch: For web requests and API calls

Example specialist roles:
- Frontend Developer (React/TypeScript expert)
- Backend Developer (Python/FastAPI expert)  
- Database Designer (SQL/schema expert)
- Security Reviewer (security best practices)
- DevOps Engineer (deployment and infrastructure)
- QA Tester (testing and quality assurance)
""",
    servers=["filesystem", "fetch"],
    dynamic_agents=True,
    max_dynamic_agents=5,
    model="haiku"
)
async def main():
    async with fast.run() as agent:
        print("=== Project Manager Demo ===\n")
        
        # Example 1: Web Development Project
        print("Example 1: Building a Todo App")
        await agent.project_manager("""
        I need to build a React todo application with the following requirements:
        1. Frontend: React with TypeScript, modern hooks, responsive design
        2. Backend: Python FastAPI with RESTful endpoints
        3. Database: PostgreSQL schema design
        4. Security: Authentication, input validation, CORS
        5. Testing: Unit tests and integration tests
        
        Please create appropriate specialists and coordinate their work to:
        - Design the application architecture
        - Create the database schema
        - Build the backend API
        - Develop the React frontend
        - Implement security measures
        - Write comprehensive tests
        
        Show me the team you create and how you delegate the work.
        """)
        
        print("\n" + "="*50 + "\n")
        
        # Example 2: Mobile App Project
        print("Example 2: Mobile App Development")
        await agent.project_manager("""
        I need to create a mobile app for a fitness tracking platform:
        1. React Native app with offline capabilities
        2. Node.js backend with real-time features
        3. MongoDB for flexible data storage
        4. Integration with health APIs (Apple Health, Google Fit)
        5. Push notifications and analytics
        
        Create a specialized team to handle this project and show how you coordinate 
        the development across mobile, backend, and integration specialists.
        """)


async def interactive_demo():
    """Run an interactive demo where users can experiment with dynamic agents."""
    async with fast.run() as agent:
        print("\n=== Interactive Project Manager Demo ===")
        print("You can now interact with the project manager!")
        print("Try commands like:")
        print("- 'Create a microservices architecture for an e-commerce platform'")
        print("- 'Build a data analysis pipeline with Python and Apache Spark'") 
        print("- 'Set up a CI/CD pipeline for a React application'")
        print("- Type 'exit' to quit\n")
        
        await agent.project_manager.interactive()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(main())