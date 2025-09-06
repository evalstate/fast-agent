"""
Dynamic Agents Example

This example demonstrates how to use dynamic agents that can be created at runtime
based on task analysis. The project manager creates specialized teams on-the-fly.
"""

import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Dynamic Agents Example")


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
    model="haiku",
)
async def main():
    async with fast.run() as agent:
        print("=== Dynamic Agents Demo ===\n")

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

        print("\n" + "=" * 50 + "\n")

        # Example 2: Code Review Project
        print("Example 2: Code Review Team")
        await agent.project_manager("""
        I have a large Python codebase that needs a comprehensive review.
        Create a specialized code review team with different focuses:
        1. Security reviewer for vulnerability assessment
        2. Performance reviewer for optimization opportunities
        3. Code quality reviewer for maintainability
        4. Architecture reviewer for design patterns
        
        Then have them review this sample code in parallel and provide a consolidated report:
        
        ```python
        import hashlib
        import sqlite3
        
        def authenticate_user(username, password):
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            query = f"SELECT * FROM users WHERE username='{username}'"
            cursor.execute(query)
            user = cursor.fetchone()
            
            if user and user[2] == password:
                return True
            return False
        
        def create_user(username, password):
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute(f"INSERT INTO users VALUES ('{username}', '{password}')")
            conn.commit()
            conn.close()
        ```
        """)


@fast.agent(
    name="simple_creator",
    instruction="""You are a simple agent that demonstrates basic dynamic agent creation.
    
You can create other agents and delegate simple tasks to them.
Show how to create, use, and manage dynamic agents step by step.""",
    servers=["filesystem"],
    dynamic_agents=True,
    max_dynamic_agents=3,
)
async def simple_example():
    async with fast.run() as agent:
        print("\n=== Simple Dynamic Agent Example ===\n")

        await agent.simple_creator("""
        Please demonstrate the dynamic agent system by:
        1. Creating a file organizer agent that can read and organize files
        2. Creating a content writer agent that can write documentation
        3. List your active agents
        4. Have the file organizer create a project structure
        5. Have the content writer create a README file
        6. Show the results of their work
        7. Clean up by terminating the agents
        """)


async def interactive_demo():
    """Run an interactive demo where users can experiment with dynamic agents."""
    async with fast.run() as agent:
        print("\n=== Interactive Dynamic Agents Demo ===")
        print("You can now interact with the project manager!")
        print("Try commands like:")
        print("- 'Create a mobile app development team'")
        print("- 'Build a data analysis pipeline'")
        print("- 'Set up a microservices architecture'")
        print("- Type 'exit' to quit\n")

        await agent.project_manager.interactive()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        asyncio.run(simple_example())
    elif len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(main())
