"""
Code Review Team Dynamic Agents Example

This example demonstrates creating specialized code review teams that can analyze
code from different perspectives simultaneously.
"""

import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Code Review Team Demo")


# Sample problematic code for review
SAMPLE_CODE = '''
import hashlib
import sqlite3
import os

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

def process_large_dataset(data):
    results = []
    for item in data:
        # Inefficient nested loops
        for i in range(len(data)):
            for j in range(len(data)):
                if data[i] == data[j]:
                    results.append(item)
    return results

class UserManager:
    def __init__(self):
        self.users = []
        
    def add_user(self, user):
        self.users.append(user)
        
    def find_user(self, username):
        for user in self.users:
            if user.username == username:
                return user
'''


@fast.agent(
    name="review_coordinator",
    instruction="""You are a code review coordinator that creates specialized review teams.

When given code to review, you should:
1. Create different types of reviewers with specific expertise
2. Have them analyze the code in parallel from their perspectives
3. Consolidate their findings into a comprehensive report

Types of reviewers you can create:
- Security Reviewer: Focuses on vulnerabilities, injection attacks, authentication
- Performance Reviewer: Looks for optimization opportunities, bottlenecks
- Code Quality Reviewer: Examines maintainability, readability, best practices
- Architecture Reviewer: Analyzes design patterns, structure, scalability

Use dynamic agent tools to create and coordinate the review team.""",
    servers=["filesystem"],
    dynamic_agents=True,
    max_dynamic_agents=6,
    model="haiku"
)
async def main():
    async with fast.run() as agent:
        print("=== Code Review Team Demo ===\n")
        
        await agent.review_coordinator(f"""
        I have a Python codebase that needs a comprehensive review.
        Create a specialized code review team with different focuses:
        1. Security reviewer for vulnerability assessment
        2. Performance reviewer for optimization opportunities
        3. Code quality reviewer for maintainability
        4. Architecture reviewer for design patterns

        Then have them review this sample code in parallel and provide a consolidated report:

        ```python
        {SAMPLE_CODE}
        ```

        Each reviewer should focus on their specialty and provide specific recommendations.
        """)


@fast.agent(
    name="security_focused_reviewer",
    instruction="""You are a security-focused code reviewer that creates specialized 
    security analysis teams.

Create agents that focus on different security aspects:
- Input validation specialist
- Authentication security expert  
- Database security analyst
- General security vulnerability scanner""",
    servers=["filesystem"],
    dynamic_agents=True,
    max_dynamic_agents=4,
    model="haiku"
)
async def security_review_example():
    async with fast.run() as agent:
        print("\n=== Security-Focused Review Example ===\n")
        
        await agent.security_focused_reviewer(f"""
        Create a specialized security review team to analyze this code for vulnerabilities:

        ```python
        {SAMPLE_CODE}
        ```

        Create different security specialists and have them each focus on their area of expertise.
        Provide a detailed security assessment with risk levels and remediation steps.
        """)


async def run_all_reviews():
    """Run all code review examples."""
    print("Running Code Review Examples...\n")
    
    await main()
    print("\n" + "="*60 + "\n")
    await security_review_example()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "security":
        asyncio.run(security_review_example())
    elif len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        asyncio.run(main())
    else:
        asyncio.run(run_all_reviews())