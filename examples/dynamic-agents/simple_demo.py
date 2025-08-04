"""
Simple Dynamic Agents Demo

This example demonstrates the basic functionality of dynamic agents with
simple use cases that are easy to understand and modify.
"""

import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Simple Dynamic Agents Demo")


@fast.agent(
    name="simple_creator",
    instruction="""You are a simple agent that demonstrates basic dynamic agent creation.
    
You can create other agents and delegate simple tasks to them.
Show how to create, use, and manage dynamic agents step by step.
Be clear about what you're doing and explain each step.""",
    servers=["filesystem"],
    dynamic_agents=True,
    max_dynamic_agents=3,
    model="haiku"
)
async def simple_example():
    async with fast.run() as agent:
        print("=== Simple Dynamic Agent Example ===\n")
        
        await agent.simple_creator("""
        Please demonstrate the dynamic agent system by:
        1. Creating a file organizer agent that can read and organize files
        2. Creating a content writer agent that can write documentation
        3. List your active agents
        4. Have the file organizer create a project structure
        5. Have the content writer create a README file
        6. Show the results of their work
        7. Clean up by terminating the agents
        
        Walk me through each step clearly.
        """)


@fast.agent(
    name="task_delegator",
    instruction="""You are a task delegation specialist. You break down complex tasks 
    into smaller pieces and create specialized agents to handle each piece.
    
    Focus on clear task division and coordination between agents.""",
    servers=["filesystem"],
    dynamic_agents=True,
    max_dynamic_agents=4,
    model="haiku"
)
async def delegation_example():
    async with fast.run() as agent:
        print("\n=== Task Delegation Example ===\n")
        
        await agent.task_delegator("""
        I need to analyze a Python project and create documentation for it.
        
        Please:
        1. Create a code analyzer agent to examine the project structure
        2. Create a documentation writer agent to write technical docs
        3. Create a readme generator agent to create user-friendly documentation
        4. Coordinate their work to produce comprehensive project documentation
        
        Show how you delegate tasks and combine their results.
        """)


async def run_all_examples():
    """Run all simple examples in sequence."""
    print("Running Simple Dynamic Agents Examples...\n")
    
    await simple_example()
    #print("\n" + "="*60 + "\n")
    #await delegation_example()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "delegation":
        asyncio.run(delegation_example())
    elif len(sys.argv) > 1 and sys.argv[1] == "basic":
        asyncio.run(simple_example())
    else:
        asyncio.run(run_all_examples())