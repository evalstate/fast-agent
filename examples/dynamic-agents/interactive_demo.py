"""
Interactive Dynamic Agents Demo

This example provides an interactive interface where users can experiment
with dynamic agents and see how they work in real-time.
"""

import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Interactive Dynamic Agents Demo")


@fast.agent(
    name="interactive_manager",
    instruction="""You are an interactive agent manager that helps users explore dynamic agents.

You can create any type of specialist agent based on the user's needs:
- Development teams (frontend, backend, DevOps, etc.)
- Analysis teams (data scientists, researchers, etc.)
- Creative teams (writers, designers, etc.)
- Business teams (product managers, marketers, etc.)

When a user asks for something:
1. Analyze what type of specialists would be helpful
2. Create appropriate dynamic agents
3. Demonstrate how they work together
4. Show the user the results

Be conversational and educational - explain what you're doing and why.""",
    servers=["filesystem", "fetch"],
    dynamic_agents=True,
    max_dynamic_agents=6,
    model="haiku"
)
async def interactive_demo():
    """Run an interactive demo where users can experiment with dynamic agents."""
    async with fast.run() as agent:
        print("=== Interactive Dynamic Agents Demo ===")
        print()
        print("🤖 Welcome to the Dynamic Agents playground!")
        print()
        print("You can ask me to create specialized teams for any task. Try:")
        print("  • 'Create a web development team for an e-commerce site'")
        print("  • 'Build a data analysis team to analyze sales data'")
        print("  • 'Set up a content creation team for a marketing campaign'")
        print("  • 'Create a code review team for a Python project'")
        print("  • 'Build a research team to analyze market trends'")
        print()
        print("Type 'help' for more examples or 'exit' to quit")
        print("=" * 60)
        print()
        
        await agent.interactive_manager.interactive()


@fast.agent(
    name="demo_guide",
    instruction="""You are a helpful guide that demonstrates dynamic agents with pre-built examples.

You have several demo scenarios ready to show:
1. Software development team
2. Content creation team  
3. Research and analysis team
4. Marketing team
5. Customer support team

When asked, create the appropriate team and walk through a realistic scenario.""",
    servers=["filesystem", "fetch"],
    dynamic_agents=True,
    max_dynamic_agents=5,
    model="haiku"
)
async def guided_demo():
    """Run a guided demo with pre-built scenarios."""
    async with fast.run() as agent:
        print("=== Guided Dynamic Agents Demo ===")
        print()
        print("🎯 Choose a demo scenario:")
        print("  1. Software Development Team")
        print("  2. Content Creation Team")
        print("  3. Research & Analysis Team")
        print("  4. Marketing Team")
        print("  5. Customer Support Team")
        print("  6. All scenarios (sequential)")
        print()
        
        choice = input("Enter your choice (1-6): ").strip()
        
        scenarios = {
            "1": "Create a full-stack development team to build a social media platform",
            "2": "Create a content team to produce a comprehensive product launch campaign",
            "3": "Create a research team to analyze competitor strategies in the AI market",
            "4": "Create a marketing team to launch a new mobile app",
            "5": "Create a customer support team to handle technical inquiries",
            "6": "all"
        }
        
        if choice == "6":
            for i, scenario in enumerate(scenarios.values(), 1):
                if scenario == "all":
                    continue
                print(f"\n{'='*60}")
                print(f"Demo {i}: {scenario}")
                print('='*60)
                await agent.demo_guide(scenario)
                if i < 5:  # Don't wait after the last demo
                    input("\nPress Enter to continue to the next demo...")
        elif choice in scenarios:
            await agent.demo_guide(scenarios[choice])
        else:
            print("Invalid choice. Running interactive demo instead...")
            await interactive_demo()


async def quick_demo():
    """A quick demonstration of dynamic agents."""
    async with fast.run() as agent:
        print("=== Quick Dynamic Agents Demo ===")
        
        await agent.demo_guide("""
        Give me a quick demonstration of dynamic agents by:
        1. Creating 2-3 different specialist agents
        2. Showing how they can work together on a simple project
        3. Demonstrating their different capabilities
        
        Keep it concise but informative.
        """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "guided":
        asyncio.run(guided_demo())
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(quick_demo())
    else:
        asyncio.run(interactive_demo())