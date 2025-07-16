import asyncio
import sys

from mcp_agent.core.fastagent import FastAgent, PromptExitError

fast_agent = FastAgent(
    name="MCP Filtering Demo",
    parse_cli_args=False,
    quiet=False
)

@fast_agent.agent(
    name="filtered_agent",    
    model="gpt-4o-mini",
    instruction="You are a creative writer with filtered access to tools, resources, and prompts.",
    servers=["creativity"],
    # Tool filtering: only string manipulation tools and coding tools
    tools={"creativity": ["reverse_string", "capitalize_string", "coding_*"]},
    # Resource filtering: only writing resources (not coding resources)
    resources={"creativity": ["resource://writing/*"]},
    # Prompt filtering: only writing prompts (not coding prompts)
    prompts={"creativity": ["writing_*"]}
)
async def filtered_agent():
    return "Filtered agent ready"

@fast_agent.agent(
    name="unfiltered_agent",
    model="gpt-4o-mini", 
    instruction="You are a creative writer with access to all tools, resources, and prompts.",
    servers=["creativity"]
    # No filtering - gets everything
)
async def unfiltered_agent():
    return "Unfiltered agent ready"

async def main():
    try:
        async with fast_agent.run() as agent:
            try:
                print("🎯 MCP Filtering Demo")
                print("=" * 50)
                
                # Show filtered agent capabilities
                print("\n📦 FILTERED AGENT:")
                filtered = agent._agent("filtered_agent")
                
                tools = await filtered.list_tools()
                print(f"✅ Available tools ({len(tools.tools)}): {[tool.name for tool in tools.tools]}")
                
                resources = await filtered.list_resources()
                resource_list = resources.get("creativity", [])
                print(f"📚 Available resources ({len(resource_list)}): {resource_list}")
                
                prompts = await filtered.list_prompts()
                prompt_list = prompts.get("creativity", [])
                prompt_names = [p.name for p in prompt_list]
                print(f"💬 Available prompts ({len(prompt_names)}): {prompt_names}")
                
                # Show unfiltered agent capabilities
                print("\n🌐 UNFILTERED AGENT:")
                unfiltered = agent._agent("unfiltered_agent")
                
                tools = await unfiltered.list_tools()
                print(f"✅ Available tools ({len(tools.tools)}): {[tool.name for tool in tools.tools]}")
                
                resources = await unfiltered.list_resources()
                resource_list = resources.get("creativity", [])
                print(f"📚 Available resources ({len(resource_list)}): {resource_list}")
                
                prompts = await unfiltered.list_prompts()
                prompt_list = prompts.get("creativity", [])
                prompt_names = [p.name for p in prompt_list]
                print(f"💬 Available prompts ({len(prompt_names)}): {prompt_names}")
                
                print("\n" + "=" * 50)
                print("🚀 Starting interactive session with filtered agent...")
                print("Type 'exit' to quit")
                
                await agent.interactive(agent_name="filtered_agent")

            except PromptExitError:
                print("👋 Goodbye!")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
