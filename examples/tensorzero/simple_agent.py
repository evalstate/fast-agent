import asyncio

from mcp_agent.core.fastagent import FastAgent

CONFIG_FILE = "fastagent.config.yaml"
fast = FastAgent("fast-agent example", config_path=CONFIG_FILE, ignore_unknown_args=True)


@fast.agent(
    name="default",
    instruction="""
        You are an agent dedicated to helping developers understand the relationship between TensoZero and fast-agent. If the user makes a request 
        that requires you to invoke the test tools, please do so. When you use the tool, describe your rationale for doing so. 
    """,
    servers=["tester"],
)
async def main():
    async with fast.run() as agent_app:
        agent_name = "default"
        print("\nStarting interactive session with template_vars set via decorator...")
        await agent_app.interactive(agent=agent_name)


if __name__ == "__main__":
    asyncio.run(main())  # type: ignore
