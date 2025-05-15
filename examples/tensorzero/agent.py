import asyncio

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Explicitly provide the path to the config file in the current directory
CONFIG_FILE = "fastagent.config.yaml"
fast = FastAgent("fast-agent example", config_path=CONFIG_FILE, ignore_unknown_args=True)

# Define T0 system variables here
my_t0_system_vars = {
    "TEST_VARIABLE_1": "Roses are red",
    "TEST_VARIABLE_2": "Violets are blue",
    "TEST_VARIABLE_3": "Sugar is sweet",
    "TEST_VARIABLE_4": "Vibe code responsibly 👍",
}


@fast.agent(
    name="default",
    instruction="""
        You are an agent dedicated to helping developers understand the relationship between TensoZero and fast-agent. If the user makes a request 
        that requires you to invoke the test tools, please do so. When you use the tool, describe your rationale for doing so. 
    """,
    servers=["tester"],
    request_params=RequestParams(template_vars=my_t0_system_vars),
)
async def main():
    async with fast.run() as agent_app:  # Get the AgentApp wrapper
        agent_name = "default"
        print("\nStarting interactive session with template_vars set via decorator...")
        await agent_app.interactive(agent=agent_name)


if __name__ == "__main__":
    asyncio.run(main())  # type: ignore
