import asyncio
from mcp_agent.core.fastagent import FastAgent

# Explicitly provide the path to the config file in the current directory
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
    async with fast.run() as agent_app:  # Get the AgentApp wrapper
        agent_name = "default"
        agent_instance = agent_app.default
        print(f"Found agent: {agent_name}")

        # --- Define the System Template Variables ---
        my_t0_system_vars = {
            "TEST_VARIABLE_1": "Roses are red",
            "TEST_VARIABLE_2": "Violets are blue",
            "TEST_VARIABLE_3": "Sugar is sweet",
            "TEST_VARIABLE_4": "Careless vibe coding will kill your app",
        }

        # --- Set the template variables on the LLM instance ---
        # TODO: For now this requires manual override of the t0_system_template_vars. An optional update to the base agent class could make sense
        agent_instance._llm.t0_system_template_vars = my_t0_system_vars

        # --- Start interactive mode using the AgentApp wrapper ---
        print("\nStarting interactive session...")
        await agent_app.interactive(agent=agent_name)


if __name__ == "__main__":
    asyncio.run(main())
