import pytest

from mcp_agent.core.fastagent import FastAgent

pytestmark = pytest.mark.usefixtures("tensorzero_docker_env", "chdir_to_tensorzero_example")


@pytest.mark.asyncio
async def test_tensorzero_simple_agent_smoke():  # Removed unused project_root fixture
    """
    Smoke test for the TensorZero simple agent interaction defined in examples/tensorzero/simple_agent.py.
    Sends a single "hi" message.
    """
    config_file = "fastagent.config.yaml"

    fast = FastAgent(
        "fast-agent simple example test", config_path=config_file, ignore_unknown_args=True
    )

    @fast.agent(
        name="simple_default",
        instruction="""
            You are an agent dedicated to helping developers understand the relationship between TensoZero and fast-agent. If the user makes a request 
            that requires you to invoke the test tools, please do so. When you use the tool, describe your rationale for doing so. 
        """,
        servers=["tester"],
        model="tensorzero.simple_chat",
    )
    async def dummy_simple_agent_func():
        pass

    message_to_send = "Hi."

    async with fast.run() as agent_app:
        agent_instance = agent_app.simple_default

        print(f"\nSending message to agent '{agent_instance._name}': '{message_to_send}'")
        await agent_instance.send(message_to_send)
        print(f"Message sent successfully to '{agent_instance._name}'.")

    print("\nSimple agent interaction smoke test completed successfully.")
