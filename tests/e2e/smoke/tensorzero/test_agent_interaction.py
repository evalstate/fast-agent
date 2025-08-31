import pytest

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

pytestmark = pytest.mark.usefixtures("tensorzero_docker_env")


@pytest.mark.asyncio
async def test_tensorzero_agent_smoke(project_root, chdir_to_tensorzero_example):
    """
    Smoke test for the TensorZero agent interaction defined in examples/tensorzero/agent.py.
    Sends a predefined sequence of messages.
    """
    config_file = "fastagent.config.yaml"

    my_t0_system_vars = {
        "TEST_VARIABLE_1": "Roses are red",
        "TEST_VARIABLE_2": "Violets are blue",
        "TEST_VARIABLE_3": "Sugar is sweet",
        "TEST_VARIABLE_4": "Vibe code responsibly 👍",
    }

    fast = FastAgent("fast-agent example test", config_path=config_file, ignore_unknown_args=True)

    @fast.agent(
        name="default",
        instruction="""
            You are an agent dedicated to helping developers understand the relationship between TensoZero and fast-agent. If the user makes a request
            that requires you to invoke the test tools, please do so. When you use the tool, describe your rationale for doing so.
        """,
        servers=["tester"],
        model="tensorzero.test_chat",
        request_params=RequestParams(template_vars=my_t0_system_vars),
    )
    async def dummy_agent_func():
        pass

    messages_to_send = [
        "Hi.",
        "Tell me a poem.",
        "Do you have any tools that you can use?",
        "Please demonstrate the use of that tool on your last response.",
        "Please summarize the conversation so far.",
        "What tool calls have you executed in this session, and what were their results?",
    ]

    async with fast.run() as agent_app:
        agent_instance = agent_app.default
        print(f"\nSending {len(messages_to_send)} messages to agent '{agent_instance._name}'...")
        for i, msg_text in enumerate(messages_to_send):
            print(f"Sending message {i + 1}: '{msg_text}'")
            await agent_instance.send(msg_text)
            print(f"Message {i + 1} sent successfully.")

    print("\nAgent interaction smoke test completed successfully.")
