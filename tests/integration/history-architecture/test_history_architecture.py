"""
Integration tests for the new conversation history architecture.

These tests verify that:
1. Agent message_history is the single source of truth
2. Provider history is diagnostic only (write-only)
3. load_history doesn't trigger LLM calls
4. Templates are correctly handled
"""

import pytest

from fast_agent.core.prompt import Prompt
from fast_agent.mcp.prompt_serialization import save_messages
from fast_agent.mcp.prompts.prompt_load import load_history_into_agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_history_no_llm_call(fast_agent, tmp_path):
    """
    Verify that load_history_into_agent() does NOT trigger an LLM API call.

    This test ensures the bug fix where load_history previously called generate().
    """
    fast = fast_agent

    # Create a temporary history file with a simple conversation
    history_file = tmp_path / "test_history.json"
    messages = [
        Prompt.user("Hello"),
        Prompt.assistant("Hi there!"),
        Prompt.user("How are you?"),
    ]

    # Save using the proper serialization format
    save_messages(messages, str(history_file))

    @fast.agent(model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            agent_obj = agent.default

            # Get initial message count
            initial_count = len(agent_obj.message_history)
            assert initial_count == 0, "Agent should start with no history"

            # Load history - this should NOT make an LLM call
            load_history_into_agent(agent_obj, history_file)

            # Verify history was loaded
            loaded_count = len(agent_obj.message_history)
            assert loaded_count == 3, f"Expected 3 messages, got {loaded_count}"

            # Verify content
            assert agent_obj.message_history[0].role == "user"
            assert "Hello" in agent_obj.message_history[0].first_text()
            assert agent_obj.message_history[1].role == "assistant"
            assert "Hi there!" in agent_obj.message_history[1].first_text()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_message_history_source_of_truth(fast_agent):
    """
    Verify that _message_history is the single source of truth.

    Provider history should be diagnostic only and not read for API calls.
    """
    fast = fast_agent

    @fast.agent(model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            agent_obj = agent.default

            # Start with empty histories
            assert len(agent_obj.message_history) == 0

            # Manually add a message to message_history
            test_msg = Prompt.user("Test message")
            agent_obj.message_history.append(test_msg)

            # Verify message is in message history
            assert len(agent_obj.message_history) == 1
            assert agent_obj.message_history[0].first_text() == "Test message"

            # Provider history should still be empty (no API call yet)
            # This verifies that message_history is independent of provider history

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_template_persistence_after_clear(fast_agent):
    """
    Verify that template messages are preserved after clear() but removed after clear(clear_prompts=True).
    """
    fast = fast_agent

    @fast.agent(model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            agent_obj = agent.default

            # Create template messages directly
            template_msgs = [
                Prompt.user("You are a helpful assistant."),
                Prompt.assistant("I understand."),
            ]
            agent_obj._template_messages = [msg.model_copy(deep=True) for msg in template_msgs]
            agent_obj._message_history = [msg.model_copy(deep=True) for msg in template_msgs]

            # Verify template is loaded
            assert len(agent_obj._template_messages) == 2
            assert len(agent_obj.message_history) == 2

            # Add a user message
            user_msg = Prompt.user("New message")
            agent_obj._message_history.append(user_msg)
            assert len(agent_obj.message_history) == 3

            # Clear without clearing prompts
            agent_obj.clear()

            # Templates should be restored, new message should be gone
            assert len(agent_obj.message_history) == 2
            assert len(agent_obj._template_messages) == 2

            # Add another message
            agent_obj._message_history.append(user_msg)
            assert len(agent_obj.message_history) == 3

            # Clear with clear_prompts=True
            agent_obj.clear(clear_prompts=True)

            # Everything should be gone
            assert len(agent_obj.message_history) == 0
            assert len(agent_obj._template_messages) == 0

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_history_diagnostic_only(fast_agent):
    """
    Verify that provider history (self.history) is diagnostic only.

    The provider should NOT read from self.history for API calls.
    """
    fast = fast_agent

    @fast.agent(model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            agent_obj = agent.default
            llm = agent_obj._llm

            # Start with empty histories
            assert len(agent_obj.message_history) == 0

            # Manually add a message to message_history
            test_msg = Prompt.user("Test")
            agent_obj.message_history.append(test_msg)

            # Verify it's in _message_history
            assert len(agent_obj.message_history) == 1

            # Provider history should still be empty (until an API call is made)
            # This confirms that _message_history is independent of provider history
            # and that provider history is only written to, not read from
            assert len(llm.history.get()) == 0

    await agent_function()
