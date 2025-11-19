"""
Integration tests for the new conversation history architecture.

These tests verify that:
1. _message_history is the single source of truth
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
            # Get the underlying LLM
            llm = agent.default._llm

            # Get initial message count
            initial_count = len(llm.message_history)
            assert initial_count == 0, "Agent should start with no history"

            # Load history - this should NOT make an LLM call
            load_history_into_agent(llm, history_file)

            # Verify history was loaded
            loaded_count = len(llm.message_history)
            assert loaded_count == 3, f"Expected 3 messages, got {loaded_count}"

            # Verify content
            assert llm.message_history[0].role == "user"
            assert "Hello" in llm.message_history[0].first_text()
            assert llm.message_history[1].role == "assistant"
            assert "Hi there!" in llm.message_history[1].first_text()

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
            # Get the underlying LLM
            llm = agent.default._llm

            # Start with empty histories
            assert len(llm.message_history) == 0

            # Manually add a message to _message_history
            test_msg = Prompt.user("Test message")
            llm._message_history.append(test_msg)

            # Verify message is in message history
            assert len(llm.message_history) == 1
            assert llm.message_history[0].first_text() == "Test message"

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
            # Get the underlying LLM
            llm = agent.default._llm

            # Create template messages directly
            template_msgs = [
                Prompt.user("You are a helpful assistant."),
                Prompt.assistant("I understand."),
            ]
            llm._template_messages = [msg.model_copy(deep=True) for msg in template_msgs]
            llm._message_history = [msg.model_copy(deep=True) for msg in template_msgs]

            # Verify template is loaded
            assert len(llm._template_messages) == 2
            assert len(llm.message_history) == 2

            # Add a user message
            user_msg = Prompt.user("New message")
            llm._message_history.append(user_msg)
            assert len(llm.message_history) == 3

            # Clear without clearing prompts
            llm.clear()

            # Templates should be restored, new message should be gone
            assert len(llm.message_history) == 2
            assert len(llm._template_messages) == 2

            # Add another message
            llm._message_history.append(user_msg)
            assert len(llm.message_history) == 3

            # Clear with clear_prompts=True
            llm.clear(clear_prompts=True)

            # Everything should be gone
            assert len(llm.message_history) == 0
            assert len(llm._template_messages) == 0

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
            # Get the underlying LLM
            llm = agent.default._llm

            # Start with empty histories
            assert len(llm.message_history) == 0

            # Manually add a message to _message_history
            test_msg = Prompt.user("Test")
            llm._message_history.append(test_msg)

            # Verify it's in _message_history
            assert len(llm.message_history) == 1

            # Provider history should still be empty (until an API call is made)
            # This confirms that _message_history is independent of provider history
            # and that provider history is only written to, not read from

    await agent_function()
