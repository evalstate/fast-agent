from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_agent.context import Context
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.context_truncation import ContextTruncation
from mcp_agent.llm.memory import SimpleMemory
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


# A mock LLM class that simulates the integration of ContextTruncation
class TruncationTestLLM(AugmentedLLM):
    def __init__(self, context, truncation_strategy="simple", max_context_tokens=1000, **kwargs):
        # Simplified init for testing
        super().__init__(context=context, provider="test")
        self.history = SimpleMemory()
        self._message_history = []
        self.instruction = kwargs.get("instruction", "")
        self.name = "test_llm"
        self.aggregator = None

        # Truncation-specific properties
        self.truncation_strategy = truncation_strategy
        self.max_context_tokens = max_context_tokens
        self.context_truncation = ContextTruncation(context=context)

        # Mock the summarization LLM to avoid real LLM calls
        self._summarization_llm = kwargs.get("summarization_llm_mock")
        if self._summarization_llm:
            self.context_truncation.get_summarization_llm = lambda: self._summarization_llm

    async def generate(self, multipart_messages, request_params=None):
        # Use consistent model name throughout
        model_name = "gpt-4"
        
        # This simulates the core logic of the LLM's generate method
        
        # 1. Combine current history with new messages for the check
        temp_memory = SimpleMemory()
        temp_memory.set(self.history.get() + multipart_messages)

        # 2. Check if truncation is needed
        if self.context_truncation.needs_truncation(
            temp_memory, self.max_context_tokens, model_name, self.instruction
        ):
            self.logger.info("Truncation needed.")
            # 3. Apply the chosen truncation strategy
            if self.truncation_strategy == "summarize":
                self.history = await self.context_truncation.summarize_and_truncate(
                    self.history, self.max_context_tokens, model_name, self.instruction
                )
            else:
                self.history = self.context_truncation.truncate(
                    self.history, self.max_context_tokens, model_name, self.instruction
                )

        # 4. Add the new user messages to the (potentially truncated) history
        self.history.extend(multipart_messages)

        # 5. Generate a dummy response (keep it short to avoid token issues)
        response_text = f"Response to: {multipart_messages[-1].first_text()[:20]}..."
        response_message = PromptMessageMultipart(
            role="assistant", content=[{"type": "text", "text": response_text}]
        )
        self.history.append(response_message)
        self._message_history = self.history.get()
        return response_message

    async def _apply_prompt_provider_specific(self, multipart_messages, request_params=None, is_template=False):
        return await self.generate(multipart_messages, request_params)

    def _precall(self, multipart_messages):
        pass

def create_message(text, role="user", repeat=1):
    return PromptMessageMultipart(
        role=role, content=[{"type": "text", "text": f"{text} " * repeat}]
    )

@pytest.fixture
def mock_context():
    return Context()

@pytest.fixture
def summarization_llm_mock(mocker):
    mock_llm = MagicMock(spec=AugmentedLLM)
    # Return a short, consistent summary
    summary_message = create_message("Short summary.", role="assistant")
    mock_llm.generate = AsyncMock(return_value=summary_message)
    return mock_llm

@pytest.mark.asyncio
async def test_e2e_summarization_lifecycle(mock_context, summarization_llm_mock):
    """
    Tests the full summarization lifecycle with a low token limit, ensuring
    the final context is valid and smaller than the maximum.
    """
    # 1. Setup: LLM with a token limit that will be exceeded
    max_tokens = 300
    llm = TruncationTestLLM(
        mock_context,
        truncation_strategy="summarize",
        max_context_tokens=max_tokens,
        summarization_llm_mock=summarization_llm_mock,
    )

    # 2. Populate history with enough content to exceed the limit
    # Use a long, varied string to ensure a high token count, as tiktoken
    # is efficient with simple repeated text.
    long_text = (
        "This is a substantially longer piece of text designed to consume a significant "
        "number of tokens for testing purposes. It discusses various concepts like context "
        "windows, large language models, and truncation strategies. By using diverse "
        "vocabulary instead of simple repetition, we can create a more realistic test "
        "scenario that accurately reflects real-world usage and ensures our token counting "
        "and summarization logic is triggered correctly."
    )
    
    llm.history.extend([
        create_message(f"First old message. {long_text}", role="user"),
        create_message(f"First old response. {long_text}", role="assistant"),
        create_message(f"Second old message. {long_text}", role="user"),
        create_message(f"Second old response. {long_text}", role="assistant")
    ])

    # 3. Get initial token count for debugging
    initial_token_count = llm.context_truncation._estimate_tokens(llm.history.get(), "gpt-4")
    print(f"Initial token count: {initial_token_count}")
    # With the updated _estimate_tokens, this count will now be much higher and more accurate.
    assert initial_token_count > max_tokens, "Initial history should exceed max_tokens to trigger truncation."

    # 4. Action: Send a new message that should trigger summarization
    new_message = create_message("This is the new message that should trigger summarization.")
    
    # Calculate what the total would be
    temp_memory = SimpleMemory()
    temp_memory.set(llm.history.get() + [new_message])
    total_before_truncation = llm.context_truncation._estimate_tokens(temp_memory.get(), "gpt-4")
    print(f"Total tokens before truncation: {total_before_truncation}")
    
    # In this E2E test, the truncation happens on the *next* turn.
    # The generate method first adds the new message, then truncates the *existing* history
    # before the next call. Let's adjust the test logic to reflect that.
    # We will pre-load the history and then the generate call will truncate it.
    
    await llm.generate([new_message])

    # 5. Assertions
    
    # Assert that the summarization was actually called
    summarization_llm_mock.generate.assert_called_once()

    final_history = llm.history.get()
    
    # Print final history for debugging
    print(f"Final history length: {len(final_history)}")
    for i, msg in enumerate(final_history):
        print(f"  {i}: {msg.role}: {msg.first_text()[:50]}...")
    
    # Find the summary message to verify its content
    summary_user_message = next(
        (msg for msg in final_history if "Here is a summary" in msg.first_text()), 
        None
    )
    
    assert summary_user_message is not None, "Summary injection message not found in history"
    summary_text = summary_user_message.first_text().split(": ", 1)[1]
    
    assert "short summary" in summary_text.lower(), f"Expected 'short summary' in '{summary_text}'"
    
    # Assert that the final token count is below the maximum limit
    final_token_count = llm.context_truncation._estimate_tokens(final_history, "gpt-4")
    print(f"Final token count: {final_token_count}, Max: {max_tokens}")
    
    # The summarization logic keeps 50% of the context for recent messages.
    # The final count should be roughly 50% (the keep buffer) + summary + new message.
    assert final_token_count <= max_tokens, (
        f"Final token count {final_token_count} exceeds limit of {max_tokens}."
    )

    # Assert that the conversation flows correctly (new message and response are last)
    assert "This is the new message" in final_history[-2].first_text()
    assert "Response to:" in final_history[-1].first_text()
    
    # Verify that old messages were actually removed/summarized
    # The only original message that might remain is the one right before the summarization point.
    old_message_full_text = f"Second old message. {long_text}"
    
    found_full_old_message = any(
        old_message_full_text in msg.first_text() for msg in final_history
    )
    
    assert not found_full_old_message, "Summarization should have removed the oldest messages."

@pytest.mark.asyncio
async def test_no_truncation_when_under_limit(mock_context, summarization_llm_mock):
    """Test that no truncation occurs when under the token limit."""
    max_tokens = 1000  # High limit
    llm = TruncationTestLLM(
        mock_context,
        truncation_strategy="summarize",
        max_context_tokens=max_tokens,
        summarization_llm_mock=summarization_llm_mock,
    )

    # Add a small amount of history
    llm.history.extend([
        create_message("Short message", repeat=1),
        create_message("Short response", repeat=1, role="assistant"),
    ])

    new_message = create_message("Another short message", repeat=1)
    await llm.generate([new_message])

    # Summarization should not have been called
    summarization_llm_mock.generate.assert_not_called()

    # All original messages plus new ones should be present
    final_history = llm.history.get()
    assert len(final_history) == 4  # 2 original + 1 new + 1 response