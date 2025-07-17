
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_agent.context import Context
from mcp_agent.llm.context_truncation import ContextTruncation
from mcp_agent.llm.memory import SimpleMemory
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


@pytest.fixture
def mock_context():
    """Fixture for a mock application context."""
    return Context()

@pytest.fixture
def context_truncation(mock_context):
    """Fixture for ContextTruncation instance."""
    return ContextTruncation(context=mock_context)

@pytest.fixture
def memory():
    """Fixture for SimpleMemory instance."""
    return SimpleMemory()

def create_message(role: str, text: str) -> PromptMessageMultipart:
    """Helper to create a PromptMessageMultipart."""
    return PromptMessageMultipart(role=role, content=[{"type": "text", "text": text}])

def test_initialization(context_truncation):
    """Test that ContextTruncation initializes correctly."""
    assert context_truncation.context is not None
    assert context_truncation.logger is not None

def test_needs_truncation_false(context_truncation, memory):
    """Test that needs_truncation returns False when context is within limits."""
    memory.extend([create_message("user", "Hello")])
    assert not context_truncation.needs_truncation(memory, max_tokens=1000, model="test-model")

def test_needs_truncation_true(context_truncation, memory):
    """Test that needs_truncation returns True when context exceeds limits."""
    large_text = "This is a realistic test sentence. " * 100 # Approx. 800 tokens
    memory.extend([create_message("user", large_text)])
    assert context_truncation.needs_truncation(memory, max_tokens=400, model="test-model") is True

def test_truncate_simple_removal(context_truncation, memory):
    """Test that truncate removes the oldest non-system messages."""
    memory.extend(
        [
            create_message("user", "First message"),
            create_message("assistant", "First response"),
            create_message("user", "Second message" * 500),  # Large message
        ]
    )
    truncated_memory = context_truncation.truncate(memory, max_tokens=500, model="test-model")
    final_messages = truncated_memory.get()
    # Should remove the first user/assistant messages
    assert len(final_messages) == 1
    assert "Second message" in final_messages[0].first_text()

@pytest.mark.asyncio
async def test_summarize_and_truncate_conversational_injection(context_truncation, memory, mocker):
    """Test that summarize_and_truncate uses the conversational injection pattern."""
    # Mock the summarization LLM call
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=create_message("assistant", "Summary of old messages."))
    context_truncation.get_summarization_llm = MagicMock(return_value=mock_llm)

    memory.extend(
        [
            create_message("user", "Old message 1 " * 100),
            create_message("assistant", "Old response 1 " * 100),
            create_message("user", "Recent message to keep"),
        ]
    )

    # Set max_tokens to trigger summarization
    truncated_memory = await context_truncation.summarize_and_truncate(
        memory, max_tokens=200, model="test-model"
    )

    # Verify summarization was called
    context_truncation.get_summarization_llm.assert_called_once()
    mock_llm.generate.assert_called_once()

    # Check the new memory content for the conversational pattern
    final_messages = truncated_memory.get()
    assert len(final_messages) == 3  # Summary User + Summary Assistant + Kept Message
    
    # 1. Conversational summary injection
    assert final_messages[0].role == "user"
    assert "Here is a summary of our conversation so far: Summary of old messages." in final_messages[0].first_text()
    assert final_messages[1].role == "assistant"
    assert "Thanks, I am caught up. Let's continue." in final_messages[1].first_text()

    # 2. Recent message is preserved
    assert final_messages[2].role == "user"
    assert final_messages[2].first_text() == "Recent message to keep"

def test_estimate_tokens_with_tiktoken(context_truncation):
    """Test token estimation using tiktoken."""
    messages = [create_message("user", "This is a test sentence.")]
    assert context_truncation._estimate_tokens(messages, "gpt-4") == 10

def test_estimate_tokens_fallback(context_truncation):
    """Test token estimation fallback for unknown models."""
    messages = [create_message("user", "This is a test sentence.")]
    assert context_truncation._estimate_tokens(messages, "unknown-model") == 10
