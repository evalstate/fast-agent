"""
Tests for OpenAIMCPTypeConverter.
"""

from mcp.types import (
    PromptMessage,
    TextContent,
)

from mcp_agent.llm.providers import OpenAISamplingConverter


class TestOpenAIMCPTypeConverter:
    def test_from_mcp_prompt_message_user(self):
        """Test converting a user PromptMessage to OpenAI ChatCompletionMessageParam."""
        # Create a user PromptMessage
        prompt_message = PromptMessage(
            role="user",
            content=TextContent(type="text", text="Please explain this concept."),
        )

        # Convert to OpenAI ChatCompletionMessageParam
        openai_param = OpenAISamplingConverter.from_prompt_message(prompt_message)

        # Verify the conversion
        assert isinstance(openai_param, dict)  # ChatCompletionMessageParam is a TypedDict
        assert openai_param["role"] == "user"
        assert "Please explain this concept." == openai_param["content"]

    def test_from_mcp_prompt_message_assistant(self):
        """Test converting an assistant PromptMessage to OpenAI ChatCompletionMessageParam."""
        # Create an assistant PromptMessage
        prompt_message = PromptMessage(
            role="assistant",
            content=TextContent(type="text", text="Here's the explanation..."),
        )

        # Convert to OpenAI ChatCompletionMessageParam
        openai_param = OpenAISamplingConverter.from_prompt_message(prompt_message)

        # Verify the conversion
        assert isinstance(openai_param, dict)  # ChatCompletionMessageParam is a TypedDict
        assert openai_param["role"] == "assistant"
        assert openai_param["content"] == "Here's the explanation..."
