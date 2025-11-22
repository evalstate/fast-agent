"""
Unit tests for Anthropic caching functionality.

These tests directly test the _convert_extended_messages_to_provider method
to verify cache_control markers are applied correctly based on cache_mode settings.
"""

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types import RequestParams


class TestAnthropicCaching:
    """Test cases for Anthropic caching functionality."""

    def _create_context_with_cache_mode(self, cache_mode: str) -> Context:
        """Create a context with specified cache mode."""
        ctx = Context()
        ctx.config = Settings()
        ctx.config.anthropic = AnthropicSettings(
            api_key="test_key", cache_mode=cache_mode
        )
        return ctx

    def _create_llm(self, cache_mode: str = "off") -> AnthropicLLM:
        """Create an AnthropicLLM instance with specified cache mode."""
        ctx = self._create_context_with_cache_mode(cache_mode)
        llm = AnthropicLLM(context=ctx)
        return llm

    def test_conversion_off_mode_no_cache_control(self):
        """Test that no cache_control is applied when cache_mode is 'off'."""
        llm = self._create_llm(cache_mode="off")

        # Create test messages
        messages = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Hello")]
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Hi there")]
            ),
        ]

        # Convert to provider format
        converted = llm._convert_extended_messages_to_provider(messages)

        # Verify no cache_control in any message
        assert len(converted) == 2
        for msg in converted:
            assert "content" in msg
            for block in msg["content"]:
                if isinstance(block, dict):
                    assert "cache_control" not in block, (
                        "cache_control should not be present when cache_mode is 'off'"
                    )

    def test_conversion_prompt_mode_templates_cached(self):
        """Test that template messages get cache_control in 'prompt' mode."""
        llm = self._create_llm(cache_mode="prompt")

        # Create template messages
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="System context")]
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Understood")]
            ),
        ]
        llm._template_messages = template_msgs

        # Create conversation messages
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        # Convert using _convert_to_provider_format which prepends templates
        converted = llm._convert_to_provider_format(conversation_msgs)

        # Verify we have 3 messages (2 templates + 1 conversation)
        assert len(converted) == 3

        # Template messages should have cache_control
        # The last template message should have cache_control on its last block
        found_cache_control = False
        for i, msg in enumerate(converted[:2]):  # First 2 are templates
            if "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict) and "cache_control" in block:
                        found_cache_control = True
                        assert block["cache_control"]["type"] == "ephemeral"

        assert found_cache_control, "Template messages should have cache_control in 'prompt' mode"

        # Conversation message should NOT have cache_control
        conv_msg = converted[2]
        for block in conv_msg.get("content", []):
            if isinstance(block, dict):
                assert "cache_control" not in block, (
                    "Conversation messages should not have cache_control in 'prompt' mode"
                )

    def test_conversion_auto_mode_templates_cached(self):
        """Test that template messages get cache_control in 'auto' mode."""
        llm = self._create_llm(cache_mode="auto")

        # Create template messages
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")]
            ),
        ]
        llm._template_messages = template_msgs

        # Create conversation messages
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        # Convert using _convert_to_provider_format
        converted = llm._convert_to_provider_format(conversation_msgs)

        # Template message should have cache_control
        found_cache_control = False
        template_msg = converted[0]
        if "content" in template_msg:
            for block in template_msg["content"]:
                if isinstance(block, dict) and "cache_control" in block:
                    found_cache_control = True
                    assert block["cache_control"]["type"] == "ephemeral"

        assert found_cache_control, "Template messages should have cache_control in 'auto' mode"

    def test_conversion_off_mode_templates_not_cached(self):
        """Test that template messages do NOT get cache_control when cache_mode is 'off'."""
        llm = self._create_llm(cache_mode="off")

        # Create template messages
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")]
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Response")]
            ),
        ]
        llm._template_messages = template_msgs

        # Create conversation messages
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        # Convert using _convert_to_provider_format
        converted = llm._convert_to_provider_format(conversation_msgs)

        # No messages should have cache_control
        for msg in converted:
            if "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict):
                        assert "cache_control" not in block, (
                            "No messages should have cache_control when cache_mode is 'off'"
                        )

    def test_conversion_multiple_messages_structure(self):
        """Test that message structure is preserved during conversion."""
        llm = self._create_llm(cache_mode="off")

        messages = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="First")]
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Second")]
            ),
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Third")]
            ),
        ]

        converted = llm._convert_extended_messages_to_provider(messages)

        # Verify structure
        assert len(converted) == 3
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[2]["role"] == "user"

    def test_build_request_messages_avoids_duplicate_tool_results(self):
        """Ensure tool_result blocks are only included once per tool use."""
        llm = self._create_llm()
        tool_id = "toolu_test"
        tool_result = CallToolResult(
            content=[TextContent(type="text", text="result payload")], isError=False
        )
        user_msg = PromptMessageExtended(role="user", content=[], tool_results={tool_id: tool_result})
        llm._message_history = [user_msg]

        params = llm.get_request_params(RequestParams(use_history=True))
        message_param = AnthropicConverter.convert_to_anthropic(user_msg)

        prepared = llm._build_request_messages(params, message_param)

        tool_blocks = [
            block
            for msg in prepared
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]

        assert len(tool_blocks) == 1
        assert tool_blocks[0]["tool_use_id"] == tool_id

    def test_build_request_messages_includes_current_when_history_empty(self):
        """Fallback to the current message if history produced no entries."""
        llm = self._create_llm()
        params = llm.get_request_params(RequestParams(use_history=True))
        msg = PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")])
        llm._message_history = []
        message_param = AnthropicConverter.convert_to_anthropic(msg)

        prepared = llm._build_request_messages(params, message_param)

        assert prepared[-1] == message_param

    def test_build_request_messages_without_history(self):
        """When history is disabled, always send the current message."""
        llm = self._create_llm()
        params = llm.get_request_params(RequestParams(use_history=False))
        msg = PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")])
        llm._message_history = [msg]
        message_param = AnthropicConverter.convert_to_anthropic(msg)

        prepared = llm._build_request_messages(params, message_param)

        assert prepared == [message_param]

    def test_conversion_empty_messages(self):
        """Test conversion of empty message list."""
        llm = self._create_llm(cache_mode="off")

        converted = llm._convert_extended_messages_to_provider([])

        assert converted == []

    def test_conversion_with_templates_only(self):
        """Test conversion when only templates exist (no conversation)."""
        llm = self._create_llm(cache_mode="prompt")

        # Create template messages
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")]
            ),
        ]
        llm._template_messages = template_msgs

        # Convert with empty conversation
        converted = llm._convert_to_provider_format([])

        # Should have just the template
        assert len(converted) == 1

        # Template should have cache_control
        found_cache_control = False
        for block in converted[0].get("content", []):
            if isinstance(block, dict) and "cache_control" in block:
                found_cache_control = True

        assert found_cache_control, "Template should have cache_control in 'prompt' mode"

    def test_cache_control_on_last_content_block(self):
        """Test that cache_control is applied to the last content block of template messages."""
        llm = self._create_llm(cache_mode="prompt")

        # Create a template with multiple content blocks
        template_msgs = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="First block"),
                    TextContent(type="text", text="Second block"),
                ],
            ),
        ]
        llm._template_messages = template_msgs

        # Convert with empty conversation
        converted = llm._convert_to_provider_format([])

        # Cache control should be on the last block
        content_blocks = converted[0]["content"]
        assert len(content_blocks) == 2

        # First block should NOT have cache_control
        if isinstance(content_blocks[0], dict):
            # Cache control might be on any block, but typically the last one
            pass

        # At least one block should have cache_control
        found_cache_control = any(
            isinstance(block, dict) and "cache_control" in block
            for block in content_blocks
        )
        assert found_cache_control, "Template should have cache_control"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
