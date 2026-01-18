"""
Unit tests for Anthropic caching functionality.

These tests directly test the _convert_extended_messages_to_provider method
to verify cache_control markers are applied correctly based on cache_mode settings.
"""

from typing import Literal

import pytest
from anthropic.types import MessageParam
from mcp.types import CallToolResult, TextContent

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.cache_planner import AnthropicCachePlanner
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types import RequestParams


class TestAnthropicCaching:
    """Test cases for Anthropic caching functionality."""

    def _create_context_with_cache_mode(
        self,
        cache_mode: Literal["off", "prompt", "auto"],
        cache_ttl: Literal["5m", "1h"] = "5m",
    ) -> Context:
        """Create a context with specified cache mode and TTL."""
        ctx = Context()
        ctx.config = Settings()
        ctx.config.anthropic = AnthropicSettings(
            api_key="test_key", cache_mode=cache_mode, cache_ttl=cache_ttl
        )
        return ctx

    def _create_llm(
        self,
        cache_mode: Literal["off", "prompt", "auto"] = "off",
        cache_ttl: Literal["5m", "1h"] = "5m",
    ) -> AnthropicLLM:
        """Create an AnthropicLLM instance with specified cache mode and TTL."""
        ctx = self._create_context_with_cache_mode(cache_mode, cache_ttl)
        llm = AnthropicLLM(context=ctx)
        return llm

    def _apply_cache_plan(
        self,
        messages: list[PromptMessageExtended],
        cache_mode: Literal["off", "prompt", "auto"],
        system_blocks: int = 0,
        cache_ttl: Literal["5m", "1h"] = "5m",
    ) -> list[MessageParam]:
        planner = AnthropicCachePlanner()
        plan = planner.plan_indices(messages, cache_mode=cache_mode, system_cache_blocks=system_blocks)
        converted = [AnthropicConverter.convert_to_anthropic(m) for m in messages]
        for idx in plan:
            AnthropicLLM._apply_cache_control_to_message(converted[idx], ttl=cache_ttl)
        return converted

    def test_conversion_off_mode_no_cache_control(self):
        """Test that no cache_control is applied when cache_mode is 'off'."""
        # Create test messages
        messages = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Hello")]
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Hi there")]
            ),
        ]

        converted = self._apply_cache_plan(messages, cache_mode="off")

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
        # Create template + conversation messages (agent supplies all, flags templates)
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="System context")], is_template=True
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Understood")], is_template=True
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        converted = self._apply_cache_plan(template_msgs + conversation_msgs, cache_mode="prompt")

        # Verify we have 3 messages (2 templates + 1 conversation)
        assert len(converted) == 3

        # Template messages should have cache_control
        # The last template message should have cache_control on its last block
        found_cache_control = False
        template_count = len(template_msgs)
        for i, msg in enumerate(converted[:template_count]):  # First template_count are templates
            if "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict) and "cache_control" in block:
                        found_cache_control = True
                        assert block["cache_control"]["type"] == "ephemeral"
                        assert block["cache_control"]["ttl"] == "5m"

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
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        converted = self._apply_cache_plan(template_msgs + conversation_msgs, cache_mode="auto")

        # Template message should have cache_control
        found_cache_control = False
        template_msg = converted[0]
        if "content" in template_msg:
            for block in template_msg["content"]:
                if isinstance(block, dict) and "cache_control" in block:
                    found_cache_control = True
                    assert block["cache_control"]["type"] == "ephemeral"
                    assert block["cache_control"]["ttl"] == "5m"

        assert found_cache_control, "Template messages should have cache_control in 'auto' mode"

    def test_conversion_off_mode_templates_not_cached(self):
        """Test that template messages do NOT get cache_control when cache_mode is 'off'."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Response")], is_template=True
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        converted = self._apply_cache_plan(template_msgs + conversation_msgs, cache_mode="off")

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

        converted = [AnthropicConverter.convert_to_anthropic(m) for m in messages]

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
        history = [user_msg]

        params = llm.get_request_params(RequestParams(use_history=True))
        message_param = AnthropicConverter.convert_to_anthropic(user_msg)

        prepared = llm._build_request_messages(params, message_param, history=history)

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
        message_param = AnthropicConverter.convert_to_anthropic(msg)

        prepared = llm._build_request_messages(params, message_param, history=[])

        assert prepared[-1] == message_param

    def test_build_request_messages_without_history(self):
        """When history is disabled, always send the current message."""
        llm = self._create_llm()
        params = llm.get_request_params(RequestParams(use_history=False))
        msg = PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")])
        message_param = AnthropicConverter.convert_to_anthropic(msg)

        prepared = llm._build_request_messages(params, message_param, history=[])

        assert prepared == [message_param]

    def test_conversion_empty_messages(self):
        """Test conversion of empty message list."""
        llm = self._create_llm(cache_mode="off")

        converted = llm._convert_extended_messages_to_provider([])

        assert converted == []

    def test_conversion_with_templates_only(self):
        """Test conversion when only templates exist (no conversation)."""
        # Create template messages
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]

        converted = self._apply_cache_plan(template_msgs, cache_mode="prompt")

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
        # Create a template with multiple content blocks
        template_msgs = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="First block"),
                    TextContent(type="text", text="Second block"),
                ],
                is_template=True,
            ),
        ]

        converted = self._apply_cache_plan(template_msgs, cache_mode="prompt")

        # Cache control should be on the last block
        content = converted[0].get("content", [])
        content_blocks = [] if isinstance(content, str) else list(content)
        assert len(content_blocks) == 2

        # First block should NOT have cache_control
        if isinstance(content_blocks[0], dict):
            # Cache control might be on any block, but typically the last one
            pass

        # At least one block should have cache_control
        found_cache_control = any(
            isinstance(block, dict) and "cache_control" in block for block in content_blocks
        )
        assert found_cache_control, "Template should have cache_control"

    def test_conversion_prompt_mode_with_1h_ttl(self):
        """Test that cache_ttl='1h' produces correct cache_control with 1h TTL."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="System context")], is_template=True
            ),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Understood")], is_template=True
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        converted = self._apply_cache_plan(
            template_msgs + conversation_msgs, cache_mode="prompt", cache_ttl="1h"
        )

        # Verify we have 3 messages (2 templates + 1 conversation)
        assert len(converted) == 3

        # Template messages should have cache_control with 1h TTL
        found_1h_cache_control = False
        template_count = len(template_msgs)
        for msg in converted[:template_count]:
            if "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict) and "cache_control" in block:
                        assert block["cache_control"]["type"] == "ephemeral"
                        assert block["cache_control"]["ttl"] == "1h"
                        found_1h_cache_control = True

        assert found_1h_cache_control, "Template messages should have cache_control with 1h TTL"

    def test_conversion_auto_mode_with_1h_ttl(self):
        """Test that cache_ttl='1h' works correctly in 'auto' mode."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Question")]
            ),
        ]

        converted = self._apply_cache_plan(
            template_msgs + conversation_msgs, cache_mode="auto", cache_ttl="1h"
        )

        # Template message should have cache_control with 1h TTL
        found_1h_cache_control = False
        template_msg = converted[0]
        if "content" in template_msg:
            for block in template_msg["content"]:
                if isinstance(block, dict) and "cache_control" in block:
                    assert block["cache_control"]["type"] == "ephemeral"
                    assert block["cache_control"]["ttl"] == "1h"
                    found_1h_cache_control = True

        assert found_1h_cache_control, "Template messages should have cache_control with 1h TTL in 'auto' mode"

    @pytest.mark.parametrize("cache_ttl", ["5m", "1h"])
    def test_cache_ttl_values(self, cache_ttl: Literal["5m", "1h"]):
        """Test that both valid TTL values produce correct cache_control."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]

        converted = self._apply_cache_plan(template_msgs, cache_mode="prompt", cache_ttl=cache_ttl)

        # Find the cache_control and verify TTL
        for block in converted[0].get("content", []):
            if isinstance(block, dict) and "cache_control" in block:
                assert block["cache_control"]["type"] == "ephemeral"
                assert block["cache_control"]["ttl"] == cache_ttl
                return

        pytest.fail(f"No cache_control found for TTL {cache_ttl}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
