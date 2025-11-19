import unittest
from typing import Any, Dict

from mcp.types import TextContent

from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def apply_cache_control_to_message(message: Dict[str, Any], position: int) -> bool:
    """
    Apply cache control to a message at the specified position.

    Args:
        message: The message dict to modify
        position: The position in the message list (for logging/tracking)

    Returns:
        True if cache control was applied, False otherwise
    """
    if not isinstance(message, dict) or "content" not in message:
        return False

    content_list = message["content"]
    if not isinstance(content_list, list) or not content_list:
        return False

    # Apply cache control to the last content block of this message
    for content_block in reversed(content_list):
        if isinstance(content_block, dict):
            content_block["cache_control"] = {"type": "ephemeral"}
            return True

    return False


class TestCacheControlApplication(unittest.TestCase):
    """Test cache control application logic without mocks."""

    def test_apply_cache_control_to_valid_message(self):
        """Test applying cache control to a valid message."""
        message = {"role": "user", "content": [{"type": "text", "text": "Hello world"}]}

        success = apply_cache_control_to_message(message, 0)

        self.assertTrue(success)
        self.assertIn("cache_control", message["content"][0])
        self.assertEqual(message["content"][0]["cache_control"]["type"], "ephemeral")

    def test_apply_cache_control_to_message_with_multiple_blocks(self):
        """Test applying cache control to message with multiple content blocks."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First block"},
                {"type": "text", "text": "Second block"},
            ],
        }

        success = apply_cache_control_to_message(message, 0)

        self.assertTrue(success)
        # Should apply to the last block
        self.assertNotIn("cache_control", message["content"][0])
        self.assertIn("cache_control", message["content"][1])
        self.assertEqual(message["content"][1]["cache_control"]["type"], "ephemeral")

    def test_apply_cache_control_to_invalid_message(self):
        """Test that cache control is not applied to invalid messages."""
        # Test with missing content
        message1 = {"role": "user"}
        success1 = apply_cache_control_to_message(message1, 0)
        self.assertFalse(success1)

        # Test with empty content
        message2 = {"role": "user", "content": []}
        success2 = apply_cache_control_to_message(message2, 0)
        self.assertFalse(success2)

        # Test with non-dict content
        message3 = {"role": "user", "content": ["not a dict"]}
        success3 = apply_cache_control_to_message(message3, 0)
        self.assertFalse(success3)

    def test_apply_cache_control_preserves_existing_content(self):
        """Test that applying cache control preserves existing message content."""
        original_text = "Important message"
        message = {"role": "user", "content": [{"type": "text", "text": original_text}]}

        success = apply_cache_control_to_message(message, 0)

        self.assertTrue(success)
        # Original content should be preserved
        self.assertEqual(message["content"][0]["text"], original_text)
        self.assertEqual(message["content"][0]["type"], "text")
        # Cache control should be added
        self.assertIn("cache_control", message["content"][0])

    def test_multipart_to_anthropic_conversion_preserves_structure(self):
        """Test that PromptMessageExtended -> Anthropic conversion preserves structure for caching."""
        # Create a multipart message
        text_content = TextContent(type="text", text="Test message for caching")
        multipart = PromptMessageExtended(role="user", content=[text_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Verify structure is suitable for cache control application
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertIsInstance(anthropic_msg["content"], list)
        self.assertTrue(len(anthropic_msg["content"]) > 0)
        self.assertIsInstance(anthropic_msg["content"][0], dict)

        # Apply cache control to the converted message
        success = apply_cache_control_to_message(anthropic_msg, 0)
        self.assertTrue(success)
        self.assertIn("cache_control", anthropic_msg["content"][0])

    def test_multiple_multipart_messages_conversion_and_caching(self):
        """Test converting multiple multipart messages and applying cache to specific position."""
        # Create multiple messages
        messages = []
        for i in range(4):
            text_content = TextContent(type="text", text=f"Message {i}")
            multipart = PromptMessageExtended(
                role="user" if i % 2 == 0 else "assistant", content=[text_content]
            )
            anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)
            messages.append(anthropic_msg)

        # Apply cache control to position 2 (3rd message)
        cache_position = 2
        success = apply_cache_control_to_message(messages[cache_position], cache_position)

        self.assertTrue(success)

        # Verify only the target message has cache control
        for i, msg in enumerate(messages):
            if i == cache_position:
                self.assertIn("cache_control", msg["content"][0])
                self.assertEqual(msg["content"][0]["cache_control"]["type"], "ephemeral")
            else:
                self.assertNotIn("cache_control", msg["content"][0])

    def test_assistant_message_caching(self):
        """Test that assistant messages can also receive cache control."""
        text_content = TextContent(type="text", text="Assistant response")
        multipart = PromptMessageExtended(role="assistant", content=[text_content])

        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)
        success = apply_cache_control_to_message(anthropic_msg, 0)

        self.assertTrue(success)
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertIn("cache_control", anthropic_msg["content"][0])

    def test_cache_control_application_idempotent(self):
        """Test that applying cache control multiple times doesn't break anything."""
        message = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}

        # Apply cache control twice
        success1 = apply_cache_control_to_message(message, 0)
        success2 = apply_cache_control_to_message(message, 0)

        self.assertTrue(success1)
        self.assertTrue(success2)

        # Should still have cache control
        self.assertIn("cache_control", message["content"][0])
        self.assertEqual(message["content"][0]["cache_control"]["type"], "ephemeral")


if __name__ == "__main__":
    unittest.main()
