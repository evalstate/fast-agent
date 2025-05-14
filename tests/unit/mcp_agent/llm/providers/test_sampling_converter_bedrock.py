"""
Unit tests for the BedrockSamplingConverter class.
"""

import unittest
from typing import Dict, Any

from mcp import StopReason
from mcp.types import PromptMessage, TextContent

from mcp_agent.llm.providers.sampling_converter_bedrock import (
    BedrockSamplingConverter,
    mcp_stop_reason_to_bedrock_stop_reason,
    bedrock_stop_reason_to_mcp_stop_reason,
)


class TestBedrockSamplingConverter(unittest.TestCase):
    """Tests for BedrockSamplingConverter."""
    
    def test_from_prompt_message(self):
        """Test converting PromptMessage to Bedrock format."""
        # Create a simple text message
        text_content = TextContent(type="text", text="Hello, world!")
        message = PromptMessage(role="user", content=text_content)
        
        # Convert to Bedrock format
        bedrock_message = BedrockSamplingConverter.from_prompt_message(message)
        
        # Assertions - check that it's converted to Bedrock format
        self.assertIsInstance(bedrock_message, Dict)
        # The default format is expected to be Claude format
        self.assertEqual(bedrock_message["role"], "user")
        
        # Content could be a list or direct field depending on model family
        if isinstance(bedrock_message["content"], list):
            # Claude format
            self.assertEqual(len(bedrock_message["content"]), 1)
            self.assertEqual(bedrock_message["content"][0]["type"], "text")
            self.assertEqual(bedrock_message["content"][0]["text"], "Hello, world!")
        else:
            # Could be Meta or Nova format
            self.assertIn(bedrock_message["content"], ["Hello, world!", bedrock_message])


class TestBedrockStopReasonConverter(unittest.TestCase):
    """Tests for BedrockStopReason conversion functions."""
    
    def test_mcp_to_bedrock_stop_reason(self):
        """Test converting MCP stop reasons to Bedrock stop reasons."""
        # Common stop reasons
        self.assertEqual(mcp_stop_reason_to_bedrock_stop_reason("endTurn"), "end_turn")
        self.assertEqual(mcp_stop_reason_to_bedrock_stop_reason("maxTokens"), "max_tokens")
        self.assertEqual(mcp_stop_reason_to_bedrock_stop_reason("stopSequence"), "stop_sequence")
        self.assertEqual(mcp_stop_reason_to_bedrock_stop_reason("toolUse"), "tool_use")
        
        # Edge cases
        self.assertEqual(mcp_stop_reason_to_bedrock_stop_reason(None), "end_turn")
        self.assertEqual(mcp_stop_reason_to_bedrock_stop_reason(""), "end_turn")
        
        # Unknown stop reason - should return the original
        self.assertEqual(mcp_stop_reason_to_bedrock_stop_reason("unknown"), "unknown")
    
    def test_bedrock_to_mcp_stop_reason(self):
        """Test converting Bedrock stop reasons to MCP stop reasons."""
        # Common stop reasons
        self.assertEqual(bedrock_stop_reason_to_mcp_stop_reason("end_turn"), "endTurn")
        self.assertEqual(bedrock_stop_reason_to_mcp_stop_reason("max_tokens"), "maxTokens")
        self.assertEqual(bedrock_stop_reason_to_mcp_stop_reason("stop_sequence"), "stopSequence")
        self.assertEqual(bedrock_stop_reason_to_mcp_stop_reason("tool_use"), "toolUse")
        
        # Edge cases
        self.assertEqual(bedrock_stop_reason_to_mcp_stop_reason(None), "endTurn")
        self.assertEqual(bedrock_stop_reason_to_mcp_stop_reason(""), "endTurn")
        
        # Unknown stop reason - should return the original
        self.assertEqual(bedrock_stop_reason_to_mcp_stop_reason("unknown"), "unknown")