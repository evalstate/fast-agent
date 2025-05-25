"""
Tests for AnthropicMCPTypeConverter.
"""


class TestAnthropicMCPTypeConverter:
    def test_stop_reason_conversions(self):
        """Test various stop reason conversions."""
        from mcp_agent.llm.providers.sampling_converter_anthropic import (
            anthropic_stop_reason_to_mcp_stop_reason,
            mcp_stop_reason_to_anthropic_stop_reason,
        )

        # Test MCP to Anthropic conversions
        assert mcp_stop_reason_to_anthropic_stop_reason("endTurn") == "end_turn"
        assert mcp_stop_reason_to_anthropic_stop_reason("maxTokens") == "max_tokens"
        assert mcp_stop_reason_to_anthropic_stop_reason("stopSequence") == "stop_sequence"
        assert mcp_stop_reason_to_anthropic_stop_reason("toolUse") == "tool_use"
        assert mcp_stop_reason_to_anthropic_stop_reason("unknown") == "unknown"

        # Test Anthropic to MCP conversions
        assert anthropic_stop_reason_to_mcp_stop_reason("end_turn") == "endTurn"
        assert anthropic_stop_reason_to_mcp_stop_reason("max_tokens") == "maxTokens"
        assert anthropic_stop_reason_to_mcp_stop_reason("stop_sequence") == "stopSequence"
        assert anthropic_stop_reason_to_mcp_stop_reason("tool_use") == "toolUse"
        assert anthropic_stop_reason_to_mcp_stop_reason("unknown") == "unknown"
