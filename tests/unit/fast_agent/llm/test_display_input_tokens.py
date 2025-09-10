import unittest

from anthropic.types import Usage as AnthropicUsage
from openai.types.completion_usage import CompletionUsage as OpenAIUsage

from fast_agent.llm.usage_tracking import TurnUsage


class TestDisplayInputTokens(unittest.TestCase):
    """Test that display_input_tokens works correctly for different providers."""

    def test_anthropic_display_input_tokens_with_cache_read(self):
        """Test Anthropic display tokens include cache read tokens."""
        usage = AnthropicUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2020,  # Cache read
            input_tokens=142,  # New tokens only
            output_tokens=161,
        )

        turn = TurnUsage.from_anthropic(usage, "claude-sonnet-4-0")

        # Anthropic input_tokens excludes cache, so display should add cache tokens
        self.assertEqual(turn.input_tokens, 142)  # New tokens only
        self.assertEqual(turn.display_input_tokens, 142 + 2020)  # New + cache = 2162
        self.assertEqual(turn.cache_usage.cache_read_tokens, 2020)

    def test_anthropic_display_input_tokens_with_cache_write(self):
        """Test Anthropic display tokens include cache write tokens."""
        usage = AnthropicUsage(
            cache_creation_input_tokens=2020,  # Cache write
            cache_read_input_tokens=0,
            input_tokens=142,  # New tokens only
            output_tokens=140,
        )

        turn = TurnUsage.from_anthropic(usage, "claude-sonnet-4-0")

        # Anthropic input_tokens excludes cache, so display should add cache tokens
        self.assertEqual(turn.input_tokens, 142)  # New tokens only
        self.assertEqual(turn.display_input_tokens, 142 + 2020)  # New + cache = 2162
        self.assertEqual(turn.cache_usage.cache_write_tokens, 2020)

    def test_anthropic_display_input_tokens_no_cache(self):
        """Test Anthropic display tokens without cache."""
        usage = AnthropicUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=142,
            output_tokens=161,
        )

        turn = TurnUsage.from_anthropic(usage, "claude-sonnet-4-0")

        # No cache tokens, so display should equal input_tokens
        self.assertEqual(turn.input_tokens, 142)
        self.assertEqual(turn.display_input_tokens, 142)  # Same as input_tokens

    def test_openai_display_input_tokens_with_cache(self):
        """Test OpenAI display tokens (input_tokens already includes cache)."""
        # Mock OpenAI usage with cache
        usage = OpenAIUsage(
            prompt_tokens=2162,  # Total tokens including cached (142 + 2020)
            completion_tokens=161,
            total_tokens=2323,
        )

        # Add prompt_tokens_details with cached_tokens
        usage.prompt_tokens_details = type("obj", (object,), {"cached_tokens": 2020})()

        turn = TurnUsage.from_openai(usage, "gpt-4")

        # OpenAI input_tokens already includes cache, so display should be the same
        self.assertEqual(turn.input_tokens, 2162)  # Already includes cache
        self.assertEqual(turn.display_input_tokens, 2162)  # Same as input_tokens
        self.assertEqual(turn.cache_usage.cache_hit_tokens, 2020)

    def test_openai_display_input_tokens_no_cache(self):
        """Test OpenAI display tokens without cache."""
        usage = OpenAIUsage(prompt_tokens=142, completion_tokens=161, total_tokens=303)

        turn = TurnUsage.from_openai(usage, "gpt-4")

        # No cache, so display should equal input_tokens
        self.assertEqual(turn.input_tokens, 142)
        self.assertEqual(turn.display_input_tokens, 142)  # Same as input_tokens
        self.assertEqual(turn.cache_usage.cache_hit_tokens, 0)

    def test_cross_provider_consistency(self):
        """Test that display_input_tokens shows total submitted tokens for both providers."""

        # Scenario: 142 new tokens + 2020 cached tokens = 2162 total

        # Anthropic
        anthropic_usage = AnthropicUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2020,
            input_tokens=142,  # New only
            output_tokens=161,
        )
        anthropic_turn = TurnUsage.from_anthropic(anthropic_usage, "claude-sonnet-4-0")

        # OpenAI
        openai_usage = OpenAIUsage(
            prompt_tokens=2162,  # Total including cache
            completion_tokens=161,
            total_tokens=2323,
        )
        openai_usage.prompt_tokens_details = type("obj", (object,), {"cached_tokens": 2020})()
        openai_turn = TurnUsage.from_openai(openai_usage, "gpt-4")

        # Both should show the same total submitted tokens for display
        self.assertEqual(anthropic_turn.display_input_tokens, 2162)
        self.assertEqual(openai_turn.display_input_tokens, 2162)

        # But their internal input_tokens representation differs
        self.assertEqual(anthropic_turn.input_tokens, 142)  # New only
        self.assertEqual(openai_turn.input_tokens, 2162)  # Total


if __name__ == "__main__":
    unittest.main()
