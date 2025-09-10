import unittest

from anthropic.types import Usage

from fast_agent.llm.usage_tracking import TurnUsage, UsageAccumulator


class TestUsageTrackingCacheBilling(unittest.TestCase):
    """Test that cache tokens are properly included in billing calculations."""

    def test_anthropic_cache_billing_calculation(self):
        """Test that cached tokens are included in cumulative billing totals."""

        # Recreate the exact scenario from the debug output
        usage = Usage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2020,  # Cache read
            input_tokens=142,  # New tokens
            output_tokens=161,
        )

        turn = TurnUsage.from_anthropic(usage, "claude-sonnet-4-0")
        accumulator = UsageAccumulator()
        accumulator.add_turn(turn)

        # Debug what we're getting
        print(f"Turn total_tokens: {turn.total_tokens}")
        print(f"Turn input_tokens: {turn.input_tokens}")
        print(f"Turn output_tokens: {turn.output_tokens}")
        print(f"Cache read tokens: {turn.cache_usage.cache_read_tokens}")
        print(f"Current context: {turn.current_context_tokens}")
        print(f"Accumulator cumulative_input: {accumulator.cumulative_input_tokens}")
        print(f"Accumulator cumulative_billing: {accumulator.cumulative_billing_tokens}")

        # The issue: cumulative_billing should include cache tokens
        # Current: 142 + 161 = 303
        # Should be: (142 + 2020) + 161 = 2323 (total context)

        # For now, let's assert what we expect based on the debug output
        self.assertEqual(turn.input_tokens, 142)
        self.assertEqual(turn.output_tokens, 161)
        self.assertEqual(turn.cache_usage.cache_read_tokens, 2020)
        self.assertEqual(turn.current_context_tokens, 2323)  # This is correct

        # Fixed: this should now include cache tokens in billing
        expected_input = 142 + 2020  # New tokens + cache read tokens
        expected_billing = expected_input + 161  # Total input + output

        self.assertEqual(accumulator.cumulative_input_tokens, expected_input)
        self.assertEqual(accumulator.cumulative_billing_tokens, expected_billing)

    def test_anthropic_cache_write_billing_calculation(self):
        """Test cache write tokens are included in billing."""

        # First turn: cache write
        usage1 = Usage(
            cache_creation_input_tokens=2020,  # Cache write
            cache_read_input_tokens=0,
            input_tokens=142,
            output_tokens=140,
        )

        turn1 = TurnUsage.from_anthropic(usage1, "claude-sonnet-4-0")
        accumulator = UsageAccumulator()
        accumulator.add_turn(turn1)

        print("\nCache write turn:")
        print(f"Input tokens: {turn1.input_tokens}")
        print(f"Cache write tokens: {turn1.cache_usage.cache_write_tokens}")
        print(f"Current context: {turn1.current_context_tokens}")
        print(f"Cumulative input: {accumulator.cumulative_input_tokens}")
        print(f"Cumulative billing: {accumulator.cumulative_billing_tokens}")

        # Cache write should also be included in billing totals
        self.assertEqual(turn1.cache_usage.cache_write_tokens, 2020)
        self.assertEqual(turn1.current_context_tokens, 2302)  # 142 + 2020 + 140

        # This should include cache write tokens in the cumulative totals
        expected_total_input = 142 + 2020  # New tokens + cache tokens
        expected_total_billing = expected_total_input + 140  # + output tokens

        # Fixed: cache tokens are now included
        self.assertEqual(accumulator.cumulative_input_tokens, expected_total_input)
        self.assertEqual(accumulator.cumulative_billing_tokens, expected_total_billing)

    def test_cumulative_with_multiple_cache_turns(self):
        """Test cumulative billing across multiple turns with cache operations."""

        # Turn 1: Cache write
        usage1 = Usage(
            cache_creation_input_tokens=2020,
            cache_read_input_tokens=0,
            input_tokens=142,
            output_tokens=140,
        )

        # Turn 2: Cache read
        usage2 = Usage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2020,
            input_tokens=289,
            output_tokens=87,
        )

        accumulator = UsageAccumulator()
        accumulator.add_turn(TurnUsage.from_anthropic(usage1, "claude-sonnet-4-0"))
        accumulator.add_turn(TurnUsage.from_anthropic(usage2, "claude-sonnet-4-0"))

        print("\nMultiple turns:")
        print(f"Cumulative input: {accumulator.cumulative_input_tokens}")
        print(f"Cumulative output: {accumulator.cumulative_output_tokens}")
        print(f"Cumulative billing: {accumulator.cumulative_billing_tokens}")
        print(f"Cache read tokens: {accumulator.cumulative_cache_read_tokens}")
        print(f"Cache write tokens: {accumulator.cumulative_cache_write_tokens}")

        # Expected totals (including cache tokens in billing):
        expected_input = 142 + 289 + 2020 + 2020  # new tokens + cache tokens from both turns
        expected_output = 140 + 87
        expected_billing = expected_input + expected_output

        # Fixed: cache tokens are now included in billing
        self.assertEqual(accumulator.cumulative_input_tokens, expected_input)
        self.assertEqual(accumulator.cumulative_output_tokens, expected_output)
        self.assertEqual(accumulator.cumulative_billing_tokens, expected_billing)


if __name__ == "__main__":
    unittest.main()
