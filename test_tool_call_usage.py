#!/usr/bin/env python3
"""
Test tool call usage tracking with PassthroughLLM.
"""

import asyncio
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM


async def test_tool_call_usage():
    """Test that tool calls are tracked in usage"""
    llm = PassthroughLLM()
    
    print("=== Testing Tool Call Usage Tracking ===")
    
    # Initially no usage
    print(f"Initial usage: {llm.usage_accumulator.turn_count} turns")
    
    # Make a regular message
    print("\n1. Regular message:")
    await llm.generate([Prompt.user("Hello there")])
    print(f"After regular message: {llm.usage_accumulator.turn_count} turns")
    
    # Make a tool call
    print("\n2. Tool call:")
    await llm.generate([Prompt.user("***CALL_TOOL some_tool {}")])
    print(f"After tool call: {llm.usage_accumulator.turn_count} turns")
    
    # Check the usage summary
    summary = llm.get_usage_summary()
    print(f"\nFinal summary:")
    print(f"  Total turns: {summary['turn_count']}")
    print(f"  Cumulative billing tokens: {summary['cumulative_billing_tokens']}")
    print(f"  Current context tokens: {summary['current_context_tokens']}")


if __name__ == "__main__":
    asyncio.run(test_tool_call_usage())