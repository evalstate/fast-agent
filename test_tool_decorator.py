"""
Quick test script for the @fast.tool decorator.

Demonstrates:
  1. Bare @fast.tool — uses function name and docstring
  2. Parameterized @fast.tool(name=..., description=...) — custom name/description
  3. Global tools are automatically available to all agents
  4. Agents with explicit function_tools only see those tools (opt-out of globals)

Uses the 'passthrough' model so no API keys are needed.
Run with: uv run test_tool_decorator.py
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent("Tool Test", parse_cli_args=False, quiet=True)


# --- Global tools (available to all agents without explicit function_tools) ---


@fast.tool
def say_hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@fast.tool(name="add_numbers", description="Add two integers together")
def add(a: int, b: int) -> int:
    return a + b


# --- A standalone function used as an explicit function_tool ---


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# --- Agents ---


@fast.agent(name="assistant", model="passthrough", instruction="You are helpful.")
@fast.agent(
    name="calculator",
    model="passthrough",
    instruction="You do math.",
    function_tools=[multiply],
)
async def main():
    async with fast.run() as agent:
        # --- Test 1: Global tools on "assistant" ---
        print("=== assistant (global @fast.tool tools) ===")
        tools = await agent.assistant.list_tools()
        tool_names = [t.name for t in tools.tools]
        print("  Available tools:", tool_names)

        assert "say_hello" in tool_names, "say_hello tool not found on assistant!"
        assert "add_numbers" in tool_names, "add_numbers tool not found on assistant!"

        result = await agent.assistant.call_tool("say_hello", {"name": "World"})
        print(f"  say_hello result: {result.content[0].text}")
        assert result.content[0].text == "Hello, World!"

        result = await agent.assistant.call_tool("add_numbers", {"a": 3, "b": 7})
        print(f"  add_numbers result: {result.content[0].text}")
        assert result.content[0].text == "10"

        # --- Test 2: Explicit function_tools on "calculator" (no globals) ---
        print("\n=== calculator (explicit function_tools only) ===")
        calc_tools = await agent.calculator.list_tools()
        calc_tool_names = [t.name for t in calc_tools.tools]
        print("  Available tools:", calc_tool_names)

        assert "multiply" in calc_tool_names, "multiply tool not found on calculator!"
        assert "say_hello" not in calc_tool_names, (
            "say_hello should NOT be on calculator (explicit function_tools overrides globals)!"
        )
        assert "add_numbers" not in calc_tool_names, (
            "add_numbers should NOT be on calculator (explicit function_tools overrides globals)!"
        )

        result = await agent.calculator.call_tool("multiply", {"a": 4, "b": 5})
        print(f"  multiply result: {result.content[0].text}")
        assert result.content[0].text == "20"

        print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
