#!/usr/bin/env python3
"""Test script for streaming buffer functionality.

This script tests the key design points:
1. Stream tokens and render as markdown
2. Truncate when hitting screen limits
3. Preserve code block fences
4. Preserve table headers
5. Add closing fence for unclosed blocks
"""

import sys
sys.path.insert(0, 'src')

from fast_agent.ui.streaming_buffer import StreamBuffer


def test_basic_streaming():
    """Test basic append and retrieve."""
    print("Test 1: Basic streaming")
    buffer = StreamBuffer()
    buffer.append("Hello ")
    buffer.append("world!")
    assert buffer.get_full_text() == "Hello world!"
    print("✓ Basic streaming works\n")


def test_no_truncation_when_fits():
    """Test that content within limits is not truncated."""
    print("Test 2: No truncation when content fits")
    buffer = StreamBuffer()
    buffer.append("Line 1\n")
    buffer.append("Line 2\n")
    buffer.append("Line 3\n")

    # Terminal height of 50 should easily fit 3 lines
    result = buffer.get_display_text(terminal_height=50)
    assert "Line 1" in result
    assert "Line 2" in result
    assert "Line 3" in result
    print("✓ No truncation when fits\n")


def test_code_block_fence_preservation():
    """Test that code block fence is preserved when truncated."""
    print("Test 3: Code block fence preservation")
    buffer = StreamBuffer()

    # Build a large code block
    buffer.append("```python\n")
    for i in range(100):
        buffer.append(f"def function_{i}():\n")
        buffer.append(f"    return {i}\n")

    # Truncate to small height - should preserve opening fence
    result = buffer.get_display_text(terminal_height=10)
    assert result.startswith("```python\n"), f"Expected to start with fence, got: {result[:50]}"
    print("✓ Code block fence preserved\n")


def test_unclosed_code_block():
    """Test that unclosed code block gets closing fence."""
    print("Test 4: Unclosed code block gets closing fence")
    buffer = StreamBuffer()

    # Start a code block but don't close it
    buffer.append("```python\n")
    buffer.append("def foo():\n")
    buffer.append("    return 42\n")

    # Should add closing fence
    result = buffer.get_display_text(terminal_height=50)
    assert result.endswith("```\n") or result.endswith("```"), f"Expected closing fence, got: {result[-20:]}"
    print("✓ Unclosed code block gets closing fence\n")


def test_table_header_preservation():
    """Test that table header is preserved when truncating data rows."""
    print("Test 5: Table header preservation")
    buffer = StreamBuffer()

    # Build a large table
    buffer.append("| Name | Size |\n")
    buffer.append("|------|------|\n")
    for i in range(100):
        buffer.append(f"| file_{i} | {i}KB |\n")

    # Truncate to small height - should preserve header
    result = buffer.get_display_text(terminal_height=10)
    assert "| Name | Size |" in result, f"Expected header, got: {result[:100]}"
    assert "|------|------|" in result, f"Expected separator, got: {result[:100]}"
    print("✓ Table header preserved\n")


def test_multiple_code_blocks():
    """Test handling of multiple code blocks."""
    print("Test 6: Multiple code blocks")
    buffer = StreamBuffer()

    # Add multiple code blocks
    buffer.append("First block:\n")
    buffer.append("```python\n")
    buffer.append("def foo(): pass\n")
    buffer.append("```\n")
    buffer.append("\nSecond block:\n")
    buffer.append("```javascript\n")
    buffer.append("function bar() {}\n")
    buffer.append("```\n")

    # Should handle both correctly
    result = buffer.get_display_text(terminal_height=50)
    assert "```python" in result
    assert "```javascript" in result
    print("✓ Multiple code blocks handled\n")


def test_streaming_scenario():
    """Simulate actual streaming scenario."""
    print("Test 7: Realistic streaming scenario")
    buffer = StreamBuffer()

    # Simulate streaming chunks
    chunks = [
        "Here's some code:\n\n",
        "```python\n",
        "def calculate_sum(a, b):\n",
        "    \"\"\"Add two numbers.\"\"\"\n",
        "    return a + b\n",
        "\n",
        "result = calculate_sum(5, 3)\n",
        "print(result)\n",
        "```\n",
        "\nThe result is 8."
    ]

    for chunk in chunks:
        buffer.append(chunk)

    result = buffer.get_display_text(terminal_height=50)
    assert "```python" in result
    assert "def calculate_sum" in result
    assert "The result is 8" in result
    print("✓ Realistic streaming works\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing StreamBuffer")
    print("=" * 60 + "\n")

    test_basic_streaming()
    test_no_truncation_when_fits()
    test_code_block_fence_preservation()
    test_unclosed_code_block()
    test_table_header_preservation()
    test_multiple_code_blocks()
    test_streaming_scenario()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
