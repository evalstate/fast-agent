#!/usr/bin/env python3
"""
MCP server WITH instructions for testing server instructions feature.
"""

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server with instructions
app = FastMCP(
    name="Server With Instructions",
    instructions="""This server provides calculation and text manipulation tools.

When using calculation tools:
- Always validate inputs are numbers
- Return results with appropriate precision
- Handle division by zero gracefully

When using text tools:
- Preserve original formatting where possible
- Be mindful of character encoding
- Return helpful error messages for invalid inputs"""
)


# Calculation tools
@app.tool(
    name="calculate_sum",
    description="Add two numbers together",
)
def calculate_sum(a: float, b: float) -> str:
    return f"The sum is: {a + b}"


@app.tool(
    name="calculate_product",
    description="Multiply two numbers",
)
def calculate_product(a: float, b: float) -> str:
    return f"The product is: {a * b}"


@app.tool(
    name="calculate_divide",
    description="Divide first number by second",
)
def calculate_divide(a: float, b: float) -> str:
    if b == 0:
        return "Error: Division by zero"
    return f"The result is: {a / b}"


# Text manipulation tools
@app.tool(
    name="text_reverse",
    description="Reverse a text string",
)
def text_reverse(text: str) -> str:
    return text[::-1]


@app.tool(
    name="text_uppercase",
    description="Convert text to uppercase",
)
def text_uppercase(text: str) -> str:
    return text.upper()


@app.tool(
    name="text_count",
    description="Count characters in text",
)
def text_count(text: str) -> str:
    return f"Character count: {len(text)}"


if __name__ == "__main__":
    import sys
    app.run(transport="stdio")
    sys.exit(0)