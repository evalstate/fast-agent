#!/usr/bin/env python3
"""
MCP server for testing filtering functionality with multiple tools, resources, and prompts.
"""

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="Filtering Test Server")

# Tools
@app.tool(
    name="math_add",
    description="Add two numbers",
)
def math_add(a: int, b: int) -> str:
    return f"Result: {a + b}"

@app.tool(
    name="math_subtract", 
    description="Subtract two numbers",
)
def math_subtract(a: int, b: int) -> str:
    return f"Result: {a - b}"

@app.tool(
    name="math_multiply",
    description="Multiply two numbers", 
)
def math_multiply(a: int, b: int) -> str:
    return f"Result: {a * b}"

@app.tool(
    name="string_upper",
    description="Convert string to uppercase",
)
def string_upper(text: str) -> str:
    return text.upper()

@app.tool(
    name="string_lower",
    description="Convert string to lowercase",
)
def string_lower(text: str) -> str:
    return text.lower()

@app.tool(
    name="utility_ping",
    description="Simple ping utility",
)
def utility_ping() -> str:
    return "pong"

@app.tool(
    name="utility_status",
    description="Get server status",
)
def utility_status() -> str:
    return "server is running"

# Resources
@app.resource("resource://math/constants")
def math_constants() -> str:
    return "π = 3.14159\ne = 2.71828\nφ = 1.618034"

@app.resource("resource://math/formulas")
def math_formulas() -> str:
    return "Area of circle: π × r²\nPythagorean theorem: a² + b² = c²"

@app.resource("resource://string/examples")
def string_examples() -> str:
    return "Hello World\nTesting 123\nCase Sensitivity Test"

@app.resource("resource://utility/info")
def utility_info() -> str:
    return "Ping: Tests server connectivity\nStatus: Shows server health"

# Prompts
@app.prompt("math_helper")
def math_helper_prompt(operation: str = "add") -> str:
    """Help with mathematical operations"""
    return f"I am a math helper. Let me help you with {operation} operations."

@app.prompt("math_teacher")
def math_teacher_prompt(level: str = "basic") -> str:
    """Math teaching assistant"""
    return f"I am a math teacher. I can teach {level} level mathematics."

@app.prompt("string_processor")
def string_processor_prompt(mode: str = "upper") -> str:
    """String processing assistant"""
    return f"I am a string processor. I can process strings in {mode} mode."

@app.prompt("utility_assistant")
def utility_assistant_prompt() -> str:
    """General utility assistant"""
    return "I am a utility assistant. I can help with various utility functions."

if __name__ == "__main__":
    app.run() 