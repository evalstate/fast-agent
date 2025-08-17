#!/usr/bin/env python3
"""
MCP server that generates large, predictable content for context truncation testing.
"""

import asyncio

from mcp.server import Server
from mcp.types import (
    CallToolResult,
    ListToolsResponse,
    TextContent,
    Tool,
)

app = Server("large-content-test")


@app.list_tools()
async def list_tools() -> ListToolsResponse:
    """List available tools for testing truncation."""
    return ListToolsResponse(
        tools=[
            Tool(
                name="generate_large_text",
                description="Generate large text content of specified size",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "size": {
                            "type": "integer", 
                            "description": "Size of text to generate (in tokens approximately)",
                            "minimum": 100,
                            "maximum": 50000
                        },
                        "content_type": {
                            "type": "string",
                            "description": "Type of content to generate",
                            "enum": ["lorem", "numbers", "alphabet"]
                        }
                    },
                    "required": ["size"]
                }
            ),
            Tool(
                name="get_conversation_length",
                description="Get the current conversation length for testing",
                inputSchema={
                    "type": "object",
                    "properties": {},
                }
            )
        ]
    )


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls for testing."""
    if name == "generate_large_text":
        size = arguments.get("size", 1000)
        content_type = arguments.get("content_type", "lorem")
        
        if content_type == "lorem":
            # Generate Lorem ipsum-style content
            base_text = (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
            )
            
            # Approximate tokens (roughly 4 chars per token)
            target_chars = size * 4
            repetitions = max(1, target_chars // len(base_text))
            content = (base_text * repetitions)[:target_chars]
            
        elif content_type == "numbers":
            # Generate numbered lines
            lines = []
            for i in range(1, size + 1):
                lines.append(f"Line number {i:05d}: This is test content for context truncation testing.")
                if len('\n'.join(lines)) * 0.25 >= size:  # Rough token estimation
                    break
            content = '\n'.join(lines)
            
        elif content_type == "alphabet":
            # Generate alphabet patterns
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            content_parts = []
            for i in range(size // 10):  # Rough token estimation
                content_parts.append(f"{alphabet} - Pattern {i:05d}")
            content = '\n'.join(content_parts)
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Generated {content_type} content (~{size} tokens):\n\n{content}"
                )
            ],
            isError=False
        )
    
    elif name == "get_conversation_length":
        return CallToolResult(
            content=[
                TextContent(
                    type="text", 
                    text="This tool would return conversation length info in a real implementation."
                )
            ],
            isError=False
        )
    
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True
        )


async def main():
    # Use stdio transport  
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())