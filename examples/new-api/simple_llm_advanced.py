import asyncio
from typing import Any, Dict

from mcp.server.fastmcp.tools.base import Tool as FastMCPTool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory


# Example 1: Simple function that will be wrapped
async def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Search results as a formatted string
    """
    # Mock implementation
    return f"Found {max_results} results for '{query}': [Result 1, Result 2, ...]"


# Example 2: Create a FastMCP Tool directly for more control
def create_calculator_tool() -> FastMCPTool:
    """Create a calculator tool with explicit schema."""

    def calculate(operation: str, a: float, b: float) -> float:
        """Perform a calculation."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float("inf"),
        }

        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")

        return operations[operation](a, b)

    # Create the tool with explicit configuration
    return FastMCPTool.from_function(
        fn=calculate,
        name="calculator",
        description="Perform basic arithmetic operations",
        # FastMCP will still generate the schema, but we could override if needed
    )


# Example 3: Complex async tool with side effects
async def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Send an email (mock implementation).

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content

    Returns:
        Dictionary with send status and message ID
    """
    # Mock async operation
    await asyncio.sleep(0.1)

    return {
        "status": "sent",
        "message_id": f"msg_{hash((to, subject))}",
        "timestamp": "2024-01-01T12:00:00Z",
    }


async def main():
    core: Core = Core()
    await core.initialize()

    # Create agent configuration
    config = AgentConfig(name="assistant", model="haiku")

    # Mix different tool types
    tools = [
        search_web,  # Async function
        create_calculator_tool(),  # Pre-configured FastMCP Tool
        send_email,  # Complex async function
    ]

    # Create tool agent
    tool_agent = ToolAgent(config, tools=tools, context=core.context)

    # Attach the LLM
    await tool_agent.attach_llm(ModelFactory.create_factory("haiku"))

    # Test various tools
    print("Testing search:")
    await tool_agent.send("Search for information about Python FastMCP")

    print("\nTesting calculator:")
    await tool_agent.send("What is 42 multiplied by 17?")

    print("\nTesting email:")
    await tool_agent.send(
        "Send an email to test@example.com with subject 'Hello' and body 'Test message'"
    )


if __name__ == "__main__":
    asyncio.run(main())
