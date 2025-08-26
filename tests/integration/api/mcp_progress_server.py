#!/usr/bin/env python3

import asyncio

from mcp.server.fastmcp import FastMCP

# Create the FastMCP server
app = FastMCP(
    name="Progress Test Server", instructions="A server for testing progress notifications"
)


@app.tool(
    name="progress_task",
    description="A task that sends progress notifications during execution.",
)
async def progress_task(steps: int = 5) -> str:
    """
    Execute a task with progress notifications.

    Args:
        steps: Number of steps to simulate (default: 5)
    """
    context = app.get_context()

    # Use the correct FastMCP method for progress reporting
    await context.report_progress(0, steps, "Starting task...")

    # Simulate work with progress updates
    for i in range(steps):
        await asyncio.sleep(0.1)  # Simulate work
        await context.report_progress(i + 1, steps, f"Completed step {i + 1} of {steps}")

    # Final completion
    await context.report_progress(steps, steps, "Task completed!")

    return f"Successfully completed {steps} steps"


@app.tool(
    name="progress_task_no_message",
    description="A task that sends progress notifications without messages.",
)
async def progress_task_no_message(steps: int = 3) -> str:
    """
    Execute a task with progress notifications but no messages.

    Args:
        steps: Number of steps to simulate (default: 3)
    """
    context = app.get_context()

    # Send progress without messages (should fall back to server/tool name display)
    for i in range(steps):
        await asyncio.sleep(0.1)  # Simulate work
        await context.report_progress(i + 1, steps)  # No message - should show server/tool name

    return f"Completed {steps} steps without messages"


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
