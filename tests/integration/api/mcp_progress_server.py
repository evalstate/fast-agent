import asyncio

from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_context

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
    context = get_context()
    await context.report_progress(0, steps, "Starting task...")

    for i in range(steps):
        await asyncio.sleep(0.1)
        await context.report_progress(
            i + 1,
            steps,
            f"Completed step {i + 1} of {steps}",
        )

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
    context = get_context()
    for i in range(steps):
        await asyncio.sleep(0.1)
        await context.report_progress(i + 1, steps)

    return f"Completed {steps} steps without messages"


async def send_progress(
    context: Context,
    progress: float,
    total: float | None = None,
    message: str | None = None,
) -> None:
    """Report progress through FastMCP's public request context."""
    await context.report_progress(progress, total, message)


@app.tool(
    name="progress_task_with_helper",
    description="A task using the helper function for progress.",
)
async def progress_task_with_helper(steps: int = 5) -> str:
    """
    Execute a task using the helper function for progress notifications.

    Args:
        steps: Number of steps to simulate (default: 5)
    """
    context = get_context()

    # Use the helper function for cleaner code
    await send_progress(context, 0, steps, "Starting task...")

    for i in range(steps):
        await asyncio.sleep(0.1)
        await send_progress(context, i + 1, steps, f"Step {i + 1}/{steps}")

    await send_progress(context, steps, steps, "Complete!")

    return f"Successfully completed {steps} steps with helper"


if __name__ == "__main__":
    # Run the server
    app.run()
