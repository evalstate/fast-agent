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

    # Get the progress token from the request metadata
    progress_token = (
        context.request_context.meta.progressToken if context.request_context.meta else None
    )

    if progress_token is None:
        # Client didn't request progress updates
        # Just do the work without progress notifications
        for i in range(steps):
            await asyncio.sleep(0.1)
        return f"Successfully completed {steps} steps (no progress tracking)"

    # Use the session directly to properly send progress with related_request_id
    session = context.request_context.session
    request_id = context.request_id

    # Send progress notifications with proper correlation
    await session.send_progress_notification(
        progress_token=progress_token,
        progress=0,
        total=steps,
        message="Starting task...",
        related_request_id=request_id,  # ✅ Properly correlate with request
    )

    # Simulate work with progress updates
    for i in range(steps):
        await asyncio.sleep(0.1)  # Simulate work
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=i + 1,
            total=steps,
            message=f"Completed step {i + 1} of {steps}",
            related_request_id=request_id,  # ✅ Properly correlate with request
        )

    # Final completion
    await session.send_progress_notification(
        progress_token=progress_token,
        progress=steps,
        total=steps,
        message="Task completed!",
        related_request_id=request_id,  # ✅ Properly correlate with request
    )

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

    # Get the progress token from the request metadata
    progress_token = (
        context.request_context.meta.progressToken if context.request_context.meta else None
    )

    if progress_token is None:
        # Client didn't request progress updates
        for i in range(steps):
            await asyncio.sleep(0.1)
        return f"Completed {steps} steps (no progress tracking)"

    # Use the session directly for proper correlation
    session = context.request_context.session
    request_id = context.request_id

    # Send progress without messages
    for i in range(steps):
        await asyncio.sleep(0.1)  # Simulate work
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=i + 1,
            total=steps,
            message=None,  # No message
            related_request_id=request_id,  # ✅ Properly correlate with request
        )

    return f"Completed {steps} steps without messages"


# Alternative: Create a helper function to wrap the correct usage
async def send_progress(
    context, progress: float, total: float | None = None, message: str | None = None
) -> None:
    """Helper function that correctly sends progress with related_request_id."""
    progress_token = (
        context.request_context.meta.progressToken if context.request_context.meta else None
    )

    if progress_token is None:
        return

    await context.request_context.session.send_progress_notification(
        progress_token=progress_token,
        progress=progress,
        total=total,
        message=message,
        related_request_id=context.request_id,  # ✅ Always include this
    )


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
    context = app.get_context()

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
