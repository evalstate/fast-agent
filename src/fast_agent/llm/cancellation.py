"""
Cancellation support for LLM provider calls.

This module previously provided a CancellationToken class for cancellation.
Now cancellation is handled natively through asyncio.Task.cancel() which raises
asyncio.CancelledError at the next await point.

Usage:
    # Store the task when starting work
    task = asyncio.current_task()

    # To cancel:
    task.cancel()  # Raises asyncio.CancelledError in the task

    # In the LLM provider, CancelledError propagates naturally:
    async for chunk in stream:
        # task.cancel() will raise CancelledError here
        process(chunk)
"""

# This module is kept for documentation purposes.
# All cancellation is now handled via asyncio.Task.cancel()
