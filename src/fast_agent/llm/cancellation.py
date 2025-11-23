"""
Cancellation support for LLM provider calls.

This module provides a CancellationToken that can be used to cancel
in-flight LLM requests, particularly useful for:
- ESC key cancellation in interactive sessions
- ACP session/cancel protocol support

Note: This module uses asyncio.CancelledError for cancellation exceptions,
which is the standard Python approach for async cancellation.
"""

import asyncio


class CancellationToken:
    """
    A token that can be used to cancel ongoing LLM operations.

    The token uses an asyncio.Event internally to signal cancellation.
    Once cancelled, it remains in the cancelled state.

    Usage:
        token = CancellationToken()

        # In the calling code:
        result = await llm.generate(messages, cancellation_token=token)

        # To cancel (e.g., from ESC key handler or ACP cancel):
        token.cancel()

        # In the LLM provider:
        async for chunk in stream:
            if token.is_cancelled:
                raise asyncio.CancelledError(token.cancel_reason)
            # process chunk
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._cancel_reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """
        Signal cancellation.

        Args:
            reason: Optional reason for cancellation (e.g., "user_cancelled", "timeout")
        """
        self._cancel_reason = reason or "cancelled"
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._event.is_set()

    @property
    def cancel_reason(self) -> str | None:
        """Get the reason for cancellation, if any."""
        return self._cancel_reason

    async def wait_for_cancellation(self) -> None:
        """Wait until cancellation is requested."""
        await self._event.wait()

    def reset(self) -> None:
        """
        Reset the token to non-cancelled state.

        Note: This should be used with caution. Generally, it's better
        to create a new token for each operation.
        """
        self._event.clear()
        self._cancel_reason = None
