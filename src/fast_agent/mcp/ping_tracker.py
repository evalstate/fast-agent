"""Shared ping failure tracking for MCP transport layers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_PING_FAILURE_RESET_THRESHOLD = 3


class PingFailureTracker:
    """Tracks consecutive ping failures and determines when to reset connection state."""

    def __init__(
        self,
        url: str,
        threshold: int = DEFAULT_PING_FAILURE_RESET_THRESHOLD,
    ) -> None:
        """Initialize ping failure tracker.

        Args:
            url: URL being tracked (for logging context)
            threshold: Number of consecutive failures before reset is recommended
        """
        self.url = url
        self.threshold = threshold
        self._count = 0

    def record_failure(self) -> tuple[int, bool]:
        """Record a ping failure and return count and reset recommendation.

        Returns:
            Tuple of (failure_count, should_reset) where should_reset is True
            if threshold has been reached.
        """
        self._count += 1
        logger.warning(
            "Ping timeout waiting for keepalive on %s (%s/%s)",
            self.url,
            self._count,
            self.threshold,
        )
        should_reset = self._count >= self.threshold
        if should_reset:
            logger.warning("Multiple ping timeouts on %s; clearing resumption state", self.url)
        return self._count, should_reset

    def reset(self) -> None:
        """Reset the failure count (called on successful ping or non-timeout error)."""
        if self._count > 0:
            logger.debug("Resetting ping failure count for %s", self.url)
        self._count = 0

    @property
    def count(self) -> int:
        """Current failure count."""
        return self._count

    def format_detail(self) -> str:
        """Format error detail string with current failure count."""
        detail = f"Ping timeout waiting for keepalive ({self._count}/{self.threshold})"
        if self._count >= self.threshold:
            detail += "; clearing resumption state"
        return detail
