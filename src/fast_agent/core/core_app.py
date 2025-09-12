from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional, TypeVar

from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction

if TYPE_CHECKING:
    # Only imported for type checking to avoid circular imports at runtime
    from fast_agent.config import Settings
    from fast_agent.context import Context
    from fast_agent.core.executor.workflow_signal import SignalWaitCallback

R = TypeVar("R")


class Core:
    """
    fast-agent core. handles application settings, config and context management.
    """

    def __init__(
        self,
        name: str = "fast-agent",
        settings: Optional[Settings] | str = None,
        signal_notification: Optional[SignalWaitCallback] = None,
    ) -> None:
        """
        Initialize the core.
        Args:
            name:
            settings: If unspecified, the settings are loaded from fastagent.config.yaml.
                If this is a string, it is treated as the path to the config file to load.
            signal_notification: Callback for getting notified on workflow signals/events.
        """
        self.name = name

        # We use these to initialize the context in initialize()
        self._config_or_path = settings
        self._signal_notification = signal_notification

        self._logger = None
        # Use forward reference for type to avoid runtime import
        self._context: Optional["Context"] = None
        self._initialized = False

    @property
    def context(self) -> "Context":
        if self._context is None:
            raise RuntimeError(
                "Core not initialized, please call initialize() first, or use async with app.run()."
            )
        return self._context

    @property
    def config(self):
        return self.context.config

    @property
    def server_registry(self):
        return self.context.server_registry

    @property
    def logger(self):
        if self._logger is None:
            self._logger = get_logger(f"fast_agent.{self.name}")
        return self._logger

    async def initialize(self) -> None:
        """Initialize the fast-agent core. Sets up context (and therefore logging and loading settings)."""
        if self._initialized:
            return

        # Import here to avoid circular imports during module initialization
        from fast_agent import context as _context_mod

        self._context = await _context_mod.initialize_context(
            self._config_or_path, store_globally=True
        )

        # Set the properties that were passed in the constructor
        self._context.signal_notification = self._signal_notification
        # Note: upstream_session support removed for now

        self._initialized = True
        self.logger.info(
            "fast-agent initialized",
            data={
                "progress_action": "Running",
                "target": self.name or "mcp_application",
                "agent_name": self.name or "fast-agent core",
            },
        )

    async def cleanup(self) -> None:
        """Cleanup application resources."""
        if not self._initialized:
            return

        self.logger.info(
            "fast-agent cleanup",
            data={
                "progress_action": ProgressAction.FINISHED,
                "target": self.name or "fast-agent",
                "agent_name": self.name or "fast-agent core",
            },
        )
        try:
            # Import here to avoid circular imports during module initialization
            from fast_agent import context as _context_mod

            await _context_mod.cleanup_context()
        except asyncio.CancelledError:
            self.logger.debug("Cleanup cancelled error during shutdown")

        self._context = None
        self._initialized = False

    @asynccontextmanager
    async def run(self):
        """
        Use core for context management

        Example:
            async with core.run() as running_app:
                # App is initialized here
                pass
        """
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
