import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Dict, Optional, Type, TypeVar, Union

from mcp_agent.config import Settings
from mcp_agent.context import Context, cleanup_context, initialize_context
from mcp_agent.event_progress import ProgressAction
from mcp_agent.executor.workflow_signal import SignalWaitCallback
from mcp_agent.human_input.handler import console_input_callback
from mcp_agent.human_input.types import HumanInputCallback
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp import ServerSession

R = TypeVar("R")


class MCPApp:
    """
    Main application class that manages global state and can host workflows.

    Example usage:
        app = MCPApp()

        @app.workflow
        class MyWorkflow(Workflow[str]):
            @app.task
            async def my_task(self):
                pass

            async def run(self):
                await self.my_task()

        async with app.run() as running_app:
            workflow = MyWorkflow()
            result = await workflow.execute()
    """

    def __init__(
        self,
        name: str = "mcp_application",
        settings: Optional[Union[Settings, str, dict]] = None,
        human_input_callback: Optional[HumanInputCallback] = console_input_callback,
        signal_notification: Optional[SignalWaitCallback] = None,
        upstream_session: Optional["ServerSession"] = None,
    ) -> None:
        """
        Initialize the application with a name and optional settings.
        Args:
            name: Name of the application
            settings: Application configuration - Can be one of:
                - None: Load from default config file
                - str: Path to config file
                - dict: JSON configuration dictionary
                - Settings: Already initialized Settings object
            human_input_callback: Callback for handling human input
            signal_notification: Callback for getting notified on workflow signals/events.
            upstream_session: Optional upstream session if the MCPApp is running as a server to an MCP client.
        """
        self.name = name

        # We use these to initialize the context in initialize()
        self._config_or_path = settings
        self._human_input_callback = human_input_callback
        self._signal_notification = signal_notification
        self._upstream_session = upstream_session

        self._workflows: Dict[str, Type] = {}  # id to workflow class
        self._logger = None
        self._context: Optional[Context] = None
        self._initialized = False

    @property
    def context(self) -> Context:
        if self._context is None:
            raise RuntimeError(
                "MCPApp not initialized, please call initialize() first, or use async with app.run()."
            )
        return self._context

    @property
    def config(self):
        return self._context.config

    @property
    def server_registry(self):
        return self._context.server_registry

    @property
    def executor(self):
        return self._context.executor

    @property
    def engine(self):
        return self.executor.execution_engine

    @property
    def upstream_session(self):
        return self._context.upstream_session

    @upstream_session.setter
    def upstream_session(self, value) -> None:
        self._context.upstream_session = value

    @property
    def workflows(self):
        return self._workflows

    @property
    def tasks(self):
        return self.context.task_registry.list_activities()

    @property
    def logger(self):
        if self._logger is None:
            self._logger = get_logger(f"mcp_agent.{self.name}")
        return self._logger

    async def initialize(self) -> None:
        """Initialize the application."""
        if self._initialized:
            return

        # Initialize the context with the provided config (which could be a dict/JSON)
        self._context = await initialize_context(self._config_or_path)

        # Set the properties that were passed in the constructor
        self._context.human_input_handler = self._human_input_callback
        self._context.signal_notification = self._signal_notification
        self._context.upstream_session = self._upstream_session

        self._initialized = True
        self.logger.info(
            "MCPAgent initialized",
            data={
                "progress_action": "Running",
                "target": self.name or "mcp_application",
                "agent_name": self.name or "fastagent loop",
            },
        )

    async def cleanup(self) -> None:
        """Cleanup application resources."""
        if not self._initialized:
            return

        # Updatre progress display before logging is shut down
        self.logger.info(
            "MCPAgent cleanup",
            data={
                "progress_action": ProgressAction.FINISHED,
                "target": self.name or "mcp_application",
                "agent_name": self.name or "fastagent loop",
            },
        )
        try:
            await cleanup_context()
        except asyncio.CancelledError:
            self.logger.debug("Cleanup cancelled error during shutdown")

        self._context = None
        self._initialized = False

    @asynccontextmanager
    async def run(self):
        """
        Run the application. Use as context manager.

        Example:
            async with app.run() as running_app:
                # App is initialized here
                pass
        """
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
