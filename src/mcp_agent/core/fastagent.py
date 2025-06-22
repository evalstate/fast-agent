"""
Direct FastAgent implementation that uses the simplified Agent architecture.
This replaces the traditional FastAgent with a more streamlined approach that
directly creates Agent instances without proxies.
"""

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

import yaml
from opentelemetry import trace

from mcp_agent import config
from mcp_agent.app import MCPApp
from mcp_agent.context import Context
from mcp_agent.core.agent_app import AgentApp
from mcp_agent.core.direct_decorators import (
    agent as agent_decorator,
)
from mcp_agent.core.direct_decorators import (
    chain as chain_decorator,
)
from mcp_agent.core.direct_decorators import (
    custom as custom_decorator,
)
from mcp_agent.core.direct_decorators import (
    evaluator_optimizer as evaluator_optimizer_decorator,
)
from mcp_agent.core.direct_decorators import (
    orchestrator as orchestrator_decorator,
)
from mcp_agent.core.direct_decorators import (
    parallel as parallel_decorator,
)
from mcp_agent.core.direct_decorators import (
    router as router_decorator,
)
from mcp_agent.core.direct_factory import (
    create_agents_in_dependency_order,
    get_model_factory,
)
from mcp_agent.core.error_handling import handle_error
from mcp_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ModelConfigError,
    PromptExitError,
    ProviderKeyError,
    ServerConfigError,
    ServerInitializationError,
)
from mcp_agent.core.usage_display import display_usage_report
from mcp_agent.core.validation import (
    validate_server_references,
    validate_workflow_references,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompts.prompt_load import load_prompt_multipart

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

F = TypeVar("F", bound=Callable[..., Any])  # For decorated functions
logger = get_logger(__name__)


class FastAgent:
    """
    A simplified FastAgent implementation that directly creates Agent instances
    without using proxies.
    """

    def __init__(
        self,
        name: str,
        config_path: str | None = None,
        ignore_unknown_args: bool = False,
        parse_cli_args: bool = True,  # Add new parameter with default True
    ) -> None:
        """
        Initialize the fast-agent application.

        Args:
            name: Name of the application
            config_path: Optional path to config file
            ignore_unknown_args: Whether to ignore unknown command line arguments
                                 when parse_cli_args is True.
            parse_cli_args: If True, parse command line arguments using argparse.
                            Set to False when embedding FastAgent in another framework
                            (like FastAPI/Uvicorn) that handles its own arguments.
        """
        self.args = argparse.Namespace()  # Initialize args always

        # --- Wrap argument parsing logic ---
        if parse_cli_args:
            # Setup command line argument parsing
            parser = argparse.ArgumentParser(description="DirectFastAgent Application")
            parser.add_argument(
                "--model",
                help="Override the default model for all agents",
            )
            parser.add_argument(
                "--agent",
                default="default",
                help="Specify the agent to send a message to (used with --message)",
            )
            parser.add_argument(
                "-m",
                "--message",
                help="Message to send to the specified agent",
            )
            parser.add_argument(
                "-p", "--prompt-file", help="Path to a prompt file to use (either text or JSON)"
            )
            parser.add_argument(
                "--quiet",
                action="store_true",
                help="Disable progress display, tool and message logging for cleaner output",
            )
            parser.add_argument(
                "--version",
                action="store_true",
                help="Show version and exit",
            )
            parser.add_argument(
                "--server",
                action="store_true",
                help="Run as an MCP server",
            )
            parser.add_argument(
                "--transport",
                choices=["sse", "http", "stdio"],
                default="http",
                help="Transport protocol to use when running as a server (sse or stdio)",
            )
            parser.add_argument(
                "--port",
                type=int,
                default=8000,
                help="Port to use when running as a server with SSE transport",
            )
            parser.add_argument(
                "--host",
                default="0.0.0.0",
                help="Host address to bind to when running as a server with SSE transport",
            )

            if ignore_unknown_args:
                known_args, _ = parser.parse_known_args()
                self.args = known_args
            else:
                # Use parse_known_args here too, to avoid crashing on uvicorn args etc.
                # even if ignore_unknown_args is False, we only care about *our* args.
                known_args, unknown = parser.parse_known_args()
                self.args = known_args
                # Optionally, warn about unknown args if not ignoring?
                # if unknown and not ignore_unknown_args:
                #     logger.warning(f"Ignoring unknown command line arguments: {unknown}")

            # Handle version flag
            if self.args.version:
                try:
                    app_version = get_version("fast-agent-mcp")
                except:  # noqa: E722
                    app_version = "unknown"
                print(f"fast-agent-mcp v{app_version}")
                sys.exit(0)
        # --- End of wrapped logic ---

        self.name = name
        self.config_path = config_path

        try:
            # Load configuration directly for this instance
            self._load_config()

            # Create the app with our local settings
            self.app = MCPApp(
                name=name,
                settings=config.Settings(**self.config) if hasattr(self, "config") else None,
            )

        except yaml.parser.ParserError as e:
            handle_error(
                e,
                "YAML Parsing Error",
                "There was an error parsing the config or secrets YAML configuration file.",
            )
            raise SystemExit(1)

        # Dictionary to store agent configurations from decorators
        self.agents: Dict[str, Dict[str, Any]] = {}

    def _load_config(self) -> None:
        """Load configuration from YAML file including secrets using get_settings
        but without relying on the global cache."""

        # Import but make a local copy to avoid affecting the global state
        from mcp_agent.config import _settings, get_settings

        # Temporarily clear the global settings to ensure a fresh load
        old_settings = _settings
        _settings = None

        try:
            # Use get_settings to load config - this handles all paths and secrets merging
            settings = get_settings(self.config_path)

            # Convert to dict for backward compatibility
            self.config = settings.model_dump() if settings else {}
        finally:
            # Restore the original global settings
            _settings = old_settings

    @property
    def context(self) -> Context:
        """Access the application context"""
        return self.app.context

    # Decorator methods with type-safe implementations
    agent = agent_decorator
    custom = custom_decorator
    orchestrator = orchestrator_decorator
    router = router_decorator
    chain = chain_decorator
    parallel = parallel_decorator
    evaluator_optimizer = evaluator_optimizer_decorator

    @asynccontextmanager
    async def run(self):
        """
        Context manager for running the application.
        Initializes all registered agents.
        """
        active_agents: Dict[str, Agent] = {}
        had_error = False
        await self.app.initialize()

        # Handle quiet mode and CLI model override safely
        # Define these *before* they are used, checking if self.args exists and has the attributes
        quiet_mode = hasattr(self.args, "quiet") and self.args.quiet
        cli_model_override = (
            self.args.model if hasattr(self.args, "model") and self.args.model else None
        )  # Define cli_model_override here
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(self.name):
            try:
                async with self.app.run():
                    # Apply quiet mode if requested
                    if (
                        quiet_mode
                        and hasattr(self.app.context, "config")
                        and hasattr(self.app.context.config, "logger")
                    ):
                        # Update our app's config directly
                        self.app.context.config.logger.progress_display = False
                        self.app.context.config.logger.show_chat = False
                        self.app.context.config.logger.show_tools = False

                        # Directly disable the progress display singleton
                        from mcp_agent.progress_display import progress_display

                        progress_display.stop()

                    # Pre-flight validation
                    if 0 == len(self.agents):
                        raise AgentConfigError(
                            "No agents defined. Please define at least one agent."
                        )
                    validate_server_references(self.context, self.agents)
                    validate_workflow_references(self.agents)

                    # Get a model factory function
                    # Now cli_model_override is guaranteed to be defined
                    def model_factory_func(model=None, request_params=None):
                        return get_model_factory(
                            self.context,
                            model=model,
                            request_params=request_params,
                            cli_model=cli_model_override,  # Use the variable defined above
                        )

                    # Create all agents in dependency order
                    active_agents = await create_agents_in_dependency_order(
                        self.app,
                        self.agents,
                        model_factory_func,
                    )

                    # Create a wrapper with all agents for simplified access
                    wrapper = AgentApp(active_agents)

                    # Handle command line options that should be processed after agent initialization

                    # Handle --server option
                    # Check if parse_cli_args was True before checking self.args.server
                    if hasattr(self.args, "server") and self.args.server:
                        try:
                            # Print info message if not in quiet mode
                            if not quiet_mode:
                                print(f"Starting FastAgent '{self.name}' in server mode")
                                print(f"Transport: {self.args.transport}")
                                if self.args.transport == "sse":
                                    print(f"Listening on {self.args.host}:{self.args.port}")
                                print("Press Ctrl+C to stop")

                            # Create the MCP server
                            from mcp_agent.mcp_server import AgentMCPServer

                            mcp_server = AgentMCPServer(
                                agent_app=wrapper,
                                server_name=f"{self.name}-MCP-Server",
                            )

                            # Run the server directly (this is a blocking call)
                            await mcp_server.run_async(
                                transport=self.args.transport,
                                host=self.args.host,
                                port=self.args.port,
                            )
                        except KeyboardInterrupt:
                            if not quiet_mode:
                                print("\nServer stopped by user (Ctrl+C)")
                        except Exception as e:
                            if not quiet_mode:
                                import traceback

                                traceback.print_exc()
                                print(f"\nServer stopped with error: {e}")

                        # Exit after server shutdown
                        raise SystemExit(0)

                    # Handle direct message sending if  --message is provided
                    if hasattr(self.args, "message") and self.args.message:
                        agent_name = self.args.agent
                        message = self.args.message

                        if agent_name not in active_agents:
                            available_agents = ", ".join(active_agents.keys())
                            print(
                                f"\n\nError: Agent '{agent_name}' not found. Available agents: {available_agents}"
                            )
                            raise SystemExit(1)

                        try:
                            # Get response from the agent
                            agent = active_agents[agent_name]
                            response = await agent.send(message)

                            # In quiet mode, just print the raw response
                            # The chat display should already be turned off by the configuration
                            if self.args.quiet:
                                print(f"{response}")

                            raise SystemExit(0)
                        except Exception as e:
                            print(f"\n\nError sending message to agent '{agent_name}': {str(e)}")
                            raise SystemExit(1)

                    if hasattr(self.args, "prompt_file") and self.args.prompt_file:
                        agent_name = self.args.agent
                        prompt: List[PromptMessageMultipart] = load_prompt_multipart(
                            Path(self.args.prompt_file)
                        )
                        if agent_name not in active_agents:
                            available_agents = ", ".join(active_agents.keys())
                            print(
                                f"\n\nError: Agent '{agent_name}' not found. Available agents: {available_agents}"
                            )
                            raise SystemExit(1)

                        try:
                            # Get response from the agent
                            agent = active_agents[agent_name]
                            response = await agent.generate(prompt)

                            # In quiet mode, just print the raw response
                            # The chat display should already be turned off by the configuration
                            if self.args.quiet:
                                print(f"{response.last_text()}")

                            raise SystemExit(0)
                        except Exception as e:
                            print(f"\n\nError sending message to agent '{agent_name}': {str(e)}")
                            raise SystemExit(1)

                    yield wrapper

            except PromptExitError as e:
                # User requested exit - not an error, show usage report
                self._handle_error(e)
                raise SystemExit(0)
            except (
                ServerConfigError,
                ProviderKeyError,
                AgentConfigError,
                ServerInitializationError,
                ModelConfigError,
                CircularDependencyError,
            ) as e:
                had_error = True
                self._handle_error(e)
                raise SystemExit(1)

            finally:
                # Print usage report before cleanup (show for user exits too)
                if active_agents and not had_error:
                    self._print_usage_report(active_agents)

                # Clean up any active agents (always cleanup, even on errors)
                if active_agents:
                    for agent in active_agents.values():
                        try:
                            await agent.shutdown()
                        except Exception:
                            pass

    def _handle_error(self, e: Exception, error_type: Optional[str] = None) -> None:
        """
        Handle errors with consistent formatting and messaging.

        Args:
            e: The exception that was raised
            error_type: Optional explicit error type
        """
        if isinstance(e, ServerConfigError):
            handle_error(
                e,
                "Server Configuration Error",
                "Please check your 'fastagent.config.yaml' configuration file and add the missing server definitions.",
            )
        elif isinstance(e, ProviderKeyError):
            handle_error(
                e,
                "Provider Configuration Error",
                "Please check your 'fastagent.secrets.yaml' configuration file and ensure all required API keys are set.",
            )
        elif isinstance(e, AgentConfigError):
            handle_error(
                e,
                "Workflow or Agent Configuration Error",
                "Please check your agent definition and ensure names and references are correct.",
            )
        elif isinstance(e, ServerInitializationError):
            handle_error(
                e,
                "MCP Server Startup Error",
                "There was an error starting up the MCP Server.",
            )
        elif isinstance(e, ModelConfigError):
            handle_error(
                e,
                "Model Configuration Error",
                "Common models: gpt-4.1, o3-mini, sonnet, haiku. for o3, set reasoning effort with o3-mini.high",
            )
        elif isinstance(e, CircularDependencyError):
            handle_error(
                e,
                "Circular Dependency Detected",
                "Check your agent configuration for circular dependencies.",
            )
        elif isinstance(e, PromptExitError):
            handle_error(
                e,
                "User requested exit",
            )
        elif isinstance(e, asyncio.CancelledError):
            handle_error(
                e,
                "Cancelled",
                "The operation was cancelled.",
            )
        else:
            handle_error(e, error_type or "Error", "An unexpected error occurred.")

    def _print_usage_report(self, active_agents: dict) -> None:
        """Print a formatted table of token usage for all agents."""
        display_usage_report(active_agents, show_if_progress_disabled=False, subdued_colors=True)

    async def start_server(
        self,
        transport: str = "sse",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
    ) -> None:
        """
        Start the application as an MCP server.
        This method initializes agents and exposes them through an MCP server.
        It is a blocking method that runs until the server is stopped.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
            host: Host address for the server when using SSE
            port: Port for the server when using SSE
            server_name: Optional custom name for the MCP server
            server_description: Optional description for the MCP server
        """
        # This method simply updates the command line arguments and uses run()
        # to ensure we follow the same initialization path for all operations

        # Store original args
        original_args = None
        if hasattr(self, "args"):
            original_args = self.args

        # Create our own args object with server settings
        from argparse import Namespace

        self.args = Namespace()
        self.args.server = True
        self.args.transport = transport
        self.args.host = host
        self.args.port = port
        self.args.quiet = (
            original_args.quiet if original_args and hasattr(original_args, "quiet") else False
        )
        self.args.model = None
        if hasattr(original_args, "model"):
            self.args.model = original_args.model

        # Run the application, which will detect the server flag and start server mode
        async with self.run():
            pass  # This won't be reached due to SystemExit in run()

        # Restore original args (if we get here)
        if original_args:
            self.args = original_args

    # Keep run_with_mcp_server for backward compatibility
    async def run_with_mcp_server(
        self,
        transport: str = "sse",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
    ) -> None:
        """
        Run the application and expose agents through an MCP server.
        This method is kept for backward compatibility.
        For new code, use start_server() instead.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
            host: Host address for the server when using SSE
            port: Port for the server when using SSE
            server_name: Optional custom name for the MCP server
            server_description: Optional description for the MCP server
        """
        await self.start_server(
            transport=transport,
            host=host,
            port=port,
            server_name=server_name,
            server_description=server_description,
        )

    async def main(self):
        """
        Helper method for checking if server mode was requested.

        Usage:
        ```python
        fast = FastAgent("My App")

        @fast.agent(...)
        async def app_main():
            # Check if server mode was requested
            # This doesn't actually do anything - the check happens in run()
            # But it provides a way for application code to know if server mode
            # was requested for conditionals
            is_server_mode = hasattr(self, "args") and self.args.server

            # Normal run - this will handle server mode automatically if requested
            async with fast.run() as agent:
                # This code only executes for normal mode
                # Server mode will exit before reaching here
                await agent.send("Hello")
        ```

        Returns:
            bool: True if --server flag is set, False otherwise
        """
        # Just check if the flag is set, no action here
        # The actual server code will be handled by run()
        return hasattr(self, "args") and self.args.server
