"""
Direct FastAgent implementation that uses the simplified Agent architecture.
This replaces the traditional FastAgent with a more streamlined approach that
directly creates Agent instances without proxies.
"""

import argparse
import asyncio
import inspect
import pathlib
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version as get_version
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
)

import yaml
from opentelemetry import trace

from fast_agent import config
from fast_agent.context import Context
from fast_agent.core import Core
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.direct_decorators import (
    agent as agent_decorator,
)
from fast_agent.core.direct_decorators import (
    chain as chain_decorator,
)
from fast_agent.core.direct_decorators import (
    custom as custom_decorator,
)
from fast_agent.core.direct_decorators import (
    evaluator_optimizer as evaluator_optimizer_decorator,
)
from fast_agent.core.direct_decorators import (
    iterative_planner as orchestrator2_decorator,
)
from fast_agent.core.direct_decorators import (
    orchestrator as orchestrator_decorator,
)
from fast_agent.core.direct_decorators import (
    parallel as parallel_decorator,
)
from fast_agent.core.direct_decorators import (
    router as router_decorator,
)
from fast_agent.core.direct_factory import (
    create_agents_in_dependency_order,
    get_model_factory,
)
from fast_agent.core.error_handling import handle_error
from fast_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ModelConfigError,
    PromptExitError,
    ProviderKeyError,
    ServerConfigError,
    ServerInitializationError,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.validation import (
    validate_provider_keys_post_creation,
    validate_server_references,
    validate_workflow_references,
)
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.skills import SkillManifest, SkillRegistry
from fast_agent.ui.console import configure_console_stream
from fast_agent.ui.usage_display import display_usage_report

if TYPE_CHECKING:
    from mcp.client.session import ElicitationFnT
    from pydantic import AnyUrl

    from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.types import PromptMessageExtended

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
        parse_cli_args: bool = True,
        quiet: bool = False,  # Add quiet parameter
        skills_directory: str | pathlib.Path | None = None,
        **kwargs,
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
            quiet: If True, disable progress display, tool and message logging for cleaner output
        """
        self.args = argparse.Namespace()  # Initialize args always
        self._programmatic_quiet = quiet  # Store the programmatic quiet setting
        self._skills_directory_override = (
            Path(skills_directory).expanduser() if skills_directory else None
        )
        self._default_skill_manifests: List[SkillManifest] = []
        self._server_instance_factory = None
        self._server_instance_dispose = None
        self._server_managed_instances: List[AgentInstance] = []

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
            parser.add_argument(
                "--instance-scope",
                choices=["shared", "connection", "request"],
                default="shared",
                help="Control MCP agent instancing behaviour (shared, connection, request)",
            )
            parser.add_argument(
                "--skills",
                help="Path to skills directory to use instead of default .claude/skills",
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

        # Apply programmatic quiet setting (overrides CLI if both are set)
        if self._programmatic_quiet:
            self.args.quiet = True

        # Apply CLI skills directory if not already set programmatically
        if (
            self._skills_directory_override is None
            and hasattr(self.args, "skills")
            and self.args.skills
        ):
            self._skills_directory_override = Path(self.args.skills).expanduser()

        self.name = name
        self.config_path = config_path

        try:
            # Load configuration directly for this instance
            self._load_config()

            # Apply programmatic quiet mode to config before creating app
            if self._programmatic_quiet and hasattr(self, "config"):
                if "logger" not in self.config:
                    self.config["logger"] = {}
                self.config["logger"]["progress_display"] = False
                self.config["logger"]["show_chat"] = False
                self.config["logger"]["show_tools"] = False

            # Create the app with our local settings
            self.app = Core(
                name=name,
                settings=config.Settings(**self.config) if hasattr(self, "config") else None,
                **kwargs,
            )

            # Stop progress display immediately if quiet mode is requested
            if self._programmatic_quiet:
                from fast_agent.ui.progress_display import progress_display

                progress_display.stop()

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
        from fast_agent.config import _settings, get_settings

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

    # Decorator methods with precise signatures for IDE completion
    # Provide annotations so IDEs can discover these attributes on instances
    if TYPE_CHECKING:  # pragma: no cover - typing aid only
        from collections.abc import Coroutine
        from pathlib import Path

        from fast_agent.skills import SkillManifest, SkillRegistry
        from fast_agent.types import RequestParams

        P = ParamSpec("P")
        R = TypeVar("R")

        def agent(
            self,
            name: str = "default",
            instruction_or_kwarg: Optional[str | Path | AnyUrl] = None,
            *,
            instruction: str | Path | AnyUrl = DEFAULT_AGENT_INSTRUCTION,
            servers: List[str] = [],
            tools: Optional[Dict[str, List[str]]] = None,
            resources: Optional[Dict[str, List[str]]] = None,
            prompts: Optional[Dict[str, List[str]]] = None,
            skills: Optional[List[SkillManifest | SkillRegistry | Path | str | None]] = None,
            model: Optional[str] = None,
            use_history: bool = True,
            request_params: RequestParams | None = None,
            human_input: bool = False,
            default: bool = False,
            elicitation_handler: Optional[ElicitationFnT] = None,
            api_key: str | None = None,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def custom(
            self,
            cls,
            name: str = "default",
            instruction_or_kwarg: Optional[str | Path | AnyUrl] = None,
            *,
            instruction: str | Path | AnyUrl = "You are a helpful agent.",
            servers: List[str] = [],
            tools: Optional[Dict[str, List[str]]] = None,
            resources: Optional[Dict[str, List[str]]] = None,
            prompts: Optional[Dict[str, List[str]]] = None,
            model: Optional[str] = None,
            use_history: bool = True,
            request_params: RequestParams | None = None,
            human_input: bool = False,
            default: bool = False,
            elicitation_handler: Optional[ElicitationFnT] = None,
            api_key: str | None = None,
        ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

        def orchestrator(
            self,
            name: str,
            *,
            agents: List[str],
            instruction: str
            | Path
            | AnyUrl = "You are an expert planner. Given an objective task and a list of Agents\n(which are collections of capabilities), your job is to break down the objective\ninto a series of steps, which can be performed by these agents.\n",
            model: Optional[str] = None,
            request_params: RequestParams | None = None,
            use_history: bool = False,
            human_input: bool = False,
            plan_type: Literal["full", "iterative"] = "full",
            plan_iterations: int = 5,
            default: bool = False,
            api_key: str | None = None,
        ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

        def iterative_planner(
            self,
            name: str,
            *,
            agents: List[str],
            instruction: str | Path | AnyUrl = "You are an expert planner. Plan iteratively.",
            model: Optional[str] = None,
            request_params: RequestParams | None = None,
            plan_iterations: int = -1,
            default: bool = False,
            api_key: str | None = None,
        ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

        def router(
            self,
            name: str,
            *,
            agents: List[str],
            instruction: Optional[str | Path | AnyUrl] = None,
            servers: List[str] = [],
            tools: Optional[Dict[str, List[str]]] = None,
            resources: Optional[Dict[str, List[str]]] = None,
            prompts: Optional[Dict[str, List[str]]] = None,
            model: Optional[str] = None,
            use_history: bool = False,
            request_params: RequestParams | None = None,
            human_input: bool = False,
            default: bool = False,
            elicitation_handler: Optional[ElicitationFnT] = None,
            api_key: str | None = None,
        ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

        def chain(
            self,
            name: str,
            *,
            sequence: List[str],
            instruction: Optional[str | Path | AnyUrl] = None,
            cumulative: bool = False,
            default: bool = False,
        ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

        def parallel(
            self,
            name: str,
            *,
            fan_out: List[str],
            fan_in: str | None = None,
            instruction: Optional[str | Path | AnyUrl] = None,
            include_request: bool = True,
            default: bool = False,
        ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

        def evaluator_optimizer(
            self,
            name: str,
            *,
            generator: str,
            evaluator: str,
            instruction: Optional[str | Path | AnyUrl] = None,
            min_rating: str = "GOOD",
            max_refinements: int = 3,
            default: bool = False,
        ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

    # Runtime bindings (actual implementations)
    agent = agent_decorator
    custom = custom_decorator
    orchestrator = orchestrator_decorator
    iterative_planner = orchestrator2_decorator
    router = router_decorator
    chain = chain_decorator
    parallel = parallel_decorator
    evaluator_optimizer = evaluator_optimizer_decorator

    def _get_acp_server_class(self):
        """Import and return the ACP server class with helpful error handling."""
        try:
            from fast_agent.acp.server import AgentACPServer

            return AgentACPServer
        except ModuleNotFoundError as exc:
            if exc.name == "acp":
                raise ServerInitializationError(
                    "ACP transport requires the 'agent-client-protocol' package. "
                    "Install it via `pip install fast-agent-mcp[acp]` or "
                    "`pip install agent-client-protocol`."
                ) from exc
            raise

    @asynccontextmanager
    async def run(self) -> AsyncIterator["AgentApp"]:
        """
        Context manager for running the application.
        Initializes all registered agents.
        """
        active_agents: Dict[str, AgentProtocol] = {}
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
                    registry = getattr(self.context, "skill_registry", None)
                    if self._skills_directory_override is not None:
                        override_registry = SkillRegistry(
                            base_dir=Path.cwd(),
                            override_directory=self._skills_directory_override,
                        )
                        self.context.skill_registry = override_registry
                        registry = override_registry

                    default_skills: List[SkillManifest] = []
                    if registry:
                        default_skills = registry.load_manifests()

                    self._apply_skills_to_agent_configs(default_skills)

                    # Apply quiet mode if requested
                    if quiet_mode:
                        cfg = self.app.context.config
                        if cfg is not None and cfg.logger is not None:
                            # Update our app's config directly
                            cfg_logger = cfg.logger
                            cfg_logger.progress_display = False
                            cfg_logger.show_chat = False
                            cfg_logger.show_tools = False

                        # Directly disable the progress display singleton
                        from fast_agent.ui.progress_display import progress_display

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

                    managed_instances: list[AgentInstance] = []
                    instance_lock = asyncio.Lock()

                    async def instantiate_agent_instance() -> AgentInstance:
                        async with instance_lock:
                            agents_map = await create_agents_in_dependency_order(
                                self.app,
                                self.agents,
                                model_factory_func,
                            )
                            validate_provider_keys_post_creation(agents_map)
                            instance = AgentInstance(AgentApp(agents_map), agents_map)
                            managed_instances.append(instance)
                            return instance

                    async def dispose_agent_instance(instance: AgentInstance) -> None:
                        async with instance_lock:
                            if instance in managed_instances:
                                managed_instances.remove(instance)
                        await instance.shutdown()

                    primary_instance = await instantiate_agent_instance()
                    wrapper = primary_instance.app
                    active_agents = primary_instance.agents

                    self._server_instance_factory = instantiate_agent_instance
                    self._server_instance_dispose = dispose_agent_instance
                    self._server_managed_instances = managed_instances

                    # Disable streaming if parallel agents are present
                    from fast_agent.agents.agent_types import AgentType

                    has_parallel = any(
                        agent.agent_type == AgentType.PARALLEL for agent in active_agents.values()
                    )
                    if has_parallel:
                        cfg = self.app.context.config
                        if cfg is not None and cfg.logger is not None:
                            cfg.logger.streaming = "none"

                    # Handle command line options that should be processed after agent initialization

                    # Handle --server option
                    # Check if parse_cli_args was True before checking self.args.server
                    if hasattr(self.args, "server") and self.args.server:
                        # For stdio/acp transports, use stderr to avoid interfering with JSON-RPC
                        import sys

                        is_stdio_transport = self.args.transport in ["stdio", "acp"]
                        configure_console_stream("stderr" if is_stdio_transport else "stdout")
                        output_stream = sys.stderr if is_stdio_transport else sys.stdout

                        try:
                            # Print info message if not in quiet mode
                            if not quiet_mode:
                                print(
                                    f"Starting fast-agent  '{self.name}' in server mode",
                                    file=output_stream,
                                )
                                print(f"Transport: {self.args.transport}", file=output_stream)
                                if self.args.transport == "sse":
                                    print(
                                        f"Listening on {self.args.host}:{self.args.port}",
                                        file=output_stream,
                                    )
                                print("Press Ctrl+C to stop", file=output_stream)

                            # Check which server type to use based on transport
                            if self.args.transport == "acp":
                                # Create and run ACP server
                                AgentACPServer = self._get_acp_server_class()

                                server_name = getattr(self.args, "server_name", None)
                                instance_scope = getattr(self.args, "instance_scope", "shared")
                                acp_server = AgentACPServer(
                                    primary_instance=primary_instance,
                                    create_instance=self._server_instance_factory,
                                    dispose_instance=self._server_instance_dispose,
                                    instance_scope=instance_scope,
                                    server_name=server_name or f"{self.name}",
                                )

                                # Run the ACP server (this is a blocking call)
                                await acp_server.run_async()
                            else:
                                # Create the MCP server
                                from fast_agent.mcp.server import AgentMCPServer

                                tool_description = getattr(self.args, "tool_description", None)
                                server_description = getattr(self.args, "server_description", None)
                                server_name = getattr(self.args, "server_name", None)
                                instance_scope = getattr(self.args, "instance_scope", "shared")
                                mcp_server = AgentMCPServer(
                                    primary_instance=primary_instance,
                                    create_instance=self._server_instance_factory,
                                    dispose_instance=self._server_instance_dispose,
                                    instance_scope=instance_scope,
                                    server_name=server_name or f"{self.name}-MCP-Server",
                                    server_description=server_description,
                                    tool_description=tool_description,
                                )

                                # Run the server directly (this is a blocking call)
                                await mcp_server.run_async(
                                    transport=self.args.transport,
                                    host=self.args.host,
                                    port=self.args.port,
                                )
                        except KeyboardInterrupt:
                            if not quiet_mode:
                                print("\nServer stopped by user (Ctrl+C)", file=output_stream)
                            raise SystemExit(0)
                        except Exception as e:
                            if not quiet_mode:
                                import traceback

                                traceback.print_exc()
                                print(f"\nServer stopped with error: {e}", file=output_stream)
                            else:
                                print(f"\nServer stopped with error: {e}", file=output_stream)
                            raise SystemExit(1)

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
                        prompt: List[PromptMessageExtended] = load_prompt(
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
                            prompt_result = await agent.generate(prompt)

                            # In quiet mode, just print the raw response
                            # The chat display should already be turned off by the configuration
                            if self.args.quiet:
                                print(f"{prompt_result.last_text()}")

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
                # Ensure progress display is stopped before showing usage summary
                try:
                    from fast_agent.ui.progress_display import progress_display

                    progress_display.stop()
                except:  # noqa: E722
                    pass

                # Print usage report before cleanup (show for user exits too)
                if (
                    getattr(self, "_server_managed_instances", None)
                    and not had_error
                    and not quiet_mode
                    and getattr(self.args, "server", False) is False
                ):
                    # Only show usage report for non-server interactive runs
                    if managed_instances:
                        instance = managed_instances[0]
                        self._print_usage_report(instance.agents)
                elif active_agents and not had_error and not quiet_mode:
                    self._print_usage_report(active_agents)

                # Clean up any active agents (always cleanup, even on errors)
                if getattr(self, "_server_managed_instances", None) and getattr(
                    self, "_server_instance_dispose", None
                ):
                    # Dispose any remaining instances
                    remaining_instances = list(self._server_managed_instances)
                    for instance in remaining_instances:
                        try:
                            await self._server_instance_dispose(instance)
                        except Exception:
                            pass
                    self._server_managed_instances.clear()
                elif active_agents:
                    for agent in active_agents.values():
                        try:
                            await agent.shutdown()
                        except Exception:
                            pass

    def _apply_skills_to_agent_configs(self, default_skills: List[SkillManifest]) -> None:
        self._default_skill_manifests = list(default_skills)

        for agent_data in self.agents.values():
            config_obj = agent_data.get("config")
            if not config_obj:
                continue

            resolved = self._resolve_skills(config_obj.skills)
            if not resolved:
                resolved = list(default_skills)
            else:
                resolved = self._deduplicate_skills(resolved)

            config_obj.skill_manifests = resolved

    def _resolve_skills(
        self,
        entry: SkillManifest
        | SkillRegistry
        | Path
        | str
        | List[SkillManifest | SkillRegistry | Path | str | None]
        | None,
    ) -> List[SkillManifest]:
        if entry is None:
            return []
        if isinstance(entry, list):
            manifests: List[SkillManifest] = []
            for item in entry:
                manifests.extend(self._resolve_skills(item))
            return manifests
        if isinstance(entry, SkillManifest):
            return [entry]
        if isinstance(entry, SkillRegistry):
            try:
                return entry.load_manifests()
            except Exception:
                logger.debug(
                    "Failed to load skills from registry",
                    data={"registry": type(entry).__name__},
                )
                return []
        if isinstance(entry, Path):
            return SkillRegistry.load_directory(entry.expanduser().resolve())
        if isinstance(entry, str):
            return SkillRegistry.load_directory(Path(entry).expanduser().resolve())

        logger.debug(
            "Unsupported skill entry type",
            data={"type": type(entry).__name__},
        )
        return []

    @staticmethod
    def _deduplicate_skills(manifests: List[SkillManifest]) -> List[SkillManifest]:
        unique: Dict[str, SkillManifest] = {}
        for manifest in manifests:
            key = manifest.name.lower()
            if key not in unique:
                unique[key] = manifest
        return list(unique.values())

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
        transport: str = "http",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
        tool_description: Optional[str] = None,
        instance_scope: str = "shared",
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
            server_description: Optional description/instructions for the MCP server
            tool_description: Optional description template for the exposed send tool.
                              Use {agent} to reference the agent name.
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
        self.args.tool_description = tool_description
        self.args.server_description = server_description
        self.args.server_name = server_name
        self.args.instance_scope = instance_scope
        self.args.quiet = (
            original_args.quiet if original_args and hasattr(original_args, "quiet") else False
        )
        self.args.model = None
        if original_args is not None and hasattr(original_args, "model"):
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
        tool_description: Optional[str] = None,
        instance_scope: str = "shared",
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
            server_description: Optional description/instructions for the MCP server
            tool_description: Optional description template for the exposed send tool.
        """
        await self.start_server(
            transport=transport,
            host=host,
            port=port,
            server_name=server_name,
            server_description=server_description,
            tool_description=tool_description,
            instance_scope=instance_scope,
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


@dataclass
class AgentInstance:
    app: AgentApp
    agents: Dict[str, "AgentProtocol"]

    async def shutdown(self) -> None:
        for agent in self.agents.values():
            try:
                shutdown = getattr(agent, "shutdown", None)
                if shutdown is None:
                    continue
                result = shutdown()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
