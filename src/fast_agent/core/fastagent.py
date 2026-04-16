"""
Direct FastAgent implementation that uses the simplified Agent architecture.
This replaces the traditional FastAgent with a more streamlined approach that
directly creates Agent instances without proxies.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
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
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)

import yaml
import yaml.parser
from opentelemetry import trace

from fast_agent import config
from fast_agent.core import Core
from fast_agent.core.agent_app import AgentApp, AgentRefreshResult
from fast_agent.core.agent_tools import add_tools_for_agents
from fast_agent.core.default_agent import agent_is_default, resolve_default_agent_name
from fast_agent.core.direct_decorators import DecoratorMixin
from fast_agent.core.direct_factory import (
    create_agents_in_dependency_order,
    get_default_model_source,
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
from fast_agent.core.instruction_utils import apply_instruction_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.core.validation import (
    validate_provider_keys_post_creation,
    validate_server_references,
    validate_workflow_references,
)
from fast_agent.mcp.connect_targets import resolve_target_entry
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.skills import SKILLS_DEFAULT, SkillManifest, SkillRegistry, SkillsDefault
from fast_agent.tools.function_tool_config import function_tool_entrypoint
from fast_agent.ui.console import configure_console_stream
from fast_agent.ui.usage_display import display_usage_report

if TYPE_CHECKING:
    from fastmcp.tools import FunctionTool

    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.core.agent_card_loader import LoadedAgentCard
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.interfaces import AgentProtocol, ModelFactoryFunctionProtocol
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
    from fast_agent.mcp.types import McpAgentProtocol
    from fast_agent.session import Session, SessionHydrationResult
    from fast_agent.types import PromptMessageExtended

F = TypeVar("F", bound=Callable[..., Any])  # For decorated functions
logger = get_logger(__name__)
SkillEntry: TypeAlias = SkillManifest | SkillRegistry | Path | str
SkillConfig: TypeAlias = SkillEntry | list[SkillEntry | None] | None | SkillsDefault
FileSignature: TypeAlias = tuple[int, int]
_PROMPT_EXIT_SHUTDOWN_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class RunSettings:
    quiet_mode: bool
    cli_model_override: str | None
    noenv_mode: bool
    server_mode: bool
    transport: str | None
    is_acp_server_mode: bool
    reload_enabled: bool


@dataclass
class RunRuntime:
    model_factory_func: ModelFactoryFunctionProtocol
    global_prompt_context: dict[str, str] | None
    is_acp_server_mode: bool
    noenv_mode: bool
    managed_instances: list[AgentInstance]
    instance_lock: asyncio.Lock


@dataclass
class ManagedRunState:
    runtime: RunRuntime
    primary_instance: AgentInstance
    wrapper: AgentApp
    active_agents: dict[str, AgentProtocol]


@dataclass(frozen=True)
class RuntimeCallbacks:
    create_instance: Callable[[], Awaitable[AgentInstance]]
    dispose_instance: Callable[[AgentInstance], Awaitable[None]]
    refresh_shared_instance: Callable[[], Awaitable[AgentRefreshResult]]
    reload_and_refresh: Callable[[], Awaitable[AgentRefreshResult]]
    reload_source: Callable[[], Awaitable[bool]] | None
    load_card_and_refresh: Callable[[str, str | None], Awaitable[tuple[list[str], list[str]]]]
    load_card_source: Callable[[str, str | None], Awaitable[tuple[list[str], list[str]]]]
    attach_agent_tools_and_refresh: Callable[[str, Sequence[str]], Awaitable[list[str]]]
    detach_agent_tools_and_refresh: Callable[[str, Sequence[str]], Awaitable[list[str]]]
    attach_agent_tools_source: Callable[[str, Sequence[str]], Awaitable[list[str]]]
    detach_agent_tools_source: Callable[[str, Sequence[str]], Awaitable[list[str]]]
    attach_mcp_server: Callable[
        [str, str, MCPServerSettings | None, MCPAttachOptions | None],
        Awaitable[MCPAttachResult],
    ]
    detach_mcp_server: Callable[[str, str], Awaitable[MCPDetachResult]]
    list_attached_mcp_servers: Callable[[str], Awaitable[list[str]]]
    list_configured_detached_mcp_servers: Callable[[str], Awaitable[list[str]]]
    dump_agent_card: Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class CardFileSnapshot:
    current_files: set[Path]
    removed_files: set[Path]
    current_stats: dict[Path, FileSignature]
    changed_files: set[Path]


@dataclass(frozen=True)
class WatchFileSnapshot:
    watch_files: set[Path]
    removed_files: set[Path]
    current_stats: dict[Path, FileSignature]
    changed_files: set[Path]
    current_tool_files: set[Path]


class FastAgent(DecoratorMixin):
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
        environment_dir: str | pathlib.Path | None = None,
        skills_directory: str | pathlib.Path | Sequence[str | pathlib.Path] | None = None,
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
        self._environment_dir_override = self._normalize_environment_dir(environment_dir)
        self._skills_directory_override = self._normalize_skill_directories(skills_directory)
        self._default_skill_manifests: list[SkillManifest] = []
        self._server_instance_factory = None
        self._server_instance_dispose = None
        self._server_managed_instances: list[AgentInstance] = []

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
                default=None,
                help="Agent name for --message/--prompt-file (defaults to the app default agent)",
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
                choices=["http", "stdio", "acp"],
                default=None,
                help="Transport protocol to use when running as a server (http, stdio, or acp)",
            )
            parser.add_argument(
                "--port",
                type=int,
                default=8000,
                help="Port to use when running as a server with HTTP transport",
            )
            parser.add_argument(
                "--host",
                default="0.0.0.0",
                help="Host address to bind to when running as a server with HTTP transport",
            )
            parser.add_argument(
                "--instance-scope",
                choices=["shared", "connection", "request"],
                default="shared",
                help=(
                    "Control MCP agent instancing behaviour (shared, connection, request). "
                    "ACP is always connection-scoped."
                ),
            )
            parser.add_argument(
                "--env",
                help="Override the base fast-agent environment directory",
            )
            parser.add_argument(
                "--skills",
                help="Path to skills directory to use instead of default skills directories",
            )
            parser.add_argument(
                "--dump",
                "--dump-agents",
                dest="dump_agents",
                help="Export all loaded agents as Markdown AgentCards into a directory",
            )
            parser.add_argument(
                "--dump-yaml",
                "--dump-agents-yaml",
                dest="dump_agents_yaml",
                help="Export all loaded agents as YAML AgentCards into a directory",
            )
            parser.add_argument(
                "--dump-agent",
                help="Export a single agent by name",
            )
            parser.add_argument(
                "--dump-agent-path",
                help="Output file path for --dump-agent",
            )
            parser.add_argument(
                "--dump-agent-yaml",
                action="store_true",
                help="Export a single agent as YAML (default: Markdown)",
            )
            parser.add_argument(
                "--reload",
                action="store_true",
                help="Enable manual AgentCard reloads (/reload)",
            )
            parser.add_argument(
                "--watch",
                action="store_true",
                help="Watch AgentCard paths and reload when files change",
            )
            parser.add_argument(
                "--card-tool",
                action="append",
                dest="card_tools",
                help="Path or URL to an AgentCard file to load as a tool (repeatable)",
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

            # Track whether CLI flags were explicitly provided
            cli_args = sys.argv[1:]
            server_flag_used = "--server" in cli_args
            transport_flag_used = any(
                arg == "--transport" or arg.startswith("--transport=") for arg in cli_args
            )

            # If a transport was provided, assume server mode even without --server
            if transport_flag_used and not getattr(self.args, "server", False):
                self.args.server = True

            # Default the transport if still unset
            if getattr(self.args, "transport", None) is None:
                self.args.transport = "http"

            # Warn that --server is deprecated when the user supplied it explicitly
            if server_flag_used:
                print(
                    "--server is deprecated; server mode is implied when --transport is provided. "
                    "This flag will be removed in a future release.",
                    file=sys.stderr,
                )

            # Handle version flag
            if self.args.version:
                try:
                    app_version = get_version("fast-agent-mcp")
                except:  # noqa: E722
                    app_version = "unknown"
                print(f"fast-agent-mcp v{app_version}")
                sys.exit(0)
        # --- End of wrapped logic ---

        # Force quiet mode automatically when running ACP transport
        transport = getattr(self.args, "transport", None)
        if transport == "acp":
            self._programmatic_quiet = True
            setattr(self.args, "quiet", True)

        # Apply programmatic quiet setting (overrides CLI if both are set)
        if self._programmatic_quiet:
            self.args.quiet = True

        # Apply CLI environment directory if not already set programmatically
        if self._environment_dir_override is None and hasattr(self.args, "env") and self.args.env:
            self._environment_dir_override = self._normalize_environment_dir(self.args.env)

        if self._environment_dir_override is not None:
            os.environ["ENVIRONMENT_DIR"] = str(self._environment_dir_override)

        # Apply CLI skills directory if not already set programmatically
        if (
            self._skills_directory_override is None
            and hasattr(self.args, "skills")
            and self.args.skills
        ):
            self._skills_directory_override = self._normalize_skill_directories(self.args.skills)

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

            # Propagate environment dir override into config so path helpers resolve consistently
            if self._environment_dir_override is not None and hasattr(self, "config"):
                self.config["environment_dir"] = str(self._environment_dir_override)

            # Propagate CLI skills override into config so resolve_skill_directories() works everywhere
            if self._skills_directory_override is not None and hasattr(self, "config"):
                if "skills" not in self.config:
                    self.config["skills"] = {}
                self.config["skills"]["directories"] = [
                    str(p) for p in self._skills_directory_override
                ]

            # Create settings and update global settings so resolve_skill_directories() works
            instance_settings = config.Settings(**self.config) if hasattr(self, "config") else None
            if instance_settings is not None:
                instance_settings._config_file = getattr(self, "_loaded_config_file", None)
                instance_settings._secrets_file = getattr(self, "_loaded_secrets_file", None)
            if instance_settings is not None:
                config.update_global_settings(instance_settings)

            # Create the app with our local settings
            self.app = Core(
                name=name,
                settings=instance_settings,
                **kwargs,
            )

            # Stop progress display immediately if quiet mode is requested
            if self._programmatic_quiet:
                if getattr(self.args, "server", False) and getattr(
                    self.args, "transport", None
                ) in ["stdio", "acp"]:
                    configure_console_stream("stderr")
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
        self.agents: dict[str, AgentCardData] = {}
        # Global function-tool registry populated by @fast.tool.
        # These are local Python tools, not AgentConfig.tools MCP filter maps.
        self._registered_tools: list[FunctionTool] = []
        # Tracking for AgentCard-loaded agents
        self._agent_card_sources: dict[str, Path] = {}
        self._agent_card_roots: dict[Path, set[str]] = {}
        self._agent_card_root_files: dict[Path, set[Path]] = {}
        self._agent_card_root_watch_files: dict[Path, set[Path]] = {}
        self._agent_card_file_cache: dict[Path, tuple[int, int]] = {}
        self._agent_card_name_by_path: dict[Path, str] = {}
        self._agent_card_histories: dict[str, list[Path]] = {}
        self._agent_card_history_mtime: dict[str, float] = {}
        self._agent_card_history_len: dict[str, int] = {}
        self._agent_card_tool_files: dict[Path, set[Path]] = {}
        self._agent_card_last_changed: set[str] = set()
        self._agent_card_last_removed: set[str] = set()
        self._agent_card_last_dependents: set[str] = set()
        self._agent_declared_servers: dict[str, list[str]] = {}
        self._card_mcp_owned_servers: dict[str, set[str]] = {}
        self._dynamic_mcp_server_names: set[str] = set()
        self._base_mcp_servers: dict[str, MCPServerSettings] | None = None
        self._agent_registry_version: int = 0
        self._agent_card_watch_task: asyncio.Task[None] | None = None
        self._agent_card_reload_lock: asyncio.Lock | None = None
        self._agent_card_watch_reload: Callable[[], Awaitable[AgentRefreshResult]] | None = None
        self._card_collision_warnings: list[str] = []

    @staticmethod
    def _normalize_skill_directories(
        value: str | Path | Sequence[str | Path] | None,
    ) -> list[Path] | None:
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            entries: list[str | Path] = [value]
        else:
            entries = list(value)
        return [Path(entry).expanduser() for entry in entries]

    @staticmethod
    def _normalize_environment_dir(value: str | Path | None) -> Path | None:
        if value is None:
            return None
        env_dir = Path(value).expanduser()
        if not env_dir.is_absolute():
            return (Path.cwd() / env_dir).resolve()
        return env_dir.resolve()

    def _load_config(self) -> None:
        """Load configuration from YAML file including secrets using get_settings
        but without relying on the global cache."""

        import fast_agent.config as _config_module

        # Temporarily clear the global settings to ensure a fresh load
        old_settings = _config_module._settings
        _config_module._settings = None

        try:
            # Use get_settings to load config - this handles all paths and secrets merging
            settings = _config_module.get_settings(self.config_path)
            self._loaded_config_file = settings._config_file if settings else None
            self._loaded_secrets_file = settings._secrets_file if settings else None
            # Convert to dict for backward compatibility
            self.config = settings.model_dump() if settings else {}
        finally:
            # Restore the original global settings
            _config_module._settings = old_settings

    def _is_acp_server_mode(self) -> bool:
        """Return True when this instance is serving the ACP transport."""
        return bool(getattr(self.args, "server", False)) and getattr(
            self.args, "transport", None
        ) == "acp"

    @property
    def context(self) -> Context:
        """Access the application context"""
        return self.app.context

    def load_agents(self, path: str | Path) -> list[str]:
        """
        Load AgentCards from a file or directory and register them as agents.

        Loading is idempotent for the provided path: any previously loaded agents
        from the same path that are no longer present are removed.

        Returns:
            Sorted list of agent names loaded from the provided path.
        """
        root = Path(path).expanduser().resolve()
        changed = self._load_agent_cards_from_root(root, incremental=False)
        if changed:
            self._agent_registry_version += 1
        return sorted(self._agent_card_roots.get(root, set()))

    def load_agents_from_url(self, url: str) -> list[str]:
        """Load an AgentCard from a URL (markdown or YAML)."""
        import tempfile

        from fast_agent.core.agent_card_loader import load_agent_cards
        from fast_agent.core.direct_decorators import _fetch_url_content

        content = _fetch_url_content(url)

        # Determine extension from URL
        suffix = ".md"
        url_lower = url.lower()
        if url_lower.endswith((".yaml", ".yml")):
            suffix = ".yaml"
        elif url_lower.endswith((".md", ".markdown")):
            suffix = ".md"

        # Write to temp file and parse
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            cards = load_agent_cards(temp_path)
            loaded_names = [card.name for card in cards]
            for card in cards:
                # Check for conflicts
                if card.name in self.agents and card.name not in self._agent_card_sources:
                    raise AgentConfigError(
                        f"Agent '{card.name}' already exists and is not from AgentCard",
                        f"URL: {url}",
                    )
                # Register the agent
                self.agents[card.name] = card.agent_data
                # Note: URL-loaded cards don't track source path (no reload support)
                if card.message_files:
                    self._agent_card_histories[card.name] = card.message_files
            # Apply skills
            if cards:
                self._apply_skills_to_agent_configs(self._default_skill_manifests)
                self._agent_card_last_changed.update(loaded_names)
                self._agent_registry_version += 1
            return loaded_names
        finally:
            temp_path.unlink(missing_ok=True)

    def attach_agent_tools(self, parent_name: str, child_names: Sequence[str]) -> list[str]:
        """Attach loaded agents to a parent agent via Agents-as-Tools."""
        parent_data = self.agents.get(parent_name)
        if not parent_data:
            raise AgentConfigError(f"Agent '{parent_name}' not found")

        if parent_data.get("type") not in ("basic", "smart", "custom"):
            raise AgentConfigError(f"Agent '{parent_name}' does not support agents-as-tools")

        missing = [
            name for name in child_names if name and name != parent_name and name not in self.agents
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise AgentConfigError(f"Agent(s) not found: {missing_list}")

        existing = list(parent_data.get("child_agents") or [])
        added: list[str] = []
        for name in child_names:
            if not name or name == parent_name:
                continue
            if name in existing or name in added:
                continue
            added.append(name)

        if added:
            parent_data["child_agents"] = existing + added
            self._agent_card_last_changed.add(parent_name)
            self._agent_registry_version += 1

        return added

    def detach_agent_tools(self, parent_name: str, child_names: Sequence[str]) -> list[str]:
        """Detach agents-as-tools from a parent agent."""
        parent_data = self.agents.get(parent_name)
        if not parent_data:
            raise AgentConfigError(f"Agent '{parent_name}' not found")

        if parent_data.get("type") not in ("basic", "smart", "custom"):
            raise AgentConfigError(f"Agent '{parent_name}' does not support agents-as-tools")

        existing = list(parent_data.get("child_agents") or [])
        removed: list[str] = []
        for name in child_names:
            if not name or name == parent_name:
                continue
            if name not in existing or name in removed:
                continue
            removed.append(name)

        if removed:
            pruned = [name for name in existing if name not in set(removed)]
            if pruned:
                parent_data["child_agents"] = pruned
            else:
                parent_data.pop("child_agents", None)
            self._agent_card_last_changed.add(parent_name)
            self._agent_registry_version += 1

        return removed

    def get_default_agent_name(self) -> str | None:
        """Find the default agent name from the registration data.

        Returns the name of the first agent with config.default=True,
        excluding tool_only agents. Falls back to the first non-tool_only
        agent if no explicit default is set.

        Returns:
            The name of the default agent, or None if no agents are registered.
        """
        return resolve_default_agent_name(
            self.agents,
            is_default=lambda _name, agent_data: bool(
                getattr(agent_data.get("config"), "default", False)
            ),
            is_tool_only=lambda _name, agent_data: bool(agent_data.get("tool_only", False)),
        )

    def dump_agent_card_text(self, name: str, *, as_yaml: bool = False) -> str:
        """Render an AgentCard as text."""
        from fast_agent.core.agent_card_loader import dump_agent_to_string

        agent_data = self.agents.get(name)
        if not agent_data:
            raise AgentConfigError(f"Agent '{name}' not found for dump")

        message_paths = self._agent_card_histories.get(name)
        return dump_agent_to_string(name, agent_data, as_yaml=as_yaml, message_paths=message_paths)

    async def reload_agents(self) -> bool:
        """Reload all previously registered AgentCard roots."""
        if not self._agent_card_roots:
            return False

        if self._agent_card_reload_lock is None:
            self._agent_card_reload_lock = asyncio.Lock()

        async with self._agent_card_reload_lock:
            self._agent_card_last_changed = set()
            self._agent_card_last_removed = set()
            self._agent_card_last_dependents = set()
            changed = False
            for root in sorted(self._agent_card_roots.keys()):
                if self._load_agent_cards_from_root(root, incremental=True):
                    changed = True

            if changed:
                self._agent_registry_version += 1
            return changed

    def _resolve_current_card_files(self, root: Path, *, incremental: bool) -> set[Path]:
        if root.exists():
            return self._collect_agent_card_files(root)
        if incremental:
            return set()
        raise AgentConfigError(f"AgentCard path not found: {root}")

    @staticmethod
    def _collect_existing_file_signatures(
        paths: set[Path],
    ) -> tuple[set[Path], dict[Path, FileSignature]]:
        existing_paths = set(paths)
        current_stats: dict[Path, FileSignature] = {}
        for path_entry in list(existing_paths):
            try:
                stat = path_entry.stat()
            except FileNotFoundError:
                existing_paths.discard(path_entry)
                continue
            current_stats[path_entry] = (stat.st_mtime_ns, stat.st_size)
        return existing_paths, current_stats

    def _select_changed_files(
        self,
        current_stats: dict[Path, FileSignature],
        *,
        incremental: bool,
    ) -> set[Path]:
        if not incremental:
            return set(current_stats.keys())
        return {
            path_entry
            for path_entry, signature in current_stats.items()
            if self._agent_card_file_cache.get(path_entry) != signature
        }

    def _build_card_file_snapshot(self, root: Path, *, incremental: bool) -> CardFileSnapshot:
        current_files = self._resolve_current_card_files(root, incremental=incremental)
        previous_files = self._agent_card_root_files.get(root, set())
        removed_files = previous_files - current_files
        current_files, current_stats = self._collect_existing_file_signatures(current_files)
        changed_files = self._select_changed_files(current_stats, incremental=incremental)
        return CardFileSnapshot(
            current_files=current_files,
            removed_files=removed_files,
            current_stats=current_stats,
            changed_files=changed_files,
        )

    def _load_agent_cards_from_file(
        self,
        path_entry: Path,
        *,
        incremental: bool,
    ) -> list[LoadedAgentCard]:
        from fast_agent.core.agent_card_loader import load_agent_cards

        try:
            return load_agent_cards(path_entry)
        except AgentConfigError as exc:
            if not incremental:
                raise
            if "Instruction is required" in exc.message:
                logger.warning(
                    "Skipping incomplete AgentCard during reload; waiting for write to complete",
                    path=str(path_entry),
                )
                return []
            logger.warning(
                "Skipping invalid AgentCard during reload",
                path=str(path_entry),
                error=str(exc),
            )
            return []
        except Exception as exc:
            if not incremental:
                raise
            logger.warning(
                "Skipping invalid AgentCard during reload",
                path=str(path_entry),
                error=str(exc),
            )
            return []

    def _record_loaded_card_tool_files(self, cards: Sequence[LoadedAgentCard]) -> None:
        for card in cards:
            config = card.agent_data.get("config")
            function_tools = getattr(config, "function_tools", None)
            self._agent_card_tool_files[card.path] = self._resolve_function_tool_paths(
                card.path,
                function_tools,
            )

    def _load_cards_for_paths(
        self,
        path_entries: set[Path],
        *,
        incremental: bool,
        cards: list[LoadedAgentCard],
        loaded_card_files: set[Path],
    ) -> None:
        for path_entry in sorted(path_entries):
            loaded_cards = self._load_agent_cards_from_file(path_entry, incremental=incremental)
            cards.extend(loaded_cards)
            loaded_card_files.add(path_entry)
            self._record_loaded_card_tool_files(loaded_cards)

    def _collect_current_tool_files(self, current_card_files: set[Path]) -> set[Path]:
        current_tool_files: set[Path] = set()
        for card_path in current_card_files:
            current_tool_files.update(self._agent_card_tool_files.get(card_path, set()))
        return current_tool_files

    def _collect_current_history_files_for_root(
        self,
        root: Path,
        cards: Sequence[LoadedAgentCard],
    ) -> set[Path]:
        current_history_files: set[Path] = set()
        for history_files in self._agent_card_histories.values():
            for history_file in history_files:
                try:
                    if history_file.is_relative_to(root):
                        current_history_files.add(history_file)
                except ValueError:
                    continue
        for card in cards:
            for history_file in card.message_files or []:
                try:
                    if history_file.is_relative_to(root):
                        current_history_files.add(history_file)
                except ValueError:
                    continue
        return current_history_files

    def _build_watch_file_snapshot(
        self,
        root: Path,
        *,
        current_card_files: set[Path],
        cards: Sequence[LoadedAgentCard],
        incremental: bool,
    ) -> WatchFileSnapshot:
        current_tool_files = self._collect_current_tool_files(current_card_files)
        current_history_files = self._collect_current_history_files_for_root(root, cards)
        watch_files = set(current_card_files) | current_tool_files | current_history_files
        previous_watch_files = self._agent_card_root_watch_files.get(root, set())
        removed_files = previous_watch_files - watch_files
        watch_files, current_stats = self._collect_existing_file_signatures(watch_files)
        changed_files = self._select_changed_files(current_stats, incremental=incremental)
        return WatchFileSnapshot(
            watch_files=watch_files,
            removed_files=removed_files,
            current_stats=current_stats,
            changed_files=changed_files,
            current_tool_files=current_tool_files,
        )

    @staticmethod
    def _get_changed_tool_files(
        current_card_files: set[Path],
        watch_snapshot: WatchFileSnapshot,
    ) -> set[Path]:
        changed_tool_files = {
            path_entry
            for path_entry in watch_snapshot.changed_files
            if path_entry in watch_snapshot.current_tool_files
        }
        removed_tool_files = {
            path_entry
            for path_entry in watch_snapshot.removed_files
            if path_entry not in current_card_files
        }
        return changed_tool_files | removed_tool_files

    def _reload_cards_affected_by_tool_changes(
        self,
        current_card_files: set[Path],
        changed_tool_files: set[Path],
        *,
        incremental: bool,
        cards: list[LoadedAgentCard],
        loaded_card_files: set[Path],
    ) -> None:
        if not changed_tool_files:
            return

        affected_card_files = {
            card_path
            for card_path in current_card_files
            if self._agent_card_tool_files.get(card_path, set()) & changed_tool_files
        }
        self._load_cards_for_paths(
            affected_card_files - loaded_card_files,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

    def _load_agent_cards_from_root(self, root: Path, *, incremental: bool) -> bool:
        card_snapshot = self._build_card_file_snapshot(root, incremental=incremental)
        for removed_path in card_snapshot.removed_files:
            self._agent_card_tool_files.pop(removed_path, None)

        cards: list[LoadedAgentCard] = []
        loaded_card_files: set[Path] = set()
        self._load_cards_for_paths(
            card_snapshot.changed_files,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

        missing_tool_cards = {
            path_entry
            for path_entry in card_snapshot.current_files
            if path_entry not in self._agent_card_tool_files
        }
        self._load_cards_for_paths(
            missing_tool_cards,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

        watch_snapshot = self._build_watch_file_snapshot(
            root,
            current_card_files=card_snapshot.current_files,
            cards=cards,
            incremental=incremental,
        )
        changed_tool_files = self._get_changed_tool_files(
            card_snapshot.current_files,
            watch_snapshot,
        )
        self._reload_cards_affected_by_tool_changes(
            card_snapshot.current_files,
            changed_tool_files,
            incremental=incremental,
            cards=cards,
            loaded_card_files=loaded_card_files,
        )

        self._apply_agent_card_updates(
            root,
            current_files=card_snapshot.current_files,
            removed_files=card_snapshot.removed_files,
            changed_cards=cards,
            current_stats=watch_snapshot.current_stats,
        )

        for path_entry in watch_snapshot.removed_files:
            self._agent_card_file_cache.pop(path_entry, None)

        self._agent_card_root_watch_files[root] = set(watch_snapshot.watch_files)

        return bool(card_snapshot.removed_files or watch_snapshot.changed_files)

    @staticmethod
    def _agent_card_extensions() -> set[str]:
        return {".md", ".markdown", ".yaml", ".yml"}

    def _collect_agent_card_files(self, root: Path) -> set[Path]:
        def _has_frontmatter(path: Path) -> bool:
            try:
                raw_text = path.read_text(encoding="utf-8")
            except Exception:
                return False
            if raw_text.startswith("\ufeff"):
                raw_text = raw_text.lstrip("\ufeff")
            for line in raw_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                return stripped in ("---", "+++")
            return False

        if root.is_dir():
            extensions = self._agent_card_extensions()
            return {
                entry
                for entry in root.iterdir()
                if entry.is_file()
                and entry.suffix.lower() in extensions
                and (entry.suffix.lower() not in {".md", ".markdown"} or _has_frontmatter(entry))
            }

        if root.suffix.lower() not in self._agent_card_extensions():
            raise AgentConfigError(f"Unsupported AgentCard file extension: {root}")
        if root.suffix.lower() in {".md", ".markdown"} and not _has_frontmatter(root):
            raise AgentConfigError(
                "AgentCard markdown files must include frontmatter",
                f"Missing frontmatter in {root}",
            )
        return {root}

    @staticmethod
    def _resolve_function_tool_paths(
        card_path: Path,
        function_tools: Sequence[object] | None,
    ) -> set[Path]:
        tool_paths: set[Path] = set()
        if not function_tools:
            return tool_paths
        for spec in function_tools:
            entrypoint = function_tool_entrypoint(spec)
            if not entrypoint or ":" not in entrypoint:
                continue
            module_path_str, _func_name = entrypoint.rsplit(":", 1)
            module_path = Path(module_path_str)
            if not module_path.is_absolute():
                module_path = (card_path.parent / module_path).resolve()
            if module_path.suffix.lower() == ".py":
                tool_paths.add(module_path)
        return tool_paths

    def _apply_agent_card_updates(
        self,
        root: Path,
        *,
        current_files: set[Path],
        removed_files: set[Path],
        changed_cards: list[LoadedAgentCard],
        current_stats: dict[Path, tuple[int, int]],
    ) -> None:
        removed_names = {
            self._agent_card_name_by_path[path]
            for path in removed_files
            if path in self._agent_card_name_by_path
        }
        removed_dependents: set[str] = set()
        if removed_names:
            from fast_agent.core.validation import get_agent_dependencies

            removed_set = set(removed_names)
            for name, agent_data in self.agents.items():
                if get_agent_dependencies(agent_data) & removed_set:
                    removed_dependents.add(name)

        new_names_by_path: dict[Path, str] = {}
        seen_names: dict[str, Path] = {}
        changed_names: set[str] = set()
        for card in changed_cards:
            changed_names.add(card.name)
            if card.name in seen_names:
                raise AgentConfigError(
                    f"Duplicate agent name '{card.name}' during reload",
                    f"Conflicts: {seen_names[card.name]} and {card.path}",
                )
            seen_names[card.name] = card.path
            new_names_by_path[card.path] = card.name

            existing_source = self._agent_card_sources.get(card.name)
            if card.name in self.agents and existing_source is None:
                raise AgentConfigError(
                    f"Agent '{card.name}' already exists and is not loaded from AgentCard",
                    f"Path: {root}",
                )
            if existing_source is not None and existing_source != card.path:
                if existing_source in removed_files:
                    continue

                # Check if this is a tool-cards vs agent-cards collision
                def _is_tool_card_path(path: Path) -> bool:
                    return path.parent.name == "tool-cards"

                existing_is_tool = _is_tool_card_path(Path(existing_source))
                new_is_tool = _is_tool_card_path(card.path)

                if existing_is_tool != new_is_tool:
                    # Cross-directory collision - tool-cards takes priority
                    if new_is_tool:
                        # New card is tool-card, existing is agent-card
                        # Override: clean up existing, let new card load
                        warning_msg = (
                            f"Agent '{card.name}' defined in both tool-cards and agent-cards. "
                            f"Using tool-card version from {card.path}. "
                            f"Skipping agent-card at {existing_source}."
                        )
                        print(f"Warning: {warning_msg}", file=sys.stderr)
                        self._card_collision_warnings.append(warning_msg)
                        # Mark existing agent-card for removal so tool-card can replace it
                        removed_names.add(card.name)
                        # Continue to let this card be processed normally
                    else:
                        # Existing is tool-card, new is agent-card
                        # Skip: keep existing, don't load new
                        warning_msg = (
                            f"Agent '{card.name}' defined in both tool-cards and agent-cards. "
                            f"Using tool-card version from {existing_source}. "
                            f"Skipping agent-card at {card.path}."
                        )
                        print(f"Warning: {warning_msg}", file=sys.stderr)
                        self._card_collision_warnings.append(warning_msg)
                        continue  # Skip loading this agent-card
                else:
                    # Same directory type collision - error as before
                    raise AgentConfigError(
                        f"Agent '{card.name}' already loaded from {existing_source}",
                        f"Path: {card.path}",
                    )

            previous_name = self._agent_card_name_by_path.get(card.path)
            if previous_name and previous_name != card.name:
                removed_names.add(previous_name)

        for name in sorted(removed_names):
            self.agents.pop(name, None)
            self._agent_card_sources.pop(name, None)
            self._agent_card_histories.pop(name, None)
            self._agent_card_history_mtime.pop(name, None)
            self._agent_card_history_len.pop(name, None)

        for path_entry in removed_files:
            self._agent_card_name_by_path.pop(path_entry, None)
            self._agent_card_file_cache.pop(path_entry, None)

        for card in changed_cards:
            self.agents[card.name] = card.agent_data

            self._agent_card_sources[card.name] = card.path
            self._agent_card_name_by_path[card.path] = card.name

            if card.message_files:
                self._agent_card_histories[card.name] = card.message_files
            else:
                self._agent_card_histories.pop(card.name, None)
                self._agent_card_history_mtime.pop(card.name, None)
                self._agent_card_history_len.pop(card.name, None)

        if removed_names:
            removed_set = set(removed_names)
            for agent_data in self.agents.values():
                child_agents = agent_data.get("child_agents")
                if not child_agents:
                    continue
                pruned = [name for name in child_agents if name not in removed_set]
                if pruned != child_agents:
                    agent_data["child_agents"] = pruned

        for path_entry, signature in current_stats.items():
            self._agent_card_file_cache[path_entry] = signature

        self._agent_card_root_files[root] = set(current_files)
        self._agent_card_roots[root] = {
            self._agent_card_name_by_path[path_entry]
            for path_entry in current_files
            if path_entry in self._agent_card_name_by_path
        }

        if changed_cards or removed_files:
            self._apply_skills_to_agent_configs(self._default_skill_manifests)
            from fast_agent.core.validation import get_agent_dependencies

            if changed_names:
                changed_dependents: set[str] = set()
                for name, agent_data in self.agents.items():
                    if name in changed_names:
                        continue
                    if get_agent_dependencies(agent_data) & changed_names:
                        changed_dependents.add(name)
                self._agent_card_last_dependents.update(changed_dependents)

            self._agent_card_last_changed.update(changed_names)
            self._agent_card_last_removed.update(removed_names)
            self._agent_card_last_dependents.update(removed_dependents)

    @staticmethod
    def _settings_signature(settings: MCPServerSettings) -> dict[str, Any]:
        return settings.model_dump(exclude_none=True)

    @staticmethod
    def _settings_equivalent(left: MCPServerSettings, right: MCPServerSettings) -> bool:
        return FastAgent._settings_signature(left) == FastAgent._settings_signature(right)

    @staticmethod
    def _copy_server_settings(settings: MCPServerSettings, *, name: str) -> MCPServerSettings:
        copied = settings.model_copy(deep=True)
        copied.name = name
        return copied

    def _sync_agent_card_mcp_servers(self) -> None:
        context = getattr(self.app, "context", None)
        if context is None:
            return

        app_config = getattr(context, "config", None)
        if app_config is None:
            return

        if app_config.mcp is None:
            app_config.mcp = config.MCPSettings()

        configured_servers = app_config.mcp.servers or {}
        if self._base_mcp_servers is None:
            self._base_mcp_servers = {
                name: self._copy_server_settings(server, name=name)
                for name, server in configured_servers.items()
            }

        registry = getattr(context, "server_registry", None)
        existing_registry = getattr(registry, "registry", {}) if registry is not None else {}

        preserved_runtime_servers: dict[str, MCPServerSettings] = {}
        if isinstance(existing_registry, dict):
            for name, server in existing_registry.items():
                if name in self._dynamic_mcp_server_names:
                    continue
                if not isinstance(server, config.MCPServerSettings):
                    continue
                preserved_runtime_servers[name] = self._copy_server_settings(server, name=name)

        effective_servers: dict[str, MCPServerSettings] = {
            name: self._copy_server_settings(server, name=name)
            for name, server in (self._base_mcp_servers or {}).items()
        }

        for name, server in preserved_runtime_servers.items():
            if name in effective_servers:
                continue
            effective_servers[name] = server

        resolved_servers_by_agent: dict[str, list[str]] = {}
        card_owners: dict[str, set[str]] = {}
        all_dynamic_server_names: set[str] = set()

        for agent_name in sorted(self.agents.keys()):
            agent_data = self.agents[agent_name]
            config_obj = agent_data.get("config")
            if config_obj is None:
                continue

            entries = list(getattr(config_obj, "mcp_connect", []) or [])
            if not entries:
                resolved_servers_by_agent[agent_name] = []
                continue

            owner = str(getattr(config_obj, "source_path", None) or f"agent:{agent_name}")
            for index, entry in enumerate(entries):
                explicit_name = entry.name

                if entry.connector_id is not None:
                    if entry.target is not None:
                        raise AgentConfigError(
                            f"Invalid mcp_connect entry for agent '{agent_name}'",
                            f"Entry {index}: target must be omitted when connector_id is set",
                        )
                    if explicit_name is None:
                        raise AgentConfigError(
                            f"Invalid mcp_connect entry for agent '{agent_name}'",
                            f"Entry {index}: name is required when connector_id is set",
                        )

                    payload: dict[str, Any] = {
                        "name": explicit_name,
                        "description": entry.description,
                        "management": entry.management,
                        "connector_id": entry.connector_id,
                        "headers": dict(entry.headers) if entry.headers is not None else None,
                        "access_token": entry.access_token,
                        "auth": dict(entry.auth) if entry.auth is not None else None,
                    }
                    if entry.defer_loading is not None:
                        payload["defer_loading"] = entry.defer_loading
                    try:
                        resolved_settings = config.MCPServerSettings.model_validate(payload)
                    except Exception as exc:  # noqa: BLE001
                        raise AgentConfigError(
                            f"Invalid mcp_connect entry for agent '{agent_name}'",
                            f"Entry {index} connector_id '{entry.connector_id}': {exc}",
                        ) from exc
                    resolved_name = explicit_name
                    target_label = entry.connector_id
                else:
                    target = entry.target
                    if target is None:
                        raise AgentConfigError(
                            f"Invalid mcp_connect entry for agent '{agent_name}'",
                            f"Entry {index}: target must be a non-empty string",
                        )

                    overrides: dict[str, Any] = {}
                    if entry.description is not None:
                        overrides["description"] = entry.description
                    if entry.management is not None:
                        overrides["management"] = entry.management
                    if entry.headers is not None:
                        overrides["headers"] = dict(entry.headers)
                    if entry.access_token is not None:
                        overrides["access_token"] = entry.access_token
                    if entry.defer_loading is not None:
                        overrides["defer_loading"] = entry.defer_loading
                    if entry.auth is not None:
                        overrides["auth"] = dict(entry.auth)

                    try:
                        resolved_name, resolved_settings = resolve_target_entry(
                            target=target,
                            default_name=explicit_name,
                            overrides=overrides,
                            source_path=f"mcp_connect[{index}].target",
                        )
                    except Exception as exc:  # noqa: BLE001
                        raise AgentConfigError(
                            f"Invalid mcp_connect entry for agent '{agent_name}'",
                            f"Entry {index} target '{target}': {exc}",
                        ) from exc
                    target_label = target

                existing = effective_servers.get(resolved_name)
                if existing is not None and not self._settings_equivalent(existing, resolved_settings):
                    raise AgentConfigError(
                        (
                            f"Server name collision for '{resolved_name}' from mcp_connect "
                            f"target '{target_label}'."
                        ),
                        "Set an explicit unique `name` or change target.",
                    )

                if existing is None:
                    effective_servers[resolved_name] = self._copy_server_settings(
                        resolved_settings,
                        name=resolved_name,
                    )

                resolved_servers_by_agent.setdefault(agent_name, []).append(resolved_name)
                card_owners.setdefault(owner, set()).add(resolved_name)
                all_dynamic_server_names.add(resolved_name)

        active_agent_names = set(self.agents.keys())
        for name, agent_data in self.agents.items():
            config_obj = agent_data.get("config")
            if config_obj is None:
                continue

            current_declared = list(getattr(config_obj, "servers", []))
            if name not in self._agent_declared_servers or name in self._agent_card_last_changed:
                self._agent_declared_servers[name] = current_declared

            base_servers = list(self._agent_declared_servers.get(name, []))
            self._agent_declared_servers[name] = base_servers
            merged_servers = list(dict.fromkeys(base_servers + resolved_servers_by_agent.get(name, [])))
            config_obj.servers = merged_servers

        for name in list(self._agent_declared_servers.keys()):
            if name not in active_agent_names:
                self._agent_declared_servers.pop(name, None)

        app_config.mcp.servers = {
            name: self._copy_server_settings(server, name=name)
            for name, server in effective_servers.items()
        }
        if registry is not None:
            registry.registry = {
                name: self._copy_server_settings(server, name=name)
                for name, server in effective_servers.items()
            }

        self._card_mcp_owned_servers = card_owners
        self._dynamic_mcp_server_names = all_dynamic_server_names

    def _get_registry_version(self) -> int:
        return self._agent_registry_version

    async def _watch_agent_cards(self) -> None:
        roots = sorted(self._agent_card_roots.keys())
        if not roots:
            return

        try:
            from watchfiles import awatch

            async for _changes in awatch(*roots):
                await self._reload_agent_cards_from_watch()
        except ImportError:
            logger.info("watchfiles not available; falling back to polling for AgentCard reloads")
            try:
                while True:
                    await asyncio.sleep(1.0)
                    await self._reload_agent_cards_from_watch()
            except asyncio.CancelledError:
                return
        except asyncio.CancelledError:
            return

    async def _reload_agent_cards_from_watch(self) -> bool:
        reload_callback = self._agent_card_watch_reload
        if reload_callback is None:
            return await self.reload_agents()
        result = await reload_callback()
        if isinstance(result, AgentRefreshResult):
            return result.changed
        return result

    # Decorator methods with precise signatures for IDE completion

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

    def _prepare_run_settings(self) -> RunSettings:
        """Compute the run-mode settings after app initialization."""
        quiet_mode = getattr(self.args, "quiet", False)
        server_mode = bool(getattr(self.args, "server", False))
        transport = getattr(self.args, "transport", None)
        if transport in ["stdio", "acp"] and server_mode:
            quiet_mode = True
            configure_console_stream("stderr")

        cli_model_override = getattr(self.args, "model", None)
        noenv_mode = bool(getattr(self.args, "noenv", False))

        cfg = self.context.config
        model_source_override = getattr(self.args, "model_source_override", None)
        model_source = model_source_override or get_default_model_source(
            config_default_model=cfg.default_model if cfg else None,
            cli_model=cli_model_override,
            model_references=cfg.model_references if cfg else None,
        )
        if cfg:
            cfg.model_source = model_source  # type: ignore[attr-defined]
            cfg.cli_model_override = cli_model_override  # type: ignore[attr-defined]
            if noenv_mode:
                cfg.session_history = False

        return RunSettings(
            quiet_mode=quiet_mode,
            cli_model_override=cli_model_override,
            noenv_mode=noenv_mode,
            server_mode=server_mode,
            transport=transport,
            is_acp_server_mode=server_mode and transport == "acp",
            reload_enabled=bool(
                getattr(self.args, "reload", False) or getattr(self.args, "watch", False)
            ),
        )

    def _load_default_skills_for_run(self) -> list[SkillManifest]:
        """Load default skill manifests, applying any run-specific overrides."""
        registry = getattr(self.context, "skill_registry", None)
        if self._skills_directory_override is not None:
            override_registry = SkillRegistry(
                base_dir=Path.cwd(),
                directories=self._skills_directory_override,
            )
            self.context.skill_registry = override_registry
            registry = override_registry

        if registry is None:
            return []

        try:
            return registry.load_manifests()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load skills; continuing without them",
                data={"error": str(exc)},
            )
            return []

    def _configure_quiet_mode_for_run(self) -> None:
        """Disable run-time progress and chat display output."""
        cfg = self.app.context.config
        if cfg is not None and cfg.logger is not None:
            cfg.logger.progress_display = False
            cfg.logger.show_chat = False
            cfg.logger.show_tools = False

        if cfg is not None:
            shell_cfg = getattr(cfg, "shell_execution", None)
            if shell_cfg is not None:
                shell_cfg.show_bash = False

        from fast_agent.ui.progress_display import progress_display

        progress_display.stop()

    def _validate_run_preconditions(self) -> None:
        """Validate the configured agents before creating instances."""
        if not self.agents:
            raise AgentConfigError("No agents defined. Please define at least one agent.")
        self._sync_agent_card_mcp_servers()
        validate_server_references(self.context, self.agents)
        validate_workflow_references(self.agents)
        self._handle_dump_requests()

    def _build_model_factory_func(
        self, cli_model_override: str | None
    ) -> ModelFactoryFunctionProtocol:
        """Build the model-factory closure used during agent instantiation."""

        def model_factory_func(model: Any = None, request_params: Any = None) -> Any:
            return get_model_factory(
                self.context,
                model=model,
                request_params=request_params,
                cli_model=cli_model_override,
            )

        return model_factory_func

    def _build_global_prompt_context(
        self, *, apply_global_prompt_context: bool
    ) -> dict[str, str] | None:
        """Build environment-derived prompt variables for non-ACP runs."""
        if not apply_global_prompt_context:
            return None

        context_variables: dict[str, str] = {}
        client_info: dict[str, str] = {"name": self.name}
        cli_name = getattr(self.args, "name", None)
        if cli_name:
            client_info["title"] = cli_name

        enrich_with_environment_context(
            context_variables,
            str(Path.cwd()),
            client_info,
            self._skills_directory_override,
        )
        return context_variables or None

    def _create_run_runtime(self, settings: RunSettings) -> RunRuntime:
        """Create the immutable/shared runtime resources for a run."""
        return RunRuntime(
            model_factory_func=self._build_model_factory_func(settings.cli_model_override),
            global_prompt_context=self._build_global_prompt_context(
                apply_global_prompt_context=not settings.is_acp_server_mode
            ),
            is_acp_server_mode=settings.is_acp_server_mode,
            noenv_mode=settings.noenv_mode,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
        )

    async def _instantiate_agent_instance(
        self,
        runtime: RunRuntime,
        app_override: AgentApp | None = None,
    ) -> AgentInstance:
        async with runtime.instance_lock:
            self.app._registered_tools = self._registered_tools  # type: ignore[attr-defined]
            agents_map = await create_agents_in_dependency_order(
                self.app,
                self.agents,
                runtime.model_factory_func,
            )
            if not runtime.is_acp_server_mode:
                validate_provider_keys_post_creation(agents_map)

            tool_only_agents = {
                name for name, data in self.agents.items() if data.get("tool_only", False)
            }
            if app_override is None:
                app = AgentApp(
                    agents_map,
                    tool_only_agents=tool_only_agents,
                    card_collision_warnings=self._card_collision_warnings,
                    noenv_mode=runtime.noenv_mode,
                )
            else:
                app_override.set_agents(
                    agents_map,
                    tool_only_agents=tool_only_agents,
                    card_collision_warnings=self._card_collision_warnings,
                )
                app_override.noenv_mode = runtime.noenv_mode
                app = app_override

            instance = AgentInstance(
                app,
                agents_map,
                registry_version=self._agent_registry_version,
            )
            runtime.managed_instances.append(instance)
            self._apply_agent_card_histories(instance.agents)
            if runtime.global_prompt_context:
                await self._apply_instruction_context(instance, runtime.global_prompt_context)
            return instance

    async def _dispose_agent_instance(self, runtime: RunRuntime, instance: AgentInstance) -> None:
        async with runtime.instance_lock:
            if instance in runtime.managed_instances:
                runtime.managed_instances.remove(instance)
        await instance.shutdown()

    async def _initialize_managed_run_state(self, runtime: RunRuntime) -> ManagedRunState:
        """Create the primary shared app instance for this run."""
        primary_instance = await self._instantiate_agent_instance(runtime)
        return ManagedRunState(
            runtime=runtime,
            primary_instance=primary_instance,
            wrapper=primary_instance.app,
            active_agents=primary_instance.agents,
        )

    def _expand_impacted_agents(
        self,
        impacted: set[str],
        removed_names: set[str],
    ) -> set[str]:
        if not impacted:
            return impacted

        from fast_agent.core.validation import get_agent_dependencies

        expanded = set(impacted)
        reverse_deps: dict[str, set[str]] = {}
        for name, agent_data in self.agents.items():
            for dep in get_agent_dependencies(agent_data):
                reverse_deps.setdefault(dep, set()).add(name)

        queue = list(expanded)
        while queue:
            current = queue.pop()
            for parent in reverse_deps.get(current, set()):
                if parent in removed_names or parent in expanded:
                    continue
                expanded.add(parent)
                queue.append(parent)
        return expanded

    async def _rebuild_impacted_agents(
        self,
        active_agents: dict[str, AgentProtocol],
        impacted: set[str],
        model_factory_func: ModelFactoryFunctionProtocol,
    ) -> None:
        if not impacted:
            return

        from fast_agent.core.direct_factory import active_agents_in_dependency_group
        from fast_agent.core.validation import get_dependencies_groups

        dependencies = get_dependencies_groups(self.agents)
        for group in dependencies:
            group_targets = [name for name in group if name in impacted]
            if not group_targets:
                continue
            await active_agents_in_dependency_group(
                self.app,
                self.agents,
                model_factory_func,
                group_targets,
                active_agents,
            )

    def _reload_updated_agent_file_histories(
        self,
        updated_agents: dict[str, AgentProtocol],
        *,
        allow_unchanged_empty: bool = False,
    ) -> None:
        for name, new_agent in updated_agents.items():
            history_files = self._agent_card_histories.get(name)
            if not history_files:
                continue

            files_mtime = self._get_history_files_mtime(history_files)
            if files_mtime is None:
                continue

            last_mtime = self._agent_card_history_mtime.get(name)
            last_len = self._agent_card_history_len.get(name)
            current_len = len(new_agent.message_history)
            if last_mtime is None:
                if current_len != 0:
                    continue
            elif allow_unchanged_empty and current_len == 0 and files_mtime == last_mtime:
                pass
            elif files_mtime <= last_mtime:
                continue
            elif last_len is not None and current_len != last_len:
                continue

            messages: list[PromptMessageExtended] = []
            for history_file in history_files:
                messages.extend(load_prompt(history_file))
            if not messages:
                continue

            new_agent.message_history.clear()
            new_agent.message_history.extend(messages)
            self._record_history_snapshot(name, len(new_agent.message_history), files_mtime)

    async def _finalize_updated_agents(
        self,
        updated_agents: dict[str, AgentProtocol],
        runtime: RunRuntime,
    ) -> None:
        if not updated_agents:
            return

        if not runtime.is_acp_server_mode:
            validate_provider_keys_post_creation(updated_agents)

        if runtime.global_prompt_context:
            await apply_instruction_context(updated_agents.values(), runtime.global_prompt_context)

    def _load_current_persisted_session(self) -> "Session | None":
        from fast_agent.session import get_session_manager

        manager = get_session_manager()
        current_session = manager.current_session
        if current_session is None:
            return None
        loaded_session = manager.load_session(current_session.info.name)
        return loaded_session or current_session

    def _log_local_hydration_messages(self, warnings: list[str]) -> None:
        for warning in warnings:
            logger.warning(
                "Shared runtime reload hydration warning",
                name="shared_reload_hydration_warning",
                warning=warning,
            )

    async def _hydrate_active_agents_from_session(
        self,
        agents: dict[str, AgentProtocol],
    ) -> "SessionHydrationResult | None":
        from fast_agent.session import SessionHydrationPolicy, SessionHydrator

        persisted_session = self._load_current_persisted_session()
        if persisted_session is None:
            return None

        fallback_agent_name = resolve_default_agent_name(
            agents,
            is_default=lambda _name, agent: agent_is_default(agent),
        )
        hydration = SessionHydrator().hydrate_session(
            session=persisted_session,
            agents=agents,
            fallback_agent_name=fallback_agent_name,
            policy=SessionHydrationPolicy.for_refresh(),
        )
        result = await hydration if inspect.isawaitable(hydration) else hydration
        warnings = [warning.message for warning in result.warnings]
        warnings.extend(result.usage_notices)
        self._log_local_hydration_messages(warnings)
        return result

    async def _refresh_result_from_session_restore(
        self,
        agents: dict[str, AgentProtocol],
        updated_agents: dict[str, AgentProtocol] | None = None,
    ) -> AgentRefreshResult:
        agents_to_hydrate = updated_agents if updated_agents is not None else agents
        hydration_result = self._hydrate_active_agents_from_session(agents_to_hydrate)
        hydration = await hydration_result if inspect.isawaitable(hydration_result) else hydration_result
        if hydration is None:
            if updated_agents:
                self._reload_updated_agent_file_histories(
                    updated_agents,
                    allow_unchanged_empty=True,
                )
            return AgentRefreshResult(changed=True)

        snapshot_agent_names = set(hydration.snapshot.continuation.agents)
        if updated_agents:
            unpersisted_agents = {
                name: agent
                for name, agent in updated_agents.items()
                if name not in snapshot_agent_names
            }
            if unpersisted_agents:
                self._reload_updated_agent_file_histories(
                    unpersisted_agents,
                    allow_unchanged_empty=True,
                )

        warnings = [warning.message for warning in hydration.warnings]
        warnings.extend(hydration.usage_notices)
        active_agent = hydration.active_agent
        if updated_agents is not None and active_agent not in updated_agents:
            active_agent = None
        return AgentRefreshResult(
            changed=True,
            active_agent=active_agent,
            warnings=warnings,
        )

    async def _refresh_shared_instance(self, state: ManagedRunState) -> AgentRefreshResult:
        if self._agent_registry_version <= state.primary_instance.registry_version:
            return AgentRefreshResult(changed=False)

        self._sync_agent_card_mcp_servers()
        changed_names = set(self._agent_card_last_changed)
        removed_names = set(self._agent_card_last_removed)
        dependent_names = set(self._agent_card_last_dependents)
        active_agents_local = state.active_agents

        if not (changed_names or removed_names or dependent_names):
            new_instance = await self._instantiate_agent_instance(
                state.runtime,
                app_override=state.wrapper,
            )
            refresh_result = await self._refresh_result_from_session_restore(new_instance.agents)
            old_instance = state.primary_instance
            state.primary_instance = new_instance
            state.active_agents = new_instance.agents
            await self._dispose_agent_instance(state.runtime, old_instance)
            return refresh_result

        async with state.runtime.instance_lock:
            impacted = set(changed_names)
            impacted.update(dependent_names)
            impacted.difference_update(removed_names)
            impacted = self._expand_impacted_agents(impacted, removed_names)

            removed_instances = [active_agents_local.pop(name, None) for name in removed_names]
            for agent in removed_instances:
                if agent is None:
                    continue
                await agent.shutdown()

            old_agents = {
                name: active_agents_local.get(name) for name in impacted if name in active_agents_local
            }

            await self._rebuild_impacted_agents(
                active_agents_local,
                impacted,
                state.runtime.model_factory_func,
            )

            for name, old_agent in old_agents.items():
                new_agent = active_agents_local.get(name)
                if old_agent is None or new_agent is None:
                    continue
                if old_agent is new_agent:
                    continue
                await old_agent.shutdown()

            if impacted:
                updated_agents = {
                    name: active_agents_local[name] for name in impacted if name in active_agents_local
                }
                await self._finalize_updated_agents(updated_agents, state.runtime)
                refresh_result = await self._refresh_result_from_session_restore(
                    active_agents_local,
                    updated_agents,
                )
            else:
                refresh_result = AgentRefreshResult(changed=True)

            state.primary_instance.registry_version = self._agent_registry_version
            state.active_agents = active_agents_local
            self._agent_card_last_changed.clear()
            self._agent_card_last_removed.clear()
            self._agent_card_last_dependents.clear()
            return refresh_result

    async def _reload_and_refresh(self, state: ManagedRunState) -> AgentRefreshResult:
        changed = await self.reload_agents()
        if not changed:
            return AgentRefreshResult(changed=False)
        return await self._refresh_shared_instance(state)

    async def _load_card_core(
        self,
        state: ManagedRunState,
        source: str,
        parent_name: str | None,
        *,
        should_refresh: bool,
    ) -> tuple[list[str], list[str]]:
        if source.startswith(("http://", "https://")):
            loaded_names = self.load_agents_from_url(source)
        else:
            loaded_names = self.load_agents(source)

        added_names: list[str] = []
        if parent_name:
            target_name = parent_name
            if target_name not in self.agents:
                target_name = next(iter(self.agents.keys()), None)
            if target_name and loaded_names:
                added_names = self.attach_agent_tools(target_name, loaded_names)

        if should_refresh:
            await self._refresh_shared_instance(state)
        return loaded_names, added_names

    async def _attach_agent_tools_and_refresh(
        self,
        state: ManagedRunState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> list[str]:
        added = self.attach_agent_tools(parent_name, child_names)
        if added:
            await self._refresh_shared_instance(state)
        return added

    async def _detach_agent_tools_and_refresh(
        self,
        state: ManagedRunState,
        parent_name: str,
        child_names: Sequence[str],
    ) -> list[str]:
        removed = self.detach_agent_tools(parent_name, child_names)
        if removed:
            await self._refresh_shared_instance(state)
        return removed

    def _resolve_runtime_mcp_agent(
        self,
        active_agents: dict[str, AgentProtocol],
        agent_name: str,
    ) -> McpAgentProtocol:
        from fast_agent.mcp.types import McpAgentProtocol

        target_agent = active_agents.get(agent_name)
        if target_agent is None:
            raise RuntimeError(f"Agent '{agent_name}' was not found")
        if not isinstance(target_agent, McpAgentProtocol):
            raise RuntimeError(f"Agent '{agent_name}' does not support MCP server management")
        return target_agent

    async def _attach_mcp_server_and_refresh(
        self,
        active_agents: dict[str, AgentProtocol],
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        from fast_agent.core.instruction_refresh import rebuild_agent_instruction

        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        result = await target_agent.attach_mcp_server(
            server_name=server_name,
            server_config=server_config,
            options=options,
        )
        await rebuild_agent_instruction(target_agent)
        return result

    async def _detach_mcp_server_and_refresh(
        self,
        active_agents: dict[str, AgentProtocol],
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        from fast_agent.core.instruction_refresh import rebuild_agent_instruction

        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        result = await target_agent.detach_mcp_server(server_name)
        await rebuild_agent_instruction(target_agent)
        return result

    async def _list_attached_mcp_servers(
        self,
        active_agents: dict[str, AgentProtocol],
        agent_name: str,
    ) -> list[str]:
        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        return target_agent.list_attached_mcp_servers()

    async def _list_configured_detached_mcp_servers(
        self,
        active_agents: dict[str, AgentProtocol],
        agent_name: str,
    ) -> list[str]:
        target_agent = self._resolve_runtime_mcp_agent(active_agents, agent_name)
        return target_agent.aggregator.list_configured_detached_servers()

    async def _dump_agent_card_callback(self, name: str) -> str:
        return self.dump_agent_card_text(name)

    def _build_runtime_callbacks(
        self,
        state: ManagedRunState,
        settings: RunSettings,
    ) -> RuntimeCallbacks:
        callbacks: RuntimeCallbacks

        async def create_instance() -> AgentInstance:
            instance = await self._instantiate_agent_instance(state.runtime)
            self._configure_runtime_mcp_callbacks(instance.app)
            return instance

        async def dispose_instance(instance: AgentInstance) -> None:
            await self._dispose_agent_instance(state.runtime, instance)

        async def refresh_shared_instance() -> AgentRefreshResult:
            return await self._refresh_shared_instance(state)

        async def reload_and_refresh() -> AgentRefreshResult:
            return await self._reload_and_refresh(state)

        async def load_card_and_refresh(
            source: str,
            parent_name: str | None,
        ) -> tuple[list[str], list[str]]:
            return await self._load_card_core(state, source, parent_name, should_refresh=True)

        async def load_card_source(
            source: str,
            parent_name: str | None,
        ) -> tuple[list[str], list[str]]:
            return await self._load_card_core(state, source, parent_name, should_refresh=False)

        async def attach_agent_tools_and_refresh(
            parent_name: str,
            child_names: Sequence[str],
        ) -> list[str]:
            return await self._attach_agent_tools_and_refresh(state, parent_name, child_names)

        async def detach_agent_tools_and_refresh(
            parent_name: str,
            child_names: Sequence[str],
        ) -> list[str]:
            return await self._detach_agent_tools_and_refresh(state, parent_name, child_names)

        async def attach_agent_tools_source(
            parent_name: str,
            child_names: Sequence[str],
        ) -> list[str]:
            return self.attach_agent_tools(parent_name, child_names)

        async def detach_agent_tools_source(
            parent_name: str,
            child_names: Sequence[str],
        ) -> list[str]:
            return self.detach_agent_tools(parent_name, child_names)

        async def attach_mcp_server(
            agent_name: str,
            server_name: str,
            server_config: MCPServerSettings | None = None,
            options: MCPAttachOptions | None = None,
        ) -> MCPAttachResult:
            return await self._attach_mcp_server_and_refresh(
                state.active_agents,
                agent_name,
                server_name,
                server_config,
                options,
            )

        async def detach_mcp_server(
            agent_name: str,
            server_name: str,
        ) -> MCPDetachResult:
            return await self._detach_mcp_server_and_refresh(
                state.active_agents, agent_name, server_name
            )

        async def list_attached_mcp_servers(agent_name: str) -> list[str]:
            return await self._list_attached_mcp_servers(state.active_agents, agent_name)

        async def list_configured_detached_mcp_servers(agent_name: str) -> list[str]:
            return await self._list_configured_detached_mcp_servers(
                state.active_agents,
                agent_name,
            )

        callbacks = RuntimeCallbacks(
            create_instance=create_instance,
            dispose_instance=dispose_instance,
            refresh_shared_instance=refresh_shared_instance,
            reload_and_refresh=reload_and_refresh,
            reload_source=self.reload_agents if settings.reload_enabled else None,
            load_card_and_refresh=load_card_and_refresh,
            load_card_source=load_card_source,
            attach_agent_tools_and_refresh=attach_agent_tools_and_refresh,
            detach_agent_tools_and_refresh=detach_agent_tools_and_refresh,
            attach_agent_tools_source=attach_agent_tools_source,
            detach_agent_tools_source=detach_agent_tools_source,
            attach_mcp_server=attach_mcp_server,
            detach_mcp_server=detach_mcp_server,
            list_attached_mcp_servers=list_attached_mcp_servers,
            list_configured_detached_mcp_servers=list_configured_detached_mcp_servers,
            dump_agent_card=self._dump_agent_card_callback,
        )
        return callbacks

    def _configure_runtime_mcp_callbacks(self, app: AgentApp) -> None:
        def active_agents() -> dict[str, AgentProtocol]:
            return cast("dict[str, AgentProtocol]", app.registered_agents())

        async def attach_mcp_server(
            agent_name: str,
            server_name: str,
            server_config: MCPServerSettings | None = None,
            options: MCPAttachOptions | None = None,
        ) -> MCPAttachResult:
            return await self._attach_mcp_server_and_refresh(
                active_agents(),
                agent_name,
                server_name,
                server_config,
                options,
            )

        async def detach_mcp_server(
            agent_name: str,
            server_name: str,
        ) -> MCPDetachResult:
            return await self._detach_mcp_server_and_refresh(
                active_agents(),
                agent_name,
                server_name,
            )

        async def list_attached_mcp_servers(agent_name: str) -> list[str]:
            return await self._list_attached_mcp_servers(active_agents(), agent_name)

        async def list_configured_detached_mcp_servers(agent_name: str) -> list[str]:
            return await self._list_configured_detached_mcp_servers(
                active_agents(),
                agent_name,
            )

        app.set_attach_mcp_server_callback(attach_mcp_server)
        app.set_detach_mcp_server_callback(detach_mcp_server)
        app.set_list_attached_mcp_servers_callback(list_attached_mcp_servers)
        app.set_list_configured_detached_mcp_servers_callback(
            list_configured_detached_mcp_servers
        )

    def _configure_wrapper_callbacks(
        self,
        state: ManagedRunState,
        callbacks: RuntimeCallbacks,
        settings: RunSettings,
    ) -> None:
        wrapper = state.wrapper
        wrapper.set_reload_callback(
            callbacks.reload_and_refresh if settings.reload_enabled else None
        )
        wrapper.set_refresh_callback(
            callbacks.refresh_shared_instance if settings.reload_enabled else None
        )
        wrapper.set_load_card_callback(callbacks.load_card_and_refresh)
        wrapper.set_attach_agent_tools_callback(callbacks.attach_agent_tools_and_refresh)
        wrapper.set_detach_agent_tools_callback(callbacks.detach_agent_tools_and_refresh)
        wrapper.set_dump_agent_callback(callbacks.dump_agent_card)
        self._configure_runtime_mcp_callbacks(wrapper)

        self._agent_card_watch_reload = (
            callbacks.reload_and_refresh if settings.reload_enabled else None
        )
        if getattr(self.args, "watch", False) and self._agent_card_roots:
            self._agent_card_watch_task = asyncio.create_task(self._watch_agent_cards())

        self._server_instance_factory = callbacks.create_instance
        self._server_instance_dispose = callbacks.dispose_instance
        self._server_managed_instances = state.runtime.managed_instances

    def _configure_streaming_for_run(self, active_agents: dict[str, AgentProtocol]) -> None:
        """Disable streaming when parallel agents are active."""
        from fast_agent.agents.agent_types import AgentType

        has_parallel = any(
            agent.agent_type == AgentType.PARALLEL for agent in active_agents.values()
        )
        if not has_parallel:
            return

        cfg = self.app.context.config
        if cfg is not None and cfg.logger is not None:
            cfg.logger.streaming = "none"

    async def _apply_card_tool_cli_option(
        self,
        state: ManagedRunState,
        refresh_callback: Callable[[], Awaitable[AgentRefreshResult]],
    ) -> None:
        card_tools = getattr(self.args, "card_tools", None)
        if not card_tools:
            return

        card_tool_agent_names: list[str] = []
        try:
            for card_source in card_tools:
                if card_source.startswith(("http://", "https://")):
                    names = self.load_agents_from_url(card_source)
                else:
                    names = self.load_agents(card_source)
                card_tool_agent_names.extend(names)
        except AgentConfigError as exc:
            self._handle_error(exc)
            raise SystemExit(1) from exc

        await refresh_callback()

        default_agent_name = getattr(self.args, "agent", None)
        if default_agent_name and default_agent_name not in state.active_agents:
            default_agent_name = None
        default_agent = state.wrapper._agent(default_agent_name)
        if default_agent is None:
            return

        add_tool_fn = getattr(default_agent, "add_agent_tool", None)
        if not callable(add_tool_fn):
            return

        tool_agents = [
            state.active_agents.get(tool_agent_name) for tool_agent_name in card_tool_agent_names
        ]
        add_tools_for_agents(add_tool_fn, tool_agents)

    def _print_server_startup(self, output_stream: Any) -> None:
        print(f"Starting fast-agent  '{self.name}' in server mode", file=output_stream)
        print(f"Transport: {self.args.transport}", file=output_stream)
        if self.args.transport == "http":
            print(f"Listening on {self.args.host}:{self.args.port}", file=output_stream)
        print("Press Ctrl+C to stop", file=output_stream)

    @staticmethod
    def _resolve_server_instance_scope(
        *,
        transport: str,
        instance_scope: str | None,
    ) -> str:
        if transport == "acp":
            if instance_scope is None:
                return "connection"
            if instance_scope != "connection":
                raise ValueError(
                    "ACP is always connection-scoped; instance_scope must be omitted or set to 'connection'."
                )
            return "connection"
        if instance_scope is None:
            return "shared"
        return instance_scope

    async def _run_acp_server(
        self,
        state: ManagedRunState,
        callbacks: RuntimeCallbacks,
    ) -> None:
        AgentACPServer = self._get_acp_server_class()

        server_name = getattr(self.args, "server_name", None)
        self._resolve_server_instance_scope(
            transport="acp",
            instance_scope=getattr(self.args, "instance_scope", None),
        )
        permissions_enabled = getattr(self.args, "permissions_enabled", True)

        acp_server = AgentACPServer(
            bootstrap_instance=state.primary_instance,
            create_instance=callbacks.create_instance,
            dispose_instance=callbacks.dispose_instance,
            server_name=server_name or f"{self.name}",
            skills_directory_override=self._skills_directory_override,
            permissions_enabled=permissions_enabled,
            load_card_callback=callbacks.load_card_source,
            attach_agent_tools_callback=callbacks.attach_agent_tools_source,
            detach_agent_tools_callback=callbacks.detach_agent_tools_source,
            dump_agent_card_callback=callbacks.dump_agent_card,
            reload_callback=callbacks.reload_source,
        )
        await acp_server.run_async()

    async def _run_mcp_server(
        self,
        state: ManagedRunState,
        callbacks: RuntimeCallbacks,
    ) -> None:
        from fast_agent.mcp.server import AgentMCPServer

        tool_description = getattr(self.args, "tool_description", None)
        tool_name_template = getattr(self.args, "tool_name_template", None)
        server_description = getattr(self.args, "server_description", None)
        server_name = getattr(self.args, "server_name", None)
        instance_scope = getattr(self.args, "instance_scope", "shared")
        mcp_server = AgentMCPServer(
            primary_instance=state.primary_instance,
            create_instance=callbacks.create_instance,
            dispose_instance=callbacks.dispose_instance,
            instance_scope=instance_scope,
            server_name=server_name or f"{self.name}-MCP-Server",
            server_description=server_description,
            tool_description=tool_description,
            tool_name_template=tool_name_template,
            host=self.args.host,
            get_registry_version=self._get_registry_version,
            reload_callback=callbacks.reload_source,
        )

        await mcp_server.run_async(
            transport=self.args.transport,
            host=self.args.host,
            port=self.args.port,
        )

    async def _handle_server_mode(
        self,
        state: ManagedRunState,
        callbacks: RuntimeCallbacks,
        settings: RunSettings,
    ) -> None:
        if not settings.server_mode:
            return

        is_stdio_transport = settings.transport in ["stdio", "acp"]
        configure_console_stream("stderr" if is_stdio_transport else "stdout")
        output_stream = sys.stderr if is_stdio_transport else sys.stdout

        try:
            if not settings.quiet_mode:
                self._print_server_startup(output_stream)

            if settings.transport == "acp":
                await self._run_acp_server(state, callbacks)
            else:
                await self._run_mcp_server(state, callbacks)
        except KeyboardInterrupt:
            if not settings.quiet_mode:
                print("\nServer stopped by user (Ctrl+C)", file=output_stream)
            raise SystemExit(0)
        except Exception as exc:
            if not settings.quiet_mode:
                import traceback

                traceback.print_exc()
            print(f"\nServer stopped with error: {exc}", file=output_stream)
            raise SystemExit(1) from exc

        raise SystemExit(0)

    def _get_selected_agent(
        self,
        wrapper: AgentApp,
        active_agents: dict[str, AgentProtocol],
        agent_name: str | None,
    ) -> AgentProtocol:
        if agent_name and agent_name not in active_agents:
            available_agents = ", ".join(active_agents.keys())
            print(f"\n\nError: Agent '{agent_name}' not found. Available agents: {available_agents}")
            raise SystemExit(1)
        return wrapper._agent(agent_name)

    async def _handle_message_mode(
        self,
        state: ManagedRunState,
        settings: RunSettings,
    ) -> None:
        message = getattr(self.args, "message", None)
        if not message:
            return

        agent_name = getattr(self.args, "agent", None)
        try:
            agent = self._get_selected_agent(state.wrapper, state.active_agents, agent_name)
            response = await agent.send(message)
            if settings.quiet_mode:
                print(f"{response}")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as exc:
            display_agent = agent_name or "<default>"
            print(f"\n\nError sending message to agent '{display_agent}': {str(exc)}")
            raise SystemExit(1) from exc

    async def _handle_prompt_file_mode(
        self,
        state: ManagedRunState,
        settings: RunSettings,
    ) -> None:
        prompt_file = getattr(self.args, "prompt_file", None)
        if not prompt_file:
            return

        agent_name = getattr(self.args, "agent", None)
        prompt: list[PromptMessageExtended] = load_prompt(Path(prompt_file))
        try:
            agent = self._get_selected_agent(state.wrapper, state.active_agents, agent_name)
            prompt_result = await agent.generate(prompt)
            if settings.quiet_mode:
                print(f"{prompt_result.last_text()}")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as exc:
            display_agent = agent_name or "<default>"
            print(f"\n\nError sending message to agent '{display_agent}': {str(exc)}")
            raise SystemExit(1) from exc

    async def _stop_watch_task(self) -> None:
        if self._agent_card_watch_task is None:
            return
        self._agent_card_watch_task.cancel()
        try:
            await self._agent_card_watch_task
        except asyncio.CancelledError:
            pass
        self._agent_card_watch_task = None
        self._agent_card_watch_reload = None

    def _print_usage_summary_for_run(
        self,
        state: ManagedRunState | None,
        active_agents: dict[str, AgentProtocol],
        *,
        had_error: bool,
        settings: RunSettings,
    ) -> None:
        if had_error or settings.quiet_mode:
            return

        managed_instances = state.runtime.managed_instances if state is not None else []
        if managed_instances and not settings.server_mode:
            self._print_usage_report(managed_instances[0].agents)
            return
        if active_agents:
            self._print_usage_report(active_agents)

    async def _dispose_managed_instances(
        self,
        active_agents: dict[str, AgentProtocol],
    ) -> None:
        if self._server_managed_instances and self._server_instance_dispose is not None:
            remaining_instances = list(self._server_managed_instances)
            for instance in remaining_instances:
                try:
                    await self._server_instance_dispose(instance)
                except Exception:
                    pass
            self._server_managed_instances.clear()
            return

        for agent in active_agents.values():
            try:
                await agent.shutdown()
            except Exception:
                pass

    async def _finalize_run(
        self,
        state: ManagedRunState | None,
        active_agents: dict[str, AgentProtocol],
        *,
        had_error: bool,
        settings: RunSettings,
        shutdown_timeout: float | None = None,
    ) -> None:
        try:
            from fast_agent.ui.progress_display import progress_display

            progress_display.stop()
        except Exception:
            pass

        await self._stop_watch_task()
        self._print_usage_summary_for_run(
            state,
            active_agents,
            had_error=had_error,
            settings=settings,
        )
        if shutdown_timeout is None:
            await self._dispose_managed_instances(active_agents)
            return

        try:
            await asyncio.wait_for(
                self._dispose_managed_instances(active_agents),
                timeout=shutdown_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out while shutting down agents after exit request",
                timeout_seconds=shutdown_timeout,
            )

    @asynccontextmanager
    async def run(self) -> AsyncIterator["AgentApp"]:
        """
        Context manager for running the application.
        Initializes all registered agents.
        """
        active_agents: dict[str, AgentProtocol] = {}
        had_error = False
        run_state: ManagedRunState | None = None
        shutdown_timeout: float | None = None
        await self.app.initialize()
        settings = self._prepare_run_settings()

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(self.name):
            try:
                async with self.app.run():
                    default_skills = self._load_default_skills_for_run()
                    self._apply_skills_to_agent_configs(default_skills)

                    if settings.quiet_mode:
                        self._configure_quiet_mode_for_run()

                    self._validate_run_preconditions()
                    runtime = self._create_run_runtime(settings)
                    run_state = await self._initialize_managed_run_state(runtime)
                    active_agents = run_state.active_agents

                    callbacks = self._build_runtime_callbacks(run_state, settings)
                    self._configure_wrapper_callbacks(run_state, callbacks, settings)
                    self._configure_streaming_for_run(run_state.active_agents)
                    await self._apply_card_tool_cli_option(
                        run_state,
                        callbacks.refresh_shared_instance,
                    )
                    await self._handle_server_mode(run_state, callbacks, settings)
                    await self._handle_message_mode(run_state, settings)
                    await self._handle_prompt_file_mode(run_state, settings)

                    yield run_state.wrapper

            except PromptExitError as e:
                # User requested exit - not an error, show usage report
                shutdown_timeout = _PROMPT_EXIT_SHUTDOWN_TIMEOUT_SECONDS
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
                await self._finalize_run(
                    run_state,
                    active_agents,
                    had_error=had_error,
                    settings=settings,
                    shutdown_timeout=shutdown_timeout,
                )

    async def _apply_instruction_context(
        self, instance: AgentInstance, context_vars: dict[str, str]
    ) -> None:
        """Resolve late-binding placeholders for all agents in the provided instance."""
        await apply_instruction_context(instance.agents.values(), context_vars)

    @staticmethod
    def _get_history_files_mtime(history_files: Sequence[Path]) -> float | None:
        mtimes: list[float] = []
        for history_file in history_files:
            try:
                mtimes.append(history_file.stat().st_mtime)
            except OSError:
                continue
        return max(mtimes) if mtimes else None

    def _record_history_snapshot(self, name: str, history_len: int, mtime: float | None) -> None:
        self._agent_card_history_len[name] = history_len
        if mtime is not None:
            self._agent_card_history_mtime[name] = mtime

    def _apply_agent_card_histories(self, agents: dict[str, "AgentProtocol"]) -> None:
        if not self._agent_card_histories:
            return
        for name, history_files in self._agent_card_histories.items():
            agent = agents.get(name)
            if agent is None:
                continue
            messages: list[PromptMessageExtended] = []
            for history_file in history_files:
                messages.extend(load_prompt(history_file))
            agent.clear(clear_prompts=True)
            agent.message_history.extend(messages)
            mtime = self._get_history_files_mtime(history_files)
            self._record_history_snapshot(name, len(messages), mtime)

    def _handle_dump_requests(self) -> None:
        dump_dir = getattr(self.args, "dump_agents", None)
        dump_dir_yaml = getattr(self.args, "dump_agents_yaml", None)
        dump_agent = getattr(self.args, "dump_agent", None)
        dump_agent_path = getattr(self.args, "dump_agent_path", None)
        dump_agent_yaml = getattr(self.args, "dump_agent_yaml", False)

        if dump_dir and dump_dir_yaml:
            raise AgentConfigError("Only one of --dump or --dump-yaml may be set")

        if dump_agent and dump_agent_path is None:
            raise AgentConfigError("--dump-agent-path is required with --dump-agent")
        if dump_agent_path is not None and not dump_agent:
            raise AgentConfigError("--dump-agent is required with --dump-agent-path")

        if dump_agent and (dump_dir or dump_dir_yaml):
            raise AgentConfigError("Use either --dump-agent or --dump/--dump-yaml, not both")

        if not (dump_dir or dump_dir_yaml or dump_agent):
            return

        if dump_dir or dump_dir_yaml:
            output_dir_raw = dump_dir if dump_dir is not None else dump_dir_yaml
            if output_dir_raw is None:
                raise AgentConfigError("Missing output directory for agent dump")
            output_dir = Path(output_dir_raw)
            self._dump_agents_to_dir(output_dir, as_yaml=bool(dump_dir_yaml))
            raise SystemExit(0)

        if dump_agent:
            if dump_agent_path is None:
                raise AgentConfigError("--dump-agent-path is required with --dump-agent")
            output_path = Path(dump_agent_path)
            self._dump_single_agent(dump_agent, output_path, as_yaml=dump_agent_yaml)
            raise SystemExit(0)

    def _dump_agents_to_dir(self, output_dir: Path, *, as_yaml: bool) -> None:
        from fast_agent.core.agent_card_loader import dump_agents_to_dir

        dump_agents_to_dir(
            self.agents,
            output_dir,
            as_yaml=as_yaml,
            message_map=self._agent_card_histories,
        )

    def _dump_single_agent(self, name: str, output_path: Path, *, as_yaml: bool) -> None:
        from fast_agent.core.agent_card_loader import dump_agent_to_path

        if name not in self.agents:
            raise AgentConfigError(
                f"Agent '{name}' not found for dump",
                f"Available agents: {', '.join(self.agents.keys())}",
            )
        message_paths = self._agent_card_histories.get(name)
        dump_agent_to_path(
            name,
            self.agents[name],
            output_path,
            as_yaml=as_yaml,
            message_paths=message_paths,
        )

    def _apply_skills_to_agent_configs(self, default_skills: list[SkillManifest]) -> None:
        self._default_skill_manifests = list(default_skills)

        for agent_data in self.agents.values():
            config_obj = agent_data.get("config")
            if not config_obj:
                continue

            if config_obj.skills is SKILLS_DEFAULT:
                resolved = list(default_skills)
            elif config_obj.skills is None:
                resolved = []
            else:
                resolved = self._resolve_skills(config_obj.skills)
                resolved = self._deduplicate_skills(resolved)

            config_obj.skill_manifests = resolved

    def _resolve_skills(
        self,
        entry: SkillConfig,
    ) -> list[SkillManifest]:
        if entry is SKILLS_DEFAULT:
            return []
        if entry is None:
            return []
        if isinstance(entry, list):
            filtered: list[SkillEntry] = []
            for item in entry:
                if isinstance(item, (SkillManifest, SkillRegistry, Path, str)):
                    filtered.append(item)
                elif item is not None:
                    logger.debug(
                        "Unsupported skill entry type",
                        data={"type": type(item).__name__},
                    )
            if not filtered:
                return []
            directory_entries = [item for item in filtered if isinstance(item, (Path, str))]
            if len(directory_entries) == len(filtered):
                directories: list[Path | str] = []
                for item in directory_entries:
                    directories.append(Path(item) if isinstance(item, str) else item)
                registry = SkillRegistry(base_dir=Path.cwd(), directories=directories)
                return registry.load_manifests()
            manifests: list[SkillManifest] = []
            for item in filtered:
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
        if isinstance(entry, (Path, str)):
            # Use instance method to preserve original path for relative path computation
            path = Path(entry) if isinstance(entry, str) else entry
            registry = SkillRegistry(base_dir=Path.cwd(), directories=[path])
            return registry.load_manifests()

        logger.debug(
            "Unsupported skill entry type",
            data={"type": type(entry).__name__},
        )
        return []

    @staticmethod
    def _deduplicate_skills(manifests: list[SkillManifest]) -> list[SkillManifest]:
        unique: dict[str, SkillManifest] = {}
        for manifest in manifests:
            key = manifest.name.lower()
            if key not in unique:
                unique[key] = manifest
        return list(unique.values())

    def _handle_error(self, e: Exception, error_type: str | None = None) -> None:
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
                "Common models: gpt-5.1, kimi, sonnet, haiku. Set reasoning effort on supported models with gpt-5-mini?reasoning=high",
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
        server_name: str | None = None,
        server_description: str | None = None,
        tool_description: str | None = None,
        instance_scope: str | None = None,
        permissions_enabled: bool = True,
        tool_name_template: str | None = None,
    ) -> None:
        """
        Start the application as an MCP server.
        This method initializes agents and exposes them through an MCP server.
        It is a blocking method that runs until the server is stopped.

        Args:
            transport: Transport protocol to use ("http" or "stdio")
            host: Host address for the server when using HTTP
            port: Port for the server when using HTTP
            server_name: Optional custom name for the MCP server
            server_description: Optional description/instructions for the MCP server
            tool_description: Optional description template for the exposed send tool.
                              Use {agent} to reference the agent name.
            permissions_enabled: Whether to request tool permissions from connected clients (default: True)
            tool_name_template: Optional template for exposed agent tool names.
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
        self.args.tool_name_template = tool_name_template
        self.args.server_description = server_description
        self.args.server_name = server_name
        self.args.instance_scope = self._resolve_server_instance_scope(
            transport=transport,
            instance_scope=instance_scope,
        )
        self.args.permissions_enabled = permissions_enabled
        # Force quiet mode for stdio/acp transports to avoid polluting the protocol stream
        self.args.quiet = (
            original_args.quiet if original_args and hasattr(original_args, "quiet") else False
        )
        if transport in ["stdio", "acp"]:
            self.args.quiet = True
        self.args.model = None
        if original_args is not None and hasattr(original_args, "model"):
            self.args.model = original_args.model
        if original_args is not None and hasattr(original_args, "agent"):
            self.args.agent = original_args.agent
        if original_args is not None and hasattr(original_args, "reload"):
            self.args.reload = original_args.reload
        if original_args is not None and hasattr(original_args, "watch"):
            self.args.watch = original_args.watch
        if original_args is not None and hasattr(original_args, "card_tools"):
            self.args.card_tools = original_args.card_tools

        # Run the application, which will detect the server flag and start server mode
        async with self.run():
            pass  # This won't be reached due to SystemExit in run()

        # Restore original args (if we get here)
        if original_args:
            self.args = original_args

    async def run_with_mcp_server(
        self,
        transport: str = "http",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: str | None = None,
        server_description: str | None = None,
        tool_description: str | None = None,
        instance_scope: str | None = None,
        tool_name_template: str | None = None,
    ) -> None:
        """
        Run the application and expose agents through an MCP server.

        Args:
            transport: Transport protocol to use ("http" or "stdio")
            host: Host address for the server when using HTTP
            port: Port for the server when using HTTP
            server_name: Optional custom name for the MCP server
            server_description: Optional description/instructions for the MCP server
            tool_description: Optional description template for the exposed send tool.
            tool_name_template: Optional template for exposed agent tool names.
                                Use {agent} to reference the agent name.
        """
        await self.start_server(
            transport=transport,
            host=host,
            port=port,
            server_name=server_name,
            server_description=server_description,
            tool_description=tool_description,
            tool_name_template=tool_name_template,
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
    agents: dict[str, "AgentProtocol"]
    registry_version: int = 0

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
