"""
Direct FastAgent implementation that uses the simplified Agent architecture.
This replaces the traditional FastAgent with a more streamlined approach that
directly creates Agent instances without proxies.
"""

from __future__ import annotations

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
    Literal,
    ParamSpec,
    Sequence,
    TypeAlias,
    TypeVar,
)

import yaml
import yaml.parser
from opentelemetry import trace

from fast_agent import config
from fast_agent.core import Core
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.agent_tools import add_tools_for_agents
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
    maker as maker_decorator,
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
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import (
    apply_template_variables,
    enrich_with_environment_context,
)
from fast_agent.core.validation import (
    validate_provider_keys_post_creation,
    validate_server_references,
    validate_workflow_references,
)
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.skills import SKILLS_DEFAULT, SkillManifest, SkillRegistry, SkillsDefault
from fast_agent.ui.console import configure_console_stream
from fast_agent.ui.usage_display import display_usage_report

if TYPE_CHECKING:
    from mcp.client.session import ElicitationFnT
    from pydantic import AnyUrl

    from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION
    from fast_agent.context import Context
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.types import PromptMessageExtended

F = TypeVar("F", bound=Callable[..., Any])  # For decorated functions
logger = get_logger(__name__)
SkillEntry: TypeAlias = SkillManifest | SkillRegistry | Path | str
SkillConfig: TypeAlias = SkillEntry | list[SkillEntry | None] | None | SkillsDefault


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
                choices=["sse", "http", "stdio", "acp"],
                default=None,
                help="Transport protocol to use when running as a server (sse, http, stdio, or acp)",
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

            # Propagate CLI skills override into config so resolve_skill_directories() works everywhere
            if self._skills_directory_override is not None and hasattr(self, "config"):
                if "skills" not in self.config:
                    self.config["skills"] = {}
                self.config["skills"]["directories"] = [str(p) for p in self._skills_directory_override]

            # Create settings and update global settings so resolve_skill_directories() works
            instance_settings = config.Settings(**self.config) if hasattr(self, "config") else None
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
        self.agents: dict[str, dict[str, Any]] = {}
        # Tracking for AgentCard-loaded agents
        self._agent_card_sources: dict[str, Path] = {}
        self._agent_card_roots: dict[Path, set[str]] = {}
        self._agent_card_root_files: dict[Path, set[Path]] = {}
        self._agent_card_root_watch_files: dict[Path, set[Path]] = {}
        self._agent_card_file_cache: dict[Path, tuple[int, int]] = {}
        self._agent_card_name_by_path: dict[Path, str] = {}
        self._agent_card_histories: dict[str, list[Path]] = {}
        self._agent_card_tool_files: dict[Path, set[Path]] = {}
        self._agent_card_last_changed: set[str] = set()
        self._agent_card_last_removed: set[str] = set()
        self._agent_card_last_dependents: set[str] = set()
        self._agent_registry_version: int = 0
        self._agent_card_watch_task: asyncio.Task[None] | None = None
        self._agent_card_reload_lock: asyncio.Lock | None = None
        self._agent_card_watch_reload: Callable[[], Awaitable[bool]] | None = None
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

        if parent_data.get("type") not in ("basic", "custom"):
            raise AgentConfigError(
                f"Agent '{parent_name}' does not support agents-as-tools"
            )

        missing = [
            name
            for name in child_names
            if name and name != parent_name and name not in self.agents
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

        if parent_data.get("type") not in ("basic", "custom"):
            raise AgentConfigError(
                f"Agent '{parent_name}' does not support agents-as-tools"
            )

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

    def dump_agent_card_text(self, name: str, *, as_yaml: bool = False) -> str:
        """Render an AgentCard as text."""
        from fast_agent.core.agent_card_loader import dump_agent_to_string

        agent_data = self.agents.get(name)
        if not agent_data:
            raise AgentConfigError(f"Agent '{name}' not found for dump")

        message_paths = self._agent_card_histories.get(name)
        return dump_agent_to_string(
            name, agent_data, as_yaml=as_yaml, message_paths=message_paths
        )

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

    def _load_agent_cards_from_root(self, root: Path, *, incremental: bool) -> bool:
        from fast_agent.core.agent_card_loader import load_agent_cards

        if not root.exists():
            if incremental:
                current_card_files: set[Path] = set()
            else:
                raise AgentConfigError(f"AgentCard path not found: {root}")
        else:
            current_card_files = self._collect_agent_card_files(root)

        previous_card_files = self._agent_card_root_files.get(root, set())
        removed_card_files = previous_card_files - current_card_files
        for removed_path in removed_card_files:
            self._agent_card_tool_files.pop(removed_path, None)

        current_card_stats: dict[Path, tuple[int, int]] = {}
        for path_entry in list(current_card_files):
            try:
                stat = path_entry.stat()
            except FileNotFoundError:
                current_card_files.discard(path_entry)
                continue
            current_card_stats[path_entry] = (stat.st_mtime_ns, stat.st_size)

        if incremental:
            changed_card_files = {
                path_entry
                for path_entry, signature in current_card_stats.items()
                if self._agent_card_file_cache.get(path_entry) != signature
            }
        else:
            changed_card_files = set(current_card_stats.keys())

        def _load_cards(path_entry: Path) -> list[Any]:
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

        cards: list[Any] = []
        loaded_card_files: set[Path] = set()
        for path_entry in sorted(changed_card_files):
            loaded_cards = _load_cards(path_entry)
            cards.extend(loaded_cards)
            loaded_card_files.add(path_entry)
            for card in loaded_cards:
                config = card.agent_data.get("config")
                function_tools = getattr(config, "function_tools", None)
                self._agent_card_tool_files[card.path] = self._resolve_function_tool_paths(
                    card.path, function_tools
                )

        missing_tool_cards = {
            path_entry
            for path_entry in current_card_files
            if path_entry not in self._agent_card_tool_files
        }
        for path_entry in sorted(missing_tool_cards):
            loaded_cards = _load_cards(path_entry)
            cards.extend(loaded_cards)
            loaded_card_files.add(path_entry)
            for card in loaded_cards:
                config = card.agent_data.get("config")
                function_tools = getattr(config, "function_tools", None)
                self._agent_card_tool_files[card.path] = self._resolve_function_tool_paths(
                    card.path, function_tools
                )

        current_tool_files: set[Path] = set()
        for card_path in current_card_files:
            current_tool_files.update(self._agent_card_tool_files.get(card_path, set()))

        watch_files = set(current_card_files) | current_tool_files
        previous_watch_files = self._agent_card_root_watch_files.get(root, set())
        removed_watch_files = previous_watch_files - watch_files

        current_stats: dict[Path, tuple[int, int]] = {}
        for path_entry in list(watch_files):
            try:
                stat = path_entry.stat()
            except FileNotFoundError:
                watch_files.discard(path_entry)
                continue
            current_stats[path_entry] = (stat.st_mtime_ns, stat.st_size)

        if incremental:
            changed_watch_files = {
                path_entry
                for path_entry, signature in current_stats.items()
                if self._agent_card_file_cache.get(path_entry) != signature
            }
        else:
            changed_watch_files = set(current_stats.keys())

        changed_tool_files = {
            path_entry for path_entry in changed_watch_files if path_entry in current_tool_files
        }
        removed_tool_files = {
            path_entry for path_entry in removed_watch_files if path_entry not in current_card_files
        }
        changed_tool_files |= removed_tool_files

        if changed_tool_files:
            affected_card_files = {
                card_path
                for card_path in current_card_files
                if self._agent_card_tool_files.get(card_path, set()) & changed_tool_files
            }
            for path_entry in sorted(affected_card_files - loaded_card_files):
                loaded_cards = _load_cards(path_entry)
                cards.extend(loaded_cards)
                loaded_card_files.add(path_entry)
                for card in loaded_cards:
                    config = card.agent_data.get("config")
                    function_tools = getattr(config, "function_tools", None)
                    self._agent_card_tool_files[card.path] = self._resolve_function_tool_paths(
                        card.path, function_tools
                    )

        self._apply_agent_card_updates(
            root,
            current_files=current_card_files,
            removed_files=removed_card_files,
            changed_cards=cards,
            current_stats=current_stats,
        )

        for path_entry in removed_watch_files:
            self._agent_card_file_cache.pop(path_entry, None)

        self._agent_card_root_watch_files[root] = set(watch_files)

        return bool(removed_card_files or changed_watch_files)

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
                and (
                    entry.suffix.lower() not in {".md", ".markdown"}
                    or _has_frontmatter(entry)
                )
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
        function_tools: Sequence[str] | None,
    ) -> set[Path]:
        tool_paths: set[Path] = set()
        if not function_tools:
            return tool_paths
        for spec in function_tools:
            if not isinstance(spec, str) or ":" not in spec:
                continue
            module_path_str, _func_name = spec.rsplit(":", 1)
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
        changed_cards: list[Any],
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
            if not hasattr(card, "name") or not hasattr(card, "path"):
                raise AgentConfigError("Invalid AgentCard payload during reload")
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

    def _get_registry_version(self) -> int:
        return self._agent_registry_version

    async def _watch_agent_cards(self) -> None:
        roots = sorted(self._agent_card_roots.keys())
        if not roots:
            return

        try:
            from watchfiles import awatch  # type: ignore[import-not-found]

            async for _changes in awatch(*roots):
                await self._reload_agent_cards_from_watch()
        except ImportError:
            logger.info(
                "watchfiles not available; falling back to polling for AgentCard reloads"
            )
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
        return await reload_callback()

    # Decorator methods with precise signatures for IDE completion
    # Provide annotations so IDEs can discover these attributes on instances
    if TYPE_CHECKING:  # pragma: no cover - typing aid only
        from collections.abc import Coroutine
        from pathlib import Path

        from fast_agent.agents.agent_types import FunctionToolsConfig
        from fast_agent.skills import SkillManifest, SkillRegistry
        from fast_agent.types import RequestParams

        P = ParamSpec("P")
        R = TypeVar("R")

        def agent(
            self,
            name: str = "default",
            instruction_or_kwarg: str | Path | AnyUrl | None = None,
            *,
            instruction: str | Path | AnyUrl = DEFAULT_AGENT_INSTRUCTION,
            agents: list[str] | None = None,
            servers: list[str] = [],
            tools: dict[str, list[str]] | None = None,
            resources: dict[str, list[str]] | None = None,
            prompts: dict[str, list[str]] | None = None,
            skills: SkillConfig = SKILLS_DEFAULT,
            function_tools: FunctionToolsConfig = None,
            model: str | None = None,
            use_history: bool = True,
            request_params: RequestParams | None = None,
            human_input: bool = False,
            default: bool = False,
            elicitation_handler: ElicitationFnT | None = None,
            api_key: str | None = None,
            history_source: Any | None = None,
            history_merge_target: Any | None = None,
            max_parallel: int | None = None,
            child_timeout_sec: int | None = None,
            max_display_instances: int | None = None,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def custom(
            self,
            cls,
            name: str = "default",
            instruction_or_kwarg: str | Path | AnyUrl | None = None,
            *,
            instruction: str | Path | AnyUrl = "You are a helpful agent.",
            servers: list[str] = [],
            tools: dict[str, list[str]] | None = None,
            resources: dict[str, list[str]] | None = None,
            prompts: dict[str, list[str]] | None = None,
            skills: SkillConfig = SKILLS_DEFAULT,
            model: str | None = None,
            use_history: bool = True,
            request_params: RequestParams | None = None,
            human_input: bool = False,
            default: bool = False,
            elicitation_handler: ElicitationFnT | None = None,
            api_key: str | None = None,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def orchestrator(
            self,
            name: str,
            *,
            agents: list[str],
            instruction: str
            | Path
            | AnyUrl = "You are an expert planner. Given an objective task and a list of Agents\n(which are collections of capabilities), your job is to break down the objective\ninto a series of steps, which can be performed by these agents.\n",
            model: str | None = None,
            request_params: RequestParams | None = None,
            use_history: bool = False,
            human_input: bool = False,
            plan_type: Literal["full", "iterative"] = "full",
            plan_iterations: int = 5,
            default: bool = False,
            api_key: str | None = None,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def iterative_planner(
            self,
            name: str,
            *,
            agents: list[str],
            instruction: str | Path | AnyUrl = "You are an expert planner. Plan iteratively.",
            model: str | None = None,
            request_params: RequestParams | None = None,
            plan_iterations: int = -1,
            default: bool = False,
            api_key: str | None = None,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def router(
            self,
            name: str,
            *,
            agents: list[str],
            instruction: str | Path | AnyUrl | None = None,
            servers: list[str] = [],
            tools: dict[str, list[str]] | None = None,
            resources: dict[str, list[str]] | None = None,
            prompts: dict[str, list[str]] | None = None,
            model: str | None = None,
            use_history: bool = False,
            request_params: RequestParams | None = None,
            human_input: bool = False,
            default: bool = False,
            elicitation_handler: ElicitationFnT | None = None,
            api_key: str | None = None,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def chain(
            self,
            name: str,
            *,
            sequence: list[str],
            instruction: str | Path | AnyUrl | None = None,
            cumulative: bool = False,
            default: bool = False,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def parallel(
            self,
            name: str,
            *,
            fan_out: list[str],
            fan_in: str | None = None,
            instruction: str | Path | AnyUrl | None = None,
            include_request: bool = True,
            default: bool = False,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def evaluator_optimizer(
            self,
            name: str,
            *,
            generator: str,
            evaluator: str,
            instruction: str | Path | AnyUrl | None = None,
            min_rating: str = "GOOD",
            max_refinements: int = 3,
            refinement_instruction: str | None = None,
            default: bool = False,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

        def maker(
            self,
            name: str,
            *,
            worker: str,
            k: int = 3,
            max_samples: int = 50,
            match_strategy: str = "exact",
            red_flag_max_length: int | None = None,
            instruction: str | Path | AnyUrl | None = None,
            default: bool = False,
        ) -> Callable[
            [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
        ]: ...

    # Runtime bindings (actual implementations)
    if not TYPE_CHECKING:
        agent = agent_decorator
        custom = custom_decorator
        orchestrator = orchestrator_decorator
        iterative_planner = orchestrator2_decorator
        router = router_decorator
        chain = chain_decorator
        parallel = parallel_decorator
        evaluator_optimizer = evaluator_optimizer_decorator
        maker = maker_decorator

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
        active_agents: dict[str, AgentProtocol] = {}
        had_error = False
        await self.app.initialize()

        # Handle quiet mode and CLI model override safely
        # Define these *before* they are used, checking if self.args exists and has the attributes
        # Force quiet mode for stdio/acp transports to avoid polluting the protocol stream
        quiet_mode = getattr(self.args, "quiet", False)
        if getattr(self.args, "transport", None) in ["stdio", "acp"] and getattr(
            self.args, "server", False
        ):
            quiet_mode = True
        cli_model_override = getattr(self.args, "model", None)

        # Store the model source for UI display
        config = self.context.config
        model_source = get_default_model_source(
            config_default_model=config.default_model if config else None,
            cli_model=cli_model_override,
        )
        if config:
            config.model_source = model_source  # type: ignore[attr-defined]
            config.cli_model_override = cli_model_override  # type: ignore[attr-defined]

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(self.name):
            try:
                async with self.app.run():
                    registry = getattr(self.context, "skill_registry", None)
                    if self._skills_directory_override is not None:
                        override_registry = SkillRegistry(
                            base_dir=Path.cwd(),
                            directories=self._skills_directory_override,
                        )
                        self.context.skill_registry = override_registry
                        registry = override_registry

                    default_skills: list[SkillManifest] = []
                    if registry:
                        try:
                            default_skills = registry.load_manifests()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Failed to load skills; continuing without them",
                                data={"error": str(exc)},
                            )

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
                    self._handle_dump_requests()

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

                    # Determine whether to apply global environment template variables.
                    apply_global_prompt_context = not (
                        getattr(self.args, "server", False)
                        and getattr(self.args, "transport", None) == "acp"
                    )
                    global_prompt_context: dict[str, str] | None = None
                    if apply_global_prompt_context:
                        context_variables: dict[str, str] = {}
                        client_info: dict[str, str] = {"name": self.name}
                        cli_name = getattr(self.args, "name", None)
                        if cli_name:
                            client_info["title"] = cli_name

                        # Pass skills directory override if configured
                        skills_override = self._skills_directory_override

                        enrich_with_environment_context(
                            context_variables, str(Path.cwd()), client_info, skills_override
                        )
                        if context_variables:
                            global_prompt_context = context_variables

                    async def instantiate_agent_instance(
                        app_override: AgentApp | None = None,
                    ) -> AgentInstance:
                        async with instance_lock:
                            agents_map = await create_agents_in_dependency_order(
                                self.app,
                                self.agents,
                                model_factory_func,
                            )
                            validate_provider_keys_post_creation(agents_map)
                            # Collect tool_only agent names
                            tool_only_agents = {
                                name
                                for name, data in self.agents.items()
                                if data.get("tool_only", False)
                            }
                            if app_override is None:
                                app = AgentApp(
                                    agents_map,
                                    tool_only_agents=tool_only_agents,
                                    card_collision_warnings=self._card_collision_warnings,
                                )
                            else:
                                app_override.set_agents(
                                    agents_map,
                                    tool_only_agents=tool_only_agents,
                                    card_collision_warnings=self._card_collision_warnings,
                                )
                                app = app_override
                            instance = AgentInstance(
                                app,
                                agents_map,
                                registry_version=self._agent_registry_version,
                            )
                            managed_instances.append(instance)
                            self._apply_agent_card_histories(instance.agents)
                            if global_prompt_context:
                                self._apply_instruction_context(instance, global_prompt_context)
                            return instance

                    async def dispose_agent_instance(instance: AgentInstance) -> None:
                        async with instance_lock:
                            if instance in managed_instances:
                                managed_instances.remove(instance)
                        await instance.shutdown()

                    primary_instance = await instantiate_agent_instance()
                    wrapper = primary_instance.app
                    active_agents = primary_instance.agents

                    async def refresh_shared_instance() -> bool:
                        nonlocal primary_instance, active_agents
                        if self._agent_registry_version <= primary_instance.registry_version:
                            return False
                        changed_names = set(self._agent_card_last_changed)
                        removed_names = set(self._agent_card_last_removed)
                        dependent_names = set(self._agent_card_last_dependents)
                        active_agents_local = active_agents

                        if not (changed_names or removed_names or dependent_names):
                            new_instance = await instantiate_agent_instance(app_override=wrapper)
                            old_instance = primary_instance
                            primary_instance = new_instance
                            active_agents = new_instance.agents
                            await dispose_agent_instance(old_instance)
                            return True

                        async with instance_lock:
                            impacted = set(changed_names)
                            impacted.update(dependent_names)
                            impacted.difference_update(removed_names)

                            if impacted:
                                from fast_agent.core.validation import (
                                    get_agent_dependencies,
                                )

                                reverse_deps: dict[str, set[str]] = {}
                                for name, agent_data in self.agents.items():
                                    for dep in get_agent_dependencies(agent_data):
                                        reverse_deps.setdefault(dep, set()).add(name)

                                queue = list(impacted)
                                while queue:
                                    current = queue.pop()
                                    for parent in reverse_deps.get(current, set()):
                                        if parent in removed_names or parent in impacted:
                                            continue
                                        impacted.add(parent)
                                        queue.append(parent)

                            removed_instances = [
                                active_agents_local.pop(name, None)
                                for name in removed_names
                            ]
                            for agent in removed_instances:
                                if agent is None:
                                    continue
                                await agent.shutdown()

                            old_agents = {
                                name: active_agents_local.get(name)
                                for name in impacted
                                if name in active_agents_local
                            }

                            if impacted:
                                from fast_agent.core.direct_factory import (
                                    active_agents_in_dependency_group,
                                )
                                from fast_agent.core.validation import (
                                    get_dependencies_groups,
                                )

                                dependencies = get_dependencies_groups(self.agents)
                                for group in dependencies:
                                    group_targets = [
                                        name for name in group if name in impacted
                                    ]
                                    if not group_targets:
                                        continue
                                    await active_agents_in_dependency_group(
                                        self.app,
                                        self.agents,
                                        model_factory_func,
                                        group_targets,
                                        active_agents_local,
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
                                    name: active_agents_local[name]
                                    for name in impacted
                                    if name in active_agents_local
                                }
                                if updated_agents:
                                    self._apply_agent_card_histories(updated_agents)
                                    validate_provider_keys_post_creation(updated_agents)

                                    if global_prompt_context:
                                        for agent in updated_agents.values():
                                            template = getattr(agent, "instruction", None)
                                            if not template:
                                                continue
                                            resolved = apply_template_variables(
                                                template, global_prompt_context
                                            )
                                            if resolved is None or resolved == template:
                                                continue
                                            agent.set_instruction(resolved)

                            primary_instance.registry_version = self._agent_registry_version
                            self._agent_card_last_changed.clear()
                            self._agent_card_last_removed.clear()
                            self._agent_card_last_dependents.clear()
                            return True

                    async def reload_and_refresh() -> bool:
                        changed = await self.reload_agents()
                        if not changed:
                            return False
                        return await refresh_shared_instance()

                    async def load_card_and_refresh(
                        source: str, parent_name: str | None
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
                                added_names = self.attach_agent_tools(
                                    target_name, loaded_names
                                )

                        await refresh_shared_instance()
                        return loaded_names, added_names

                    async def load_card_source(
                        source: str, parent_name: str | None
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
                                added_names = self.attach_agent_tools(
                                    target_name, loaded_names
                                )

                        return loaded_names, added_names

                    async def attach_agent_tools_and_refresh(
                        parent_name: str, child_names: Sequence[str]
                    ) -> list[str]:
                        added = self.attach_agent_tools(parent_name, child_names)
                        if added:
                            await refresh_shared_instance()
                        return added

                    async def detach_agent_tools_and_refresh(
                        parent_name: str, child_names: Sequence[str]
                    ) -> list[str]:
                        removed = self.detach_agent_tools(parent_name, child_names)
                        if removed:
                            await refresh_shared_instance()
                        return removed

                    async def attach_agent_tools_source(
                        parent_name: str, child_names: Sequence[str]
                    ) -> list[str]:
                        return self.attach_agent_tools(parent_name, child_names)

                    async def dump_agent_card(name: str) -> str:
                        return self.dump_agent_card_text(name)

                    reload_enabled = bool(
                        getattr(self.args, "reload", False)
                        or getattr(self.args, "watch", False)
                    )
                    reload_callback = self.reload_agents if reload_enabled else None
                    wrapper.set_reload_callback(reload_and_refresh if reload_enabled else None)
                    wrapper.set_refresh_callback(
                        refresh_shared_instance if reload_enabled else None
                    )
                    wrapper.set_load_card_callback(load_card_and_refresh)
                    wrapper.set_attach_agent_tools_callback(attach_agent_tools_and_refresh)
                    wrapper.set_detach_agent_tools_callback(detach_agent_tools_and_refresh)
                    wrapper.set_dump_agent_callback(dump_agent_card)
                    self._agent_card_watch_reload = reload_and_refresh if reload_enabled else None

                    if getattr(self.args, "watch", False) and self._agent_card_roots:
                        self._agent_card_watch_task = asyncio.create_task(
                            self._watch_agent_cards()
                        )

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

                    # Handle --card-tool: load card agents and add them as tools to the default agent
                    # This must happen before server startup since server mode exits after running
                    card_tools = getattr(self.args, "card_tools", None)
                    if card_tools:
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

                        # Refresh the instance to include newly loaded agents
                        await refresh_shared_instance()

                        # Get the default agent to add tools to (reuse AgentApp's default logic)
                        default_agent_name = getattr(self.args, "agent", None)
                        # Only use explicit agent_name if it actually exists in loaded agents
                        if default_agent_name and default_agent_name not in active_agents:
                            default_agent_name = None
                        default_agent = wrapper._agent(default_agent_name)

                        if default_agent:
                            add_tool_fn = getattr(default_agent, "add_agent_tool", None)
                            if callable(add_tool_fn):
                                tool_agents = [
                                    active_agents.get(tool_agent_name)
                                    for tool_agent_name in card_tool_agent_names
                                ]
                                add_tools_for_agents(add_tool_fn, tool_agents)

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
                                permissions_enabled = getattr(
                                    self.args, "permissions_enabled", True
                                )

                                # Pass skills directory override if configured
                                skills_override = self._skills_directory_override

                                acp_server = AgentACPServer(
                                    primary_instance=primary_instance,
                                    create_instance=self._server_instance_factory,
                                    dispose_instance=self._server_instance_dispose,
                                    instance_scope=instance_scope,
                                    server_name=server_name or f"{self.name}",
                                    get_registry_version=self._get_registry_version,
                                    skills_directory_override=skills_override,
                                    permissions_enabled=permissions_enabled,
                                    load_card_callback=load_card_source,
                                    attach_agent_tools_callback=attach_agent_tools_source,
                                    dump_agent_card_callback=dump_agent_card,
                                    reload_callback=reload_callback,
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
                                    host=self.args.host,
                                    get_registry_version=self._get_registry_version,
                                    reload_callback=reload_callback,
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
                        prompt: list[PromptMessageExtended] = load_prompt(
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

                if self._agent_card_watch_task is not None:
                    self._agent_card_watch_task.cancel()
                    try:
                        await self._agent_card_watch_task
                    except asyncio.CancelledError:
                        pass
                    self._agent_card_watch_task = None
                self._agent_card_watch_reload = None

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

    def _apply_instruction_context(
        self, instance: AgentInstance, context_vars: dict[str, str]
    ) -> None:
        """Resolve late-binding placeholders for all agents in the provided instance."""
        if not context_vars:
            return

        for agent in instance.agents.values():
            template = getattr(agent, "instruction", None)
            if not template:
                continue

            resolved = apply_template_variables(template, context_vars)
            if resolved is None or resolved == template:
                continue

            # Use set_instruction() which handles syncing request_params and LLM
            agent.set_instruction(resolved)

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

    def _handle_dump_requests(self) -> None:
        dump_dir = getattr(self.args, "dump_agents", None)
        dump_dir_yaml = getattr(self.args, "dump_agents_yaml", None)
        dump_agent = getattr(self.args, "dump_agent", None)
        dump_agent_path = getattr(self.args, "dump_agent_path", None)
        dump_agent_yaml = getattr(self.args, "dump_agent_yaml", False)

        if dump_dir and dump_dir_yaml:
            raise AgentConfigError(
                "Only one of --dump or --dump-yaml may be set"
            )

        if dump_agent and dump_agent_path is None:
            raise AgentConfigError("--dump-agent-path is required with --dump-agent")
        if dump_agent_path is not None and not dump_agent:
            raise AgentConfigError("--dump-agent is required with --dump-agent-path")

        if dump_agent and (dump_dir or dump_dir_yaml):
            raise AgentConfigError(
                "Use either --dump-agent or --dump/--dump-yaml, not both"
            )

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
            directory_entries = [
                item for item in filtered if isinstance(item, (Path, str))
            ]
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
                "Common models: gpt-5.1, kimi, sonnet, haiku. Set reasoning effort on supported models with gpt-5-mini.high",
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
        instance_scope: str = "shared",
        permissions_enabled: bool = True,
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
            permissions_enabled: Whether to request tool permissions from ACP clients (default: True)
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
        server_name: str | None = None,
        server_description: str | None = None,
        tool_description: str | None = None,
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
