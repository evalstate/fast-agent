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
from dataclasses import dataclass
from importlib.metadata import version as get_version
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    cast,
)

import yaml
import yaml.parser

from fast_agent import config
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core import Core
from fast_agent.core.agent_app import AgentRefreshResult as AgentRefreshResult
from fast_agent.core.agent_card_runtime import AgentCardRuntimeMixin
from fast_agent.core.agent_instance_factory import CallableAgentInstanceFactory
from fast_agent.core.default_agent import resolve_default_agent_name
from fast_agent.core.direct_decorators import DecoratorMixin
from fast_agent.core.direct_factory import (
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
from fast_agent.core.managed_runtime import ManagedRuntimeMixin
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.core.run_runtime import FastAgentRunMixin
from fast_agent.core.validation import validate_server_references, validate_workflow_references
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.skills import SKILLS_DEFAULT, SkillManifest, SkillRegistry, SkillsDefault
from fast_agent.ui.console import configure_console_stream
from fast_agent.utils.count_display import plural_label
from fast_agent.utils.text import strip_casefold
from fast_agent.utils.transports import uses_protocol_stdio

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from fastmcp.tools import FunctionTool

    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.core.agent_app import AgentApp, AgentCardLoadResult
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.core.harness import AgentHarness
    from fast_agent.interfaces import AgentProtocol, ModelFactoryFunctionProtocol
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
    from fast_agent.tools.session_environment import ShellExecutor
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)
SkillEntry: TypeAlias = SkillManifest | SkillRegistry | Path | str
SkillConfig: TypeAlias = SkillEntry | list[SkillEntry | None] | None | SkillsDefault
_DEFAULT_CLI_AGENT_PLACEHOLDER = "agent"

__all__ = [
    "AgentRefreshResult",
]


@dataclass(frozen=True)
class RunSettings:
    quiet_mode: bool
    cli_model_override: str | None
    noenv_mode: bool
    server_mode: bool
    transport: str | None
    is_acp_server_mode: bool
    reload_enabled: bool
    resume_requested: bool = False
    resume_session_id: str | None = None
    target_agent_name: str | None = None


@dataclass
class RunRuntime:
    model_factory_func: ModelFactoryFunctionProtocol
    global_prompt_context: dict[str, str] | None
    is_acp_server_mode: bool
    noenv_mode: bool
    managed_instances: list[AgentInstance]
    instance_lock: asyncio.Lock
    shell_executor: ShellExecutor
    resume_requested: bool = False
    resume_session_id: str | None = None
    target_agent_name: str | None = None


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
    load_card_and_refresh: Callable[[str, str | None], Awaitable[AgentCardLoadResult]]
    load_card_source: Callable[[str, str | None], Awaitable[AgentCardLoadResult]]
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

    def instance_factory(self) -> CallableAgentInstanceFactory:
        return CallableAgentInstanceFactory(
            create=self.create_instance,
            dispose=self.dispose_instance,
        )


class FastAgent(AgentCardRuntimeMixin, ManagedRuntimeMixin, FastAgentRunMixin, DecoratorMixin):
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
        noenv: bool = False,
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
        self._initialize_runtime_options(
            quiet=quiet,
            environment_dir=environment_dir,
            skills_directory=skills_directory,
        )
        if parse_cli_args:
            self._parse_constructor_cli_args(ignore_unknown_args=ignore_unknown_args)
        self._apply_constructor_runtime_flags(noenv=noenv)

        self.name = name
        self.config_path = config_path

        try:
            instance_settings = self._load_instance_settings()
            self.app = Core(name=name, settings=instance_settings, **kwargs)
            self._stop_progress_display_if_quiet()

        except yaml.parser.ParserError as e:
            handle_error(
                e,
                "YAML Parsing Error",
                "There was an error parsing the config or secrets YAML configuration file.",
            )
            raise SystemExit(1) from e

        self._initialize_agent_registries()

    def _initialize_runtime_options(
        self,
        *,
        quiet: bool,
        environment_dir: str | pathlib.Path | None,
        skills_directory: str | pathlib.Path | Sequence[str | pathlib.Path] | None,
    ) -> None:
        self._programmatic_quiet = quiet
        self._environment_dir_override = self._normalize_environment_dir(environment_dir)
        self._skills_directory_override = self._normalize_skill_directories(skills_directory)
        self._default_skill_manifests: list[SkillManifest] = []
        self._extra_prompt_context: dict[str, str] = {}
        self._server_instance_dispose = None
        self._server_managed_instances: list[AgentInstance] = []

    def set_prompt_context(self, values: dict[str, str]) -> None:
        """Set additional run-scoped instruction template variables."""
        self._extra_prompt_context = dict(values)

    def _parse_constructor_cli_args(self, *, ignore_unknown_args: bool) -> None:
        parser = self._constructor_arg_parser()
        known_args, _unknown = parser.parse_known_args()
        self.args = known_args
        self._normalize_constructor_cli_server_flags()
        self._handle_constructor_version_flag()

    @staticmethod
    def _constructor_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="DirectFastAgent Application")
        parser.add_argument("--model", help="Override the default model for all agents")
        parser.add_argument(
            "--agent",
            default=None,
            help="Agent name for --message/--prompt-file (defaults to the app default agent)",
        )
        parser.add_argument("-m", "--message", help="Message to send to the specified agent")
        parser.add_argument(
            "-p",
            "--prompt-file",
            help="Path to a prompt file to use (either text or JSON)",
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Disable progress display, tool and message logging for cleaner output",
        )
        parser.add_argument("--version", action="store_true", help="Show version and exit")
        parser.add_argument(
            "--transport",
            choices=["http", "stdio", "acp", "a2a"],
            default=None,
            help=("Transport protocol to use when running as a server (http, stdio, acp, or a2a)"),
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to use when running as a server with HTTP transport",
        )
        parser.add_argument(
            "--host",
            default="127.0.0.1",
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
        parser.add_argument("--env", help="Override the base fast-agent environment directory")
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
        parser.add_argument("--dump-agent", help="Export a single agent by name")
        parser.add_argument("--dump-agent-path", help="Output file path for --dump-agent")
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
            "--noenv",
            "--no-env",
            action="store_true",
            help="Disable fast-agent home/environment directory use",
        )
        parser.add_argument(
            "--card-tool",
            action="append",
            dest="card_tools",
            help=(
                "Path, HTTP(S) URL, file:// URI, or hf:// URI to an AgentCard file to load "
                "as a tool (repeatable)"
            ),
        )
        return parser

    def _normalize_constructor_cli_server_flags(self) -> None:
        cli_args = sys.argv[1:]
        transport_flag_used = any(
            arg == "--transport" or arg.startswith("--transport=") for arg in cli_args
        )

        self.args.server = False
        if transport_flag_used:
            self.args.server = True
        if getattr(self.args, "transport", None) is None:
            self.args.transport = "http"

    def _handle_constructor_version_flag(self) -> None:
        if not getattr(self.args, "version", False):
            return

        try:
            app_version = get_version("fast-agent-mcp")
        except Exception:
            app_version = "unknown"
        print(f"fast-agent-mcp v{app_version}")
        sys.exit(0)

    def _apply_constructor_runtime_flags(self, *, noenv: bool) -> None:
        if getattr(self.args, "transport", None) == "acp":
            self._programmatic_quiet = True
            self.args.quiet = True

        if self._programmatic_quiet:
            self.args.quiet = True

        if noenv:
            self.args.noenv = True
        elif "noenv" not in vars(self.args):
            self.args.noenv = False

        self._apply_constructor_environment_override()
        self._apply_constructor_skills_override()

    def _apply_constructor_environment_override(self) -> None:
        args_env = vars(self.args).get("env")
        if self._environment_dir_override is None and args_env:
            self._environment_dir_override = self._normalize_environment_dir(args_env)

        if self._environment_dir_override is None:
            return

        from fast_agent.constants import FAST_AGENT_RUNTIME_ENVIRONMENT

        os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = str(self._environment_dir_override)
        os.environ["ENVIRONMENT_DIR"] = str(self._environment_dir_override)

    def _apply_constructor_skills_override(self) -> None:
        args_skills = vars(self.args).get("skills")
        if self._skills_directory_override is None and args_skills:
            self._skills_directory_override = self._normalize_skill_directories(args_skills)

    def _load_instance_settings(self) -> config.Settings:
        instance_settings = self._load_config()
        self._apply_constructor_config_overrides(instance_settings)
        self.config = instance_settings.model_dump()
        config.update_global_settings(instance_settings)
        return instance_settings

    def _apply_constructor_config_overrides(self, settings: config.Settings) -> None:
        if self._programmatic_quiet:
            settings.logger.progress_display = False
            settings.logger.show_chat = False
            settings.logger.show_tools = False

        if self._environment_dir_override is not None:
            settings.environment_dir = str(self._environment_dir_override)

        if self._skills_directory_override is not None:
            settings.skills.directories = [str(path) for path in self._skills_directory_override]

    def _stop_progress_display_if_quiet(self) -> None:
        if not self._programmatic_quiet:
            return

        if getattr(self.args, "server", False) and uses_protocol_stdio(
            getattr(self.args, "transport", None)
        ):
            configure_console_stream("stderr")

        from fast_agent.ui.progress_display import progress_display

        progress_display.stop()

    def _initialize_agent_registries(self) -> None:
        self.agents: dict[str, AgentCardData] = {}
        self._registered_tools: list[FunctionTool] = []
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

    def _load_config(self) -> config.Settings:
        """Load configuration from YAML file including secrets using get_settings
        but without relying on the global cache."""

        import fast_agent.config as _config_module
        from fast_agent.io.source_resolver import materialize_text_source

        # Temporarily clear the global settings to ensure a fresh load
        old_settings = _config_module._settings
        _config_module._settings = None

        try:
            # Use get_settings to load config - this handles all paths and secrets merging
            resolved_config_path = (
                materialize_text_source(self.config_path, label="config file", suffix=".yaml")
                if self.config_path is not None
                else None
            )
            settings = _config_module.get_settings(
                resolved_config_path,
                env_dir=self._environment_dir_override,
                noenv=bool(getattr(self.args, "noenv", False)),
            )
            return settings
        finally:
            # Restore the original global settings
            _config_module._settings = old_settings

    @property
    def context(self) -> Context:
        """Access the application context"""
        return self.app.context

    def load_agents(self, path: str | Path) -> list[str]:
        """
        Load AgentCards from a file, directory, or URI and register them as agents.

        Loading is idempotent for the provided path: any previously loaded agents
        from the same path that are no longer present are removed.

        Returns:
            Sorted list of agent names loaded from the provided path.
        """
        from urllib.parse import urlparse

        from fast_agent.io.source_resolver import REMOTE_TEXT_SCHEMES, materialize_text_source

        source = str(path)
        parsed = urlparse(source)
        if parsed.scheme in REMOTE_TEXT_SCHEMES:
            return self.load_agents_from_url(source)

        root = (
            (
                materialize_text_source(source, label="AgentCard source")
                if parsed.scheme == "file"
                else Path(path)
            )
            .expanduser()
            .resolve()
        )
        changed = self._load_agent_cards_from_root(root, incremental=False)
        if changed:
            self._agent_registry_version += 1
        return sorted(self._agent_card_roots.get(root, set()))

    def load_agents_from_url(self, url: str) -> list[str]:
        """Load an AgentCard from a remote URL or hf:// URI (markdown or YAML)."""
        from urllib.parse import urlparse

        from fast_agent.core.agent_card_loader import load_agent_cards
        from fast_agent.io.source_resolver import REMOTE_TEXT_SCHEMES, materialize_text_source

        parsed = urlparse(url)
        if parsed.scheme not in REMOTE_TEXT_SCHEMES:
            return self.load_agents(url)

        suffix = Path(parsed.path).suffix or ".md"
        temp_path = materialize_text_source(url, label="AgentCard URL", suffix=suffix)

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
            raise AgentConfigError(
                f"{plural_label(len(missing), 'Agent')} not found: {missing_list}"
            )

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
            is_default=lambda _name, agent_data: (
                isinstance(config := agent_data.get("config"), AgentConfig) and config.default
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

    def _get_registry_version(self) -> int:
        return self._agent_registry_version

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

    def _prepare_run_settings(
        self,
        *,
        model_override: str | None = None,
        force_headless: bool = False,
    ) -> RunSettings:
        """Compute the run-mode settings after app initialization."""
        quiet_mode = getattr(self.args, "quiet", False)
        server_mode = False if force_headless else bool(getattr(self.args, "server", False))
        transport = None if force_headless else getattr(self.args, "transport", None)
        if uses_protocol_stdio(transport) and server_mode:
            quiet_mode = True
            configure_console_stream("stderr")

        cli_model_arg = (
            model_override if model_override is not None else getattr(self.args, "model", None)
        )
        cli_model_override = cli_model_arg if isinstance(cli_model_arg, str) else None
        noenv_mode = bool(getattr(self.args, "noenv", False))
        resume_requested = bool(getattr(self.args, "resume_requested", False))
        resume_session_id_arg = getattr(self.args, "resume_session_id", None)
        resume_session_id = (
            resume_session_id_arg if isinstance(resume_session_id_arg, str) else None
        )
        target_agent_name_arg = getattr(self.args, "agent", None)
        target_agent_name = (
            target_agent_name_arg if isinstance(target_agent_name_arg, str) else None
        )

        cfg = self.context.config
        model_source_override_arg = getattr(self.args, "model_source_override", None)
        model_source_override = (
            model_source_override_arg if isinstance(model_source_override_arg, str) else None
        )
        model_source = model_source_override or get_default_model_source(
            config_default_model=cfg.default_model if cfg else None,
            cli_model=cli_model_override,
            model_references=cfg.model_references if cfg else None,
        )
        if cfg:
            cfg.model_source = model_source
            cfg.cli_model_override = cli_model_override
            if noenv_mode:
                cfg.session_history = False

        return RunSettings(
            quiet_mode=quiet_mode,
            cli_model_override=cli_model_override,
            resume_requested=resume_requested,
            resume_session_id=resume_session_id,
            target_agent_name=target_agent_name,
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
        except Exception as exc:
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
        self, *, apply_global_prompt_context: bool, noenv_mode: bool
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
            noenv=noenv_mode,
        )
        context_variables.update(self._extra_prompt_context)
        return context_variables or None

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

    def harness(
        self,
        *,
        model: str | None = None,
    ) -> "AgentHarness":
        """Create a headless session harness for this fast-agent app."""
        from fast_agent.core.harness import AgentHarness

        return AgentHarness(self, model=model)

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
        if entry is SKILLS_DEFAULT or entry is None:
            return []
        if isinstance(entry, list):
            return self._resolve_skill_entries(cast("list[SkillEntry | None]", entry))
        return self._resolve_single_skill_entry(cast("SkillEntry", entry))

    def _resolve_single_skill_entry(self, entry: SkillEntry) -> list[SkillManifest]:
        if isinstance(entry, SkillManifest):
            resolved = [entry]
        elif isinstance(entry, SkillRegistry):
            resolved = self._load_skill_registry(entry)
        elif isinstance(entry, (Path, str)):
            # Use instance method to preserve original path for relative path computation
            resolved = self._load_skill_directories([entry])
        else:
            logger.debug(
                "Unsupported skill entry type",
                data={"type": type(entry).__name__},
            )
            resolved = []
        return resolved

    def _resolve_skill_entries(self, entries: list[SkillEntry | None]) -> list[SkillManifest]:
        filtered = self._supported_skill_entries(entries)
        if not filtered:
            return []
        directory_entries = [item for item in filtered if isinstance(item, (Path, str))]
        if len(directory_entries) == len(filtered):
            return self._load_skill_directories(directory_entries)

        manifests: list[SkillManifest] = []
        for item in filtered:
            manifests.extend(self._resolve_skills(item))
        return manifests

    @staticmethod
    def _supported_skill_entries(entries: list[SkillEntry | None]) -> list[SkillEntry]:
        filtered: list[SkillEntry] = []
        for item in entries:
            if isinstance(item, (SkillManifest, SkillRegistry, Path, str)):
                filtered.append(item)
            elif item is not None:
                logger.debug(
                    "Unsupported skill entry type",
                    data={"type": type(item).__name__},
                )
        return filtered

    @staticmethod
    def _load_skill_registry(registry: SkillRegistry) -> list[SkillManifest]:
        try:
            return registry.load_manifests()
        except Exception:
            logger.debug(
                "Failed to load skills from registry",
                data={"registry": type(registry).__name__},
            )
            return []

    @staticmethod
    def _load_skill_directories(entries: list[Path | str]) -> list[SkillManifest]:
        directories = [Path(item) if isinstance(item, str) else item for item in entries]
        registry = SkillRegistry(base_dir=Path.cwd(), directories=directories)
        return registry.load_manifests()

    @staticmethod
    def _deduplicate_skills(manifests: list[SkillManifest]) -> list[SkillManifest]:
        unique: dict[str, SkillManifest] = {}
        for manifest in manifests:
            key = strip_casefold(manifest.name)
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
                "Please check your 'fast-agent.yaml' configuration file and add the missing server definitions.",
            )
        elif isinstance(e, ProviderKeyError):
            handle_error(
                e,
                "Provider Configuration Error",
                "Please check your 'fast-agent.secrets.yaml' configuration file and ensure all required API keys are set.",
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
                "Common models: gpt-5.5, kimi, sonnet, haiku. Set reasoning effort on supported models with gpt-5.4-mini?reasoning=high",
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
