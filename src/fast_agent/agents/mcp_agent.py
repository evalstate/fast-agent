"""
Base Agent class that implements the AgentProtocol interface.

This class provides default implementations of the standard agent methods
and delegates operations to an attached FastAgentLLMProtocol instance.
"""

import asyncio
import fnmatch
import time
from abc import ABC
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

import mcp
from a2a.types import AgentCard, AgentSkill
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    GetPromptResult,
    ListToolsResult,
    PromptMessage,
    ReadResourceResult,
    TextContent,
    Tool,
)
from pydantic import BaseModel

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import DEFAULT_CAPABILITIES
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import MCPServerSettings
from fast_agent.constants import (
    HUMAN_INPUT_TOOL_NAME,
    SHELL_NOTICE_PREFIX,
    should_parallelize_tool_calls,
)
from fast_agent.core.exceptions import PromptExitError
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.terminal_output_limits import calculate_terminal_output_limit_for_model
from fast_agent.mcp.common import (
    get_resource_name,
    get_server_name,
    is_namespaced_name,
)
from fast_agent.mcp.experimental_session_client import ExperimentalSessionClient
from fast_agent.mcp.mcp_aggregator import (
    MCPAggregator,
    MCPAttachOptions,
    MCPAttachResult,
    MCPDetachResult,
    NamespacedTool,
    ServerStatus,
)
from fast_agent.skills import SKILLS_DEFAULT, SkillManifest
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.elicitation import (
    get_elicitation_tool,
    run_elicitation_form,
    set_elicitation_input_callback,
)
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.tools.skill_reader import SkillReader
from fast_agent.types import (
    PromptMessageExtended,
    RequestParams,
    ToolTimingInfo,
)
from fast_agent.ui import console
from fast_agent.utils.async_utils import gather_with_cancel

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)
ItemT = TypeVar("ItemT")

LLM = TypeVar("LLM", bound=FastAgentLLMProtocol)

# Display name overrides for tools shown in the bottom bar
TOOL_DISPLAY_NAMES: dict[str, str] = {
    "read_skill": "skill",
}

if TYPE_CHECKING:
    from rich.text import Text

    from fast_agent.context import Context
    from fast_agent.llm.usage_tracking import UsageAccumulator


class McpAgent(ABC, ToolAgent):
    """
    A base Agent class that implements the AgentProtocol interface.

    This class provides default implementations of the standard agent methods
    and delegates LLM operations to an attached FastAgentLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        connection_persistence: bool = True,
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            context=context,
            **kwargs,
        )

        # Create aggregator with composition
        self._aggregator = MCPAggregator(
            server_names=self.config.servers,
            connection_persistence=connection_persistence,
            name=self.config.name,
            context=context,
            config=self.config,  # Pass the full config for access to elicitation_handler
            **kwargs,
        )

        # Store the original template - resolved instruction set after build()
        self._instruction_template = self.config.instruction
        self._instruction = self.config.instruction  # Will be replaced by builder output
        self.executor = context.executor if context else None
        self.logger = get_logger(f"{__name__}.{self._name}")
        manifests: list[SkillManifest] = list(getattr(self.config, "skill_manifests", []) or [])
        if (
            self.config.skills is SKILLS_DEFAULT
            and not manifests
            and context
            and context.skill_registry
        ):
            try:
                manifests = list(context.skill_registry.load_manifests())  # type: ignore[assignment]
            except Exception:
                manifests = []

        self._shell_runtime: ShellRuntime | None = None
        self._shell_notice_emitted = False
        self._allow_shell_notice = False
        self._shell_runtime_enabled = False
        self._show_shell_tool_call_id = False
        self._defer_shell_display_to_tool_result = False
        self._shell_access_modes: tuple[str, ...] = ()
        self._bash_tool: Tool | None = None
        # Allow external runtime injection (e.g., for ACP terminal support)
        self._external_runtime = None

        # Allow filesystem runtime injection (e.g., for ACP filesystem support)
        self._filesystem_runtime = None

        self._skill_manifests: list[SkillManifest] = []
        self._skill_map: dict[str, SkillManifest] = {}
        self._skill_reader: SkillReader | None = None
        self.set_skill_manifests(manifests)
        self.skill_registry: SkillRegistry | None = None
        if isinstance(self.config.skills, SkillRegistry):
            self.skill_registry = self.config.skills
        elif self.config.skills is SKILLS_DEFAULT and context and context.skill_registry:
            self.skill_registry = context.skill_registry
        self._warnings: list[str] = []
        self._warning_messages_seen: set[str] = set()
        shell_flag_requested = bool(context and getattr(context, "shell_runtime", False))
        shell_config_requested = bool(self.config.shell)
        skills_configured = bool(self._skill_manifests)
        self._shell_runtime_activation_reason: str | None = None

        reasons: list[str] = []
        if shell_flag_requested:
            reasons.append("--shell flag")
        if shell_config_requested:
            reasons.append("agent config")
        if skills_configured:
            reasons.append("agent skills configuration")

        if reasons:
            if reasons == ["agent skills configuration"]:
                self._shell_runtime_activation_reason = "because agent skills are configured"
            else:
                self._shell_runtime_activation_reason = "via " + " and ".join(reasons)

        # Derive skills directory from this agent's manifests (respects per-agent config)
        skills_directory = None
        if self._skill_manifests:
            # Get the skills directory from the first manifest's path
            # Path structure: <env>/skills/skill-name/SKILL.md
            # So we need parent.parent of the manifest path
            first_manifest = self._skill_manifests[0]
            if first_manifest.path:
                skills_directory = first_manifest.path.parent.parent

        self._shell_access_modes: tuple[str, ...] = ()
        if self._shell_runtime_activation_reason is not None:
            modes: list[str] = []
            if skills_configured:
                modes.append("skills")
            if shell_flag_requested:
                modes.append("switch")
            self._shell_access_modes = tuple(modes)

        self._activate_shell_runtime(
            self._shell_runtime_activation_reason,
            working_directory=self.config.cwd,
            skills_directory=skills_directory,
            access_modes=self._shell_access_modes,
        )

        # Store instruction context for template resolution
        self._instruction_context: dict[str, str] = {}

        self._allow_shell_notice = True

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # set with the "attach" method
        self._llm: FastAgentLLMProtocol | None = None

        # Instantiate human input tool once if enabled in config
        self._human_input_tool: Tool | None = None
        if self.config.human_input:
            try:
                self._human_input_tool = get_elicitation_tool()
            except Exception:
                self._human_input_tool = None

        # Register the MCP UI handler as the elicitation callback so fast_agent.tools can call it
        # without importing MCP types. This avoids circular imports and ensures the callback is ready.
        try:
            from fast_agent.human_input.elicitation_handler import elicitation_input_callback
            from fast_agent.human_input.types import HumanInputRequest

            async def _mcp_elicitation_adapter(
                request_payload: dict,
                agent_name: str | None = None,
                server_name: str | None = None,
                server_info: dict | None = None,
            ) -> str:
                req = HumanInputRequest(**request_payload)
                resp = await elicitation_input_callback(
                    request=req,
                    agent_name=agent_name,
                    server_name=server_name,
                    server_info=server_info,
                )
                return resp.response if isinstance(resp.response, str) else str(resp.response)

            set_elicitation_input_callback(_mcp_elicitation_adapter)
        except Exception:
            # If UI handler import fails, leave callback unset; tool will error with a clear message
            pass

    async def __aenter__(self):
        """Initialize the agent and its MCP aggregator."""
        await self._aggregator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the agent and its MCP aggregator."""
        await self._aggregator.__aexit__(exc_type, exc_val, exc_tb)

    async def initialize(self) -> None:
        """
        Initialize the agent and connect to the MCP servers.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await self.__aenter__()

        # Apply template substitution to the instruction with server instructions
        await self._apply_instruction_templates()

        await super().initialize()

    async def shutdown(self) -> None:
        """
        Shutdown the agent and close all MCP server connections.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await self._run_lifecycle_hook("on_shutdown")
        await self._aggregator.close()
        await self._finalize_shutdown(run_hook=False)

    def enable_shell(self, working_directory: Path | None = None) -> None:
        """
        Enable shell runtime on this agent after creation.

        This allows adding shell access to agents loaded from cards or created dynamically.

        Args:
            working_directory: Optional custom working directory for shell commands.
                              If not specified, uses the current working directory.
        """
        if self._shell_runtime_enabled:
            # Already enabled, but update working directory if specified
            shell_runtime = self._shell_runtime
            if working_directory is not None and shell_runtime is not None:
                shell_runtime._working_directory = working_directory
            return

        self._activate_shell_runtime(
            "via enable_shell() call",
            working_directory=working_directory,
            access_modes=("[red]direct[/red]",),
        )

    async def get_server_status(self) -> dict[str, ServerStatus]:
        """Expose server status details for UI and diagnostics consumers."""
        if not self._aggregator:
            return {}
        return await self._aggregator.collect_server_status()

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        return await self._aggregator.attach_server(
            server_name=server_name,
            server_config=server_config,
            options=options,
        )

    async def detach_mcp_server(self, server_name: str) -> MCPDetachResult:
        return await self._aggregator.detach_server(server_name)

    def list_attached_mcp_servers(self) -> list[str]:
        return self._aggregator.list_attached_servers()

    @property
    def aggregator(self) -> MCPAggregator:
        """Expose the MCP aggregator for UI integrations."""
        return self._aggregator

    @property
    def experimental_sessions(self) -> ExperimentalSessionClient:
        """Expose focused experimental-session cookie controls for this agent."""
        return self._aggregator.experimental_sessions

    @property
    def instruction_template(self) -> str:
        """The original instruction template with placeholders."""
        return self._instruction_template or ""

    @property
    def instruction_context(self) -> dict[str, str]:
        """Context values for instruction template resolution."""
        return self._instruction_context

    @property
    def skill_manifests(self) -> list[SkillManifest]:
        """List of skill manifests configured for this agent."""
        return self._skill_manifests

    @property
    def has_filesystem_runtime(self) -> bool:
        """Whether filesystem runtime is available (affects skill tool names)."""
        return self._filesystem_runtime is not None

    @property
    def initialized(self) -> bool:
        """Check if both the agent and aggregator are initialized."""
        return self._initialized and self._aggregator.initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set the initialized state of both agent and aggregator."""
        self._initialized = value
        self._aggregator.initialized = value

    async def _apply_instruction_templates(self) -> None:
        """
        Apply template substitution to the instruction, including server instructions.
        This is called during initialization after servers are connected.
        """
        from fast_agent.core.instruction_refresh import build_instruction

        if not self._instruction_template:
            return

        # Build the instruction using the central helper
        new_instruction = await build_instruction(
            self._instruction_template,
            aggregator=self._aggregator,
            skill_manifests=self._skill_manifests,
            has_filesystem_runtime=self.has_filesystem_runtime,
            context=self._instruction_context,
            source=self._name,
        )
        self.set_instruction(new_instruction)

        # Warn if skills configured but placeholder missing
        if self._skill_manifests and "{{agentSkills}}" not in self._instruction_template:
            warning_message = f"[dim]Agent '{self._name}' skills are configured but no {{{{agentSkills}}}} in system prompt.[/dim]"
            self._record_warning(warning_message, surface="startup_once")

        self.logger.debug(f"Applied instruction templates for agent {self._name}")

    @staticmethod
    def _resolve_shell_working_directory(path: Path) -> Path:
        """Resolve a configured shell working directory for validation messages."""
        if path.is_absolute():
            return path.resolve()
        return (Path.cwd() / path).resolve()

    def _warn_if_invalid_shell_working_directory(self, working_directory: Path | None) -> None:
        """Emit a startup warning when a configured shell cwd is missing/invalid."""
        if working_directory is None:
            return

        resolved = self._resolve_shell_working_directory(working_directory)
        if not resolved.exists():
            self._record_warning(
                " ".join(
                    [
                        f"[dim]Agent '{self._name}' has shell cwd that does not exist: {resolved}.",
                        f"Configured cwd: {working_directory}.",
                        "Shell commands will fail until this path exists.[/dim]",
                    ]
                ),
                surface="startup_once",
            )
            return

        if not resolved.is_dir():
            self._record_warning(
                " ".join(
                    [
                        f"[dim]Agent '{self._name}' has shell cwd that is not a directory: {resolved}.",
                        f"Configured cwd: {working_directory}.",
                        "Shell commands will fail until this points to a directory.[/dim]",
                    ]
                ),
                surface="startup_once",
            )

    def set_skill_manifests(self, manifests: Sequence[SkillManifest]) -> None:
        self._skill_manifests = list(manifests)
        self._skill_map = {manifest.name: manifest for manifest in self._skill_manifests}
        if self._skill_manifests:
            self._skill_reader = SkillReader(self._skill_manifests, self.logger)
            self._ensure_shell_runtime_for_skills()
        else:
            self._skill_reader = None

    def _ensure_shell_runtime_for_skills(self) -> None:
        if self._shell_runtime_enabled:
            return
        if self._external_runtime is not None:
            return

        # Derive skills directory from manifests (respects per-agent config)
        skills_directory = None
        if self._skill_manifests:
            first_manifest = self._skill_manifests[0]
            if first_manifest.path:
                skills_directory = first_manifest.path.parent.parent

        self._activate_shell_runtime(
            "because agent skills are configured",
            skills_directory=skills_directory,
            working_directory=self.config.cwd,
            access_modes=("skills",),
            show_shell_notice=True,
        )

    def _resolve_shell_runtime_settings(self) -> tuple[int, int, int]:
        timeout_seconds = 90
        warning_interval_seconds = 30
        config_output_byte_limit = None
        shell_config = None
        if self._context and self._context.config:
            shell_config = getattr(self._context.config, "shell_execution", None)
        if shell_config:
            timeout_seconds = getattr(shell_config, "timeout_seconds", 90)
            warning_interval_seconds = getattr(shell_config, "warning_interval_seconds", 30)
            config_output_byte_limit = getattr(shell_config, "output_byte_limit", None)

        if config_output_byte_limit is not None:
            output_byte_limit = config_output_byte_limit
        else:
            model_name = self.config.model
            if not model_name and self._context and self._context.config:
                model_name = getattr(self._context.config, "default_model", None)
            output_byte_limit = calculate_terminal_output_limit_for_model(model_name)
        return timeout_seconds, warning_interval_seconds, output_byte_limit

    def _shell_output_limit_overridden(self) -> bool:
        """Return True when shell output byte limit is explicitly configured."""
        if not self._context or not self._context.config:
            return False
        shell_config = getattr(self._context.config, "shell_execution", None)
        if shell_config is None:
            return False
        return getattr(shell_config, "output_byte_limit", None) is not None

    def _on_llm_attached(self, llm: FastAgentLLMProtocol) -> None:
        super()._on_llm_attached(llm)
        if self._shell_runtime is None:
            return
        if self._shell_output_limit_overridden():
            return

        self._shell_runtime.set_output_byte_limit(
            calculate_terminal_output_limit_for_model(llm.model_name)
        )

    def _activate_shell_runtime(
        self,
        activation_reason: str | None,
        *,
        working_directory: Path | None = None,
        skills_directory: Path | None = None,
        access_modes: tuple[str, ...] = (),
        show_shell_notice: bool = False,
    ) -> None:
        if activation_reason is not None and self._external_runtime is not None:
            return

        self._warn_if_invalid_shell_working_directory(working_directory)

        timeout_seconds, warning_interval_seconds, output_byte_limit = (
            self._resolve_shell_runtime_settings()
        )

        self._shell_runtime_activation_reason = activation_reason
        self._shell_runtime = ShellRuntime(
            activation_reason,
            self.logger,
            timeout_seconds=timeout_seconds,
            warning_interval_seconds=warning_interval_seconds,
            skills_directory=skills_directory,
            working_directory=working_directory,
            output_byte_limit=output_byte_limit,
            config=self._context.config if self._context else None,
            agent_name=self._name,
        )
        self._shell_runtime_enabled = self._shell_runtime.enabled
        self._bash_tool = self._shell_runtime.tool
        self._shell_access_modes = access_modes if self._shell_runtime_enabled else ()
        if self._shell_runtime_enabled:
            self._shell_runtime.announce()
            if show_shell_notice and self._allow_shell_notice and not self._shell_notice_emitted:
                self._shell_notice_emitted = True
                try:
                    console.console.print(SHELL_NOTICE_PREFIX)
                except Exception:  # pragma: no cover - console fallback
                    pass

    @property
    def shell_runtime_enabled(self) -> bool:
        return self._shell_runtime_enabled

    @property
    def shell_access_modes(self) -> tuple[str, ...]:
        return self._shell_access_modes

    @property
    def shell_runtime(self) -> ShellRuntime | None:
        return self._shell_runtime

    def shell_notice_line(self) -> "Text | None":
        if not self._shell_runtime_enabled or self._shell_runtime is None:
            return None
        from fast_agent.ui.shell_notice import format_shell_notice

        return format_shell_notice(self._shell_access_modes, self._shell_runtime)

    def _record_warning(
        self,
        message: str,
        *,
        surface: Literal["runtime_toolbar", "startup_once"] = "runtime_toolbar",
    ) -> None:
        if message in self._warning_messages_seen:
            return
        self._warning_messages_seen.add(message)
        self._warnings.append(message)
        self.logger.warning(message)
        try:
            from fast_agent.ui import notification_tracker

            notification_tracker.add_warning(message, surface=surface)
        except Exception:
            pass

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def set_instruction_context(self, context: dict[str, str]) -> None:
        """
        Set session-level context variables for instruction template resolution.

        This should be called when an ACP session is established to provide
        variables like {{env}}, {{workspaceRoot}} etc. that are resolved per-session.

        Args:
            context: Dict mapping placeholder names to values (e.g., {"env": "...", "workspaceRoot": "/path"})
        """
        self._instruction_context.update(context)
        self.logger.debug(f"Set instruction context for agent {self._name}: {list(context.keys())}")

    async def __call__(
        self,
        message: Union[
            str,
            PromptMessage,
            PromptMessageExtended,
            Sequence[Union[str, PromptMessage, PromptMessageExtended]],
        ],
    ) -> str:
        return await self.send(message)

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """
        Check if a name matches a pattern for a specific server.

        Args:
            name: The name to match (could be tool name, resource URI, or prompt name)
            pattern: The pattern to match against (e.g., "add", "math*", "resource://math/*")

        Returns:
            True if the name matches the pattern
        """

        # For resources and prompts, match directly against the pattern
        return fnmatch.fnmatch(name, pattern)

    def _filter_namespaced_tools(self, tools: Sequence[Tool] | None) -> list[Tool]:
        """
        Apply configuration-based filtering to a collection of tools.
        """
        if not tools:
            return []

        return [
            tool
            for tool in tools
            if is_namespaced_name(tool.name) and self._tool_matches_filter(tool.name)
        ]

    def _filter_server_collections(
        self,
        items_by_server: Mapping[str, Sequence[ItemT]],
        filters: Mapping[str, Sequence[str]] | None,
        value_getter: Callable[[ItemT], str],
    ) -> dict[str, list[ItemT]]:
        """
        Apply server-specific filters to a mapping of collections.
        """
        if not items_by_server:
            return {}

        if not filters:
            return {server: list(items) for server, items in items_by_server.items()}

        filtered: dict[str, list[ItemT]] = {}
        for server, items in items_by_server.items():
            patterns = filters.get(server)
            if patterns is None:
                filtered[server] = list(items)
                continue

            matches = [
                item
                for item in items
                if any(self._matches_pattern(value_getter(item), pattern) for pattern in patterns)
            ]
            if matches:
                filtered[server] = matches

        return filtered

    def _filter_server_tools(self, tools: list[Tool] | None, namespace: str) -> list[Tool]:
        """
        Filter items for a Server (not namespaced)
        """
        if not tools:
            return []

        filters = self.config.tools
        if not filters:
            return list(tools)

        if namespace not in filters:
            return list(tools)

        filtered = self._filter_server_collections(
            {namespace: tools}, filters, lambda tool: tool.name
        )
        return filtered.get(namespace, [])

    async def _get_filtered_mcp_tools(self) -> list[Tool]:
        """
        Get the list of tools available to this agent, applying configured filters.

        Returns:
            List of Tool objects
        """
        aggregator_result = await self._aggregator.list_tools()
        return self._filter_namespaced_tools(aggregator_result.tools)

    def _tool_matches_filter(self, packed_name: str) -> bool:
        """
        Check if a tool name matches the agent's tool configuration.

        Args:
            tool_name: The name of the tool to check (namespaced)
        """
        server_name = get_server_name(packed_name)
        config_tools = self.config.tools or {}
        if server_name not in config_tools:
            return True
        resource_name = get_resource_name(packed_name)
        patterns = config_tools.get(server_name, [])
        return any(self._matches_pattern(resource_name, pattern) for pattern in patterns)

    def set_external_runtime(self, runtime) -> None:
        """
        Set an external runtime (e.g., ACPTerminalRuntime) to replace ShellRuntime.

        This allows ACP mode to inject terminal support that uses the client's
        terminal capabilities instead of local process execution.

        Args:
            runtime: Runtime instance with tool and execute() method
        """
        self._external_runtime = runtime
        self.logger.info(
            f"External runtime injected: {type(runtime).__name__}",
            runtime_type=type(runtime).__name__,
        )

    def set_filesystem_runtime(self, runtime) -> None:
        """
        Set a filesystem runtime (e.g., ACPFilesystemRuntime) to add filesystem tools.

        This allows ACP mode to inject filesystem support that uses the client's
        filesystem capabilities for reading and writing files.

        Args:
            runtime: Runtime instance with tools property and read_text_file/write_text_file methods
        """
        self._filesystem_runtime = runtime
        self.logger.info(
            f"Filesystem runtime injected: {type(runtime).__name__}",
            runtime_type=type(runtime).__name__,
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult:
        """
        Call a tool by name with the given arguments.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            Result of the tool call
        """
        # Check external runtime first (e.g., ACP terminal)
        if self._external_runtime and hasattr(self._external_runtime, "tool"):
            if self._external_runtime.tool and name == self._external_runtime.tool.name:
                return await self._external_runtime.execute(arguments, tool_use_id)

        # Check filesystem runtime (e.g., ACP filesystem)
        if self._filesystem_runtime and hasattr(self._filesystem_runtime, "tools"):
            for tool in self._filesystem_runtime.tools:
                if tool.name == name:
                    # Route to the appropriate method based on tool name
                    if name == "read_text_file":
                        return await self._filesystem_runtime.read_text_file(arguments, tool_use_id)
                    elif name == "write_text_file":
                        return await self._filesystem_runtime.write_text_file(
                            arguments, tool_use_id
                        )

        # Check skill reader (non-ACP context with skills)
        if self._skill_reader and name == "read_skill":
            return await self._skill_reader.execute(arguments)

        # Fall back to shell runtime
        if (
            self._shell_runtime
            and self._shell_runtime.tool
            and name == self._shell_runtime.tool.name
        ):
            return await self._shell_runtime.execute(
                arguments,
                tool_use_id,
                show_tool_call_id=self._show_shell_tool_call_id,
                defer_display_to_tool_result=self._defer_shell_display_to_tool_result,
            )

        if name == HUMAN_INPUT_TOOL_NAME:
            # Call the elicitation-backed human input tool
            return await self._call_human_input_tool(arguments)

        if name in self._execution_tools:
            return await super().call_tool(
                name,
                arguments,
                tool_use_id,
                request_params=request_params,
            )
        request_tool_handler = None
        if request_params and request_params.tool_execution_handler:
            request_tool_handler = request_params.tool_execution_handler
        return await self._aggregator.call_tool(
            name,
            arguments,
            tool_use_id,
            request_tool_handler=request_tool_handler,
        )

    async def _call_human_input_tool(
        self, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Handle human input via an elicitation form.

        Expected inputs:
        - Either an object with optional 'message' and a 'schema' JSON Schema (object), or
        - The JSON Schema (object) itself as the arguments.

        Constraints:
        - No more than 7 top-level properties are allowed in the schema.
        """
        try:
            # Run via shared tool runner
            resp_text = await run_elicitation_form(arguments or {}, agent_name=self._name)
            if resp_text == "__DECLINED__":
                return CallToolResult(
                    isError=False,
                    content=[TextContent(type="text", text="The Human declined the input request")],
                )
            if resp_text in ("__CANCELLED__", "__DISABLE_SERVER__"):
                return CallToolResult(
                    isError=False,
                    content=[
                        TextContent(type="text", text="The Human cancelled the input request")
                    ],
                )
            # Success path: return the (JSON) response as-is
            return CallToolResult(
                isError=False,
                content=[TextContent(type="text", text=resp_text)],
            )

        except PromptExitError:
            raise
        except asyncio.TimeoutError as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: Human input request timed out: {str(e)}",
                    )
                ],
            )
        except Exception as e:
            import traceback

            print(f"Error in _call_human_input_tool: {traceback.format_exc()}")
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Error requesting human input: {str(e)}")],
            )

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
        namespace: str | None = None,
        server_name: str | None = None,
    ) -> GetPromptResult:
        """
        Get a prompt from a server.

        Args:
            prompt_name: Name of the prompt, optionally namespaced
            arguments: Optional dictionary of arguments to pass to the prompt template
            namespace: Optional namespace (server) to get the prompt from

        Returns:
            GetPromptResult containing the prompt information
        """
        target = namespace if namespace is not None else server_name
        return await self._aggregator.get_prompt(prompt_name, arguments, target)

    async def apply_prompt(
        self,
        prompt: Union[str, GetPromptResult],
        arguments: dict[str, str] | None = None,
        as_template: bool = False,
        namespace: str | None = None,
        **_: Any,
    ) -> str:
        """
        Apply an MCP Server Prompt by name or GetPromptResult and return the assistant's response.
        Will search all available servers for the prompt if not namespaced and no server_name provided.

        If the last message in the prompt is from a user, this will automatically
        generate an assistant response to ensure we always end with an assistant message.

        Args:
            prompt: The name of the prompt to apply OR a GetPromptResult object
            arguments: Optional dictionary of string arguments to pass to the prompt template
            as_template: If True, store as persistent template (always included in context)
            namespace: Optional namespace/server to resolve the prompt from

        Returns:
            The assistant's response or error message
        """

        # Handle both string and GetPromptResult inputs
        if isinstance(prompt, str):
            prompt_name = prompt
            # Get the prompt - this will search all servers if needed
            self.logger.debug(f"Loading prompt '{prompt_name}'")
            prompt_result: GetPromptResult = await self.get_prompt(
                prompt_name, arguments, namespace
            )

            if not prompt_result or not prompt_result.messages:
                error_msg = f"Prompt '{prompt_name}' could not be found or contains no messages"
                self.logger.warning(error_msg)
                return error_msg

            # Get the display name (namespaced version)
            namespaced_name = getattr(prompt_result, "namespaced_name", prompt_name)
        else:
            # prompt is a GetPromptResult object
            prompt_result = prompt
            if not prompt_result or not prompt_result.messages:
                error_msg = "Provided GetPromptResult contains no messages"
                self.logger.warning(error_msg)
                return error_msg

            # Use a reasonable display name
            namespaced_name = getattr(prompt_result, "namespaced_name", "provided_prompt")

        self.logger.debug(f"Using prompt '{namespaced_name}'")

        # Convert prompt messages to multipart format using the safer method
        multipart_messages = PromptMessageExtended.from_get_prompt_result(prompt_result)

        if as_template:
            # Use apply_prompt_template to store as persistent prompt messages
            return await self.apply_prompt_template(prompt_result, namespaced_name)
        else:
            # Always call generate to ensure LLM implementations can handle prompt templates
            # This is critical for stateful LLMs like PlaybackLLM
            response = await self.generate(multipart_messages, None)
            return response.first_text()

    async def get_embedded_resources(
        self, resource_uri: str, server_name: str | None = None
    ) -> list[EmbeddedResource]:
        """
        Get a resource from an MCP server and return it as a list of embedded resources ready for use in prompts.

        Args:
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from

        Returns:
            List of EmbeddedResource objects ready to use in a PromptMessageExtended

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        # Get the raw resource result
        result: ReadResourceResult = await self._aggregator.get_resource(resource_uri, server_name)

        # Convert each resource content to an EmbeddedResource
        embedded_resources: list[EmbeddedResource] = []
        for resource_content in result.contents:
            embedded_resource = EmbeddedResource(
                type="resource", resource=resource_content, annotations=None
            )
            embedded_resources.append(embedded_resource)

        return embedded_resources

    async def get_resource(
        self, resource_uri: str, namespace: str | None = None, server_name: str | None = None
    ) -> ReadResourceResult:
        """
        Get a resource from an MCP server.

        Args:
            resource_uri: URI of the resource to retrieve
            namespace: Optional namespace (server) to retrieve the resource from

        Returns:
            ReadResourceResult containing the resource data

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        # Get the raw resource result
        target = namespace if namespace is not None else server_name
        result: ReadResourceResult = await self._aggregator.get_resource(resource_uri, target)
        return result

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageExtended],
        resource_uri: str,
        namespace: str | None = None,
        server_name: str | None = None,
    ) -> str:
        """
        Create a prompt with the given content and resource, then send it to the agent.

        Args:
            prompt_content: Content in various formats:
                - String: Converted to a user message with the text
                - PromptMessage: Converted to PromptMessageExtended
                - PromptMessageExtended: Used directly
            resource_uri: URI of the resource to retrieve
            namespace: Optional namespace (server) to retrieve the resource from

        Returns:
            The agent's response as a string
        """
        # Get the embedded resources
        embedded_resources: list[EmbeddedResource] = await self.get_embedded_resources(
            resource_uri, namespace if namespace is not None else server_name
        )

        # Create or update the prompt message
        prompt: PromptMessageExtended
        if isinstance(prompt_content, str):
            # Create a new prompt with the text and resources
            content = [TextContent(type="text", text=prompt_content)]
            content.extend(embedded_resources)
            prompt = PromptMessageExtended(role="user", content=content)
        elif isinstance(prompt_content, PromptMessage):
            # Convert PromptMessage to PromptMessageExtended and add resources
            content = [prompt_content.content]
            content.extend(embedded_resources)
            prompt = PromptMessageExtended(role=prompt_content.role, content=content)
        elif isinstance(prompt_content, PromptMessageExtended):
            # Add resources to the existing prompt
            prompt = prompt_content
            prompt.content.extend(embedded_resources)
        else:
            raise TypeError(
                "prompt_content must be a string, PromptMessage, or PromptMessageExtended"
            )

        response: PromptMessageExtended = await self.generate([prompt], None)
        return response.first_text()

    async def run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        """Override ToolAgent's run_tools to use MCP tools via aggregator."""
        if not request.tool_calls:
            self.logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}
        tool_timings: dict[str, ToolTimingInfo] = {}
        tool_loop_error: str | None = None

        # Cache available tool names exactly as advertised to the LLM for display/highlighting
        try:
            listed_tools = await self.list_tools()
        except Exception as exc:  # pragma: no cover - defensive guard, should not happen
            self.logger.warning(f"Failed to list tools before execution: {exc}")
            listed_tools = ListToolsResult(tools=[])

        available_tools: list[str] = []
        seen_tool_names: set[str] = set()
        for tool_schema in listed_tools.tools:
            if tool_schema.name in seen_tool_names:
                continue
            available_tools.append(tool_schema.name)
            seen_tool_names.add(tool_schema.name)

        # Cache namespaced tools for routing/metadata
        namespaced_tools = self._aggregator._namespaced_tool_map

        tool_call_items = list(request.tool_calls.items())
        should_parallel = should_parallelize_tool_calls(len(tool_call_items))
        if should_parallel and tool_call_items:
            subagent_calls = self._count_agent_tool_calls(tool_call_items)
            if subagent_calls > 1:
                did_close = self.close_active_streaming_display(
                    reason="parallel subagent tool calls"
                )
                if did_close:
                    self.logger.info(
                        "Closing streaming display due to parallel subagent tool calls",
                        agent_name=self._name,
                        tool_call_count=len(tool_call_items),
                        subagent_call_count=subagent_calls,
                    )

        smart_parallel_calls = 0
        if getattr(self, "agent_type", None) == AgentType.SMART:
            smart_parallel_calls = sum(
                1
                for _, tool_request in tool_call_items
                if tool_request.params.name == "smart"
            )

        planned_calls: list[dict[str, Any]] = []

        # Plan each tool call using our aggregator
        for correlation_id, tool_request in tool_call_items:
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}
            # correlation_id is the tool_use_id from the LLM

            # Determine which tool we are calling (namespaced MCP, local, etc.)
            namespaced_tool = namespaced_tools.get(tool_name)
            local_tool = self._execution_tools.get(tool_name)
            candidate_namespaced_tool = None
            if namespaced_tool is None and local_tool is None:
                candidate_namespaced_tool = next(
                    (
                        candidate
                        for candidate in namespaced_tools.values()
                        if candidate.tool.name == tool_name
                    ),
                    None,
                )

            # Select display/highlight names
            active_namespaced = namespaced_tool or candidate_namespaced_tool
            if active_namespaced is not None:
                display_tool_name = active_namespaced.namespaced_tool_name
            else:
                display_tool_name = tool_name

            # Check if tool is available from various sources
            is_external_runtime_tool = (
                self._external_runtime
                and hasattr(self._external_runtime, "tool")
                and self._external_runtime.tool
                and tool_name == self._external_runtime.tool.name
            )
            is_filesystem_runtime_tool = (
                self._filesystem_runtime
                and hasattr(self._filesystem_runtime, "tools")
                and any(tool.name == tool_name for tool in self._filesystem_runtime.tools)
            )
            is_skill_reader_tool = (
                self._skill_reader and self._skill_reader.enabled and tool_name == "read_skill"
            )

            tool_available = (
                tool_name == HUMAN_INPUT_TOOL_NAME
                or (
                    self._shell_runtime
                    and self._shell_runtime.tool
                    and tool_name == self._shell_runtime.tool.name
                )
                or is_external_runtime_tool
                or is_filesystem_runtime_tool
                or is_skill_reader_tool
                or namespaced_tool is not None
                or local_tool is not None
                or candidate_namespaced_tool is not None
            )

            if not tool_available:
                error_message = f"Tool '{display_tool_name}' is not available"
                self.logger.error(error_message)
                tool_loop_error = self._mark_tool_loop_error(
                    correlation_id=correlation_id,
                    error_message=error_message,
                    tool_results=tool_results,
                    tool_call_id=correlation_id if should_parallel else None,
                )
                break

            metadata: dict[str, Any] | None = None
            if (
                self._shell_runtime_enabled
                and self._shell_runtime
                and self._shell_runtime.tool
                and tool_name == self._shell_runtime.tool.name
            ):
                metadata = self._shell_runtime.metadata(tool_args.get("command"))
            elif is_external_runtime_tool and hasattr(self._external_runtime, "metadata"):
                metadata = self._external_runtime.metadata()
            elif is_filesystem_runtime_tool and hasattr(self._filesystem_runtime, "metadata"):
                metadata = self._filesystem_runtime.metadata()

            display_tool_name, bottom_items, highlight_index = self._prepare_tool_display(
                tool_name=tool_name,
                namespaced_tool=namespaced_tool,
                candidate_namespaced_tool=candidate_namespaced_tool,
                local_tool=local_tool,
                fallback_order=self._unique_preserving_order(available_tools),
            )

            self.display.show_tool_call(
                name=self._name,
                tool_args=tool_args,
                bottom_items=bottom_items,
                tool_name=display_tool_name,
                highlight_index=highlight_index,
                max_item_length=12,
                metadata=metadata,
                tool_call_id=correlation_id if should_parallel else None,
                show_hook_indicator=self.has_before_tool_call_hook,
            )

            planned_calls.append(
                {
                    "correlation_id": correlation_id,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "display_tool_name": display_tool_name,
                    "namespaced_tool": namespaced_tool,
                    "candidate_namespaced_tool": candidate_namespaced_tool,
                }
            )

        if should_parallel and planned_calls:
            smart_parallel_active = smart_parallel_calls > 1
            previous_shell_tool_call_id_setting = self._show_shell_tool_call_id
            previous_shell_display_setting = self._defer_shell_display_to_tool_result
            self._show_shell_tool_call_id = True
            self._defer_shell_display_to_tool_result = True
            if smart_parallel_active:
                setattr(self, "_parallel_smart_tool_calls", True)
                self.logger.info(
                    "Parallel smart tool calls detected",
                    agent_name=self._name,
                    tool_call_count=len(tool_call_items),
                    smart_call_count=smart_parallel_calls,
                )
            try:

                async def run_one(call: dict[str, Any]) -> tuple[str, CallToolResult, float]:
                    start_time = time.perf_counter()
                    result = await self.call_tool(
                        call["tool_name"],
                        call["tool_args"],
                        call["correlation_id"],
                        request_params=request_params,
                    )
                    end_time = time.perf_counter()
                    return call["correlation_id"], result, round((end_time - start_time) * 1000, 2)

                results = await gather_with_cancel(run_one(call) for call in planned_calls)
            finally:
                self._show_shell_tool_call_id = previous_shell_tool_call_id_setting
                self._defer_shell_display_to_tool_result = previous_shell_display_setting
                if smart_parallel_active:
                    setattr(self, "_parallel_smart_tool_calls", False)

            for i, item in enumerate(results):
                call = planned_calls[i]
                correlation_id = call["correlation_id"]
                display_tool_name = call["display_tool_name"]
                namespaced_tool = call["namespaced_tool"]
                candidate_namespaced_tool = call["candidate_namespaced_tool"]

                if isinstance(item, BaseException):
                    self.logger.error(f"MCP tool {display_tool_name} failed: {item}")
                    result = CallToolResult(
                        content=[TextContent(type="text", text=f"Error: {str(item)}")],
                        isError=True,
                    )
                    duration_ms = 0.0
                else:
                    _, result, duration_ms = item

                tool_results[correlation_id] = result
                tool_timings[correlation_id] = ToolTimingInfo(
                    timing_ms=duration_ms,
                    transport_channel=getattr(result, "transport_channel", None),
                )

                skybridge_config = None
                skybridge_tool = namespaced_tool or candidate_namespaced_tool
                if skybridge_tool:
                    try:
                        skybridge_config = await self._aggregator.get_skybridge_config(
                            skybridge_tool.server_name
                        )
                    except Exception:
                        skybridge_config = None

                if not getattr(result, "_suppress_display", False):
                    self.display.show_tool_result(
                        name=self._name,
                        result=result,
                        tool_name=display_tool_name,
                        skybridge_config=skybridge_config,
                        timing_ms=duration_ms,
                        tool_call_id=correlation_id,
                        show_hook_indicator=self.has_after_tool_call_hook,
                    )

            return self._finalize_tool_results(
                tool_results, tool_timings=tool_timings, tool_loop_error=tool_loop_error
            )

        for call in planned_calls:
            correlation_id = call["correlation_id"]
            tool_name = call["tool_name"]
            tool_args = call["tool_args"]
            display_tool_name = call["display_tool_name"]
            namespaced_tool = call["namespaced_tool"]
            candidate_namespaced_tool = call["candidate_namespaced_tool"]

            try:
                start_time = time.perf_counter()
                result = await self.call_tool(
                    tool_name, tool_args, correlation_id, request_params=request_params
                )
                end_time = time.perf_counter()
                duration_ms = round((end_time - start_time) * 1000, 2)

                tool_results[correlation_id] = result
                tool_timings[correlation_id] = ToolTimingInfo(
                    timing_ms=duration_ms,
                    transport_channel=getattr(result, "transport_channel", None),
                )

                skybridge_config = None
                skybridge_tool = namespaced_tool or candidate_namespaced_tool
                if skybridge_tool:
                    skybridge_config = await self._aggregator.get_skybridge_config(
                        skybridge_tool.server_name
                    )

                if not getattr(result, "_suppress_display", False):
                    self.display.show_tool_result(
                        name=self._name,
                        result=result,
                        tool_name=display_tool_name,
                        skybridge_config=skybridge_config,
                        timing_ms=duration_ms,
                        show_hook_indicator=self.has_after_tool_call_hook,
                    )

                self.logger.debug(f"MCP tool {display_tool_name} executed successfully")
            except Exception as e:
                self.logger.error(f"MCP tool {display_tool_name} failed: {e}")
                error_result = CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True,
                )
                tool_results[correlation_id] = error_result
                self.display.show_tool_result(
                    name=self._name,
                    result=error_result,
                    show_hook_indicator=self.has_after_tool_call_hook,
                )

        return self._finalize_tool_results(
            tool_results, tool_timings=tool_timings, tool_loop_error=tool_loop_error
        )

    def _prepare_tool_display(
        self,
        *,
        tool_name: str,
        namespaced_tool: "NamespacedTool | None",
        candidate_namespaced_tool: "NamespacedTool | None",
        local_tool: Any | None,
        fallback_order: list[str],
    ) -> tuple[str, list[str] | None, int | None]:
        """
        Determine how we present tool metadata for the console display.

        Returns a tuple of (display_tool_name, bottom_items, highlight_index).
        """
        active_namespaced = namespaced_tool or candidate_namespaced_tool
        display_tool_name = (
            active_namespaced.namespaced_tool_name if active_namespaced is not None else tool_name
        )

        bottom_items: list[str] | None = None
        highlight_target: str | None = None

        if active_namespaced is not None:
            server_tools = self._aggregator._server_to_tool_map.get(
                active_namespaced.server_name, []
            )
            if server_tools:
                bottom_items = self._unique_preserving_order(
                    tool_entry.tool.name for tool_entry in server_tools
                )
            highlight_target = active_namespaced.tool.name
        elif local_tool is not None:
            bottom_items = self._unique_preserving_order(self._execution_tools.keys())
            highlight_target = tool_name
        elif tool_name == HUMAN_INPUT_TOOL_NAME:
            bottom_items = [HUMAN_INPUT_TOOL_NAME]
            highlight_target = HUMAN_INPUT_TOOL_NAME

        highlight_index: int | None = None
        if bottom_items and highlight_target:
            try:
                highlight_index = bottom_items.index(highlight_target)
            except ValueError:
                highlight_index = None

        if bottom_items is None and fallback_order:
            bottom_items = fallback_order
            fallback_target = display_tool_name if display_tool_name in bottom_items else tool_name
            try:
                highlight_index = bottom_items.index(fallback_target)
            except ValueError:
                highlight_index = None

        if bottom_items is not None:
            bottom_items = [TOOL_DISPLAY_NAMES.get(name, name) for name in bottom_items]
        return display_tool_name, bottom_items, highlight_index

    @staticmethod
    def _unique_preserving_order(items: Iterable[str]) -> list[str]:
        """Return a list of unique items while preserving original order."""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    async def apply_prompt_template(self, prompt_result: GetPromptResult, prompt_name: str) -> str:
        """
        Apply a prompt template as persistent context that will be included in all future conversations.
        Delegates to the attached LLM.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated
        """
        assert self._llm
        with self._tracer.start_as_current_span(f"Agent: '{self._name}' apply_prompt_template"):
            return await self._llm.apply_prompt_template(prompt_result, prompt_name)

    async def apply_prompt_messages(
        self, prompts: list[PromptMessageExtended], request_params: RequestParams | None = None
    ) -> str:
        """
        Apply a list of prompt messages and return the result.

        Args:
            prompts: List of PromptMessageExtended messages
            request_params: Optional request parameters

        Returns:
            The text response from the LLM
        """

        response = await self.generate(prompts, request_params)
        return response.first_text()

    async def list_prompts(
        self, namespace: str | None = None, server_name: str | None = None
    ) -> Mapping[str, list[mcp.types.Prompt]]:
        """
        List all prompts available to this agent, filtered by configuration.

        Args:
            namespace: Optional namespace (server) to list prompts from

        Returns:
            Dictionary mapping server names to lists of Prompt objects
        """
        # Get all prompts from the aggregator
        target = namespace if namespace is not None else server_name
        result = await self._aggregator.list_prompts(target)

        return self._filter_server_collections(
            result,
            self.config.prompts,
            lambda prompt: prompt.name,
        )

    async def list_resources(
        self, namespace: str | None = None, server_name: str | None = None
    ) -> dict[str, list[str]]:
        """
        List all resources available to this agent, filtered by configuration.

        Args:
            namespace: Optional namespace (server) to list resources from

        Returns:
            Dictionary mapping server names to lists of resource URIs
        """
        # Get all resources from the aggregator
        target = namespace if namespace is not None else server_name
        result = await self._aggregator.list_resources(target)

        return self._filter_server_collections(
            result,
            self.config.resources,
            lambda resource: resource,
        )

    async def list_mcp_tools(self, namespace: str | None = None) -> Mapping[str, list[Tool]]:
        """
        List all tools available to this agent, grouped by server and filtered by configuration.

        Args:
            namespace: Optional namespace (server) to list tools from

        Returns:
            Dictionary mapping server names to lists of Tool objects (with original names, not namespaced)
        """
        # Get all tools from the aggregator
        result = await self._aggregator.list_mcp_tools(namespace)
        filtered_result: dict[str, list[Tool]] = {}

        for server, server_tools in result.items():
            filtered_result[server] = self._filter_server_tools(server_tools, server)

        # Add elicitation-backed human input tool to a special server if enabled and available
        if self.config.human_input and self._human_input_tool:
            special_server_name = "__human_input__"
            filtered_result.setdefault(special_server_name, []).append(self._human_input_tool)

        return filtered_result

    async def list_tools(self) -> ListToolsResult:
        """
        List all tools available to this agent, filtered by configuration.

        Returns:
            ListToolsResult with available tools
        """
        # Start with filtered aggregator tools and merge in subclass/local tools
        merged_tools: list[Tool] = await self._get_filtered_mcp_tools()
        existing_names = {tool.name for tool in merged_tools}

        local_tools = (await super().list_tools()).tools
        for tool in local_tools:
            if tool.name not in existing_names:
                merged_tools.append(tool)
                existing_names.add(tool.name)

        # Add external runtime tool (e.g., ACP terminal) if available, otherwise bash tool
        if self._external_runtime and hasattr(self._external_runtime, "tool"):
            external_tool = self._external_runtime.tool
            if external_tool and external_tool.name not in existing_names:
                merged_tools.append(external_tool)
                existing_names.add(external_tool.name)
        elif self._bash_tool and self._bash_tool.name not in existing_names:
            merged_tools.append(self._bash_tool)
            existing_names.add(self._bash_tool.name)

        # Add filesystem runtime tools (e.g., ACP filesystem) if available
        if self._filesystem_runtime and hasattr(self._filesystem_runtime, "tools"):
            for fs_tool in self._filesystem_runtime.tools:
                if fs_tool and fs_tool.name not in existing_names:
                    merged_tools.append(fs_tool)
                    existing_names.add(fs_tool.name)
        elif self._skill_reader and self._skill_reader.enabled:
            # Non-ACP context with skills: provide read_skill tool
            skill_tool = self._skill_reader.tool
            if skill_tool.name not in existing_names:
                merged_tools.append(skill_tool)
                existing_names.add(skill_tool.name)

        if self.config.human_input:
            human_tool = getattr(self, "_human_input_tool", None)
            if human_tool and human_tool.name not in existing_names:
                merged_tools.append(human_tool)
                existing_names.add(human_tool.name)

        return ListToolsResult(tools=merged_tools)

    @property
    def agent_type(self) -> AgentType:
        """
        Return the type of this agent.
        """
        return AgentType.BASIC

    async def agent_card(self) -> AgentCard:
        """
        Return an A2A card describing this Agent
        """

        skills: list[AgentSkill] = []
        tools: ListToolsResult = await self.list_tools()
        for tool in tools.tools:
            skills.append(await self.convert(tool))

        return AgentCard(
            skills=skills,
            name=self._name,
            description=self.config.description or self.instruction,
            url=f"fast-agent://agents/{self._name}/",
            version="0.1",
            capabilities=DEFAULT_CAPABILITIES,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            provider=None,
            documentation_url=None,
        )

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_items: str | list[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Union["Text", None] = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
    ) -> None:
        """
        Display an assistant message with MCP servers in the bottom bar.

        This override adds the list of connected MCP servers to the bottom bar
        and highlights servers that were used for tool calls in this message.
        """
        # Get the list of MCP servers (if not provided)
        if bottom_items is None:
            if self._aggregator and self._aggregator.server_names:
                server_names = list(self._aggregator.server_names)
            else:
                server_names = []
        else:
            server_names = list(bottom_items)

        server_names = self._unique_preserving_order(server_names)

        shell_label = self._shell_server_label()
        if shell_label:
            server_names = [shell_label, *(name for name in server_names if name != shell_label)]

        skills_label = self._skills_tool_label()
        if skills_label and skills_label not in server_names:
            server_names.append(skills_label)

        card_tools_label = self._card_tools_label()
        if card_tools_label and card_tools_label not in server_names:
            if skills_label and skills_label in server_names:
                insert_at = server_names.index(skills_label) + 1
                server_names.insert(insert_at, card_tools_label)
            else:
                server_names.append(card_tools_label)

        # Add agent-as-tool names to the bottom bar (they aren't MCP servers but should be shown)
        for tool_name in self._agent_tools:
            # Extract the agent name from tool_name (e.g., "agent__foo" -> "foo")
            agent_label = tool_name[7:] if tool_name.startswith("agent__") else tool_name
            if agent_label not in server_names:
                server_names.append(agent_label)

        # Also check _child_agents (used by AgentsAsToolsAgent)
        # Import at runtime to avoid circular import
        from fast_agent.agents.workflow.agents_as_tools_agent import AgentsAsToolsAgent

        if isinstance(self, AgentsAsToolsAgent):
            for agent_name in self._child_agents:
                agent_label = agent_name[7:] if agent_name.startswith("agent__") else agent_name
                if agent_label not in server_names:
                    server_names.append(agent_label)

        # Extract servers from tool calls in the message for highlighting
        if highlight_items is None:
            highlight_servers = self._extract_servers_from_message(message)
        else:
            # Convert to list if needed
            if isinstance(highlight_items, str):
                highlight_servers = [highlight_items]
            else:
                highlight_servers = list(highlight_items)

        if card_tools_label and self._card_tools_used(message):
            if card_tools_label not in highlight_servers:
                highlight_servers.append(card_tools_label)

        # Call parent's implementation with server information
        await super().show_assistant_message(
            message=message,
            bottom_items=server_names,
            highlight_items=highlight_servers,
            max_item_length=max_item_length or 12,
            name=name,
            model=model,
            additional_message=additional_message,
            render_markdown=render_markdown,
            show_hook_indicator=show_hook_indicator,
        )

    def _extract_servers_from_message(self, message: PromptMessageExtended) -> list[str]:
        """
        Extract server names from tool calls in the message.

        Args:
            message: The message containing potential tool calls

        Returns:
            List of server names that were called
        """
        servers: list[str] = []

        # Check if message has tool calls
        if message.tool_calls:
            for tool_request in message.tool_calls.values():
                tool_name = tool_request.params.name

                if tool_name in self._agent_tools:
                    agent_label = tool_name[7:] if tool_name.startswith("agent__") else tool_name
                    if agent_label not in servers:
                        servers.append(agent_label)
                    continue

                child_agents = getattr(self, "_child_agents", None)
                if isinstance(child_agents, dict) and tool_name in child_agents:
                    agent_label = tool_name[7:] if tool_name.startswith("agent__") else tool_name
                    if agent_label not in servers:
                        servers.append(agent_label)
                    continue

                if (
                    self._shell_runtime_enabled
                    and self._shell_runtime
                    and self._shell_runtime.tool
                    and tool_name == self._shell_runtime.tool.name
                ):
                    shell_label = self._shell_server_label()
                    if shell_label and shell_label not in servers:
                        servers.append(shell_label)
                    continue

                # Check if this is a skills reader tool call
                if self._skill_reader and self._skill_reader.enabled:
                    if tool_name == self._skill_reader.tool.name:
                        skills_label = self._skills_tool_label()
                        if skills_label and skills_label not in servers:
                            servers.append(skills_label)
                        continue

                # Check if this is a skills reader tool call
                if self._skill_reader and self._skill_reader.enabled:
                    if tool_name == self._skill_reader.tool.name:
                        skills_label = self._skills_tool_label()
                        if skills_label and skills_label not in servers:
                            servers.append(skills_label)
                        continue

                # Use aggregator's mapping to find the server for this tool
                if tool_name in self._aggregator._namespaced_tool_map:
                    namespaced_tool = self._aggregator._namespaced_tool_map[tool_name]
                    if namespaced_tool.server_name not in servers:
                        servers.append(namespaced_tool.server_name)

        return servers

    def _shell_server_label(self) -> str | None:
        """Return the display label for the local shell runtime."""
        shell_runtime = self._shell_runtime
        if not self._shell_runtime_enabled or not shell_runtime or not shell_runtime.tool:
            return None

        runtime_info = shell_runtime.runtime_info()
        runtime_name = runtime_info.get("name")
        return runtime_name or "shell"

    def _skills_tool_label(self) -> str | None:
        if self._skill_reader and self._skill_reader.enabled:
            name = self._skill_reader.tool.name
            return TOOL_DISPLAY_NAMES.get(name, name)
        return None

    async def _parse_resource_name(self, name: str, resource_type: str) -> tuple[str | None, str]:
        """Delegate resource name parsing to the aggregator."""
        return await self._aggregator._parse_resource_name(name, resource_type)

    async def convert(self, tool: Tool) -> AgentSkill:
        """
        Convert a Tool to an AgentSkill.
        """

        if tool.name in self._skill_map:
            manifest = self._skill_map[tool.name]
            return AgentSkill(
                id=f"skill:{manifest.name}",
                name=manifest.name,
                description=manifest.description or "",
                tags=["skill"],
                examples=None,
                input_modes=None,
                output_modes=None,
            )

        _, tool_without_namespace = await self._parse_resource_name(tool.name, "tool")
        return AgentSkill(
            id=tool.name,
            name=tool_without_namespace,
            description=tool.description or "",
            tags=["tool"],
            examples=None,
            input_modes=None,  # ["text/plain"],
            # cover TextContent | ImageContent ->
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/223
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/93
            output_modes=None,  # ,["text/plain", "image/*"],
        )

    @property
    def message_history(self) -> list[PromptMessageExtended]:
        """
        Return the agent's message history as PromptMessageExtended objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageExtended objects representing the conversation history
        """
        # Conversation history is maintained at the agent layer; LLM history is diagnostic only.
        return super().message_history

    @property
    def usage_accumulator(self) -> Union["UsageAccumulator", None]:
        """
        Return the usage accumulator for tracking token usage across turns.

        Returns:
            UsageAccumulator object if LLM is attached, None otherwise
        """
        if self.llm:
            return self.llm.usage_accumulator
        return None
