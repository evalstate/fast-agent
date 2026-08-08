"""
Base Agent class that implements the AgentProtocol interface.

This class provides default implementations of the standard agent methods
and delegates operations to an attached FastAgentLLMProtocol instance.
"""

import asyncio
import fnmatch
import re
import time
from abc import ABC
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
)

import mcp_types
from a2a.types import AgentCard, AgentSkill
from mcp_types import (
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    GetPromptResult,
    ListToolsResult,
    PromptMessage,
    ReadResourceResult,
    TextContent,
    Tool,
)
from pydantic import BaseModel

from fast_agent.agents.agent_card import build_fast_agent_card
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.mcp_tool_planning import (
    PlannedMcpToolCall,
    build_mcp_tool_route,
    listed_tool_names,
)
from fast_agent.agents.mcp_tool_presentation import (
    attach_read_text_file_display_metadata,
    build_mcp_tool_presentation,
    tool_result_type_label,
    unique_preserving_order,
)
from fast_agent.agents.subagent_directive import resolve_subagent_directive
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.commands.model_capabilities import (
    resolve_model_name,
    resolve_model_params,
    resolve_resolved_model,
)
from fast_agent.config import MCPServerSettings, ShellSettings
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    HUMAN_INPUT_TOOL_NAME,
    should_parallelize_tool_calls,
)
from fast_agent.core.exceptions import AgentConfigError, ModelConfigError, PromptExitError
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.resolved_model import ResolvedModelSpec
from fast_agent.llm.terminal_output_limits import (
    calculate_terminal_output_limit_for_max_tokens,
    calculate_terminal_output_limit_for_model,
    calculate_terminal_output_limit_for_resolved_model,
)
from fast_agent.mcp.app_integrations import AppServerConfig
from fast_agent.mcp.common import (
    create_namespaced_name,
    get_resource_name,
    get_server_name,
    is_namespaced_name,
)
from fast_agent.mcp.mcp_aggregator import (
    MCPAggregator,
    MCPAttachOptions,
    MCPAttachResult,
    MCPDetachResult,
    MCPToolCatalog,
    NamespacedTool,
    ServerStatus,
)
from fast_agent.mcp.prompt_metadata import prompt_display_name
from fast_agent.mcp.provider_management import (
    ProviderManagedMCPState,
    build_provider_managed_mcp_state,
    split_managed_server_names,
)
from fast_agent.mcp.tool_result_metadata import tool_result_display_metadata
from fast_agent.mcp.tool_result_truncation import truncate_tool_result_for_llm
from fast_agent.skills import SKILLS_DEFAULT, SkillManifest
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.apply_patch_tool import APPLY_PATCH_TOOL_NAME
from fast_agent.tools.composite_filesystem_runtime import CompositeFilesystemRuntime
from fast_agent.tools.edit_file_tool import EDIT_FILE_TOOL_NAME
from fast_agent.tools.elicitation import (
    get_elicitation_tool,
    run_elicitation_form,
    set_elicitation_input_callback,
)
from fast_agent.tools.environment_filesystem_runtime import EnvironmentFilesystemRuntime
from fast_agent.tools.execution_environment import (
    EnvironmentFilesystem,
    EnvironmentTemporaryArtifacts,
)
from fast_agent.tools.external_runtime_protocol import ExternalRuntime
from fast_agent.tools.filesystem_runtime_protocol import FilesystemRuntime
from fast_agent.tools.filesystem_tool_definitions import (
    READ_TEXT_FILE_TOOL_NAME,
    WRITE_TEXT_FILE_TOOL_NAME,
)
from fast_agent.tools.local_filesystem_runtime import LocalFilesystemRuntime
from fast_agent.tools.shell_profiles import ResolvedShellToolProfile, ShellToolProfile
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.tools.skill_reader import READ_SKILL_TOOL_NAME, SkillReader
from fast_agent.types import (
    PromptMessageExtended,
    RequestParams,
    ToolTimingInfo,
)
from fast_agent.ui import console
from fast_agent.ui.shell_notice import format_shell_notice
from fast_agent.ui.tool_display import ToolCallDisplayRequest, ToolResultDisplayRequest
from fast_agent.utils.async_utils import gather_with_cancel
from fast_agent.utils.text import strip_casefold, strip_to_none
from fast_agent.utils.tool_names import BASH_TOOL_NAME, is_read_text_file_tool_name

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)
ItemT = TypeVar("ItemT")

LLM = TypeVar("LLM", bound=FastAgentLLMProtocol)

# Display name overrides for tools shown in the bottom bar
TOOL_DISPLAY_NAMES: dict[str, str] = {
    READ_SKILL_TOOL_NAME: "skill",
}


class ShellEditToolMode(StrEnum):
    WRITE_TEXT_FILE = WRITE_TEXT_FILE_TOOL_NAME
    EDIT_FILE = EDIT_FILE_TOOL_NAME
    APPLY_PATCH = APPLY_PATCH_TOOL_NAME
    OFF = "off"


@dataclass(frozen=True, slots=True)
class ShellEditToolFlags:
    write_text_file: bool
    apply_patch: bool
    edit_file: bool

    @classmethod
    def from_mode(cls, mode: ShellEditToolMode) -> "ShellEditToolFlags":
        write_text_file = mode is ShellEditToolMode.WRITE_TEXT_FILE
        return cls(
            write_text_file=write_text_file,
            apply_patch=mode is ShellEditToolMode.APPLY_PATCH,
            edit_file=write_text_file or mode is ShellEditToolMode.EDIT_FILE,
        )


@dataclass(frozen=True, slots=True)
class _ShellRuntimeSettings:
    timeout_seconds: int
    warning_interval_seconds: int
    output_byte_limit: int
    process_poll_default_wait_seconds: int
    tool_profile: ShellToolProfile
    model_tool_profile: ResolvedShellToolProfile | None


@dataclass(frozen=True, slots=True)
class _ManagedMcpSetup:
    provider_state: ProviderManagedMCPState
    client_managed_servers: list[str]
    provider_managed_servers: list[str]


if TYPE_CHECKING:
    from rich.text import Text

    from fast_agent.agents.llm_decorator import LlmDecorator
    from fast_agent.context import Context
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.tools.execution_environment import ShellEnvironment


def _effective_configured_servers(
    config: AgentConfig,
    context: "Context | None",
) -> tuple[str, ...]:
    runtime_servers = (
        context.runtime_mcp_server_names.get(config.name, ()) if context is not None else ()
    )
    return tuple(dict.fromkeys((*config.servers, *runtime_servers)))


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
        shell_environment: "ShellEnvironment | None" = None,
        **kwargs,
    ) -> None:
        self._shell_environment = shell_environment
        super().__init__(
            config=config,
            context=context,
            **kwargs,
        )

        configured_servers = _effective_configured_servers(self.config, context)
        managed_mcp = self._managed_mcp_setup(configured_servers, context)
        self._provider_managed_mcp_state = managed_mcp.provider_state
        self._configured_server_names = tuple(configured_servers)
        self._provider_managed_server_keys = tuple(managed_mcp.provider_managed_servers)

        # Create aggregator with composition
        self._aggregator = MCPAggregator(
            server_names=managed_mcp.client_managed_servers,
            connection_persistence=connection_persistence,
            name=self.config.name,
            context=context,
            config=self.config,  # Pass the full config for access to elicitation_handler
            **kwargs,
        )
        self._provider_managed_server_names = tuple(
            self._aggregator.server_display_name(name)
            for name in self._provider_managed_server_keys
        )
        self._aggregator.set_supplemental_attached_servers(self._provider_managed_server_names)

        # Store the original template - resolved instruction set after build()
        self._instruction_template = self.config.instruction
        self._instruction = self.config.instruction  # Will be replaced by builder output
        self._subagent_directive_found = False
        self.executor = context.executor if context else None
        self.logger = get_logger(f"{__name__}.{self._name}")
        manifests = self._initial_skill_manifests(context)

        self._initialize_runtime_slots(context)
        self._skill_manifests: list[SkillManifest] = []
        self._skill_map: dict[str, SkillManifest] = {}
        self._skill_reader: SkillReader | None = None
        self.set_skill_manifests(manifests)
        self.skill_registry = self._resolve_skill_registry(context)
        self._warnings: list[str] = []
        self._warning_messages_seen: set[str] = set()
        self._configure_initial_shell_runtime(context)

        # Store instruction context for template resolution
        self._instruction_context: dict[str, str] = {}

        self._allow_shell_notice = True

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # set with the "attach" method
        self._llm: FastAgentLLMProtocol | None = None

        # Instantiate human input tool once if enabled in config
        self._human_input_tool = self._initial_human_input_tool()

        # Register the interactive elicitation handler so local tools can call it
        # without importing MCP types. This avoids circular imports and ensures the callback is ready.
        self._register_mcp_elicitation_adapter()

    def _managed_mcp_setup(
        self,
        configured_servers: tuple[str, ...],
        context: "Context | None",
    ) -> _ManagedMcpSetup:
        server_settings_by_name = None
        if context and context.config and context.config.mcp:
            server_settings_by_name = context.config.mcp.servers

        if server_settings_by_name is None:
            return _ManagedMcpSetup(
                provider_state=ProviderManagedMCPState(),
                client_managed_servers=list(configured_servers),
                provider_managed_servers=[],
            )

        managed_server_names = split_managed_server_names(
            configured_servers,
            server_settings_by_name,
        )
        return _ManagedMcpSetup(
            provider_state=build_provider_managed_mcp_state(
                agent_config=self.config,
                server_settings_by_name=server_settings_by_name,
            ),
            client_managed_servers=managed_server_names.client_managed,
            provider_managed_servers=managed_server_names.provider_managed,
        )

    def _initial_skill_manifests(self, context: "Context | None") -> list[SkillManifest]:
        manifests: list[SkillManifest] = list(self.config.skill_manifests or [])
        if self.config.skills_resolved_for_run:
            return manifests
        if self.config.skills is not SKILLS_DEFAULT or manifests:
            return manifests
        if not context or not context.skill_registry:
            return []

        try:
            return list(context.skill_registry.load_manifests())
        except Exception:
            return []

    def _initialize_runtime_slots(self, context: "Context | None") -> None:
        self._shell_runtime: ShellRuntime | None = None
        self._shell_notice_emitted = False
        self._allow_shell_notice = False
        self._shell_runtime_enabled = False
        self._show_shell_tool_call_id = False
        self._defer_shell_display_to_tool_result = False
        self._shell_access_modes: tuple[str, ...] = ()
        self._bash_tool: Tool | None = None
        self._external_runtime: ExternalRuntime | None = None
        self._filesystem_runtime: FilesystemRuntime | None = None
        self._no_shell_requested = bool(context and getattr(context, "no_shell", False))
        self._shell_runtime_activation_reason: str | None = None

    def _resolve_skill_registry(self, context: "Context | None") -> SkillRegistry | None:
        if isinstance(self.config.skills, SkillRegistry):
            return self.config.skills
        if self.config.skills is SKILLS_DEFAULT and context and context.skill_registry:
            return context.skill_registry
        return None

    def _configure_initial_shell_runtime(self, context: "Context | None") -> None:
        shell_flag_requested = (
            bool(context and getattr(context, "shell_runtime", False))
            and not self._no_shell_requested
        )
        shell_config_requested = bool(self.config.shell) and not self._no_shell_requested
        skills_configured = bool(self._skill_manifests)
        self._shell_runtime_activation_reason = self._shell_activation_reason(
            shell_flag_requested=shell_flag_requested,
            shell_config_requested=shell_config_requested,
            skills_configured=skills_configured,
        )
        if self._shell_runtime_activation_reason is None:
            return

        self._shell_access_modes = self._shell_access_modes_for_activation(
            shell_flag_requested=shell_flag_requested,
            skills_configured=skills_configured,
        )
        self._activate_shell_runtime(
            self._shell_runtime_activation_reason,
            working_directory=self.config.cwd,
            access_modes=self._shell_access_modes,
        )

    def _shell_activation_reason(
        self,
        *,
        shell_flag_requested: bool,
        shell_config_requested: bool,
        skills_configured: bool,
    ) -> str | None:
        reasons: list[str] = []
        if shell_flag_requested:
            reasons.append("--shell flag")
        if shell_config_requested:
            reasons.append("agent config")
        if skills_configured and not self._no_shell_requested:
            reasons.append("agent skills configuration")
        if not reasons:
            return None
        if reasons == ["agent skills configuration"]:
            return "because agent skills are configured"
        return "via " + " and ".join(reasons)

    @staticmethod
    def _shell_access_modes_for_activation(
        *,
        shell_flag_requested: bool,
        skills_configured: bool,
    ) -> tuple[str, ...]:
        modes: list[str] = []
        if skills_configured:
            modes.append("skills")
        if shell_flag_requested:
            modes.append("switch")
        return tuple(modes)

    def _initial_human_input_tool(self) -> Tool | None:
        if not self.config.human_input:
            return None
        try:
            return get_elicitation_tool()
        except Exception:
            return None

    @staticmethod
    def _register_mcp_elicitation_adapter() -> None:
        try:

            async def _mcp_elicitation_adapter(
                request_payload: dict,
                agent_name: str | None = None,
                server_name: str | None = None,
                server_info: dict | None = None,
            ) -> str:
                from fast_agent.human_input.elicitation_handler import elicitation_input_callback
                from fast_agent.human_input.types import HumanInputRequest

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
        await self._close_transient_artifact_store()
        if self._shell_runtime is not None:
            await self._shell_runtime.close()
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
        if self._shutdown_complete:
            return
        await self._run_lifecycle_hook("on_shutdown")
        await self._close_transient_artifact_store()
        if self._shell_runtime is not None:
            await self._shell_runtime.close()
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
                shell_runtime.set_working_directory(working_directory)
                local_runtime = self._local_filesystem_runtime()
                if local_runtime is not None:
                    local_runtime.set_working_directory(working_directory)

            self._maybe_enable_local_filesystem_runtime(working_directory)
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
        status_map = await self._aggregator.collect_server_status()

        server_settings_by_name = None
        if self._context and self._context.config and self._context.config.mcp:
            server_settings_by_name = self._context.config.mcp.servers

        if not server_settings_by_name:
            return status_map

        auto_sampling = True
        if self._context and self._context.config and self._context.config.mcp:
            auto_sampling = self._context.config.mcp.client.auto_sampling

        for server_name in self._provider_managed_server_keys:
            if server_name in status_map:
                continue
            server_cfg = server_settings_by_name.get(server_name)
            if server_cfg is None:
                continue

            roots = server_cfg.roots
            elicitation = server_cfg.elicitation
            sampling_cfg = server_cfg.sampling
            status_map[server_name] = ServerStatus(
                server_name=server_name,
                transport=server_cfg.transport,
                is_connected=True,
                instructions_enabled=server_cfg.include_instructions,
                roots_configured=bool(roots),
                roots_count=len(roots) if roots else 0,
                elicitation_mode=elicitation.mode if elicitation else None,
                sampling_mode=(
                    "configured"
                    if sampling_cfg is not None
                    else ("auto" if auto_sampling else "off")
                ),
            )

        visible_status: dict[str, ServerStatus] = {}
        for server_name, status in status_map.items():
            visible_name = self._aggregator.server_display_name(server_name)
            visible_status[visible_name] = status.model_copy(update={"server_name": visible_name})
        return visible_status

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        resolved_server_config = server_config
        if (
            resolved_server_config is None
            and self._context
            and self._context.config
            and self._context.config.mcp
        ):
            resolved_server_config = self._context.config.mcp.servers.get(server_name)
        if resolved_server_config is not None and resolved_server_config.management == "provider":
            raise AgentConfigError(
                f"Provider-managed MCP server '{server_name}' cannot be attached locally."
            )
        return await self._aggregator.attach_server(
            server_name=server_name,
            server_config=server_config,
            options=options,
        )

    async def detach_mcp_server(self, server_name: str) -> MCPDetachResult:
        return await self._aggregator.detach_server(server_name)

    def list_attached_mcp_servers(self) -> list[str]:
        return unique_preserving_order(self._aggregator.list_attached_servers())

    async def list_servers(self) -> list[str]:
        return unique_preserving_order(
            [
                *(
                    self._aggregator.server_display_name(name)
                    for name in self._aggregator.server_names
                ),
                *self._provider_managed_server_names,
            ]
        )

    @property
    def aggregator(self) -> MCPAggregator:
        """Expose the MCP aggregator for UI integrations."""
        return self._aggregator

    @property
    def instruction_template(self) -> str:
        """The original instruction template with placeholders."""
        return self._instruction_template or ""

    def set_instruction_template(self, instruction: str) -> None:
        """Replace this instance's source instruction template."""
        self._instruction_template = instruction

    @property
    def subagent_directive_found(self) -> bool:
        return self._subagent_directive_found

    def process_rendered_instruction(self, instruction: str) -> str:
        """Record and hide built-in subagent directives after rendering."""
        directive = resolve_subagent_directive(instruction)
        self._subagent_directive_found |= directive.found
        if self.config.subagent_child:
            return directive.subagent_instruction
        return directive.instruction

    def _clone_config(self) -> AgentConfig:
        config = super()._clone_config()
        config.instruction = self.instruction_template
        return config

    def _clone_constructor_kwargs(self) -> dict[str, Any]:
        kwargs = super()._clone_constructor_kwargs()
        kwargs["shell_environment"] = self._shell_environment
        return kwargs

    def _temporary_artifact_environment(self) -> EnvironmentTemporaryArtifacts | None:
        environment = self._shell_environment
        if not isinstance(environment, EnvironmentTemporaryArtifacts):
            return None
        if not self._shell_runtime_enabled or self._external_runtime is not None:
            return None
        filesystem_runtime = self._filesystem_runtime
        if (
            filesystem_runtime is not None
            and filesystem_runtime is not self._environment_filesystem_runtime()
        ):
            return None
        return environment

    async def _configure_cloned_instance(self, clone: "LlmDecorator") -> None:
        await super()._configure_cloned_instance(clone)
        mcp_clone = cast("McpAgent", clone)
        attached = set(mcp_clone.list_attached_mcp_servers())
        for server_name in self.list_attached_mcp_servers():
            if server_name not in attached:
                await mcp_clone.attach_mcp_server(server_name=server_name)

    @property
    def instruction_context(self) -> dict[str, str]:
        """Context values for instruction template resolution."""
        return self._instruction_context

    @property
    def skill_manifests(self) -> list[SkillManifest]:
        """List of skill manifests configured for this agent."""
        return self._skill_manifests

    def _local_filesystem_runtime(self) -> LocalFilesystemRuntime | None:
        runtime = self._filesystem_runtime
        if isinstance(runtime, LocalFilesystemRuntime):
            return runtime
        if isinstance(runtime, CompositeFilesystemRuntime):
            primary = runtime.primary
            if isinstance(primary, LocalFilesystemRuntime):
                return primary
            fallback = runtime.fallback
            if isinstance(fallback, LocalFilesystemRuntime):
                return fallback
        return None

    def _environment_filesystem_runtime(self) -> EnvironmentFilesystemRuntime | None:
        runtime = self._filesystem_runtime
        if isinstance(runtime, EnvironmentFilesystemRuntime):
            return runtime
        if isinstance(runtime, CompositeFilesystemRuntime):
            primary = runtime.primary
            if isinstance(primary, EnvironmentFilesystemRuntime):
                return primary
            fallback = runtime.fallback
            if isinstance(fallback, EnvironmentFilesystemRuntime):
                return fallback
        return None

    def _drop_local_filesystem_runtime(self) -> None:
        local_runtime = self._local_filesystem_runtime()
        if local_runtime is None:
            return
        runtime = self._filesystem_runtime
        if runtime is local_runtime:
            self._filesystem_runtime = None
            return
        if isinstance(runtime, CompositeFilesystemRuntime):
            if runtime.primary is local_runtime:
                self._filesystem_runtime = runtime.fallback
            elif runtime.fallback is local_runtime:
                self._filesystem_runtime = runtime.primary

    def _consume_pending_media_attachments(self) -> list[ContentBlock]:
        local_runtime = self._local_filesystem_runtime()
        if local_runtime is not None:
            return local_runtime.consume_pending_media_attachments()
        environment_runtime = self._environment_filesystem_runtime()
        if environment_runtime is not None:
            return environment_runtime.consume_pending_media_attachments()
        return []

    @property
    def has_filesystem_read_text_file_tool(self) -> bool:
        """Whether the active filesystem runtime currently exposes read_text_file."""
        if self._filesystem_runtime is None:
            return False
        return any(
            tool.name == READ_TEXT_FILE_TOOL_NAME for tool in self._filesystem_runtime.tools if tool
        )

    @property
    def skill_read_tool_name(self) -> str:
        """Return the tool name that should be referenced for reading skill content.

        Skills are discovered from the active environment filesystem, so the
        environment ``read_text_file`` tool reads skill paths directly;
        ``read_skill`` remains the fallback when no read tool is exposed.
        """
        return (
            READ_TEXT_FILE_TOOL_NAME
            if self.has_filesystem_read_text_file_tool
            else READ_SKILL_TOOL_NAME
        )

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
        from fast_agent.core.instruction_refresh import (
            build_instruction,
            format_agent_skills,
            resolve_instruction_skill_manifests,
        )

        if not self._instruction_template:
            return

        # Build the instruction using the central helper
        new_instruction = await build_instruction(
            self._instruction_template,
            aggregator=self._aggregator,
            skill_manifests=resolve_instruction_skill_manifests(self, self._skill_manifests),
            skill_read_tool_name=self.skill_read_tool_name,
            context=self._instruction_context,
            source=self._name,
        )
        new_instruction = self.process_rendered_instruction(new_instruction)
        self.set_instruction(new_instruction)

        # Warn when skills are configured but not surfaced in the final instruction.
        # This check must use the rendered instruction to account for internal
        # internal templates may include {{agentSkills}}.
        if self._skill_manifests and "{{agentSkills}}" not in self._instruction_template:
            formatted_skills = format_agent_skills(
                self._skill_manifests,
                self.skill_read_tool_name,
            )
            if formatted_skills and formatted_skills not in new_instruction:
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
        if self._no_shell_requested:
            return
        if self._shell_runtime_enabled:
            return
        if self._external_runtime is not None:
            return

        self._activate_shell_runtime(
            "because agent skills are configured",
            working_directory=self.config.cwd,
            access_modes=("skills",),
            show_shell_notice=True,
        )

    def _resolve_shell_runtime_settings(self) -> _ShellRuntimeSettings:
        timeout_seconds = 90
        warning_interval_seconds = 30
        config_output_byte_limit = None
        shell_config = None
        if self._context and self._context.config:
            shell_config = self._context.config.shell_execution
        if shell_config:
            timeout_seconds = shell_config.timeout_seconds
            warning_interval_seconds = shell_config.warning_interval_seconds
            config_output_byte_limit = shell_config.output_byte_limit

        output_limit_selection = (
            shell_config.output_byte_limit_selection if shell_config is not None else "auto"
        )
        configured_profile = shell_config.tool_profile if shell_config is not None else "auto"
        model_params = self._resolve_shell_model_params()
        model_tool_profile = model_params.shell_tool_profile if model_params is not None else None
        model_name = self._resolve_shell_tool_model_name()

        if output_limit_selection == "explicit" and config_output_byte_limit is not None:
            output_byte_limit = config_output_byte_limit
        elif output_limit_selection == "auto":
            max_output_tokens = (
                ModelDatabase.get_max_output_tokens(model_name) if model_name else None
            )
            output_byte_limit = calculate_terminal_output_limit_for_max_tokens(max_output_tokens)
        else:
            model_override = (
                ModelDatabase.get_shell_output_byte_limit(model_name) if model_name else None
            )
            output_byte_limit = (
                model_override
                if model_override is not None
                else config_output_byte_limit or DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
            )
        return _ShellRuntimeSettings(
            timeout_seconds=timeout_seconds,
            warning_interval_seconds=warning_interval_seconds,
            output_byte_limit=output_byte_limit,
            process_poll_default_wait_seconds=(self._model_process_poll_default_wait_seconds()),
            tool_profile=configured_profile,
            model_tool_profile=model_tool_profile,
        )

    def _model_process_poll_default_wait_seconds(
        self,
        llm: FastAgentLLMProtocol | None = None,
    ) -> int:
        active_llm = llm or self._llm
        resolved_model = resolve_resolved_model(active_llm) if active_llm is not None else None
        if isinstance(resolved_model, ResolvedModelSpec):
            return resolved_model.process_poll_default_wait_seconds
        model_params = resolve_model_params(active_llm)
        if model_params is not None:
            return model_params.process_poll_default_wait_seconds
        model_name = (
            resolve_model_name(active_llm)
            if active_llm is not None
            else self._resolve_shell_tool_model_name()
        )
        params = ModelDatabase.get_model_params(model_name) if model_name else None
        return params.process_poll_default_wait_seconds if params is not None else 0

    def _shell_read_text_file_enabled(self) -> bool:
        """Return whether shell-enabled agents should expose local read_text_file."""
        if not self._context or not self._context.config:
            return True
        return self._context.config.shell_execution.enable_read_text_file

    def _shell_attach_media_mode(self) -> Literal["auto", "on", "off"]:
        """Return whether shell-enabled agents should expose local attach_media."""
        if not self._context or not self._context.config:
            return "auto"
        return self._context.config.shell_execution.enable_attach_media

    def _resolve_shell_edit_tool_mode(
        self,
        llm: FastAgentLLMProtocol | None = None,
    ) -> ShellEditToolMode:
        """Return which shell edit tool should be exposed for the current model/config."""
        mode_raw = (
            self._context.config.shell_execution.write_text_file_mode
            if self._context and self._context.config
            else None
        )
        mode = strip_casefold(mode_raw) if isinstance(mode_raw, str) else None
        configured_modes = {
            "on": ShellEditToolMode.WRITE_TEXT_FILE,
            ShellEditToolMode.OFF.value: ShellEditToolMode.OFF,
            ShellEditToolMode.APPLY_PATCH.value: ShellEditToolMode.APPLY_PATCH,
            ShellEditToolMode.EDIT_FILE.value: ShellEditToolMode.EDIT_FILE,
        }
        if mode in configured_modes:
            return configured_modes[mode]

        model_params = self._resolve_shell_model_params(llm)
        if model_params is not None and model_params.shell_edit_tool is not None:
            return ShellEditToolMode(model_params.shell_edit_tool)

        model_name = (
            resolve_model_name(llm) if llm is not None else self._resolve_shell_tool_model_name()
        )
        if self._prefers_apply_patch_model(model_name):
            return ShellEditToolMode.APPLY_PATCH
        if self._prefers_anthropic_edit_file_model(model_name):
            return ShellEditToolMode.EDIT_FILE
        return ShellEditToolMode.WRITE_TEXT_FILE

    def _shell_edit_tool_flags(
        self,
        llm: FastAgentLLMProtocol | None = None,
    ) -> ShellEditToolFlags:
        return ShellEditToolFlags.from_mode(self._resolve_shell_edit_tool_mode(llm))

    def _resolve_shell_tool_model_name(self) -> str | None:
        """Resolve the best-available model name for shell tool policy decisions."""
        llm = self._llm
        llm_model = llm.model_name if llm is not None else None
        if (model_name := strip_to_none(llm_model)) is not None:
            return model_name

        model_name = self.config.model
        if not model_name and self._context and self._context.config:
            model_name = self._context.config.default_model
        return model_name

    def _resolve_shell_model_params(
        self,
        llm: FastAgentLLMProtocol | None = None,
    ) -> ModelParameters | None:
        active_llm = llm or self._llm
        if active_llm is not None:
            model_params = resolve_model_params(active_llm)
            if model_params is not None:
                return model_params
        model_name = self._resolve_shell_tool_model_name()
        return ModelDatabase.get_model_params(model_name) if model_name else None

    def _resolve_minimal_shell_tool_contract(
        self,
        llm: FastAgentLLMProtocol | None = None,
    ) -> tuple[str, bool]:
        model_params = self._resolve_shell_model_params(llm)
        if model_params is None:
            return BASH_TOOL_NAME, False
        return (
            model_params.shell_tool_name or BASH_TOOL_NAME,
            model_params.shell_tool_requires_description,
        )

    @staticmethod
    def _prefers_apply_patch_model(model_name: str | None) -> bool:
        """Return True for Codex and GPT-5.2+ models."""
        if not model_name:
            return False

        normalized = ModelDatabase.normalize_model_name(model_name)
        if "codex" in normalized:
            return True

        match = re.match(r"^gpt-5(?:\.(\d+))?", normalized)
        if match is None:
            return False

        minor = match.group(1)
        if minor is None:
            return False
        return int(minor) >= 2

    @staticmethod
    def _prefers_extended_shell_guidance(model_name: str | None) -> bool:
        """Return True for GPT-5.6-class models."""
        if not model_name:
            return False
        normalized = ModelDatabase.normalize_model_name(model_name)
        return re.match(r"^gpt-5\.6(?:$|[-.])", normalized) is not None

    @staticmethod
    def _prefers_anthropic_edit_file_model(model_name: str | None) -> bool:
        """Return True for Anthropic-series models."""
        if not model_name:
            return False

        normalized = ModelDatabase.normalize_model_name(model_name)
        params = ModelDatabase.get_model_params(normalized)
        if params is not None and params.default_provider in {
            Provider.ANTHROPIC,
            Provider.ANTHROPIC_VERTEX,
        }:
            return True
        return re.search(r"(?:^|[./:])claude-", normalized) is not None

    def _maybe_enable_local_filesystem_runtime(self, working_directory: Path | None = None) -> None:
        """Enable local filesystem runtime when shell mode is active and configured."""
        if not self._shell_runtime_enabled:
            return

        enable_read = self._shell_read_text_file_enabled()
        enable_attach_media = self._shell_attach_media_mode()
        model_info = self.llm.model_info if self.llm else None
        edit_flags = self._shell_edit_tool_flags()
        environment_filesystem = self._shell_environment
        environment_runtime = self._environment_filesystem_runtime()
        if isinstance(environment_filesystem, EnvironmentFilesystem):
            if environment_runtime is not None:
                environment_runtime.set_enabled_tools(
                    enable_read=enable_read,
                    enable_write=edit_flags.write_text_file,
                    enable_apply_patch=edit_flags.apply_patch,
                    enable_edit_file=edit_flags.edit_file,
                    enable_attach_media=enable_attach_media,
                )
                environment_runtime.set_model_info(model_info)
                environment_runtime.set_tool_handler_resolver(self._get_tool_handler)
                return

            environment_runtime = EnvironmentFilesystemRuntime(
                environment_filesystem,
                enable_read=enable_read,
                enable_write=edit_flags.write_text_file,
                enable_apply_patch=edit_flags.apply_patch,
                enable_edit_file=edit_flags.edit_file,
                enable_attach_media=enable_attach_media,
                model_info=model_info,
                tool_handler_resolver=self._get_tool_handler,
            )
            if self._filesystem_runtime is None:
                self._filesystem_runtime = environment_runtime
            else:
                self._filesystem_runtime = CompositeFilesystemRuntime(
                    primary=self._filesystem_runtime,
                    fallback=environment_runtime,
                )
            self.logger.info(
                "Environment filesystem runtime enabled",
                runtime_type=type(self._filesystem_runtime).__name__,
                read_enabled=enable_read,
                write_enabled=edit_flags.write_text_file,
                apply_patch_enabled=edit_flags.apply_patch,
                edit_file_enabled=edit_flags.edit_file,
                attach_media_enabled=enable_attach_media,
            )
            return

        if self._shell_environment is not None:
            self._drop_local_filesystem_runtime()
            return

        local_runtime = self._local_filesystem_runtime()
        if local_runtime is not None:
            if working_directory is not None:
                local_runtime.set_working_directory(working_directory)
            local_runtime.set_model_info(model_info)
            local_runtime.set_tool_handler_resolver(self._get_tool_handler)
            local_runtime.set_enabled_tools(
                enable_read=enable_read,
                enable_write=edit_flags.write_text_file,
                enable_apply_patch=edit_flags.apply_patch,
                enable_edit_file=edit_flags.edit_file,
                enable_attach_media=enable_attach_media,
            )
            return

        runtime_working_directory = (
            working_directory if working_directory is not None else self.config.cwd
        )
        local_runtime = LocalFilesystemRuntime(
            self.logger,
            working_directory=runtime_working_directory,
            enable_read=enable_read,
            enable_write=edit_flags.write_text_file,
            enable_apply_patch=edit_flags.apply_patch,
            enable_edit_file=edit_flags.edit_file,
            enable_attach_media=enable_attach_media,
            model_info=model_info,
            tool_handler_resolver=self._get_tool_handler,
        )
        if self._filesystem_runtime is None:
            self._filesystem_runtime = local_runtime
        else:
            self._filesystem_runtime = CompositeFilesystemRuntime(
                primary=self._filesystem_runtime,
                fallback=local_runtime,
            )
        self.logger.info(
            "Local filesystem runtime enabled",
            runtime_type=type(self._filesystem_runtime).__name__,
            read_enabled=enable_read,
            write_enabled=edit_flags.write_text_file,
            apply_patch_enabled=edit_flags.apply_patch,
            edit_file_enabled=edit_flags.edit_file,
            attach_media_enabled=enable_attach_media,
        )

    def _shell_output_limit_overridden(self) -> bool:
        """Return True when shell output byte limit is explicitly configured."""
        if not self._context or not self._context.config:
            return False
        shell_config = self._context.config.shell_execution
        return shell_config.output_byte_limit_selection == "explicit"

    def _validate_llm_attachment(self, llm: FastAgentLLMProtocol) -> None:
        super()._validate_llm_attachment(llm)
        resolved_model = resolve_resolved_model(llm)
        if not isinstance(resolved_model, ResolvedModelSpec):
            return
        poll_period = resolved_model.model_config.process_poll_default_wait_seconds
        if poll_period is None:
            return
        maximum = (
            self._context.config.shell_execution.process_poll_max_wait_seconds
            if self._context is not None and self._context.config is not None
            else ShellSettings().process_poll_max_wait_seconds
        )
        if poll_period > maximum:
            raise ModelConfigError(
                f"Model query poll_period={poll_period} exceeds "
                f"shell_execution.process_poll_max_wait_seconds={maximum}. "
                "Lower poll_period or raise the configured maximum."
            )

    def _on_llm_attached(self, llm: FastAgentLLMProtocol) -> None:
        super()._on_llm_attached(llm)

        if self._provider_managed_mcp_state.has_servers():
            if (
                self._provider_managed_mcp_state.has_connectors()
                and llm.provider != Provider.RESPONSES
            ):
                raise AgentConfigError(
                    "Provider-managed connectors are only supported for the OpenAI Responses provider."
                )
            if llm.provider not in {Provider.ANTHROPIC, Provider.RESPONSES}:
                raise AgentConfigError(
                    "Provider-managed MCP is only supported for Anthropic Messages "
                    "and the OpenAI Responses provider."
                )
            llm.set_provider_managed_mcp_state(self._provider_managed_mcp_state)

        local_runtime = self._local_filesystem_runtime()
        if local_runtime is not None:
            edit_flags = self._shell_edit_tool_flags(llm)
            local_runtime.set_model_info(llm.model_info)
            local_runtime.set_enabled_tools(
                enable_read=self._shell_read_text_file_enabled(),
                enable_write=edit_flags.write_text_file,
                enable_apply_patch=edit_flags.apply_patch,
                enable_edit_file=edit_flags.edit_file,
                enable_attach_media=self._shell_attach_media_mode(),
            )
        environment_runtime = self._environment_filesystem_runtime()
        if environment_runtime is not None:
            edit_flags = self._shell_edit_tool_flags(llm)
            environment_runtime.set_enabled_tools(
                enable_read=self._shell_read_text_file_enabled(),
                enable_write=edit_flags.write_text_file,
                enable_apply_patch=edit_flags.apply_patch,
                enable_edit_file=edit_flags.edit_file,
                enable_attach_media=self._shell_attach_media_mode(),
            )
            environment_runtime.set_model_info(llm.model_info)

        if self._shell_runtime is None:
            return
        model_name = resolve_model_name(llm)
        shell_tool_name, require_description = self._resolve_minimal_shell_tool_contract(llm)
        self._shell_runtime.set_minimal_shell_tool_contract(
            tool_name=shell_tool_name,
            require_description=require_description,
            extended_guidance=self._prefers_extended_shell_guidance(model_name),
        )
        shell_config = (
            self._context.config.shell_execution
            if self._context is not None and self._context.config is not None
            else None
        )
        configured_profile = shell_config.tool_profile if shell_config is not None else "auto"
        model_params = self._resolve_shell_model_params(llm)
        self._shell_runtime.set_tool_profile(
            configured_profile,
            model_profile=(model_params.shell_tool_profile if model_params is not None else None),
        )
        self._bash_tool = self._shell_runtime.tool
        self._shell_runtime.set_process_poll_default_wait_seconds(
            self._model_process_poll_default_wait_seconds(llm)
        )
        if self._shell_output_limit_overridden():
            return

        self._shell_runtime.set_output_byte_limit(self._model_tool_output_byte_limit(llm))

    def _model_tool_output_byte_limit(
        self,
        llm: FastAgentLLMProtocol | None = None,
    ) -> int:
        active_llm = llm or self._llm
        resolved_model = resolve_resolved_model(active_llm) if active_llm is not None else None
        shell_config = (
            self._context.config.shell_execution
            if self._context is not None and self._context.config is not None
            else None
        )
        automatic_sizing = (
            shell_config is None or shell_config.output_byte_limit_selection == "auto"
        )
        if resolved_model is not None:
            if automatic_sizing:
                return calculate_terminal_output_limit_for_max_tokens(
                    resolved_model.max_output_tokens
                )
            model_override = (
                resolved_model.model_params.shell_output_byte_limit
                if resolved_model.model_params is not None
                else None
            )
            if model_override is not None:
                return calculate_terminal_output_limit_for_resolved_model(resolved_model)
        model_name = (
            resolve_model_name(active_llm)
            if active_llm is not None
            else self._resolve_shell_tool_model_name()
        )
        model_override = (
            ModelDatabase.get_shell_output_byte_limit(model_name) if model_name else None
        )
        if automatic_sizing:
            max_output_tokens = (
                ModelDatabase.get_max_output_tokens(model_name) if model_name else None
            )
            return calculate_terminal_output_limit_for_max_tokens(max_output_tokens)
        if model_override is not None:
            return calculate_terminal_output_limit_for_model(model_name)
        return shell_config.output_byte_limit or DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

    def _activate_shell_runtime(
        self,
        activation_reason: str | None,
        *,
        working_directory: Path | None = None,
        access_modes: tuple[str, ...] = (),
        show_shell_notice: bool = False,
    ) -> None:
        if activation_reason is not None and self._external_runtime is not None:
            return

        self._warn_if_invalid_shell_working_directory(working_directory)

        shell_settings = self._resolve_shell_runtime_settings()
        shell_tool_name, require_description = self._resolve_minimal_shell_tool_contract()

        self._shell_runtime_activation_reason = activation_reason
        self._shell_runtime = ShellRuntime(
            activation_reason,
            self.logger,
            timeout_seconds=shell_settings.timeout_seconds,
            warning_interval_seconds=shell_settings.warning_interval_seconds,
            working_directory=working_directory,
            output_byte_limit=shell_settings.output_byte_limit,
            process_poll_default_wait_seconds=(shell_settings.process_poll_default_wait_seconds),
            tool_profile=shell_settings.tool_profile,
            model_tool_profile=shell_settings.model_tool_profile,
            config=self._context.config if self._context else None,
            agent_name=self._name,
            shell_environment=self._shell_environment,
            minimal_shell_tool_name=shell_tool_name,
            minimal_shell_tool_requires_description=require_description,
            extended_guidance=self._prefers_extended_shell_guidance(
                self._resolve_shell_tool_model_name()
            ),
        )
        self._shell_runtime_enabled = self._shell_runtime.enabled
        self._bash_tool = self._shell_runtime.tool
        self._shell_access_modes = access_modes if self._shell_runtime_enabled else ()
        self._maybe_enable_local_filesystem_runtime(working_directory)
        if self._shell_runtime_enabled:
            self._shell_runtime.announce()
            if show_shell_notice and self._allow_shell_notice and not self._shell_notice_emitted:
                self._shell_notice_emitted = True
                with suppress(Exception):
                    console.console.print(
                        format_shell_notice(self._shell_access_modes, self._shell_runtime)
                    )

    @property
    def shell_runtime_enabled(self) -> bool:
        return self._shell_runtime_enabled

    @property
    def shell_access_modes(self) -> tuple[str, ...]:
        return self._shell_access_modes

    @property
    def shell_runtime(self) -> ShellRuntime | None:
        return self._shell_runtime

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
        message: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
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

    def set_external_runtime(self, runtime: ExternalRuntime | None) -> None:
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

    def set_filesystem_runtime(self, runtime: FilesystemRuntime) -> None:
        """
        Set a filesystem runtime (e.g., ACPFilesystemRuntime) to add filesystem tools.

        This allows ACP mode to inject filesystem support that uses the client's
        filesystem capabilities for reading and writing files while preserving
        local shell edit tools when shell mode is enabled.

        Args:
            runtime: Runtime instance with tools property and read_text_file/write_text_file methods
        """
        local_runtime = self._local_filesystem_runtime()
        if isinstance(runtime, (LocalFilesystemRuntime, CompositeFilesystemRuntime)):
            self._filesystem_runtime = runtime
        elif local_runtime is not None and runtime is not local_runtime:
            self._filesystem_runtime = CompositeFilesystemRuntime(
                primary=runtime, fallback=local_runtime
            )
        else:
            self._filesystem_runtime = runtime
        current_local_runtime = self._local_filesystem_runtime()
        if current_local_runtime is not None:
            current_local_runtime.set_tool_handler_resolver(self._get_tool_handler)
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
        local_result = await self._call_local_tool(
            name,
            arguments,
            tool_use_id,
            request_params=request_params,
        )
        if local_result is not None:
            return local_result

        request_tool_handler = None
        if request_params and request_params.tool_execution_handler:
            request_tool_handler = request_params.tool_execution_handler
        return await self._aggregator.call_tool(
            name,
            arguments,
            tool_use_id,
            request_tool_handler=request_tool_handler,
        )

    async def _call_local_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult | None:
        if self._external_runtime is not None and name == self._external_runtime.tool.name:
            return await self._external_runtime.execute(arguments or {}, tool_use_id)

        if self._filesystem_runtime and any(
            tool.name == name for tool in self._filesystem_runtime.tools
        ):
            return await self._filesystem_runtime.call_tool(
                name,
                arguments,
                tool_use_id,
                request_params=request_params,
            )

        if self._skill_reader and name == READ_SKILL_TOOL_NAME:
            return await self._skill_reader.execute(arguments)

        if self._shell_runtime and self._shell_runtime.owns_tool(name):
            return await self._shell_runtime.call_tool(
                name,
                arguments,
                tool_use_id,
                show_tool_call_id=self._show_shell_tool_call_id,
                defer_display_to_tool_result=self._defer_shell_display_to_tool_result,
            )

        return await self._call_builtin_tool(
            name,
            arguments,
            tool_use_id,
            request_params=request_params,
        )

    async def _call_builtin_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult | None:
        if name == HUMAN_INPUT_TOOL_NAME:
            return await self._call_human_input_tool(arguments)

        if name in self._execution_tools:
            return await super().call_tool(
                name,
                arguments,
                tool_use_id,
                request_params=request_params,
            )
        return None

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
                    is_error=False,
                    content=[TextContent(type="text", text="The Human declined the input request")],
                )
            if resp_text in ("__CANCELLED__", "__DISABLE_SERVER__"):
                return CallToolResult(
                    is_error=False,
                    content=[
                        TextContent(type="text", text="The Human cancelled the input request")
                    ],
                )
            # Success path: return the (JSON) response as-is
            return CallToolResult(
                is_error=False,
                content=[TextContent(type="text", text=resp_text)],
            )

        except PromptExitError:
            raise
        except asyncio.TimeoutError as e:
            return CallToolResult(
                is_error=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: Human input request timed out: {e!s}",
                    )
                ],
            )
        except Exception as e:
            import traceback

            print(f"Error in _call_human_input_tool: {traceback.format_exc()}")
            return CallToolResult(
                is_error=True,
                content=[TextContent(type="text", text=f"Error requesting human input: {e!s}")],
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
        prompt: str | GetPromptResult,
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
            namespaced_name = prompt_display_name(prompt_result, prompt_name)
        else:
            # prompt is a GetPromptResult object
            prompt_result = prompt
            if not prompt_result or not prompt_result.messages:
                error_msg = "Provided GetPromptResult contains no messages"
                self.logger.warning(error_msg)
                return error_msg

            # Use a reasonable display name
            namespaced_name = prompt_display_name(prompt_result, "provided_prompt")

        self.logger.debug(f"Using prompt '{namespaced_name}'")

        # Convert prompt messages to multipart format using the safer method
        multipart_messages = PromptMessageExtended.from_get_prompt_result(prompt_result)

        if as_template:
            # Use apply_prompt_template to store as persistent prompt messages
            return await self.apply_prompt_template(prompt_result, namespaced_name)

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
        prompt_content: str | PromptMessage | PromptMessageExtended,
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
            content: list[ContentBlock] = [TextContent(type="text", text=prompt_content)]
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
        tool_metadata: dict[str, dict[str, Any]] = {}
        tool_loop_error: str | None = None

        available_tools = await self._available_tool_names_for_run_tools()
        tool_catalog = self._aggregator.tool_catalog()

        tool_call_items = list(request.tool_calls.items())
        should_parallel = should_parallelize_tool_calls(len(tool_call_items))
        self._maybe_close_display_for_parallel_subagent_tools(tool_call_items, should_parallel)
        planned_calls, tool_loop_error = self._plan_mcp_tool_calls(
            tool_call_items=tool_call_items,
            tool_catalog=tool_catalog,
            available_tools=available_tools,
            should_parallel=should_parallel,
            tool_results=tool_results,
            tool_metadata=tool_metadata,
        )

        if should_parallel and planned_calls:
            self.display.show_parallel_tool_calls(
                [
                    request
                    for call in planned_calls
                    if (request := self._planned_mcp_tool_call_display_request(call)) is not None
                ]
            )
            await self._run_parallel_planned_tool_calls(
                planned_calls,
                request_params=request_params,
                tool_results=tool_results,
                tool_timings=tool_timings,
            )

            return self._finalize_tool_results(
                tool_results,
                tool_timings=tool_timings,
                tool_metadata=tool_metadata,
                tool_loop_error=tool_loop_error,
            )

        for call in planned_calls:
            display_request = self._planned_mcp_tool_call_display_request(call)
            if display_request is not None:
                self.display.show_tool_call(
                    display_request.tool_name,
                    display_request.tool_args,
                    bottom_items=display_request.bottom_items,
                    highlight_indexes=display_request.highlight_indexes,
                    max_item_length=display_request.max_item_length,
                    name=display_request.name,
                    metadata=display_request.metadata,
                    tool_call_id=display_request.tool_call_id,
                    source_label=display_request.source_label,
                    server_name=display_request.server_name,
                    show_hook_indicator=display_request.show_hook_indicator,
                )
        await self._run_sequential_planned_tool_calls(
            planned_calls,
            request_params=request_params,
            tool_results=tool_results,
            tool_timings=tool_timings,
        )

        return self._finalize_tool_results(
            tool_results,
            tool_timings=tool_timings,
            tool_metadata=tool_metadata,
            tool_loop_error=tool_loop_error,
        )

    async def _available_tool_names_for_run_tools(self) -> list[str]:
        try:
            listed_tools = await self.list_tools()
        except Exception as exc:  # pragma: no cover - defensive guard, should not happen
            self.logger.warning(f"Failed to list tools before execution: {exc}")
            listed_tools = ListToolsResult(tools=[])
        return listed_tool_names(listed_tools)

    def _maybe_close_display_for_parallel_subagent_tools(
        self,
        tool_call_items: list[tuple[str, Any]],
        should_parallel: bool,
    ) -> None:
        if not should_parallel or not tool_call_items:
            return

        subagent_calls = self._count_agent_tool_calls(tool_call_items)
        if subagent_calls <= 1:
            return

        did_close = self.close_active_streaming_display(reason="parallel subagent tool calls")
        if did_close:
            self.logger.info(
                "Closing streaming display due to parallel subagent tool calls",
                agent_name=self._name,
                tool_call_count=len(tool_call_items),
                subagent_call_count=subagent_calls,
            )

    def _plan_mcp_tool_calls(
        self,
        *,
        tool_call_items: list[tuple[str, Any]],
        tool_catalog: MCPToolCatalog,
        available_tools: list[str],
        should_parallel: bool,
        tool_results: dict[str, CallToolResult],
        tool_metadata: dict[str, dict[str, Any]],
    ) -> tuple[list[PlannedMcpToolCall], str | None]:
        planned_calls: list[PlannedMcpToolCall] = []
        for correlation_id, tool_request in tool_call_items:
            try:
                planned_call = self._plan_mcp_tool_call(
                    correlation_id=correlation_id,
                    tool_request=tool_request,
                    tool_catalog=tool_catalog,
                    available_tools=available_tools,
                    should_parallel=should_parallel,
                )
            except ValueError as exc:
                error_message = str(exc)
                self.logger.error(error_message)
                return planned_calls, self._mark_tool_loop_error(
                    correlation_id=correlation_id,
                    error_message=error_message,
                    tool_results=tool_results,
                    tool_call_id=correlation_id if should_parallel else None,
                )

            if planned_call.metadata:
                tool_metadata[correlation_id] = planned_call.metadata
            planned_calls.append(planned_call)

        return planned_calls, None

    async def _run_parallel_planned_tool_calls(
        self,
        planned_calls: list[PlannedMcpToolCall],
        *,
        request_params: RequestParams | None,
        tool_results: dict[str, CallToolResult],
        tool_timings: dict[str, ToolTimingInfo],
    ) -> None:
        previous_shell_tool_call_id_setting = self._show_shell_tool_call_id
        previous_shell_display_setting = self._defer_shell_display_to_tool_result
        self._show_shell_tool_call_id = True
        self._defer_shell_display_to_tool_result = True
        try:
            results = await gather_with_cancel(
                self._execute_mcp_planned_tool_call(call, request_params=request_params)
                for call in planned_calls
            )
        finally:
            self._show_shell_tool_call_id = previous_shell_tool_call_id_setting
            self._defer_shell_display_to_tool_result = previous_shell_display_setting

        display_requests: list[ToolResultDisplayRequest] = []
        for call, item in zip(planned_calls, results, strict=True):
            if isinstance(item, BaseException):
                self.logger.error(f"MCP tool {call.display_tool_name} failed: {item}")
                result = CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {item!s}")],
                    is_error=True,
                )
                duration_ms = 0.0
            else:
                _, result, duration_ms = item
            display_request = await self._record_planned_tool_result(
                call,
                result,
                duration_ms=duration_ms,
                tool_results=tool_results,
                tool_timings=tool_timings,
            )
            if display_request is not None:
                display_requests.append(display_request)
        self.display.show_parallel_tool_results(display_requests)

    async def _run_sequential_planned_tool_calls(
        self,
        planned_calls: list[PlannedMcpToolCall],
        *,
        request_params: RequestParams | None,
        tool_results: dict[str, CallToolResult],
        tool_timings: dict[str, ToolTimingInfo],
    ) -> None:
        for call in planned_calls:
            try:
                _, result, duration_ms = await self._execute_mcp_planned_tool_call(
                    call,
                    request_params=request_params,
                )
                display_request = await self._record_planned_tool_result(
                    call,
                    result,
                    duration_ms=duration_ms,
                    tool_results=tool_results,
                    tool_timings=tool_timings,
                )
                if display_request is not None:
                    self._show_tool_result_display_request(display_request)
                self.logger.debug(f"MCP tool {call.display_tool_name} executed successfully")
            except Exception as e:
                self.logger.error(f"MCP tool {call.display_tool_name} failed: {e}")
                error_result = CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {e!s}")],
                    is_error=True,
                )
                display_request = await self._record_planned_tool_result(
                    call,
                    error_result,
                    duration_ms=0.0,
                    tool_results=tool_results,
                    tool_timings=tool_timings,
                )
                if display_request is not None:
                    self._show_tool_result_display_request(display_request)

    async def _execute_mcp_planned_tool_call(
        self,
        call: PlannedMcpToolCall,
        *,
        request_params: RequestParams | None,
    ) -> tuple[str, CallToolResult, float]:
        start_time = time.perf_counter()
        result = await self.call_tool(
            call.execution_tool_name,
            call.tool_args,
            call.correlation_id,
            request_params=request_params,
        )
        end_time = time.perf_counter()
        return call.correlation_id, result, round((end_time - start_time) * 1000, 2)

    async def _record_planned_tool_result(
        self,
        call: PlannedMcpToolCall,
        result: CallToolResult,
        *,
        duration_ms: float,
        tool_results: dict[str, CallToolResult],
        tool_timings: dict[str, ToolTimingInfo],
    ) -> ToolResultDisplayRequest | None:
        if not call.is_local_shell:
            result = truncate_tool_result_for_llm(
                result,
                byte_limit=self._model_tool_output_byte_limit(),
            )
        attach_read_text_file_display_metadata(
            result,
            display_tool_name=call.display_tool_name,
            tool_args=call.tool_args,
        )
        tool_results[call.correlation_id] = result
        display_metadata = tool_result_display_metadata(result)
        tool_timings[call.correlation_id] = ToolTimingInfo(
            timing_ms=duration_ms,
            transport_channel=display_metadata.get("transport_channel"),
        )

        if display_metadata.get("suppress_display", False):
            return None
        if self._is_builtin_subagent_tool(call.metadata):
            await self._show_subagent_result(result)
            return None

        return ToolResultDisplayRequest(
            result=result,
            name=self._name,
            tool_name=call.display_tool_name,
            app_integration_config=await self._app_integration_config_for_planned_tool(call),
            timing_ms=duration_ms,
            tool_call_id=call.correlation_id,
            type_label=tool_result_type_label(call.display_tool_name),
            source_label=call.source_label,
            server_name=call.server_name,
            show_hook_indicator=self.has_after_tool_call_hook,
        )

    def _show_tool_result_display_request(self, request: ToolResultDisplayRequest) -> None:
        self.display.show_tool_result(
            request.result,
            name=request.name,
            tool_name=request.tool_name,
            app_integration_config=request.app_integration_config,
            timing_ms=request.timing_ms,
            tool_call_id=request.tool_call_id,
            type_label=request.type_label,
            source_label=request.source_label,
            server_name=request.server_name,
            show_hook_indicator=request.show_hook_indicator,
        )

    async def _app_integration_config_for_planned_tool(
        self,
        call: PlannedMcpToolCall,
    ) -> AppServerConfig | None:
        namespaced_tool = call.namespaced_tool or call.candidate_namespaced_tool
        if namespaced_tool is None:
            return None
        try:
            return await self._aggregator.get_app_integration_config(namespaced_tool.server_name)
        except Exception:
            return None

    def _plan_mcp_tool_call(
        self,
        *,
        correlation_id: str,
        tool_request: Any,
        tool_catalog: MCPToolCatalog,
        available_tools: list[str],
        should_parallel: bool,
    ) -> PlannedMcpToolCall:
        del should_parallel
        tool_name = tool_request.params.name
        tool_args = tool_request.params.arguments or {}
        local_tool = self._execution_tools.get(tool_name)
        is_external_runtime_tool = self._is_external_runtime_tool(tool_name)
        is_filesystem_runtime_tool = self._is_filesystem_runtime_tool(tool_name)
        route = build_mcp_tool_route(
            requested_name=tool_name,
            catalog=tool_catalog,
            local_tool_exists=local_tool is not None,
            is_filesystem_runtime_tool=is_filesystem_runtime_tool,
        )
        if not self._is_planned_tool_available(
            tool_name=tool_name,
            namespaced_tool=route.namespaced_tool,
            local_tool=local_tool,
            candidate_namespaced_tool=route.candidate_namespaced_tool,
            is_external_runtime_tool=is_external_runtime_tool,
            is_filesystem_runtime_tool=is_filesystem_runtime_tool,
        ):
            raise ValueError(f"Tool '{route.display_name}' is not available")

        metadata = self._metadata_for_planned_tool(
            tool_name=tool_name,
            tool_args=tool_args,
            local_tool=local_tool,
            is_external_runtime_tool=is_external_runtime_tool,
            is_filesystem_runtime_tool=is_filesystem_runtime_tool,
            route_to_namespaced_candidate=route.route_to_namespaced_candidate,
        )
        presentation = build_mcp_tool_presentation(
            route,
            tool_catalog,
            local_tool_names=self._execution_tools if local_tool is not None else None,
            fallback_order=available_tools,
            display_name_overrides=TOOL_DISPLAY_NAMES,
        )
        active_namespaced = route.active_namespaced_tool
        return PlannedMcpToolCall(
            correlation_id=correlation_id,
            route=route,
            tool_args=tool_args,
            bottom_items=presentation.bottom_items,
            highlight_indexes=presentation.highlight_indexes,
            source_label=(
                "MCP"
                if active_namespaced is not None
                else self._tool_display_source_label(presentation.display_name, metadata)
            ),
            server_name=active_namespaced.server_name if active_namespaced is not None else None,
            is_local_shell=(
                self._bash_tool is not None
                and tool_name == self._bash_tool.name
                and route.namespaced_tool is None
            ),
            metadata=metadata,
        )

    def _is_external_runtime_tool(self, tool_name: str) -> bool:
        return self._external_runtime is not None and tool_name == self._external_runtime.tool.name

    def _is_filesystem_runtime_tool(self, tool_name: str) -> bool:
        return bool(
            self._filesystem_runtime
            and any(tool.name == tool_name for tool in self._filesystem_runtime.tools)
        )

    def _is_planned_tool_available(
        self,
        *,
        tool_name: str,
        namespaced_tool: NamespacedTool | None,
        local_tool: Any | None,
        candidate_namespaced_tool: NamespacedTool | None,
        is_external_runtime_tool: bool,
        is_filesystem_runtime_tool: bool,
    ) -> bool:
        is_shell_tool = bool(self._shell_runtime and self._shell_runtime.owns_tool(tool_name))
        is_skill_reader_tool = bool(
            self._skill_reader and self._skill_reader.enabled and tool_name == READ_SKILL_TOOL_NAME
        )
        return (
            tool_name == HUMAN_INPUT_TOOL_NAME
            or is_shell_tool
            or is_external_runtime_tool
            or is_filesystem_runtime_tool
            or is_skill_reader_tool
            or namespaced_tool is not None
            or local_tool is not None
            or candidate_namespaced_tool is not None
        )

    def _metadata_for_planned_tool(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        local_tool: Any | None,
        is_external_runtime_tool: bool,
        is_filesystem_runtime_tool: bool,
        route_to_namespaced_candidate: bool,
    ) -> dict[str, Any] | None:
        if self._shell_runtime_enabled and self._shell_runtime:
            if self._shell_runtime.tool and tool_name == self._shell_runtime.tool.name:
                return self._shell_runtime.metadata(tool_args)
            if self._shell_runtime.owns_tool(tool_name):
                return self._shell_runtime.process_tool_metadata(tool_name, tool_args)
        if is_external_runtime_tool and self._external_runtime is not None:
            return self._external_runtime.metadata()
        if (
            is_filesystem_runtime_tool
            and self._filesystem_runtime
            and not route_to_namespaced_candidate
        ):
            return self._filesystem_runtime.metadata()
        if local_tool is not None:
            return self._jsonable_tool_metadata(self._tool_display_metadata(tool_name))
        return None

    def _planned_mcp_tool_call_display_request(
        self,
        call: PlannedMcpToolCall,
    ) -> ToolCallDisplayRequest | None:
        if is_read_text_file_tool_name(call.display_tool_name):
            return None
        if self._is_builtin_subagent_tool(call.metadata):
            self._show_subagent_message(call.tool_args)
            return None
        return ToolCallDisplayRequest(
            tool_name=call.display_tool_name,
            tool_args=call.tool_args,
            bottom_items=call.bottom_items,
            highlight_indexes=call.highlight_indexes,
            max_item_length=12,
            name=self._name,
            metadata=call.metadata,
            tool_call_id=call.correlation_id,
            source_label=call.source_label,
            server_name=call.server_name,
            show_hook_indicator=self.has_before_tool_call_hook,
        )

    def resolve_stream_tool_metadata(self, tool_name: str) -> Mapping[str, Any] | None:
        metadata = super().resolve_stream_tool_metadata(tool_name)
        if metadata:
            return metadata

        lookup_name = tool_name.strip()
        if not lookup_name:
            return None
        if (
            self._shell_runtime_enabled
            and self._shell_runtime
            and self._shell_runtime.owns_tool(lookup_name)
        ):
            if self._shell_runtime.tool and lookup_name == self._shell_runtime.tool.name:
                return self._shell_runtime.metadata({})
            return self._shell_runtime.process_tool_metadata(lookup_name, {})

        if not is_namespaced_name(lookup_name) and "/" in lookup_name:
            server_name, base_tool_name = lookup_name.split("/", 1)
            if server_name and base_tool_name:
                lookup_name = create_namespaced_name(server_name, base_tool_name)

        namespaced_tool = self._aggregator.tool_catalog().namespaced_tool(lookup_name)
        if namespaced_tool is None or not isinstance(namespaced_tool.tool.meta, Mapping):
            return None

        metadata = dict(namespaced_tool.tool.meta)
        return self._jsonable_tool_metadata(metadata)

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
        with self._tracer.start_as_current_span(f"Agent: '{self._name}' apply_prompt_template"):
            return await self._require_llm().apply_prompt_template(prompt_result, prompt_name)

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
    ) -> Mapping[str, list[mcp_types.Prompt]]:
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

        self._append_unique_tools(
            merged_tools,
            existing_names,
            await self._additional_runtime_tools(),
        )

        return ListToolsResult(tools=merged_tools)

    @staticmethod
    def _append_unique_tools(
        merged_tools: list[Tool],
        existing_names: set[str],
        tools: Iterable[Tool],
    ) -> None:
        for tool in tools:
            if tool.name in existing_names:
                continue
            merged_tools.append(tool)
            existing_names.add(tool.name)

    async def _additional_runtime_tools(self) -> list[Tool]:
        tools = list((await super().list_tools()).tools)
        tools.extend(self._terminal_runtime_tools())
        tools.extend(self._filesystem_runtime_tools())
        skill_tool = self._skill_reader_fallback_tool()
        if skill_tool is not None:
            tools.append(skill_tool)
        human_tool = self._human_input_runtime_tool()
        if human_tool is not None:
            tools.append(human_tool)
        return tools

    def _terminal_runtime_tools(self) -> list[Tool]:
        if self._external_runtime is not None:
            return [self._external_runtime.tool]
        if self._shell_runtime is not None:
            return self._shell_runtime.tools
        return []

    def _filesystem_runtime_tools(self) -> list[Tool]:
        if not self._filesystem_runtime:
            return []
        return [tool for tool in self._filesystem_runtime.tools if tool is not None]

    def _skill_reader_fallback_tool(self) -> Tool | None:
        if (
            self._skill_reader
            and self._skill_reader.enabled
            and self.skill_read_tool_name == READ_SKILL_TOOL_NAME
        ):
            return self._skill_reader.tool
        return None

    def _human_input_runtime_tool(self) -> Tool | None:
        if not self.config.human_input:
            return None
        return self._human_input_tool

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

        tools: ListToolsResult = await self.list_tools()
        skills = [await self.convert(tool) for tool in tools.tools]

        return build_fast_agent_card(
            skills=skills,
            name=self._name,
            description=self.config.description or self.instruction,
        )

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_items: str | list[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: "Text | None" = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
        render_message: bool = True,
        show_reprint_banner: bool = False,
    ) -> None:
        """
        Display an assistant message with MCP servers in the bottom bar.

        This override adds the list of connected MCP servers to the bottom bar
        and highlights servers that were used for tool calls in this message.
        """
        # Get the list of MCP servers (if not provided)
        server_names = self._assistant_bottom_items(bottom_items)
        card_tools_label = self._card_tools_label()

        # Extract servers from tool calls in the message for highlighting
        highlight_servers = self._assistant_highlight_items(
            message,
            highlight_items=highlight_items,
            card_tools_label=card_tools_label,
        )

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
            render_message=render_message,
            show_reprint_banner=show_reprint_banner,
        )

    def _assistant_bottom_items(self, bottom_items: list[str] | None) -> list[str]:
        server_names = (
            list(self.list_attached_mcp_servers()) if bottom_items is None else list(bottom_items)
        )
        server_names = unique_preserving_order(server_names)
        server_names = self._with_shell_label_first(server_names)
        self._append_optional_server_label(server_names, self._skills_tool_label())
        self._append_card_tools_label(server_names)
        self._append_agent_tool_labels(server_names)
        return server_names

    def _with_shell_label_first(self, server_names: list[str]) -> list[str]:
        shell_label = self._shell_server_label()
        if not shell_label:
            return server_names
        return [shell_label, *(name for name in server_names if name != shell_label)]

    @staticmethod
    def _append_optional_server_label(server_names: list[str], label: str | None) -> None:
        if label and label not in server_names:
            server_names.append(label)

    def _append_card_tools_label(self, server_names: list[str]) -> None:
        card_tools_label = self._card_tools_label()
        if not card_tools_label or card_tools_label in server_names:
            return

        skills_label = self._skills_tool_label()
        if skills_label and skills_label in server_names:
            insert_at = server_names.index(skills_label) + 1
            server_names.insert(insert_at, card_tools_label)
            return
        server_names.append(card_tools_label)

    def _append_agent_tool_labels(self, server_names: list[str]) -> None:
        for tool_name in self.agent_backed_tools:
            agent_label = tool_name.removeprefix("agent__")
            if agent_label not in server_names:
                server_names.append(agent_label)

    def _assistant_highlight_items(
        self,
        message: PromptMessageExtended,
        *,
        highlight_items: str | list[str] | None,
        card_tools_label: str | None,
    ) -> list[str]:
        if highlight_items is None:
            highlight_servers = self._extract_servers_from_message(message)
        elif isinstance(highlight_items, str):
            highlight_servers = [highlight_items]
        else:
            highlight_servers = list(highlight_items)

        if (
            card_tools_label
            and self._card_tools_used(message)
            and card_tools_label not in highlight_servers
        ):
            highlight_servers.append(card_tools_label)
        return highlight_servers

    def _extract_servers_from_message(self, message: PromptMessageExtended) -> list[str]:
        """
        Extract server names from tool calls in the message.

        Args:
            message: The message containing potential tool calls

        Returns:
            List of server names that were called
        """
        servers: list[str] = []
        for tool_request in (message.tool_calls or {}).values():
            server_label = self._server_label_for_tool_call(tool_request.params.name)
            if server_label and server_label not in servers:
                servers.append(server_label)
        return servers

    def _server_label_for_tool_call(self, tool_name: str) -> str | None:
        if tool_name in self.agent_backed_tools:
            return tool_name.removeprefix("agent__")
        if self._shell_runtime and self._shell_runtime.owns_tool(tool_name):
            return self._shell_server_label()
        if self._skill_reader_tool_called(tool_name):
            return self._skills_tool_label()
        namespaced_tool = self._aggregator.tool_catalog().namespaced_tool(tool_name)
        return namespaced_tool.server_name if namespaced_tool is not None else None

    def _skill_reader_tool_called(self, tool_name: str) -> bool:
        return bool(
            self._skill_reader
            and self._skill_reader.enabled
            and tool_name == self._skill_reader.tool.name
        )

    def _shell_server_label(self) -> str | None:
        """Return the display label for the local shell runtime."""
        shell_runtime = self._shell_runtime
        if not self._shell_runtime_enabled or not shell_runtime or not shell_runtime.tool:
            return None

        runtime_info = shell_runtime.runtime_info()
        runtime_name = runtime_info.name
        return runtime_name or "shell"

    def _shell_tool_name_for_display(self) -> str | None:
        shell_runtime = self._shell_runtime
        if not self._shell_runtime_enabled or not shell_runtime or not shell_runtime.tool:
            return None
        return shell_runtime.tool.name

    def _skills_tool_label(self) -> str | None:
        if self._skill_reader and self._skill_reader.enabled:
            name = self._skill_reader.tool.name
            return TOOL_DISPLAY_NAMES.get(name, name)
        return None

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

        tool_name_resolution = self._aggregator.resolve_tool_name(tool.name)
        return AgentSkill(
            id=tool.name,
            name=tool_name_resolution.local_name,
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
    def usage_accumulator(self) -> "UsageAccumulator | None":
        """
        Return the usage accumulator for tracking token usage across turns.

        Returns:
            UsageAccumulator object if LLM is attached, None otherwise
        """
        if self.llm:
            return self.llm.usage_accumulator
        return None
