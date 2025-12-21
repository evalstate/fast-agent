"""
Slash Commands for ACP

Provides slash command support for the ACP server, allowing clients to
discover and invoke special commands with the /command syntax.

Session commands (status, tools, save, clear, load) are always available.
Agent-specific commands are queried from the current agent if it implements
ACPAwareProtocol.
"""

from __future__ import annotations

import textwrap
import time
import uuid
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from acp.helpers import text_block, tool_content
from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    ToolCallProgress,
    ToolCallStart,
    UnstructuredCommandInput,
)

from fast_agent.agents.agent_types import AgentType
from fast_agent.config import get_settings
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.core.logging.logger import get_logger
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.interfaces import ACPAwareProtocol, AgentProtocol
from fast_agent.llm.model_info import ModelInfo
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompts.prompt_load import load_history_into_agent
from fast_agent.skills.manager import (
    candidate_marketplace_urls,
    fetch_marketplace_skills,
    fetch_marketplace_skills_with_source,
    format_marketplace_display_url,
    get_manager_directory,
    get_marketplace_url,
    install_marketplace_skill,
    list_local_skills,
    reload_skill_manifests,
    remove_local_skill,
    resolve_skill_directories,
    select_manifest_by_name_or_index,
    select_skill_by_name_or_index,
)
from fast_agent.skills.registry import format_skills_for_prompt
from fast_agent.types.conversation_summary import ConversationSummary
from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from mcp.types import ListToolsResult, Tool

    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.skills.registry import SkillRegistry


@runtime_checkable
class WarningAwareAgent(Protocol):
    @property
    def warnings(self) -> list[str]: ...

    @property
    def skill_registry(self) -> "SkillRegistry | None": ...


@runtime_checkable
class InstructionAwareAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def instruction(self) -> str | None: ...


@runtime_checkable
class InstructionRefreshAgent(InstructionAwareAgent, Protocol):
    async def rebuild_instruction_templates(self) -> None: ...


class SlashCommandHandler:
    """Handles slash command execution for ACP sessions."""

    def __init__(
        self,
        session_id: str,
        instance: AgentInstance,
        primary_agent_name: str,
        *,
        history_exporter: type[HistoryExporter] | HistoryExporter | None = None,
        client_info: dict | None = None,
        client_capabilities: dict | None = None,
        protocol_version: int | None = None,
        session_instructions: dict[str, str] | None = None,
    ):
        """
        Initialize the slash command handler.

        Args:
            session_id: The ACP session ID
            instance: The agent instance for this session
            primary_agent_name: Name of the primary agent
            history_exporter: Optional history exporter
            client_info: Client information from ACP initialize
            client_capabilities: Client capabilities from ACP initialize
            protocol_version: ACP protocol version
        """
        self.session_id = session_id
        self.instance = instance
        self.primary_agent_name = primary_agent_name
        self._logger = get_logger(__name__)
        # Track current agent (can change via setSessionMode). Ensure it exists.
        if primary_agent_name in instance.agents:
            self.current_agent_name = primary_agent_name
        else:
            # Fallback: pick the first registered agent to enable agent-specific commands.
            self.current_agent_name = next(iter(instance.agents.keys()), primary_agent_name)
        self.history_exporter = history_exporter or HistoryExporter
        self._created_at = time.time()
        self.client_info = client_info
        self.client_capabilities = client_capabilities
        self.protocol_version = protocol_version
        self._session_instructions = session_instructions or {}

        # Session-level commands (always available, operate on current agent)
        self._session_commands: dict[str, AvailableCommand] = {
            "status": AvailableCommand(
                name="status",
                description="Show fast-agent diagnostics",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="[system|auth|authreset]")
                ),
            ),
            "tools": AvailableCommand(
                name="tools",
                description="List available tools",
                input=None,
            ),
            "skills": AvailableCommand(
                name="skills",
                description="List or manage local skills (add/remove/registry)",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="[add|remove|registry] [name|number|url]")
                ),
            ),
            "save": AvailableCommand(
                name="save",
                description="Save conversation history",
                input=None,
            ),
            "clear": AvailableCommand(
                name="clear",
                description="Clear history (`last` for prev. turn)",
                input=AvailableCommandInput(root=UnstructuredCommandInput(hint="[last]")),
            ),
            "load": AvailableCommand(
                name="load",
                description="Load conversation history from file",
                input=AvailableCommandInput(root=UnstructuredCommandInput(hint="<filename>")),
            ),
        }

    def get_available_commands(self) -> list[AvailableCommand]:
        """Get combined session commands and current agent's commands."""
        commands = list(self._get_allowed_session_commands().values())

        # Add agent-specific commands if current agent is ACP-aware
        agent = self._get_current_agent()
        if isinstance(agent, ACPAwareProtocol):
            for name, cmd in agent.acp_commands.items():
                # Convert ACPCommand to AvailableCommand
                cmd_input = None
                if cmd.input_hint:
                    cmd_input = AvailableCommandInput(
                        root=UnstructuredCommandInput(hint=cmd.input_hint)
                    )
                commands.append(
                    AvailableCommand(name=name, description=cmd.description, input=cmd_input)
                )

        return commands

    def _get_allowed_session_commands(self) -> dict[str, AvailableCommand]:
        """
        Return session-level commands filtered by the current agent's policy.

        By default, all session commands are available. ACP-aware agents can restrict
        session commands (e.g. Setup/wizard flows) by defining either:
        - `acp_session_commands_allowlist: set[str] | None` attribute, or
        - `acp_session_commands_allowlist() -> set[str] | None` method
        """
        agent = self._get_current_agent()
        if not isinstance(agent, ACPAwareProtocol):
            return self._session_commands

        allowlist = getattr(agent, "acp_session_commands_allowlist", None)
        if callable(allowlist):
            try:
                allowlist = allowlist()
            except Exception:
                allowlist = None

        if allowlist is None:
            return self._session_commands

        try:
            allowset = {str(name) for name in allowlist}
        except Exception:
            return self._session_commands

        return {name: cmd for name, cmd in self._session_commands.items() if name in allowset}

    def set_current_agent(self, agent_name: str) -> None:
        """
        Update the current agent for this session.

        This is called when the user switches modes via setSessionMode.

        Args:
            agent_name: Name of the agent to use for slash commands
        """
        self.current_agent_name = agent_name

    def update_session_instruction(self, agent_name: str, instruction: str | None) -> None:
        """
        Update the cached session instruction for an agent.

        Call this when an agent's system prompt has been rebuilt (e.g., after
        connecting new MCP servers) to keep the /system command output current.

        Args:
            agent_name: Name of the agent whose instruction was updated
            instruction: The new instruction (or None to remove from cache)
        """
        if instruction:
            self._session_instructions[agent_name] = instruction
        elif agent_name in self._session_instructions:
            del self._session_instructions[agent_name]

    def _get_current_agent(self) -> AgentProtocol | None:
        """Return the current agent or None if it does not exist."""
        return self.instance.agents.get(self.current_agent_name)

    def _get_current_agent_or_error(
        self,
        heading: str,
        missing_template: str | None = None,
    ) -> tuple[AgentProtocol | None, str | None]:
        """
        Return the current agent or an error response string if it is missing.

        Args:
            heading: Heading for the error message.
            missing_template: Optional custom missing-agent message.
        """
        agent = self._get_current_agent()
        if agent:
            return agent, None

        message = (
            missing_template or f"Agent '{self.current_agent_name}' not found for this session."
        )
        return None, "\n".join([heading, "", message])

    def is_slash_command(self, prompt_text: str) -> bool:
        """Check if the prompt text is a slash command."""
        return prompt_text.strip().startswith("/")

    def parse_command(self, prompt_text: str) -> tuple[str, str]:
        """
        Parse a slash command into command name and arguments.

        Args:
            prompt_text: The full prompt text starting with /

        Returns:
            Tuple of (command_name, arguments)
        """
        text = prompt_text.strip()
        if not text.startswith("/"):
            return "", text

        # Remove leading slash
        text = text[1:]

        # Split on first whitespace
        command_name, _, arguments = text.partition(" ")
        arguments = arguments.lstrip()

        return command_name, arguments

    async def execute_command(self, command_name: str, arguments: str) -> str:
        """
        Execute a slash command and return the response.

        Args:
            command_name: Name of the command to execute
            arguments: Arguments passed to the command

        Returns:
            The command response as a string
        """
        # Check session-level commands first (filtered by agent policy)
        allowed_session_commands = self._get_allowed_session_commands()
        if command_name in allowed_session_commands:
            if command_name == "status":
                return await self._handle_status(arguments)
            if command_name == "tools":
                return await self._handle_tools()
            if command_name == "skills":
                return await self._handle_skills(arguments)
            if command_name == "save":
                return await self._handle_save(arguments)
            if command_name == "clear":
                return await self._handle_clear(arguments)
            if command_name == "load":
                return await self._handle_load(arguments)

        # Check agent-specific commands
        agent = self._get_current_agent()
        if isinstance(agent, ACPAwareProtocol):
            agent_commands = agent.acp_commands
            if command_name in agent_commands:
                return await agent_commands[command_name].handler(arguments)

        # Unknown command
        available = self.get_available_commands()
        return f"Unknown command: /{command_name}\n\nAvailable commands:\n" + "\n".join(
            f"  /{cmd.name} - {cmd.description}" for cmd in available
        )

    async def _handle_status(self, arguments: str | None = None) -> str:
        """Handle the /status command."""
        # Check for subcommands
        normalized = (arguments or "").strip().lower()
        if normalized == "system":
            return await self._handle_status_system()
        if normalized == "auth":
            return self._handle_status_auth()
        if normalized == "authreset":
            return self._handle_status_authreset()

        # Get fast-agent version
        try:
            fa_version = get_version("fast-agent-mcp")
        except Exception:
            fa_version = "unknown"

        # Get model information from current agent (not primary)
        agent = self._get_current_agent()

        # Check if this is a PARALLEL agent
        is_parallel_agent = (
            agent and hasattr(agent, "agent_type") and agent.agent_type == AgentType.PARALLEL
        )

        # For non-parallel agents, extract standard model info
        model_name = "unknown"
        model_provider = "unknown"
        model_provider_display = "unknown"
        context_window = "unknown"
        capabilities_line = "Capabilities: unknown"

        if agent and not is_parallel_agent and agent.llm:
            model_info = ModelInfo.from_llm(agent.llm)
            if model_info:
                model_name = model_info.name
                model_provider = str(model_info.provider.value)
                model_provider_display = getattr(
                    model_info.provider, "display_name", model_provider
                )
                if model_info.context_window:
                    context_window = f"{model_info.context_window} tokens"
                capability_parts = []
                if model_info.supports_text:
                    capability_parts.append("Text")
                if model_info.supports_document:
                    capability_parts.append("Document")
                if model_info.supports_vision:
                    capability_parts.append("Vision")
                if capability_parts:
                    capabilities_line = f"Capabilities: {', '.join(capability_parts)}"

        # Get conversation statistics
        summary_stats = self._get_conversation_stats(agent)

        # Format the status response
        status_lines = [
            "# fast-agent ACP status",
            "",
            "## Version",
            f"fast-agent-mcp: {fa_version} - https://fast-agent.ai/",
            "",
        ]

        # Add client information if available
        if self.client_info or self.client_capabilities:
            status_lines.extend(["## Client Information", ""])

            if self.client_info:
                client_name = self.client_info.get("name", "unknown")
                client_version = self.client_info.get("version", "unknown")
                client_title = self.client_info.get("title")

                if client_title:
                    status_lines.append(f"Client: {client_title} ({client_name})")
                else:
                    status_lines.append(f"Client: {client_name}")
                status_lines.append(f"Client Version: {client_version}")

            if self.protocol_version:
                status_lines.append(f"ACP Protocol Version: {self.protocol_version}")

            if self.client_capabilities:
                # Filesystem capabilities
                if "fs" in self.client_capabilities:
                    fs_caps = self.client_capabilities["fs"]
                    if fs_caps:
                        for key, value in fs_caps.items():
                            status_lines.append(f"  - {key}: {value}")

                # Terminal capability
                if "terminal" in self.client_capabilities:
                    status_lines.append(f"  - Terminal: {self.client_capabilities['terminal']}")

                # Meta capabilities
                if "_meta" in self.client_capabilities:
                    meta_caps = self.client_capabilities["_meta"]
                    if meta_caps:
                        status_lines.append("Meta:")
                        for key, value in meta_caps.items():
                            status_lines.append(f"  - {key}: {value}")

            status_lines.append("")

        # Build model section based on agent type
        if is_parallel_agent:
            # Special handling for PARALLEL agents
            status_lines.append("## Active Models (Parallel Mode)")
            status_lines.append("")

            # Display fan-out agents
            if hasattr(agent, "fan_out_agents") and agent.fan_out_agents:
                status_lines.append(f"### Fan-Out Agents ({len(agent.fan_out_agents)})")
                for idx, fan_out_agent in enumerate(agent.fan_out_agents, 1):
                    agent_name = getattr(fan_out_agent, "name", f"agent-{idx}")
                    status_lines.append(f"**{idx}. {agent_name}**")

                    # Get model info for this fan-out agent
                    if fan_out_agent.llm:
                        model_info = ModelInfo.from_llm(fan_out_agent.llm)
                        if model_info:
                            provider_display = getattr(
                                model_info.provider, "display_name", str(model_info.provider.value)
                            )
                            status_lines.append(f"  - Provider: {provider_display}")
                            status_lines.append(f"  - Model: {model_info.name}")
                            if model_info.context_window:
                                status_lines.append(
                                    f"  - Context Window: {model_info.context_window} tokens"
                                )
                    else:
                        status_lines.append("  - Model: unknown")

                    status_lines.append("")
            else:
                status_lines.append("Fan-Out Agents: none configured")
                status_lines.append("")

            # Display fan-in agent
            if hasattr(agent, "fan_in_agent") and agent.fan_in_agent:
                fan_in_agent = agent.fan_in_agent
                fan_in_name = getattr(fan_in_agent, "name", "aggregator")
                status_lines.append(f"### Fan-In Agent: {fan_in_name}")

                # Get model info for fan-in agent
                if fan_in_agent.llm:
                    model_info = ModelInfo.from_llm(fan_in_agent.llm)
                    if model_info:
                        provider_display = getattr(
                            model_info.provider, "display_name", str(model_info.provider.value)
                        )
                        status_lines.append(f"  - Provider: {provider_display}")
                        status_lines.append(f"  - Model: {model_info.name}")
                        if model_info.context_window:
                            status_lines.append(
                                f"  - Context Window: {model_info.context_window} tokens"
                            )
                else:
                    status_lines.append("  - Model: unknown")

                status_lines.append("")
            else:
                status_lines.append("Fan-In Agent: none configured")
                status_lines.append("")

        else:
            # Standard single-model display
            provider_line = f"{model_provider}"
            if model_provider_display != "unknown":
                provider_line = f"{model_provider_display} ({model_provider})"

            # For HuggingFace, add the routing provider info
            if agent and agent.llm:
                get_hf_info = getattr(agent.llm, "get_hf_display_info", None)
                if callable(get_hf_info):
                    hf_info = get_hf_info()
                    hf_provider = hf_info.get("provider", "auto-routing")
                    provider_line = f"{model_provider_display} ({model_provider}) / {hf_provider}"

            status_lines.extend(
                [
                    "## Active Model",
                    f"- Provider: {provider_line}",
                    f"- Model: {model_name}",
                    f"- Context Window: {context_window}",
                    f"- {capabilities_line}",
                    "",
                ]
            )

        # Add conversation statistics
        status_lines.append(
            f"## Conversation Statistics ({getattr(agent, 'name', self.current_agent_name) if agent else 'Unknown'})"
        )

        uptime_seconds = max(time.time() - self._created_at, 0.0)
        status_lines.extend(summary_stats)
        status_lines.extend(["", f"ACP Agent Uptime: {format_duration(uptime_seconds)}"])
        status_lines.extend(["", "## Error Handling"])
        status_lines.extend(self._get_error_handling_report(agent))
        warning_report = self._get_warning_report(agent)
        if warning_report:
            status_lines.append("")
            status_lines.extend(warning_report)

        return "\n".join(status_lines)

    async def _handle_status_system(self) -> str:
        """Handle the /status system command to show the system prompt."""
        heading = "# system prompt"

        agent, error = self._get_current_agent_or_error(heading)
        if error:
            return error

        if isinstance(agent, InstructionRefreshAgent):
            try:
                await agent.rebuild_instruction_templates()
            except Exception:
                pass

        system_prompt = agent.instruction if isinstance(agent, InstructionAwareAgent) else None
        if not system_prompt:
            return "\n".join(
                [
                    heading,
                    "",
                    "No system prompt available for this agent.",
                ]
            )

        # Format the response
        agent_name = (
            agent.name if isinstance(agent, InstructionAwareAgent) else self.current_agent_name
        )
        lines = [
            heading,
            "",
            f"**Agent:** {agent_name}",
            "",
            system_prompt,
        ]

        return "\n".join(lines)

    def _handle_status_auth(self) -> str:
        """Handle the /status auth command to show permissions from auths.md."""
        heading = "# permissions"
        auths_path = Path("./.fast-agent/auths.md")
        resolved_path = auths_path.resolve()

        if not auths_path.exists():
            return "\n".join(
                [
                    heading,
                    "",
                    "No permissions set",
                    "",
                    f"Path: `{resolved_path}`",
                ]
            )

        try:
            content = auths_path.read_text(encoding="utf-8")
            return "\n".join(
                [
                    heading,
                    "",
                    content.strip() if content.strip() else "No permissions set",
                    "",
                    f"Path: `{resolved_path}`",
                ]
            )
        except Exception as exc:
            return "\n".join(
                [
                    heading,
                    "",
                    f"Failed to read permissions file: {exc}",
                    "",
                    f"Path: `{resolved_path}`",
                ]
            )

    def _handle_status_authreset(self) -> str:
        """Handle the /status authreset command to remove the auths.md file."""
        heading = "# reset permissions"
        auths_path = Path("./.fast-agent/auths.md")
        resolved_path = auths_path.resolve()

        if not auths_path.exists():
            return "\n".join(
                [
                    heading,
                    "",
                    "No permissions file exists.",
                    "",
                    f"Path: `{resolved_path}`",
                ]
            )

        try:
            auths_path.unlink()
            return "\n".join(
                [
                    heading,
                    "",
                    "Permissions file removed successfully.",
                    "",
                    f"Path: `{resolved_path}`",
                ]
            )
        except Exception as exc:
            return "\n".join(
                [
                    heading,
                    "",
                    f"Failed to remove permissions file: {exc}",
                    "",
                    f"Path: `{resolved_path}`",
                ]
            )

    async def _handle_tools(self) -> str:
        """List available tools for the current agent."""
        heading = "# tools"

        agent, error = self._get_current_agent_or_error(heading)
        if error:
            return error

        if not isinstance(agent, AgentProtocol):
            return "\n".join(
                [
                    heading,
                    "",
                    "This agent does not support tool listing.",
                ]
            )

        try:
            tools_result: "ListToolsResult" = await agent.list_tools()
        except Exception as exc:
            return "\n".join(
                [
                    heading,
                    "",
                    "Failed to fetch tools from the agent.",
                    f"Details: {exc}",
                ]
            )

        tools = tools_result.tools if tools_result else None
        if not tools:
            return "\n".join(
                [
                    heading,
                    "",
                    "No MCP tools available for this agent.",
                ]
            )

        lines = [heading, ""]
        for index, tool in enumerate(tools, start=1):
            lines.extend(self._format_tool_lines(tool, index))
            lines.append("")

        return "\n".join(lines).strip()

    async def _handle_skills(self, arguments: str | None = None) -> str:
        """Manage local skills (list/add/remove)."""
        tokens = (arguments or "").strip().split(maxsplit=1)
        action = tokens[0].lower() if tokens else "list"
        remainder = tokens[1] if len(tokens) > 1 else ""

        if action in {"list", ""}:
            return self._handle_skills_list()
        if action in {"add", "install"}:
            return await self._handle_skills_add(remainder)
        if action in {"registry", "marketplace", "source"}:
            return await self._handle_skills_registry(remainder)
        if action in {"remove", "rm", "delete", "uninstall"}:
            return await self._handle_skills_remove(remainder)

        return "Unknown /skills action. Use `/skills`, `/skills add`, or `/skills remove`."

    async def _handle_skills_registry(self, argument: str) -> str:
        heading = "# skills registry"
        argument = argument.strip()

        # Get configured registries from settings
        settings = get_settings()
        skills_settings = getattr(settings, "skills", None)
        configured_urls: list[str] = []
        if skills_settings:
            configured_urls = getattr(skills_settings, "marketplace_urls", None) or []

        if not argument:
            current = get_marketplace_url(settings)
            display_current = format_marketplace_display_url(current)

            lines = [heading, "", f"Registry: {display_current}", ""]

            # Show numbered list if registries configured
            if configured_urls:
                lines.append("Available registries:")
                for i, reg_url in enumerate(configured_urls, 1):
                    display = format_marketplace_display_url(reg_url)
                    lines.append(f"- [{i}] {display}")
                lines.append("")

            lines.append(
                "Usage: `/skills registry [number|URL]`.\n\n URL should point to a repo with a valid `marketplace.json`"
            )
            return "\n".join(lines)

        # Check if argument is a number (select from configured registries)
        if argument.isdigit():
            index = int(argument)
            if not configured_urls:
                return f"{heading}\n\nNo registries configured."
            if 1 <= index <= len(configured_urls):
                url = configured_urls[index - 1]
            else:
                return f"{heading}\n\nInvalid registry number. Use 1-{len(configured_urls)}."
        else:
            url = argument

        candidates = candidate_marketplace_urls(url)
        try:
            marketplace, resolved_url = await fetch_marketplace_skills_with_source(url)
        except Exception as exc:  # noqa: BLE001
            display_url = format_marketplace_display_url(url)
            self._logger.warning(
                "Failed to load skills registry",
                data={
                    "registry": url,
                    "candidates": candidates,
                    "error": str(exc),
                },
            )
            return "\n".join(
                [
                    heading,
                    "",
                    f"Failed to load registry: {exc}",
                    f"Registry: {display_url}",
                ]
            )

        if not marketplace:
            display_url = format_marketplace_display_url(url)
            return "\n".join(
                [
                    heading,
                    "",
                    "No skills found in the registry; registry unchanged.",
                    f"Registry: {display_url}",
                ]
            )

        # Update only the active registry, preserve the configured list
        if skills_settings is not None:
            skills_settings.marketplace_url = resolved_url

        display_url = format_marketplace_display_url(resolved_url)
        if candidates:
            self._logger.debug(
                "Resolved skills registry",
                data={
                    "input": url,
                    "resolved": resolved_url,
                    "candidates": candidates,
                },
            )
        response_lines = [
            heading,
            "",
            f"Registry set to: `{display_url}`",
            "",
            f"Skills discovered: {len(marketplace)}",
        ]

        return "\n".join(response_lines)

    def _handle_skills_list(self) -> str:
        manager_dir = get_manager_directory()
        manifests = list_local_skills(manager_dir)
        return self._format_local_skills(manifests, manager_dir)

    async def _handle_skills_add(self, argument: str) -> str:
        if argument.strip().lower() in {"q", "quit", "exit"}:
            return "Cancelled."

        agent, error = self._get_current_agent_or_error("# skills add")
        if error:
            return error

        tool_call_id = self._build_tool_call_id()
        await self._send_skills_update(
            agent,
            tool_call_id,
            title="Install skill",
            status="in_progress",
            message="Fetching marketplace…",
            start=True,
        )

        marketplace_url = get_marketplace_url()
        try:
            marketplace = await fetch_marketplace_skills(marketplace_url)
        except Exception as exc:  # noqa: BLE001
            return (
                "# skills add\n\n"
                f"Failed to load marketplace: {exc}\n\n"
                f"Repository: `{format_marketplace_display_url(marketplace_url)}`"
            )

        if not marketplace:
            return "# skills add\n\nNo skills found in the marketplace."

        if not argument:
            lines = [
                "# skills add",
                "",
                f"Repository: `{format_marketplace_display_url(marketplace_url)}`",
            ]
            # repo_hint = self._get_marketplace_repo_hint(marketplace)
            # if repo_hint:
            #     lines.append(f"Repository: `{repo_hint}`")
            lines.extend(["", "Available skills:  ", ""])
            lines.extend(self._format_marketplace_list(marketplace))
            lines.append("")
            lines.append("Install with `/skills add <number|name>`.")
            lines.append("Change registry with `/skills registry`.")
            return "\n".join(lines)

        skill = select_skill_by_name_or_index(marketplace, argument)
        if not skill:
            return "Skill not found. Use `/skills add` to list available skills."

        manager_dir = get_manager_directory()
        repo_label = self._format_repo_label(skill)
        await self._send_skills_update(
            agent,
            tool_call_id,
            title="Installing skill",
            status="in_progress",
            message=(
                f"Cloning {repo_label} ({skill.repo_subdir})"
                if repo_label
                else f"Cloning skill source ({skill.repo_subdir})"
            ),
        )
        try:
            install_path = await install_marketplace_skill(skill, destination_root=manager_dir)
        except Exception as exc:  # noqa: BLE001
            await self._send_skills_update(
                agent,
                tool_call_id,
                title="Install failed",
                status="completed",
                message=f"Failed to install skill: {exc}",
            )
            return f"# skills add\n\nFailed to install skill: {exc}"

        await self._refresh_agent_skills(agent)
        await self._send_skills_update(
            agent,
            tool_call_id,
            title="Install complete",
            status="completed",
            message=f"Installed {skill.name}",
        )

        return "\n".join(
            [
                "# skills add",
                "",
                f"Installed: {skill.name}",
                f"Location: `{install_path}`",
            ]
        )

    async def _handle_skills_remove(self, argument: str) -> str:
        if argument.strip().lower() in {"q", "quit", "exit"}:
            return "Cancelled."

        manager_dir = get_manager_directory()
        manifests = list_local_skills(manager_dir)
        if not manifests:
            return "# skills remove\n\nNo local skills to remove."

        if not argument:
            lines = [
                "# skills remove",
                "",
                "Installed skills:",
            ]
            lines.extend(self._format_local_list(manifests))
            lines.append("")
            lines.append(
                "Remove with `/skills remove <number|name>` or `/skills remove q` to cancel."
            )
            return "\n".join(lines)

        manifest = select_manifest_by_name_or_index(manifests, argument)
        if not manifest:
            return "Skill not found. Use `/skills remove` to list installed skills."

        try:
            skill_dir = Path(manifest.path).parent
            remove_local_skill(skill_dir, destination_root=manager_dir)
        except Exception as exc:  # noqa: BLE001
            return f"# skills remove\n\nFailed to remove skill: {exc}"

        agent, error = self._get_current_agent_or_error("# skills remove")
        if error:
            return error

        await self._refresh_agent_skills(agent)

        return "\n".join(
            [
                "# skills remove",
                "",
                f"Removed: {manifest.name}",
            ]
        )

    async def _refresh_agent_skills(self, agent: AgentProtocol) -> None:
        override_dirs = resolve_skill_directories(get_settings())
        registry, manifests = reload_skill_manifests(
            base_dir=Path.cwd(), override_directories=override_dirs
        )
        instruction_context = None
        try:
            skills_text = format_skills_for_prompt(manifests, read_tool_name="read_text_file")
            instruction_context = {"agentSkills": skills_text}
        except Exception:
            instruction_context = None

        await rebuild_agent_instruction(
            agent,
            skill_manifests=manifests,
            instruction_context=instruction_context,
            skill_registry=registry,
        )

    def _format_local_skills(self, manifests: list[Any], manager_dir: Path) -> str:
        lines = ["# skills", "", f"Directory: `{manager_dir}`", ""]
        if not manifests:
            lines.append("No skills available in the manager directory.")
            lines.append("")
            lines.append("Use `/skills add` to list available skills to install.")
            return "\n".join(lines)
        lines.append("Installed skills:")
        lines.extend(self._format_local_list(manifests))
        lines.append("")
        lines.append("Use `/skills add` to list available skills to install\n")
        lines.append("Remove a skill with `/skills remove <number|name>`.\n")
        lines.append("Change skills registry with `/skills registry <url>`.\n")
        return "\n".join(lines)

    def _format_local_list(self, manifests: list[Any]) -> list[str]:
        lines: list[str] = []
        for index, manifest in enumerate(manifests, 1):
            name = getattr(manifest, "name", "")
            description = getattr(manifest, "description", "")
            path = Path(getattr(manifest, "path", Path()))
            source_path = path.parent if path.is_file() else path
            try:
                display_path = source_path.relative_to(Path.cwd())
            except ValueError:
                display_path = source_path

            lines.append(f"- [{index}] {name}")
            if description:
                wrapped = textwrap.fill(description, width=76, subsequent_indent="    ")
                lines.append(f"  - {wrapped}")
            lines.append(f"  - source: `{display_path}`")
        return lines

    def _format_marketplace_list(self, marketplace: list[Any]) -> list[str]:
        lines: list[str] = []
        current_bundle = None
        for index, entry in enumerate(marketplace, 1):
            bundle_name = getattr(entry, "bundle_name", None)
            bundle_description = getattr(entry, "bundle_description", None)
            if bundle_name and bundle_name != current_bundle:
                current_bundle = bundle_name
                if lines:
                    lines.append("")
                lines.append(f"**{bundle_name}**  ")
                if bundle_description:
                    wrapped = textwrap.fill(bundle_description, width=76)
                    lines.append(wrapped)
                lines.append("")
            lines.append(f"- [{index}] **{entry.name}**")
            if entry.description:
                wrapped = textwrap.fill(entry.description, width=76, subsequent_indent="    ")
                lines.append(f"  - {wrapped}")
            if entry.source_url:
                lines.append(f"  - source: [link]({entry.source_url})")
        return lines

    def _format_repo_label(self, entry: Any) -> str | None:
        repo_url = getattr(entry, "repo_url", None)
        if not repo_url:
            return None
        repo_ref = getattr(entry, "repo_ref", None)
        if repo_ref:
            return f"{repo_url}@{repo_ref}"
        return repo_url

    def _get_marketplace_repo_hint(self, marketplace: list[Any]) -> str | None:
        if not marketplace:
            return None
        return self._format_repo_label(marketplace[0])

    def _build_tool_call_id(self) -> str:
        return str(uuid.uuid4())

    async def _send_skills_update(
        self,
        agent: AgentProtocol,
        tool_call_id: str,
        *,
        title: str,
        status: str,
        message: str | None = None,
        start: bool = False,
    ) -> None:
        acp = getattr(agent, "acp", None)
        if not acp:
            return
        try:
            if start:
                await acp.send_session_update(
                    ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=title,
                        kind="fetch",
                        status="in_progress",
                        session_update="tool_call",
                    )
                )
            content = [tool_content(text_block(message))] if message else None
            await acp.send_session_update(
                ToolCallProgress(
                    tool_call_id=tool_call_id,
                    title=title,
                    status=status,  # type: ignore[arg-type]
                    content=content,
                    session_update="tool_call_update",
                )
            )
        except Exception:
            return

    def _format_tool_lines(self, tool: "Tool", index: int) -> list[str]:
        """
        Convert a Tool into markdown-friendly lines.

        We avoid fragile getattr usage by relying on the typed attributes
        provided by mcp.types.Tool. Additional guards are added for optional fields.
        """
        lines: list[str] = []

        meta = tool.meta or {}
        name = tool.name or "unnamed"
        title = (tool.title or "").strip()

        header = f"{index}. **{name}**"
        if title:
            header = f"{header} - {title}"
        if meta.get("openai/skybridgeEnabled"):
            header = f"{header} _(skybridge)_"
        lines.append(header)

        description = (tool.description or "").strip()
        if description:
            wrapped = textwrap.wrap(description, width=92)
            if wrapped:
                indent = "    "
                lines.extend(f"{indent}{desc_line}" for desc_line in wrapped[:6])
                if len(wrapped) > 6:
                    lines.append(f"{indent}...")

        args_line = self._format_tool_arguments(tool)
        if args_line:
            lines.append(f"    - Args: {args_line}")

        template = meta.get("openai/skybridgeTemplate")
        if template:
            lines.append(f"    - Template: `{template}`")

        return lines

    def _format_tool_arguments(self, tool: "Tool") -> str | None:
        """Render tool input schema fields as inline-code argument list."""
        schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else None
        if not schema:
            return None

        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return None

        required_raw = schema.get("required", [])
        required = set(required_raw) if isinstance(required_raw, list) else set()

        args: list[str] = []
        for prop_name in properties.keys():
            suffix = "*" if prop_name in required else ""
            args.append(f"`{prop_name}{suffix}`")

        return ", ".join(args) if args else None

    async def _handle_save(self, arguments: str | None = None) -> str:
        """Handle the /save command by persisting conversation history."""
        heading = "# save conversation"

        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error

        filename = arguments.strip() if arguments and arguments.strip() else None

        try:
            saved_path = await self.history_exporter.save(agent, filename)
        except Exception as exc:
            return "\n".join(
                [
                    heading,
                    "",
                    "Failed to save conversation history.",
                    f"Details: {exc}",
                ]
            )

        return "\n".join(
            [
                heading,
                "",
                "Conversation history saved successfully.",
                f"Filename: `{saved_path}`",
            ]
        )

    async def _handle_load(self, arguments: str | None = None) -> str:
        """Handle the /load command by loading conversation history from a file."""
        heading = "# load conversation"

        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error

        filename = arguments.strip() if arguments and arguments.strip() else None

        if not filename:
            return "\n".join(
                [
                    heading,
                    "",
                    "Filename required for /load command.",
                    "Usage: /load <filename>",
                ]
            )

        file_path = Path(filename)
        if not file_path.exists():
            return "\n".join(
                [
                    heading,
                    "",
                    f"File not found: `{filename}`",
                ]
            )

        try:
            if hasattr(agent, "rebuild_instruction_templates"):
                await agent.rebuild_instruction_templates()
            load_history_into_agent(agent, file_path)
        except Exception as exc:
            return "\n".join(
                [
                    heading,
                    "",
                    "Failed to load conversation history.",
                    f"Details: {exc}",
                ]
            )

        message_count = len(agent.message_history) if hasattr(agent, "message_history") else 0
        if hasattr(agent, "rebuild_instruction_templates"):
            await agent.rebuild_instruction_templates()

        return "\n".join(
            [
                heading,
                "",
                "Conversation history loaded successfully.",
                f"Filename: `{filename}`",
                f"Messages: {message_count}",
            ]
        )

    async def _handle_clear(self, arguments: str | None = None) -> str:
        """Handle /clear and /clear last commands."""
        normalized = (arguments or "").strip().lower()
        if normalized == "last":
            return self._handle_clear_last()
        return self._handle_clear_all()

    def _handle_clear_all(self) -> str:
        """Clear the entire conversation history."""
        heading = "# clear conversation"
        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error

        try:
            history = getattr(agent, "message_history", None)
            original_count = len(history) if isinstance(history, list) else None

            cleared = False
            clear_method = getattr(agent, "clear", None)
            if callable(clear_method):
                clear_method()
                cleared = True
            elif isinstance(history, list):
                history.clear()
                cleared = True
        except Exception as exc:
            return "\n".join(
                [
                    heading,
                    "",
                    "Failed to clear conversation history.",
                    f"Details: {exc}",
                ]
            )

        if not cleared:
            return "\n".join(
                [
                    heading,
                    "",
                    "Agent does not expose a clear() method or message history list.",
                ]
            )

        removed_text = (
            f"Removed {original_count} message(s)." if isinstance(original_count, int) else ""
        )

        response_lines = [
            heading,
            "",
            "Conversation history cleared.",
        ]

        if removed_text:
            response_lines.append(removed_text)

        return "\n".join(response_lines)

    def _handle_clear_last(self) -> str:
        """Remove the most recent conversation message."""
        heading = "# clear last conversation turn"
        agent, error = self._get_current_agent_or_error(
            heading,
            missing_template=f"Unable to locate agent '{self.current_agent_name}' for this session.",
        )
        if error:
            return error

        try:
            removed = None
            pop_method = getattr(agent, "pop_last_message", None)
            if callable(pop_method):
                removed = pop_method()
            else:
                history = getattr(agent, "message_history", None)
                if isinstance(history, list) and history:
                    removed = history.pop()
        except Exception as exc:
            return "\n".join(
                [
                    heading,
                    "",
                    "Failed to remove the last message.",
                    f"Details: {exc}",
                ]
            )

        if removed is None:
            return "\n".join(
                [
                    heading,
                    "",
                    "No messages available to remove.",
                ]
            )

        role = getattr(removed, "role", "message")
        return "\n".join(
            [
                heading,
                "",
                f"Removed last {role} message.",
            ]
        )

    def _get_conversation_stats(self, agent) -> list[str]:
        """Get conversation statistics from the agent's message history."""
        if not agent or not hasattr(agent, "message_history"):
            return [
                "- Turns: 0",
                "- Tool Calls: 0",
                "- Context Used: 0%",
            ]

        try:
            # Create a conversation summary from message history
            summary = ConversationSummary(messages=agent.message_history)

            # Calculate turns (user + assistant message pairs)
            turns = min(summary.user_message_count, summary.assistant_message_count)

            # Get tool call statistics
            tool_calls = summary.tool_calls
            tool_errors = summary.tool_errors
            tool_successes = summary.tool_successes
            context_usage_line = self._context_usage_line(summary, agent)

            stats = [
                f"- Turns: {turns}",
                f"- Messages: {summary.message_count} (user: {summary.user_message_count}, assistant: {summary.assistant_message_count})",
                f"- Tool Calls: {tool_calls} (successes: {tool_successes}, errors: {tool_errors})",
                context_usage_line,
            ]

            # Add timing information if available
            if summary.total_elapsed_time_ms > 0:
                stats.append(
                    f"- Total LLM Time: {format_duration(summary.total_elapsed_time_ms / 1000)}"
                )

            if summary.conversation_span_ms > 0:
                span_seconds = summary.conversation_span_ms / 1000
                stats.append(
                    f"- Conversation Runtime (LLM + tools): {format_duration(span_seconds)}"
                )

            # Add tool breakdown if there were tool calls
            if tool_calls > 0 and summary.tool_call_map:
                stats.append("")
                stats.append("### Tool Usage Breakdown")
                for tool_name, count in sorted(
                    summary.tool_call_map.items(), key=lambda x: x[1], reverse=True
                ):
                    stats.append(f"  - {tool_name}: {count}")

            return stats

        except Exception as e:
            return [
                "- Turns: error",
                "- Tool Calls: error",
                f"- Context Used: error ({e})",
            ]

    def _get_error_handling_report(self, agent, max_entries: int = 3) -> list[str]:
        """Summarize error channel availability and recent entries."""
        channel_label = f"Error Channel: {FAST_AGENT_ERROR_CHANNEL}"
        if not agent or not hasattr(agent, "message_history"):
            return ["_No errors recorded_"]

        recent_entries: list[str] = []
        history = getattr(agent, "message_history", []) or []

        for message in reversed(history):
            channels = getattr(message, "channels", None) or {}
            channel_blocks = channels.get(FAST_AGENT_ERROR_CHANNEL)
            if not channel_blocks:
                continue

            for block in channel_blocks:
                text = get_text(block)
                if text:
                    cleaned = text.replace("\n", " ").strip()
                    if cleaned:
                        recent_entries.append(cleaned)
                else:
                    # Truncate long content (e.g., base64 image data)
                    block_str = str(block)
                    if len(block_str) > 60:
                        recent_entries.append(f"{block_str[:60]}... ({len(block_str)} characters)")
                    else:
                        recent_entries.append(block_str)
                if len(recent_entries) >= max_entries:
                    break
            if len(recent_entries) >= max_entries:
                break

        if recent_entries:
            lines = [channel_label, "Recent Entries:"]
            lines.extend(f"- {entry}" for entry in recent_entries)
            return lines

        return ["_No errors recorded_"]

    def _get_warning_report(self, agent, max_entries: int = 5) -> list[str]:
        warnings: list[str] = []
        if isinstance(agent, WarningAwareAgent):
            warnings.extend(agent.warnings)
            if agent.skill_registry:
                warnings.extend(agent.skill_registry.warnings)

        cleaned: list[str] = []
        seen: set[str] = set()
        for warning in warnings:
            message = str(warning).strip()
            if message and message not in seen:
                cleaned.append(message)
                seen.add(message)

        if not cleaned:
            return []

        lines = ["Warnings:"]
        for message in cleaned[:max_entries]:
            lines.append(f"- {message}")
        if len(cleaned) > max_entries:
            lines.append(f"- ... ({len(cleaned) - max_entries} more)")
        return lines

    def _context_usage_line(self, summary: ConversationSummary, agent) -> str:
        """Generate a context usage line with token estimation and fallbacks."""
        # Prefer usage accumulator when available (matches enhanced/interactive prompt display)
        usage = getattr(agent, "usage_accumulator", None)
        if usage:
            window = usage.context_window_size
            tokens = usage.current_context_tokens
            pct = usage.context_usage_percentage
            if window and pct is not None:
                return f"- Context Used: {min(pct, 100.0):.1f}% (~{tokens:,} tokens of {window:,})"
            if tokens:
                return f"- Context Used: ~{tokens:,} tokens (window unknown)"

        # Fallback to tokenizing the actual conversation text
        token_count, char_count = self._estimate_tokens(summary, agent)

        model_info = ModelInfo.from_llm(agent.llm) if getattr(agent, "llm", None) else None
        if model_info and model_info.context_window:
            percentage = (
                (token_count / model_info.context_window) * 100
                if model_info.context_window
                else 0.0
            )
            percentage = min(percentage, 100.0)
            return f"- Context Used: {percentage:.1f}% (~{token_count:,} tokens of {model_info.context_window:,})"

        token_text = f"~{token_count:,} tokens" if token_count else "~0 tokens"
        return f"- Context Used: {char_count:,} chars ({token_text} est.)"

    def _estimate_tokens(self, summary: ConversationSummary, agent) -> tuple[int, int]:
        """Estimate tokens and return (tokens, characters) for the conversation history."""
        text_parts: list[str] = []
        for message in summary.messages:
            for content in getattr(message, "content", []) or []:
                text = get_text(content)
                if text:
                    text_parts.append(text)

        combined = "\n".join(text_parts)
        char_count = len(combined)
        if not combined:
            return 0, 0

        model_name = None
        llm = getattr(agent, "llm", None)
        if llm:
            model_name = getattr(llm, "model_name", None)

        token_count = self._count_tokens_with_tiktoken(combined, model_name)
        return token_count, char_count

    def _count_tokens_with_tiktoken(self, text: str, model_name: str | None) -> int:
        """Try to count tokens with tiktoken; fall back to a rough chars/4 estimate."""
        try:
            import tiktoken

            if model_name:
                encoding = tiktoken.encoding_for_model(model_name)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except Exception:
            # Rough heuristic: ~4 characters per token (matches default bytes/token constant)
            return max(1, (len(text) + 3) // 4)
