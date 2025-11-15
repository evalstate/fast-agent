"""
Slash Commands for ACP

Provides slash command support for the ACP server, allowing clients to
discover and invoke special commands with the /command syntax.
"""

from __future__ import annotations

import textwrap
import time
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING

from acp.schema import AvailableCommand, AvailableCommandInput, CommandInputHint

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.llm.model_info import ModelInfo
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types.conversation_summary import ConversationSummary
from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from mcp.types import ListToolsResult, Tool

    from fast_agent.core.fastagent import AgentInstance


class SlashCommandHandler:
    """Handles slash command execution for ACP sessions."""

    def __init__(
        self,
        session_id: str,
        instance: AgentInstance,
        primary_agent_name: str,
        *,
        current_agent_name: str | None = None,
        history_exporter: type[HistoryExporter] | HistoryExporter | None = None,
        client_info: dict | None = None,
        client_capabilities: dict | None = None,
        protocol_version: str | None = None,
    ):
        """
        Initialize the slash command handler.

        Args:
            session_id: The ACP session ID
            instance: The agent instance for this session
            primary_agent_name: Name of the primary agent
            current_agent_name: Name of the currently active agent (defaults to primary)
            history_exporter: Optional history exporter
            client_info: Client information from ACP initialize
            client_capabilities: Client capabilities from ACP initialize
            protocol_version: ACP protocol version
        """
        self.session_id = session_id
        self.instance = instance
        self.primary_agent_name = primary_agent_name
        self.current_agent_name = current_agent_name or primary_agent_name
        self.history_exporter = history_exporter or HistoryExporter
        self._created_at = time.time()
        self.client_info = client_info
        self.client_capabilities = client_capabilities
        self.protocol_version = protocol_version

        # Register available commands using SDK's AvailableCommand type
        self.commands: dict[str, AvailableCommand] = {
            "status": AvailableCommand(
                name="status",
                description="Show fast-agent diagnostics",
                input=AvailableCommandInput(root=CommandInputHint(hint="[agent]")),
            ),
            "tools": AvailableCommand(
                name="tools",
                description="List available MCP tools",
                input=None,
            ),
            "save": AvailableCommand(
                name="save",
                description="Save conversation history",
                input=None,
            ),
            "clear": AvailableCommand(
                name="clear",
                description="Clear history (`last` for prev. turn)",
                input=AvailableCommandInput(root=CommandInputHint(hint="[last]")),
            ),
        }

    def get_available_commands(self) -> list[AvailableCommand]:
        """Get the list of available commands for this session."""
        return list(self.commands.values())

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
        parts = text.split(None, 1)
        command_name = parts[0] if parts else ""
        arguments = parts[1] if len(parts) > 1 else ""

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
        if command_name not in self.commands:
            return f"Unknown command: /{command_name}\n\nAvailable commands:\n" + "\n".join(
                f"  /{cmd.name} - {cmd.description}" for cmd in self.commands.values()
            )

        # Route to specific command handler
        if command_name == "status":
            return await self._handle_status(arguments)
        if command_name == "tools":
            return await self._handle_tools()
        if command_name == "save":
            return await self._handle_save(arguments)
        if command_name == "clear":
            return await self._handle_clear(arguments)

        return f"Command /{command_name} is not yet implemented."

    async def _handle_status(self, arguments: str | None = None) -> str:
        """
        Handle the /status command.

        Args:
            arguments: Optional agent name to show detailed info for specific agent

        Returns:
            Status information as formatted markdown
        """
        # If arguments provided, show detailed agent info
        agent_name_arg = arguments.strip() if arguments else None
        if agent_name_arg:
            return self._handle_status_agent(agent_name_arg)

        # Otherwise show general status with mode awareness
        return self._handle_status_general()

    def _handle_status_general(self) -> str:
        """Handle /status with no arguments - show general status with mode info."""
        # Get fast-agent version
        try:
            fa_version = get_version("fast-agent-mcp")
        except Exception:
            fa_version = "unknown"

        # Get current agent
        agent = self.instance.agents.get(self.current_agent_name)

        # Get model information from current agent
        model_name = "unknown"
        model_provider = "unknown"
        model_provider_display = "unknown"
        context_window = "unknown"
        capabilities_line = "Capabilities: unknown"

        if agent and hasattr(agent, "_llm") and agent._llm:
            model_info = ModelInfo.from_llm(agent._llm)
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

        # Get conversation statistics for current agent
        summary_stats = self._get_conversation_stats(agent)

        # Format the status response
        status_lines = [
            "# fast-agent ACP status",
            "",
            "## Version",
            f"fast-agent: {fa_version}",
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
                status_lines.extend(["", "### Client Capabilities"])

                # Filesystem capabilities
                if "fs" in self.client_capabilities:
                    fs_caps = self.client_capabilities["fs"]
                    if fs_caps:
                        status_lines.append("Filesystem:")
                        for key, value in fs_caps.items():
                            status_lines.append(f"  - {key}: {value}")

                # Terminal capability
                if "terminal" in self.client_capabilities:
                    status_lines.append(f"Terminal: {self.client_capabilities['terminal']}")

                # Meta capabilities
                if "_meta" in self.client_capabilities:
                    meta_caps = self.client_capabilities["_meta"]
                    if meta_caps:
                        status_lines.append("Meta:")
                        for key, value in meta_caps.items():
                            status_lines.append(f"  - {key}: {value}")

            status_lines.append("")

        # Add mode information
        status_lines.extend(self._get_mode_info())
        status_lines.append("")

        # Add active model information for current agent
        provider_line = f"{model_provider}"
        if model_provider_display != "unknown":
            provider_line = f"{model_provider_display} ({model_provider})"

        status_lines.extend(
            [
                "## Active Model",
                f"- Provider: {provider_line}",
                f"- Model: {model_name}",
                f"- Context Window: {context_window}",
                f"- {capabilities_line}",
                "",
                "## Conversation Statistics",
            ]
        )

        uptime_seconds = max(time.time() - self._created_at, 0.0)
        status_lines.extend(summary_stats)
        status_lines.extend(["", f"ACP Agent Uptime: {format_duration(uptime_seconds)}"])
        status_lines.extend(["", "## Error Handling"])
        status_lines.extend(self._get_error_handling_report(agent))

        return "\n".join(status_lines)

    def _handle_status_agent(self, agent_name: str) -> str:
        """Handle /status <agent> - show detailed info for a specific agent."""
        # Check if agent exists
        if agent_name not in self.instance.agents:
            available = ", ".join(self.instance.agents.keys())
            return "\n".join([
                f"# status {agent_name}",
                "",
                f"Agent '{agent_name}' not found.",
                "",
                f"Available agents: {available}",
            ])

        agent = self.instance.agents[agent_name]

        # Get agent configuration
        agent_config = getattr(agent, "_config", None) or getattr(agent, "config", None)
        instruction = ""
        agent_type = "unknown"
        is_default = False
        mcp_servers = []

        if agent_config:
            instruction = getattr(agent_config, "instruction", "")
            agent_type = str(getattr(agent_config, "agent_type", "unknown"))
            is_default = getattr(agent_config, "default", False)
            mcp_servers = getattr(agent_config, "servers", [])

        # Get model information
        model_name = "unknown"
        model_provider = "unknown"
        model_provider_display = "unknown"
        context_window = "unknown"
        capabilities_line = "Capabilities: unknown"

        if hasattr(agent, "_llm") and agent._llm:
            model_info = ModelInfo.from_llm(agent._llm)
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

        # Format agent name for display
        formatted_name = self._format_agent_name_as_title(agent_name)

        # Build response
        status_lines = [
            f"# status {agent_name}",
            "",
            f"**{formatted_name}**",
            "",
        ]

        if is_default:
            status_lines.append("*Default Agent*")
            status_lines.append("")

        if self.current_agent_name == agent_name:
            status_lines.append("*Currently Active*")
            status_lines.append("")

        # Add agent configuration
        status_lines.extend([
            "## Configuration",
            f"- Type: {agent_type}",
        ])

        if instruction:
            # Show first line of instruction
            first_line = instruction.split("\n")[0]
            if len(first_line) > 100:
                first_line = first_line[:97] + "..."
            status_lines.append(f"- Instruction: {first_line}")

        # Add model information
        provider_line = f"{model_provider}"
        if model_provider_display != "unknown":
            provider_line = f"{model_provider_display} ({model_provider})"

        status_lines.extend([
            "",
            "## Model",
            f"- Provider: {provider_line}",
            f"- Model: {model_name}",
            f"- Context Window: {context_window}",
            f"- {capabilities_line}",
        ])

        # Add MCP servers if any
        if mcp_servers:
            status_lines.extend([
                "",
                "## MCP Servers",
            ])
            for server in mcp_servers:
                status_lines.append(f"- {server}")

        # Add conversation statistics
        status_lines.extend([
            "",
            "## Conversation Statistics",
        ])
        status_lines.extend(summary_stats)

        # Add error handling report
        status_lines.extend(["", "## Error Handling"])
        status_lines.extend(self._get_error_handling_report(agent))

        return "\n".join(status_lines)

    def _get_mode_info(self) -> list[str]:
        """Get current mode and available modes information."""
        lines = ["## Session Modes"]

        # Current mode
        current_formatted = self._format_agent_name_as_title(self.current_agent_name)
        lines.append(f"**Current Mode**: {current_formatted} (`{self.current_agent_name}`)")

        # Available modes count
        mode_count = len(self.instance.agents)
        lines.append(f"**Available Modes**: {mode_count}")

        if mode_count > 1:
            lines.append("")
            lines.append("### All Available Modes")
            for agent_name in sorted(self.instance.agents.keys()):
                formatted_name = self._format_agent_name_as_title(agent_name)

                # Get agent description (first line of instruction)
                agent = self.instance.agents[agent_name]
                agent_config = getattr(agent, "_config", None) or getattr(agent, "config", None)
                description = ""
                if agent_config:
                    instruction = getattr(agent_config, "instruction", "")
                    if instruction:
                        first_line = instruction.split("\n")[0].strip()
                        if len(first_line) > 80:
                            first_line = first_line[:77] + "..."
                        description = first_line

                # Mark current agent
                marker = " *(active)*" if agent_name == self.current_agent_name else ""

                if description:
                    lines.append(f"- **{formatted_name}** (`{agent_name}`){marker}: {description}")
                else:
                    lines.append(f"- **{formatted_name}** (`{agent_name}`){marker}")

        return lines

    def _format_agent_name_as_title(self, agent_name: str) -> str:
        """
        Format agent name as title case for display.

        Examples:
            code_expert -> Code Expert
            general_assistant -> General Assistant

        Args:
            agent_name: The agent name (typically snake_case)

        Returns:
            Title-cased version of the name
        """
        return agent_name.replace("_", " ").title()

    async def _handle_tools(self) -> str:
        """List available MCP tools for the primary agent."""
        heading = "# tools"

        agent = self.instance.agents.get(self.primary_agent_name)
        if not agent:
            return "\n".join(
                [
                    heading,
                    "",
                    f"Agent '{self.primary_agent_name}' not found for this session.",
                ]
            )

        list_tools = getattr(agent, "list_tools", None)
        if not callable(list_tools):
            return "\n".join(
                [
                    heading,
                    "",
                    "This agent does not expose a list_tools() method.",
                ]
            )

        try:
            tools_result: "ListToolsResult" = await list_tools()
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

        agent = self.instance.agents.get(self.primary_agent_name)
        if not agent:
            return "\n".join(
                [
                    heading,
                    "",
                    f"Unable to locate agent '{self.primary_agent_name}' for this session.",
                ]
            )

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

    async def _handle_clear(self, arguments: str | None = None) -> str:
        """Handle /clear and /clear last commands."""
        normalized = (arguments or "").strip().lower()
        if normalized == "last":
            return self._handle_clear_last()
        return self._handle_clear_all()

    def _handle_clear_all(self) -> str:
        """Clear the entire conversation history."""
        heading = "# clear conversation"
        agent = self.instance.agents.get(self.primary_agent_name)
        if not agent:
            return "\n".join(
                [
                    heading,
                    "",
                    f"Unable to locate agent '{self.primary_agent_name}' for this session.",
                ]
            )

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
        agent = self.instance.agents.get(self.primary_agent_name)
        if not agent:
            return "\n".join(
                [
                    heading,
                    "",
                    f"Unable to locate agent '{self.primary_agent_name}' for this session.",
                ]
            )

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

            # Calculate context usage percentage (estimate)
            # This is a rough estimate based on message count and typical token usage
            # A more accurate calculation would require token counting
            context_used_pct = self._estimate_context_usage(summary, agent)

            stats = [
                f"- Turns: {turns}",
                f"- Messages: {summary.message_count} (user: {summary.user_message_count}, assistant: {summary.assistant_message_count})",
                f"- Tool Calls: {tool_calls} (successes: {tool_successes}, errors: {tool_errors})",
                f"- Context Used: ~{context_used_pct:.1f}%",
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
            return [channel_label, "Recent Entries: unavailable (no agent history)"]

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
                    recent_entries.append(str(block))
                if len(recent_entries) >= max_entries:
                    break
            if len(recent_entries) >= max_entries:
                break

        if recent_entries:
            lines = [channel_label, "Recent Entries:"]
            lines.extend(f"- {entry}" for entry in recent_entries)
            return lines

        return [channel_label, "Recent Entries: none recorded"]

    def _estimate_context_usage(self, summary: ConversationSummary, agent) -> float:
        """
        Estimate context usage as a percentage.

        This is a rough estimate based on message count.
        A more accurate calculation would require actual token counting.
        """
        if not hasattr(agent, "_llm") or not agent._llm:
            return 0.0

        model_info = ModelInfo.from_llm(agent._llm)
        if not model_info or not model_info.context_window:
            return 0.0

        # Very rough estimate: assume average of 500 tokens per message
        # This includes both user and assistant messages
        estimated_tokens = summary.message_count * 500

        context_window = model_info.context_window
        percentage = (estimated_tokens / context_window) * 100

        # Cap at 100%
        return min(percentage, 100.0)
