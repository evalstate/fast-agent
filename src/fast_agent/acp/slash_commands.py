"""
Slash Commands for ACP

Provides slash command support for the ACP server, allowing clients to
discover and invoke special commands with the /command syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING, Optional

from fast_agent.llm.model_info import ModelInfo
from fast_agent.types.conversation_summary import ConversationSummary

if TYPE_CHECKING:
    from fast_agent.core.fastagent import AgentInstance


@dataclass
class AvailableCommand:
    """Represents a slash command available in the session."""

    name: str
    description: str
    input_hint: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format for ACP notification."""
        result = {
            "name": self.name,
            "description": self.description,
        }
        if self.input_hint:
            result["input"] = {"hint": self.input_hint}
        return result


class SlashCommandHandler:
    """Handles slash command execution for ACP sessions."""

    def __init__(self, session_id: str, instance: AgentInstance, primary_agent_name: str):
        """
        Initialize the slash command handler.

        Args:
            session_id: The ACP session ID
            instance: The agent instance for this session
            primary_agent_name: Name of the primary agent
        """
        self.session_id = session_id
        self.instance = instance
        self.primary_agent_name = primary_agent_name

        # Register available commands
        self.commands: dict[str, AvailableCommand] = {
            "status": AvailableCommand(
                name="status",
                description="Show fast-agent version, model, and context usage statistics",
                input_hint=None,
            ),
        }

    def get_available_commands(self) -> list[dict]:
        """Get the list of available commands for this session."""
        return [cmd.to_dict() for cmd in self.commands.values()]

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
            return await self._handle_status()

        return f"Command /{command_name} is not yet implemented."

    async def _handle_status(self) -> str:
        """Handle the /status command."""
        # Get fast-agent version
        try:
            fa_version = get_version("fast-agent-mcp")
        except Exception:
            fa_version = "unknown"

        # Get model information
        agent = self.instance.agents.get(self.primary_agent_name)
        model_name = "unknown"
        model_provider = "unknown"
        context_window = "unknown"

        if agent and hasattr(agent, "_llm") and agent._llm:
            model_info = ModelInfo.from_llm(agent._llm)
            if model_info:
                model_name = model_info.name
                model_provider = str(model_info.provider.value)
                context_window = (
                    str(model_info.context_window) if model_info.context_window else "unknown"
                )

        # Get conversation statistics
        summary_stats = self._get_conversation_stats(agent)

        # Format the status response
        status_lines = [
            "# Fast-Agent Status",
            "",
            "## Version",
            f"fast-agent: {fa_version}",
            "",
            "## Model",
            f"Name: {model_name}",
            f"Provider: {model_provider}",
            f"Context Window: {context_window} tokens",
            "",
            "## Conversation Statistics",
        ]

        status_lines.extend(summary_stats)

        return "\n".join(status_lines)

    def _get_conversation_stats(self, agent) -> list[str]:
        """Get conversation statistics from the agent's message history."""
        if not agent or not hasattr(agent, "message_history"):
            return [
                "Turns: 0",
                "Tool Calls: 0",
                "Context Used: 0%",
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
                f"Turns: {turns}",
                f"Messages: {summary.message_count} (user: {summary.user_message_count}, assistant: {summary.assistant_message_count})",
                f"Tool Calls: {tool_calls} (successes: {tool_successes}, errors: {tool_errors})",
                f"Context Used: ~{context_used_pct:.1f}%",
            ]

            # Add timing information if available
            if summary.total_elapsed_time_ms > 0:
                stats.append(
                    f"Total LLM Time: {summary.total_elapsed_time_ms / 1000:.2f}s"
                )

            if summary.conversation_span_ms > 0:
                stats.append(
                    f"Conversation Duration: {summary.conversation_span_ms / 1000:.2f}s"
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
                "Turns: error",
                "Tool Calls: error",
                f"Context Used: error ({e})",
            ]

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
