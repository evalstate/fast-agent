"""ACP adapter implementation for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from rich.text import Text

from fast_agent.commands.context import AgentProvider, CommandIO
from fast_agent.commands.results import CommandMessage
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types.conversation_summary import ConversationSummary

if TYPE_CHECKING:
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


@dataclass(slots=True)
class AcpCommandIO(CommandIO):
    """Command IO implementation for ACP slash commands."""

    agent_provider: AgentProvider
    agent_name: str
    session_instructions: dict[str, str] | None = None
    messages: list[CommandMessage] = field(default_factory=list)

    async def emit(self, message: CommandMessage) -> None:
        self.messages.append(message)

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        _ = allow_empty
        await self.emit(
            CommandMessage(
                text=f"Interactive input is unavailable for ACP. Provide text in the command: {prompt}",
                channel="warning",
                agent_name=self.agent_name,
            )
        )
        return default

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        _ = (options, allow_cancel)
        await self.emit(
            CommandMessage(
                text=f"Interactive selection is unavailable for ACP. Provide a value with the command: {prompt}",
                channel="warning",
                agent_name=self.agent_name,
            )
        )
        return default

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        detail = f" ({description})" if description else ""
        required_note = "required" if required else "optional"
        await self.emit(
            CommandMessage(
                text=(
                    "Prompt arguments are not interactive in ACP. "
                    f"Provide {required_note} argument '{arg_name}'{detail} in the command payload."
                ),
                channel="warning",
                agent_name=self.agent_name,
            )
        )
        return None

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list["PromptMessageExtended"],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        heading = "# history turn"
        if turn_index is not None:
            heading = f"# history turn {turn_index}"
            if total_turns is not None:
                heading = f"{heading}/{total_turns}"

        lines = [heading, "", f"Agent: {agent_name}", ""]
        for message in turn:
            role = getattr(message, "role", "message")
            if hasattr(role, "value"):
                role = role.value
            text = ""
            if hasattr(message, "all_text"):
                text = message.all_text() or message.first_text() or ""
            if not text:
                content = getattr(message, "content", None)
                if isinstance(content, list) and content:
                    text = get_text(content[0]) or ""
                elif content is not None:
                    text = get_text(content) or ""
            text = " ".join(text.split()) if text else "<no text>"
            lines.append(f"- {role}: {text}")

        await self.emit(CommandMessage(text="\n".join(lines), agent_name=agent_name))

    async def display_history_overview(
        self,
        agent_name: str,
        history: list["PromptMessageExtended"],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        _ = usage
        summary = ConversationSummary(messages=history)
        lines = [
            "# conversation history",
            "",
            f"Agent: {agent_name}",
            "",
            (
                "Messages: "
                f"{summary.message_count} (user: {summary.user_message_count}, "
                f"assistant: {summary.assistant_message_count})"
            ),
            (
                "Tool Calls: "
                f"{summary.tool_calls} (successes: {summary.tool_successes}, "
                f"errors: {summary.tool_errors})"
            ),
        ]

        recent_messages = history[-5:]
        if recent_messages:
            lines.append("")
            lines.append(f"Recent {len(recent_messages)} messages:")
            for message in recent_messages:
                role = getattr(message, "role", "message")
                if hasattr(role, "value"):
                    role = role.value
                text = ""
                if hasattr(message, "all_text"):
                    text = message.all_text() or message.first_text() or ""
                if not text:
                    content = getattr(message, "content", None)
                    if isinstance(content, list) and content:
                        text = get_text(content[0]) or ""
                    elif content is not None:
                        text = get_text(content) or ""
                snippet = " ".join(text.split())
                if not snippet:
                    snippet = "(no text content)"
                if len(snippet) > 60:
                    snippet = f"{snippet[:57]}..."
                lines.append(f"- {role}: {snippet}")
        else:
            lines.append("")
            lines.append("No messages yet.")

        await self.emit(CommandMessage(text="\n".join(lines), agent_name=agent_name))

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        lines = ["# usage", ""]
        usage_lines: list[str] = []

        for name, agent in agents.items():
            usage = getattr(agent, "usage_accumulator", None)
            if not usage:
                continue
            summary = usage.get_summary()
            if summary.get("turn_count", 0) <= 0:
                continue

            model = getattr(usage, "model", None)
            if not model:
                llm = getattr(agent, "llm", None)
                model = getattr(llm, "model_name", None) if llm else None
            model_text = f" ({model})" if model else ""

            context_pct = getattr(usage, "context_usage_percentage", None)
            context_text = f", context {context_pct:.1f}%" if context_pct is not None else ""

            usage_lines.append(
                "- "
                f"{name}{model_text}: input {summary.get('cumulative_input_tokens', 0)}, "
                f"output {summary.get('cumulative_output_tokens', 0)}, "
                f"total {summary.get('cumulative_billing_tokens', 0)}, "
                f"turns {summary.get('turn_count', 0)}, "
                f"tools {summary.get('cumulative_tool_calls', 0)}"
                f"{context_text}"
            )

        if not usage_lines:
            await self.emit(
                CommandMessage(
                    text="No usage data available",
                    channel="warning",
                    agent_name=self.agent_name,
                )
            )
            return

        lines.extend(usage_lines)
        await self.emit(CommandMessage(text="\n".join(lines), agent_name=self.agent_name))

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        override = None
        if self.session_instructions:
            override = self.session_instructions.get(agent_name)
        prompt = override or system_prompt
        heading = "# system prompt"

        if not prompt:
            await self.emit(
                CommandMessage(
                    text="\n".join([heading, "", "No system prompt available for this agent."]),
                    channel="warning",
                    agent_name=agent_name,
                )
            )
            return

        server_line = f"MCP servers: {server_count}" if server_count else None
        lines = [heading, "", f"Agent: {agent_name}"]
        if server_line:
            lines.append(server_line)
        lines.extend(["", prompt])

        await self.emit(CommandMessage(text="\n".join(lines), agent_name=agent_name))

    def render_response(self) -> str:
        if not self.messages:
            return ""
        parts: list[str] = []
        for message in self.messages:
            text = message.text
            if isinstance(text, Text):
                text_value = text.plain
            else:
                text_value = str(text)

            prefix = ""
            if message.channel == "error":
                prefix = "Error: "
            elif message.channel == "warning":
                prefix = "Warning: "
            elif message.channel == "info":
                prefix = "Info: "

            if message.title:
                title = message.title
                if text_value:
                    parts.append(f"{title}\n{text_value}")
                else:
                    parts.append(title)
                continue

            parts.append(f"{prefix}{text_value}" if prefix else text_value)

        return "\n\n".join(part for part in parts if part)
