"""Textual example that streams an LLM response with familiar console styling."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text as RichText
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from fast_agent import FastAgent
from fast_agent.constants import REASONING
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.markdown_helpers import prepare_markdown_content
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from fast_agent.interfaces import AgentProtocol

DEFAULT_PROMPT = (
    "Provide a short markdown summary with a heading and bullet list describing how "
    "Textual can be paired with fast-agent to build rich terminal apps."
)
DEFAULT_MODEL = "kimi"
CHAT_AGENT_NAME = "textual_markdown_chat"
CONFIG_PATH = Path(__file__).with_name("fastagent.config.yaml")

fast = FastAgent(
    "Textual Markdown Demo",
    config_path=str(CONFIG_PATH),
    parse_cli_args=False,
    quiet=True,
)


@fast.agent(
    name=CHAT_AGENT_NAME,
    instruction="You are a friendly assistant that responds in concise, well-formatted markdown.",
    servers=["filesystem", "fetch"],
    model=DEFAULT_MODEL,
    default=True,
)
async def textual_markdown_agent() -> None:
    """Placeholder callable for registering the chat agent with FastAgent."""
    pass


def _format_prompt(prompt: str) -> str:
    """Trim and format a prompt as a markdown quote block."""
    stripped_lines = [line.rstrip() for line in prompt.strip().splitlines()]
    return "\n".join(f"> {line if line else ' '}" for line in stripped_lines)


@dataclass
class AppOptions:
    """Runtime options for the Textual application."""

    prompt: str = DEFAULT_PROMPT
    model: str = DEFAULT_MODEL


@dataclass
class ChatMessage:
    """Represents a chat message rendered in the Textual log."""

    role: str
    content: str = ""
    name: str | None = None
    right_info: str | None = None
    bottom_metadata: list[str] | None = None
    highlight_index: int | None = None
    max_item_length: int | None = None
    pre_content: str | None = None
    block_color_override: str | None = None
    arrow_style_override: str | None = None
    arrow_override: str | None = None
    highlight_color_override: str | None = None


ROLE_TO_MESSAGE_TYPE = {
    "user": MessageType.USER,
    "assistant": MessageType.ASSISTANT,
    "system": MessageType.SYSTEM,
    "tool_call": MessageType.TOOL_CALL,
    "tool_result": MessageType.TOOL_RESULT,
}


class ChatDisplay(RichLog):
    """Rich log that renders chat messages with familiar console styling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args, auto_scroll=False, highlight=False, markup=False, wrap=False, **kwargs
        )
        self._messages: list[ChatMessage] = []
        self.can_focus = True
        self.show_vertical_scrollbar = True
        self.show_horizontal_scrollbar = False

    def update_messages(self, messages: Sequence[ChatMessage]) -> None:
        self._messages = list(messages)
        auto_scroll = self.is_vertical_scroll_end
        self.clear()

        if not self._messages:
            self.write(RichText("_No messages yet. Type a prompt below to begin._", style="dim"))
        else:
            for message in self._messages:
                renderable = self._build_renderable(message)
                self.write(renderable)

        if auto_scroll:
            self.scroll_end()

    def _build_renderable(self, message: ChatMessage) -> Group:
        message_type = ROLE_TO_MESSAGE_TYPE.get(message.role, MessageType.ASSISTANT)
        config = MESSAGE_CONFIGS[message_type]

        header = self._build_header(message, config)
        body = self._build_body(message)
        bottom_bar = self._build_bottom_bar(message, config)

        segments = [header]
        if body is not None:
            segments.append(body)
        if bottom_bar is not None:
            segments.append(bottom_bar)
        segments.append(RichText(""))
        return Group(*segments)

    def _build_header(self, message: ChatMessage, config: dict) -> RichText:
        display_name = message.name
        if not display_name:
            if message.role == "user":
                display_name = "You"
            elif message.role == "assistant":
                display_name = "Assistant"
            else:
                display_name = message.role.capitalize()

        block_color = message.block_color_override or config["block_color"]
        arrow_style = message.arrow_style_override or config["arrow_style"]
        arrow = message.arrow_override or config["arrow"]

        left_parts = [
            f"[{block_color}]▎[/{block_color}]",
            f"[{arrow_style}]{arrow}[/{arrow_style}]",
        ]
        if display_name:
            left_parts.append(f" [{block_color}]{display_name}[/{block_color}]")
        left_markup = "".join(left_parts)

        right_markup = ""
        if message.right_info:
            right_markup = f"[dim]{message.right_info}[/dim]"

        width = self.size.width or (self.app.size.width if self.app else 0) or 80
        width = max(width, 40)

        left_text = RichText.from_markup(left_markup)

        if right_markup and right_markup.strip():
            right_text = RichText()
            right_text.append("[", style="dim")
            right_text.append_text(RichText.from_markup(right_markup))
            right_text.append("]", style="dim")
            separator_count = width - left_text.cell_len - right_text.cell_len
            if separator_count < 1:
                separator_count = 1
        else:
            right_text = RichText()
            separator_count = width - left_text.cell_len

        combined = RichText()
        combined.append_text(left_text)
        combined.append(" ", style="default")
        combined.append("─" * max(1, separator_count - 1), style="dim")
        combined.append_text(right_text)

        return combined

    def _build_body(self, message: ChatMessage):
        segments: list = []

        if message.pre_content:
            segments.append(RichText(message.pre_content, style="dim"))

        content = message.content or ""

        if message.role == "user":
            formatted = _format_prompt(content) if content else ""
            if formatted:
                segments.append(Markdown(formatted))
        else:
            prepared = prepare_markdown_content(content, True) if content else ""
            if prepared:
                segments.append(Markdown(prepared))

        return Group(*segments) if segments else None

    def _build_bottom_bar(self, message: ChatMessage, config: dict) -> RichText | None:
        width = self.size.width or (self.app.size.width if self.app else 0) or 80

        items = message.bottom_metadata or []
        if not items:
            return RichText("─" * width, style="dim")

        display_items = items
        if message.max_item_length:
            display_items = self._shorten_items(display_items, message.max_item_length)

        prefix = RichText("─| ", style="dim")
        suffix = RichText(" |", style="dim")
        available = max(0, width - prefix.cell_len - suffix.cell_len)

        highlight_color = message.highlight_color_override or config["highlight_color"]

        metadata_text = self._format_metadata(
            display_items,
            message.highlight_index,
            highlight_color,
            available,
        )

        line = RichText()
        line.append_text(prefix)
        line.append_text(metadata_text)
        line.append_text(suffix)
        remaining = width - line.cell_len
        if remaining > 0:
            line.append("─" * remaining, style="dim")
        return line

    @staticmethod
    def _shorten_items(items: Sequence[str], max_length: int) -> list[str]:
        return [item[: max_length - 1] + "…" if len(item) > max_length else item for item in items]

    def _format_metadata(
        self,
        items: Sequence[str],
        highlight_index: int | None,
        highlight_color: str,
        max_width: int,
    ) -> RichText:
        formatted = RichText()

        if max_width <= 0:
            return RichText()

        def will_fit(additional_len: int) -> bool:
            return formatted.cell_len + additional_len <= max_width

        for index, item in enumerate(items):
            sep = RichText(" | ", style="dim") if index > 0 else RichText()
            item_style = (
                highlight_color
                if highlight_index is not None and index == highlight_index
                else "dim"
            )
            item_text = RichText(item, style=item_style)

            additional_len = sep.cell_len + item_text.cell_len
            if not will_fit(additional_len):
                if formatted.cell_len == 0 and max_width > 1:
                    formatted.append("…", style="dim")
                elif formatted.cell_len < max_width:
                    formatted.append(" …", style="dim")
                break

            if sep.plain:
                formatted.append_text(sep)
            formatted.append_text(item_text)

        return formatted


class TextualDisplay(ConsoleDisplay):
    """ConsoleDisplay replacement that forwards events to the Textual app."""

    def __init__(self, app: "MarkdownLLMApp", config=None) -> None:
        super().__init__(config=config)
        self._app = app

    def show_user_message(
        self,
        message: str | RichText,
        model: str | None = None,
        chat_turn: int = 0,
        name: str | None = None,
    ) -> None:
        content = message.plain if isinstance(message, RichText) else str(message)
        self._app.handle_display_user_message(content, model, chat_turn, name)

    async def show_assistant_message(
        self,
        message_text: str | RichText | PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: RichText | None = None,
    ) -> None:
        pre_content: str | None = None
        display_text: str = ""

        if isinstance(message_text, PromptMessageExtended):
            display_text = message_text.last_text() or ""
            channels = message_text.channels or {}
            reasoning_blocks = channels.get(REASONING) or []
            reasoning_segments = []
            for block in reasoning_blocks:
                text = get_text(block)
                if text:
                    reasoning_segments.append(text)
            if reasoning_segments:
                pre_content = "\n".join(reasoning_segments)
        elif isinstance(message_text, RichText):
            display_text = message_text.plain
        else:
            display_text = str(message_text) if message_text is not None else ""

        if isinstance(additional_message, RichText):
            additional_text = additional_message.plain
        elif additional_message is not None:
            additional_text = str(additional_message)
        else:
            additional_text = None

        await self._app.handle_display_assistant_message(
            content=display_text,
            pre_content=pre_content,
            bottom_items=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            name=name,
            model=model,
            additional_text=additional_text,
        )

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict | None,
        bottom_items: list[str] | None = None,
        name: str | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._app.handle_display_tool_call(
            agent_name=name,
            tool_name=tool_name,
            tool_args=tool_args or {},
            bottom_items=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            metadata=metadata,
        )

    def show_tool_result(
        self,
        result: CallToolResult,
        name: str | None = None,
        tool_name: str | None = None,
        skybridge_config=None,
    ) -> None:
        self._app.handle_display_tool_result(result, agent_name=name, tool_name=tool_name)


class MarkdownLLMApp(App[None]):
    """Textual application that displays an LLM response in a chat-style Markdown widget."""

    CSS = """
    Screen {
        layout: vertical;
    }

    RichLog#chat {
        height: 1fr;
        padding: 0 2;
    }

    Input#prompt {
        margin: 1 2;
    }
    """
    BINDINGS = [
        ("r", "regenerate", "Regenerate"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, options: AppOptions) -> None:
        super().__init__()
        self._options = options
        self._active_prompt: str = options.prompt
        self._messages: list[ChatMessage] = []
        self._chat_display: ChatDisplay | None = None
        self._prompt_input: Input | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._agent: AgentProtocol | None = None
        self._agent_app = None
        self._agent_context = None
        self._chat_turn = 0
        self._active_user_message: ChatMessage | None = None
        self._active_assistant_message: ChatMessage | None = None

        if CHAT_AGENT_NAME in fast.agents:
            fast.agents[CHAT_AGENT_NAME]["config"].model = options.model

    def compose(self) -> ComposeResult:
        header = Header(show_clock=True)
        chat = ChatDisplay(id="chat")
        prompt_input = Input(
            value=self._options.prompt,
            placeholder="Type a prompt and press Enter (R to regenerate)",
            id="prompt",
        )
        footer = Footer()
        self._chat_display = chat
        self._prompt_input = prompt_input
        yield header
        yield chat
        yield prompt_input
        yield footer

    async def on_mount(self) -> None:
        try:
            await self._ensure_agent()
        except Exception as exc:  # pragma: no cover - runtime feedback only
            self._append_error_response(f"```text\n{exc}\n```")
            self._set_status("Error")
            return

        if self._prompt_input:
            self._prompt_input.focus()

        self._refresh_chat()

        if self._options.prompt:
            self._start_generation(self._options.prompt)
        else:
            self._set_status("Ready")

    async def on_unmount(self) -> None:
        if self._current_task:
            self._current_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._current_task
            self._current_task = None
        await self._shutdown_agent()

    async def _ensure_agent(self) -> AgentProtocol | None:
        if self._agent:
            return self._agent
        if not self._agent_context:
            self._agent_context = fast.run()
            self._agent_app = await self._agent_context.__aenter__()
        agent = getattr(self._agent_app, CHAT_AGENT_NAME, None) if self._agent_app else None
        self._agent = agent
        if agent and not isinstance(getattr(agent, "display", None), TextualDisplay):
            context = getattr(agent, "_context", None)
            config = getattr(context, "config", None) if context else None
            agent.display = TextualDisplay(self, config=config)
        return agent

    async def _shutdown_agent(self) -> None:
        if self._agent_context:
            with suppress(Exception):
                await self._agent_context.__aexit__(None, None, None)
            self._agent_context = None
            self._agent_app = None
            self._agent = None

    def _start_generation(self, prompt: str | None = None) -> None:
        if self._current_task and not self._current_task.done():
            return
        self._active_assistant_message = None
        self._active_user_message = None
        prompt_value = (prompt or self._current_prompt_value()).strip()
        if not prompt_value:
            prompt_value = DEFAULT_PROMPT
        self._active_prompt = prompt_value
        self._options.prompt = prompt_value
        if self._prompt_input and self._prompt_input.value != prompt_value:
            self._prompt_input.value = prompt_value
        self._chat_turn += 1
        user_message = ChatMessage(
            role="user",
            content=prompt_value,
            name="You",
            right_info=self._user_right_info(self._options.model, self._chat_turn),
        )
        self._messages.append(user_message)
        self._active_user_message = user_message
        self._refresh_chat()
        self._set_status("Preparing response…")
        self._current_task = asyncio.create_task(self._generate_and_render())

    async def _generate_and_render(self) -> None:
        try:
            agent = await self._ensure_agent()
            if not agent:
                self._append_error_response("```text\nAgent failed to initialize.\n```")
                self._set_status("Error")
                return

            queue: asyncio.Queue[str] = asyncio.Queue()
            response_text: str | None = None
            received_stream_chunks = False

            def on_chunk(chunk: str) -> None:
                queue.put_nowait(chunk)

            def remove_listener():
                return None

            try:
                remove_listener = agent.llm.add_stream_listener(on_chunk)
                self._ensure_assistant_message()
                self._set_status("Connecting to model…")
                send_task = asyncio.create_task(agent.send(self._active_prompt))
                self._set_status("Streaming response…")

                while True:
                    if send_task.done() and queue.empty():
                        break
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if not chunk:
                        continue
                    message = self._ensure_assistant_message()
                    message.content += chunk
                    self._refresh_chat()
                    received_stream_chunks = True

                response_text = await send_task

                if not received_stream_chunks:
                    fallback = response_text or "_No response returned._"
                    message = self._ensure_assistant_message()
                    message.content = fallback
                    self._refresh_chat()

                self._set_status("Response ready")
            except Exception as exc:  # pragma: no cover - runtime feedback only
                if self._messages and self._messages[-1].role == "assistant":
                    self._messages[-1].content = f"```text\n{exc}\n```"
                    self._refresh_chat()
                else:
                    self._append_error_response(f"```text\n{exc}\n```")
                self._set_status("Error")
            finally:
                remove_listener()
        finally:
            self._current_task = None

    def _ensure_assistant_message(self) -> ChatMessage:
        if self._active_assistant_message is None:
            right_info = self._assistant_right_info(self._options.model)
            name = getattr(self._agent, "name", None)
            message = ChatMessage(role="assistant", content="", name=name, right_info=right_info)
            self._messages.append(message)
            self._active_assistant_message = message
            self._refresh_chat()
        return self._active_assistant_message

    def _refresh_chat(self) -> None:
        if not self._chat_display:
            return
        self._chat_display.update_messages(self._messages)
        if self._chat_display.is_vertical_scroll_end:
            self._scroll_chat_to_bottom()

    def _scroll_chat_to_bottom(self) -> None:
        if not self._chat_display:
            return

        def _scroll() -> None:
            self._chat_display.scroll_end(animate=False)

        self.call_after_refresh(_scroll)

    @staticmethod
    def _user_right_info(model: str | None, chat_turn: int) -> str | None:
        parts: list[str] = []
        if model:
            parts.append(model)
        if chat_turn > 0:
            parts.append(f"turn {chat_turn}")
        return " ".join(parts) if parts else None

    def _assistant_right_info(self, model: str | None) -> str | None:
        actual_model = model
        agent = self._agent
        if agent and getattr(agent, "llm", None):
            actual_model = getattr(agent.llm, "model_name", None) or model
        return actual_model or None

    def handle_display_user_message(
        self,
        content: str,
        model: str | None,
        chat_turn: int,
        name: str | None,
    ) -> None:
        if self._active_user_message:
            self._active_user_message.content = content or self._active_user_message.content
            self._active_user_message.name = name or self._active_user_message.name
            self._active_user_message.right_info = self._user_right_info(model, chat_turn)
            self._active_user_message = None
        else:
            message = ChatMessage(
                role="user",
                content=content,
                name=name or "You",
                right_info=self._user_right_info(model, chat_turn),
            )
            self._messages.append(message)
        self._refresh_chat()

    async def handle_display_assistant_message(
        self,
        *,
        content: str,
        pre_content: str | None,
        bottom_items: list[str] | None,
        highlight_index: int | None,
        max_item_length: int | None,
        name: str | None,
        model: str | None,
        additional_text: str | None,
    ) -> None:
        message = self._active_assistant_message
        if not message:
            message = ChatMessage(role="assistant", content="", name=name, right_info=None)
            self._messages.append(message)

        message.name = name or message.name
        message.right_info = self._assistant_right_info(model)
        if content:
            message.content = content
        if pre_content:
            message.pre_content = pre_content
        if additional_text:
            extra = f"\n\n{additional_text}" if message.content else additional_text
            message.content = (message.content or "") + extra
        message.bottom_metadata = bottom_items
        message.highlight_index = highlight_index
        message.max_item_length = max_item_length
        message.block_color_override = None
        message.arrow_style_override = None
        message.arrow_override = None
        message.highlight_color_override = None
        self._active_assistant_message = None
        self._refresh_chat()

    def handle_display_tool_call(
        self,
        *,
        agent_name: str | None,
        tool_name: str,
        tool_args: dict,
        bottom_items: list[str] | None,
        highlight_index: int | None,
        max_item_length: int | None,
        metadata: dict | None,
    ) -> None:
        metadata = metadata or {}

        if metadata.get("variant") == "shell":
            command = metadata.get("command") or tool_args.get("command")
            command_display = command if isinstance(command, str) and command.strip() else None
            if command_display:
                content = f"```shell\n$ {command_display}\n```"
            else:
                content = "_No shell command provided._"

            details: list[str] = []
            shell_name = metadata.get("shell_name")
            shell_path = metadata.get("shell_path")
            if shell_name or shell_path:
                if shell_name and shell_path and shell_path != shell_name:
                    details.append(f"shell: {shell_name} ({shell_path})")
                elif shell_path:
                    details.append(f"shell: {shell_path}")
                elif shell_name:
                    details.append(f"shell: {shell_name}")
            working_dir = metadata.get("working_dir_display") or metadata.get("working_dir")
            if working_dir:
                details.append(f"cwd: {working_dir}")

            capability_bits: list[str] = []
            if metadata.get("streams_output"):
                capability_bits.append("streams stdout/stderr")
            if metadata.get("returns_exit_code"):
                capability_bits.append("reports exit code")

            if capability_bits:
                details.append("; ".join(capability_bits))

            if details:
                bullet_points = "\n".join(f"- {line}" for line in details)
                content = f"{content}\n\n{bullet_points}"
        else:
            if tool_args:
                try:
                    args_text = json.dumps(tool_args, indent=2, sort_keys=True)
                except TypeError:  # pragma: no cover - fallback for unserializable args
                    args_text = str(tool_args)
                content = f"```json\n{args_text}\n```"
            else:
                content = "_No arguments provided._"

        self._active_assistant_message = None

        right_info = "shell command" if metadata.get("variant") == "shell" else f"tool request - {tool_name}"

        message = ChatMessage(
            role="tool_call",
            content=content,
            name=agent_name or "Tool",
            right_info=right_info,
            bottom_metadata=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
        )
        self._messages.append(message)
        self._refresh_chat()

    def handle_display_tool_result(
        self,
        result: CallToolResult,
        *,
        agent_name: str | None,
        tool_name: str | None,
    ) -> None:
        content_blocks = []
        for block in result.content or []:
            text = get_text(block)
            if text:
                content_blocks.append(text)
        content = "\n\n".join(content_blocks) if content_blocks else "_No content returned._"

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                structured_text = json.dumps(structured, indent=2)
            except Exception:  # pragma: no cover - best effort
                structured_text = str(structured)
            content += f"\n\n```json\n{structured_text}\n```"

        status = "ERROR" if result.isError else "success"
        right_info = f"tool result - {status}"

        bottom_metadata: list[str] = []
        channel = getattr(result, "transport_channel", None)
        if channel:
            bottom_metadata.append(self._format_transport_channel(channel))

        self._active_assistant_message = None

        message = ChatMessage(
            role="tool_result",
            content=content,
            name=agent_name or "Tool",
            right_info=right_info,
            bottom_metadata=bottom_metadata or None,
            highlight_color_override="red" if result.isError else None,
            block_color_override="red" if result.isError else None,
            arrow_style_override="dim red" if result.isError else None,
        )
        self._messages.append(message)
        self._refresh_chat()

    @staticmethod
    def _format_transport_channel(channel: str) -> str:
        mapping = {
            "post-json": "HTTP (JSON-RPC)",
            "post-sse": "Legacy SSE",
            "get": "Legacy SSE",
            "resumption": "Resumption",
            "stdio": "STDIO",
        }
        return mapping.get(channel, channel.upper())

    @staticmethod
    def _format_elapsed_seconds(elapsed: float) -> str:
        if elapsed < 0:
            elapsed = 0.0
        if elapsed < 0.001:
            return "<1ms"
        if elapsed < 1:
            return f"{elapsed * 1000:.0f}ms"
        if elapsed < 10:
            return f"{elapsed:.2f}s"
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes, seconds = divmod(elapsed, 60)
        if minutes < 60:
            return f"{int(minutes)}m {seconds:02.0f}s"
        hours, minutes = divmod(int(minutes), 60)
        return f"{hours}h {minutes:02d}m"

    def _append_error_response(self, message: str) -> None:
        self._messages.append(
            ChatMessage(
                role="assistant",
                content=message,
                name=getattr(self._agent, "name", None),
                right_info=self._assistant_right_info(self._options.model),
                block_color_override="red",
                arrow_style_override="dim red",
                highlight_color_override="red",
            )
        )
        self._active_assistant_message = self._messages[-1]
        self._refresh_chat()

    def _current_prompt_value(self) -> str:
        if self._prompt_input:
            return self._prompt_input.value or ""
        return self._options.prompt

    def _set_status(self, message: str) -> None:
        self.sub_title = message

    async def action_regenerate(self) -> None:
        if self._current_task and not self._current_task.done():
            return
        for message in reversed(self._messages):
            if message.role == "user":
                self._start_generation(message.content)
                return

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        submitted = event.value.strip() or DEFAULT_PROMPT
        self._options.prompt = submitted
        if self._prompt_input:
            self._prompt_input.value = submitted
        self._start_generation(submitted)


def parse_args(argv: Sequence[str] | None = None) -> AppOptions:
    """Parse CLI arguments for the textual demo."""
    parser = argparse.ArgumentParser(
        description="Render a chat-style LLM conversation inside Textual."
    )
    parser.add_argument(
        "--prompt",
        help="Prompt to send to the LLM (markdown is rendered directly).",
        default=DEFAULT_PROMPT,
    )
    parser.add_argument(
        "--model",
        help="Model name configured in your fast-agent settings.",
        default=DEFAULT_MODEL,
    )
    args = parser.parse_args(argv)
    return AppOptions(prompt=args.prompt, model=args.model)


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_args(argv)
    app = MarkdownLLMApp(options)
    app.run()


if __name__ == "__main__":
    main()
