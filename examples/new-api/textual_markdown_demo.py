"""Textual example that streams an LLM response into a Markdown widget."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, Markdown

from fast_agent import FastAgent
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


class MarkdownLLMApp(App[None]):
    """Textual application that displays an LLM response in a chat-style Markdown widget."""

    CSS = """
    Screen {
        layout: vertical;
    }

    Markdown#chat {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
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
        self._messages: list[dict[str, str]] = []
        self._chat_display: Markdown | None = None
        self._header: Header | None = None
        self._prompt_input: Input | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._agent: AgentProtocol | None = None
        self._agent_app = None
        self._agent_context = None

        if CHAT_AGENT_NAME in fast.agents:
            fast.agents[CHAT_AGENT_NAME]["config"].model = options.model

    def compose(self) -> ComposeResult:
        header = Header(show_clock=True)
        chat = Markdown(self._build_chat_markdown(), id="chat")
        prompt_input = Input(
            value=self._options.prompt,
            placeholder="Type a prompt and press Enter (R to regenerate)",
            id="prompt",
        )
        footer = Footer()
        self._header = header
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
        prompt_value = (prompt or self._current_prompt_value()).strip()
        if not prompt_value:
            prompt_value = DEFAULT_PROMPT
        self._active_prompt = prompt_value
        self._options.prompt = prompt_value
        if self._prompt_input and self._prompt_input.value != prompt_value:
            self._prompt_input.value = prompt_value
        self._messages.append({"role": "user", "content": prompt_value})
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

            remove_listener = lambda: None
            stream = None
            try:
                remove_listener = agent.llm.add_stream_listener(on_chunk)
                stream = self._begin_assistant_message()
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
                    if stream:
                        await stream.write(chunk)
                    if self._messages and self._messages[-1]["role"] == "assistant":
                        self._messages[-1]["content"] += chunk
                    received_stream_chunks = True
                    self._scroll_chat_to_bottom()

                response_text = await send_task

                if not received_stream_chunks:
                    fallback = response_text or "_No response returned._"
                    if self._messages and self._messages[-1]["role"] == "assistant":
                        self._messages[-1]["content"] = fallback
                    else:
                        self._append_error_response(fallback)
                    self._refresh_chat()

                self._set_status("Response ready")
                self._scroll_chat_to_bottom()
            except Exception as exc:  # pragma: no cover - runtime feedback only
                if self._messages and self._messages[-1]["role"] == "assistant":
                    self._messages[-1]["content"] = f"```text\n{exc}\n```"
                    self._refresh_chat()
                else:
                    self._append_error_response(f"```text\n{exc}\n```")
                self._set_status("Error")
            finally:
                remove_listener()
                if stream:
                    with suppress(Exception):
                        await stream.stop()
        finally:
            self._current_task = None

    def _begin_assistant_message(self):
        self._messages.append({"role": "assistant", "content": ""})
        if not self._chat_display:
            return None
        self._chat_display.update(self._build_chat_markdown(include_last_body=False))
        self._scroll_chat_to_bottom()
        return Markdown.get_stream(self._chat_display)

    def _refresh_chat(self) -> None:
        if not self._chat_display:
            return
        self._chat_display.update(self._build_chat_markdown())
        self._scroll_chat_to_bottom()

    def _build_chat_markdown(self, *, include_last_body: bool = True) -> str:
        model_label = (
            (self._agent.llm.model_name or self._options.model)
            if self._agent and getattr(self._agent, "llm", None)
            else self._options.model
        )
        parts: list[str] = [
            "# Textual LLM Chat Demo",
            f"Model: `{model_label}`",
            "",
        ]
        if not self._messages:
            parts.append("_No messages yet. Type a prompt below to begin._")
            return "\n".join(parts)

        for index, message in enumerate(self._messages):
            role_label = "You" if message["role"] == "user" else "Assistant"
            parts.append(f"**{role_label}**")
            parts.append("")

            if message["role"] == "user":
                parts.append(_format_prompt(message["content"]))
            else:
                include_body = include_last_body or index < len(self._messages) - 1
                if include_body and message["content"]:
                    parts.append(message["content"])

            parts.append("")

        return "\n".join(parts)

    def _scroll_chat_to_bottom(self) -> None:
        if not self._chat_display:
            return

        def _scroll() -> None:
            self._chat_display.scroll_end(animate=False)

        self.call_after_refresh(_scroll)

    def _append_error_response(self, message: str) -> None:
        self._messages.append({"role": "assistant", "content": message})
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
            if message["role"] == "user":
                self._start_generation(message["content"])
                return

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        submitted = event.value.strip() or DEFAULT_PROMPT
        self._options.prompt = submitted
        if self._prompt_input:
            self._prompt_input.value = submitted
        self._start_generation(submitted)


def parse_args(argv: Sequence[str] | None = None) -> AppOptions:
    """Parse CLI arguments for the textual demo."""
    parser = argparse.ArgumentParser(description="Render a chat-style LLM conversation inside Textual.")
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
