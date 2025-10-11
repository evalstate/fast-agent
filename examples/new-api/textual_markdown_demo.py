"""Textual example that streams an LLM response into a Markdown widget."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Sequence

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, Markdown

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory

DEFAULT_PROMPT = (
    "Provide a short markdown summary with a heading and bullet list describing how "
    "Textual can be paired with fast-agent to build rich terminal apps."
)
DEFAULT_MODEL = "haiku"


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
    """Textual application that displays an LLM response in a Markdown widget."""

    CSS = """
    Input {
        margin: 1 2;
    }

    Markdown {
        padding: 1 2;
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
        self._markdown: Markdown | None = None
        self._header: Header | None = None
        self._prompt_input: Input | None = None
        self._current_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        header = Header(show_clock=True)
        prompt_input = Input(
            value=self._options.prompt,
            placeholder="Edit prompt and press Enter or R to regenerate",
        )
        markdown = Markdown("# Textual LLM Demo\n\nPreparing to contact the model…")
        self._markdown = markdown
        self._header = header
        self._prompt_input = prompt_input
        yield header
        yield prompt_input
        yield markdown
        yield Footer()

    async def on_mount(self) -> None:
        if self._prompt_input:
            self._prompt_input.focus()
        self._render_markdown(status="Response")
        self._start_generation()

    async def on_unmount(self) -> None:
        if self._current_task:
            self._current_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._current_task

    def _render_markdown(self, status: str, body: str | None = None) -> None:
        if not self._markdown:
            return

        prompt_text = _format_prompt(self._active_prompt)
        sections = [
            "# Textual LLM Demo",
            f"Model: `{self._options.model}`",
            "",
            "## Prompt",
            prompt_text,
            "",
            f"## {status}",
        ]
        if body:
            sections.extend(["", body])

        self._markdown.update("\n".join(sections))

    def _start_generation(self) -> None:
        if self._current_task and not self._current_task.done():
            return
        prompt = self._current_prompt_value().strip() or DEFAULT_PROMPT
        self._active_prompt = prompt
        self._options.prompt = prompt
        if self._prompt_input and self._prompt_input.value != prompt:
            self._prompt_input.value = prompt
        self._current_task = asyncio.create_task(self._generate_and_render())

    async def _generate_and_render(self) -> None:
        self._render_markdown("Response")
        self._set_status("Connecting to model…")
        received_stream_chunks = False
        try:
            queue: asyncio.Queue[str] = asyncio.Queue()

            def on_chunk(chunk: str) -> None:
                queue.put_nowait(chunk)

            stream = None
            core = Core()
            async with core.run():
                agent = ToolAgent(
                    AgentConfig(
                        name="textual_markdown_demo",
                        model=self._options.model,
                        use_history=False,
                    ),
                    tools=(),
                    context=core.context,
                )
                await agent.attach_llm(ModelFactory.create_factory(self._options.model))

                remove_listener = agent.llm.add_stream_listener(on_chunk)
                try:
                    stream = Markdown.get_stream(self._markdown) if self._markdown else None
                    if stream:
                        # Separate the response section visually before streaming chunks
                        await stream.write("\n")

                    send_task = asyncio.create_task(agent.send(self._active_prompt))
                    self._set_status("Streaming response…")

                    while True:
                        if send_task.done() and queue.empty():
                            break
                        try:
                            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        if stream:
                            await stream.write(chunk)
                        received_stream_chunks = True

                    response = await send_task
                finally:
                    remove_listener()

            if stream:
                # Flush any pending fragments and stop the helper task
                await stream.stop()

            if not received_stream_chunks:
                if self._markdown:
                    fallback_stream = Markdown.get_stream(self._markdown)
                    try:
                        await fallback_stream.write("\n" + (response or "_No response returned._"))
                    finally:
                        await fallback_stream.stop()

            self._set_status("Response ready")
        except Exception as exc:  # pragma: no cover - runtime feedback only
            self._set_status("Error")
            self._render_markdown("Error", f"```text\n{exc}\n```")
            return

    def _current_prompt_value(self) -> str:
        if self._prompt_input:
            return self._prompt_input.value or ""
        return self._options.prompt

    def _set_status(self, message: str) -> None:
        self.sub_title = message

    async def action_regenerate(self) -> None:
        """Trigger a new LLM generation."""
        self._start_generation()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle prompt submission via Enter."""
        submitted = event.value.strip()
        if not submitted:
            submitted = DEFAULT_PROMPT
        self._options.prompt = submitted
        if self._prompt_input:
            self._prompt_input.value = submitted
        self._start_generation()


def parse_args(argv: Sequence[str] | None = None) -> AppOptions:
    """Parse CLI arguments for the textual demo."""
    parser = argparse.ArgumentParser(
        description="Render an LLM response inside Textual Markdown."
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
