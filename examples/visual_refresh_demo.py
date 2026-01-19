from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType

console = Console()


@dataclass(frozen=True)
class DemoMessage:
    message_type: MessageType
    text: str
    name: str


DEMO_MESSAGES = [
    DemoMessage(
        message_type=MessageType.USER,
        name="user",
        text="you can use the lsp tools for navigating python code",
    ),
    DemoMessage(
        message_type=MessageType.ASSISTANT,
        name="fast-agent",
        text=(
            "Short answer: the error is being logged inside the generic LLM call prep, so it "
            "will fire during normal tool loops even when everything is healthy."
        ),
    ),
    DemoMessage(
        message_type=MessageType.USER,
        name="user",
        text="this is whats needed i think;",
    ),
]

DEMO_BOTTOM_ITEMS = ["bash", "agent__ripg…", "agent__acp-…", "agent__hf-s…", "card_tools"]


def _separator_header_only(left_content: str, right_info: str = "") -> Text:
    width = console.size.width
    left_text = Text.from_markup(left_content)
    right_text = Text.from_markup(right_info) if right_info.strip() else Text("")

    combined = Text()
    combined.append_text(left_text)
    if right_text.plain:
        right_block = Text()
        right_block.append_text(right_text)
        right_block.stylize("dim")
        padding = width - left_text.cell_len - right_block.cell_len
        if padding < 1:
            padding = 1
        combined.append(" " * padding, style="default")
        combined.append_text(right_block)
    return combined


def _separator_header_inline(left_content: str, right_info: str = "") -> Text:
    left_text = Text.from_markup(left_content)
    combined = Text()
    combined.append_text(left_text)
    if right_info.strip():
        combined.append(" ", style="default")
        combined.append_text(Text(right_info, style="dim"))
    return combined


def _separator_line(left_content: str, right_info: str = "") -> Text:
    width = console.size.width
    left_text = Text.from_markup(left_content)

    if right_info.strip():
        right_text = Text()
        right_text.append("[", style="dim")
        right_text.append_text(Text.from_markup(right_info))
        right_text.append("]", style="dim")
        separator_count = width - left_text.cell_len - right_text.cell_len
        if separator_count < 1:
            separator_count = 1
    else:
        right_text = Text("")
        separator_count = width - left_text.cell_len

    combined = Text()
    combined.append_text(left_text)
    combined.append(" ", style="default")
    combined.append("─" * max(separator_count - 1, 1), style="dim")
    combined.append_text(right_text)
    return combined


def _render_bottom_bar(items: list[str], highlight_index: int | None = None) -> None:
    if not items:
        return

    console.print()
    bar = Text()
    bar.append("▎• ", style="dim")
    for idx, item in enumerate(items):
        if idx > 0:
            bar.append(" • ", style="dim")

        style = "white dim"
        if highlight_index is not None and idx == highlight_index:
            style = "green"

        bar.append(item, style=style)

    console.print(bar)


def render_palette() -> None:
    console.print("\n[bold]ANSI palette (theme-aware)[/bold]\n", markup=True)
    colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    header = Text()
    header.append("color", style="bold")
    header.append("  ")
    header.append("normal", style="bold")
    header.append("  ")
    header.append("dim", style="bold")
    header.append("  ")
    header.append("reverse dim", style="bold")
    header.append("  ")
    header.append("reverse bold", style="bold")
    console.print(header)

    for color in colors:
        row = Text()
        row.append(color.ljust(8), style="dim")
        row.append("  ")
        row.append(" sample ", style=color)
        row.append("  ")
        row.append(" sample ", style=f"{color} dim")
        row.append("  ")
        row.append(" sample ", style=f"{color} reverse dim")
        row.append("  ")
        row.append(" sample ", style=f"{color} reverse bold")
        console.print(row)
    console.print()


def render_current_style(messages: list[DemoMessage]) -> None:
    console.print("\n[bold]Option A — current separator style[/bold]\n", markup=True)
    for message in messages:
        config = MESSAGE_CONFIGS[message.message_type]
        left = (
            f"[{config['block_color']}]▎[/{config['block_color']}]"
            f"[{config['arrow_style']}]{config['arrow']}[/{config['arrow_style']}]"
            f" [{config['block_color']}]{message.name}[/{config['block_color']}]"
        )
        console.print(_separator_line(left, right_info="demo"), markup=True)
        console.print(message.text)
        if message.message_type != MessageType.USER:
            _render_bottom_bar(DEMO_BOTTOM_ITEMS, highlight_index=2)
        console.print()


def render_header_only_style(messages: list[DemoMessage]) -> None:
    console.print("\n[bold]Option A2 — header + right label (no rule)[/bold]\n", markup=True)
    for message in messages:
        config = MESSAGE_CONFIGS[message.message_type]
        left = (
            f"[{config['block_color']}]▎[/{config['block_color']}]"
            f"[{config['arrow_style']}]{config['arrow']}[/{config['arrow_style']}]"
            f" [{config['block_color']}]{message.name}[/{config['block_color']}]"
        )
        console.print(_separator_header_only(left, right_info="demo"), markup=True)
        console.print(message.text)
        if message.message_type != MessageType.USER:
            _render_bottom_bar(DEMO_BOTTOM_ITEMS, highlight_index=2)
        console.print()


def render_header_inline_style(messages: list[DemoMessage]) -> None:
    console.print("\n[bold]Option A3 — header + inline label[/bold]\n", markup=True)
    for message in messages:
        config = MESSAGE_CONFIGS[message.message_type]
        left = (
            f"[{config['block_color']}]▎[/{config['block_color']}]"
            f"[{config['arrow_style']}]{config['arrow']}[/{config['arrow_style']}]"
            f" [{config['block_color']}]{message.name}[/{config['block_color']}]"
        )
        console.print(_separator_header_inline(left, right_info="demo"), markup=True)
        console.print(message.text)
        if message.message_type != MessageType.USER:
            _render_bottom_bar(DEMO_BOTTOM_ITEMS, highlight_index=2)
        console.print()


def main() -> None:
    console.print("[bold cyan]fast-agent visual refresh demo[/bold cyan]\n")
    console.print("Sample conversation drawn from recent session history.\n", style="dim")
    render_palette()
    render_current_style(DEMO_MESSAGES)
    render_header_only_style(DEMO_MESSAGES)
    render_header_inline_style(DEMO_MESSAGES)


if __name__ == "__main__":
    main()
