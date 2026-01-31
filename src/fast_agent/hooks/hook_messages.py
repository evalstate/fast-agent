"""Shared UI helpers for hook output and failures."""

from __future__ import annotations

from typing import Literal

from rich.text import Text

from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay

logger = get_logger(__name__)

HookKind = Literal["tool", "agent"]


def _resolve_display(agent: object) -> ConsoleDisplay:
    display = getattr(agent, "display", None)
    if isinstance(display, ConsoleDisplay):
        return display

    config = None
    agent_context = getattr(agent, "context", None)
    if agent_context is not None:
        config = getattr(agent_context, "config", None)

    return ConsoleDisplay(config=config)


def _normalize_message_lines(message: str | Text | None) -> list[Text]:
    if message is None:
        return []

    if isinstance(message, Text):
        if "\n" in message.plain:
            return [line for line in message.split("\n") if line.plain != ""]
        return [message]

    text = str(message)
    if not text:
        return []
    return [Text(line) for line in text.splitlines() if line.strip() != ""]


def _build_hook_header(hook_kind: HookKind, hook_name: str | None, *, style: str) -> Text:
    header = Text()
    header.append(f"{hook_kind} hook", style=f"bold {style}")
    if hook_name:
        header.append(" ")
        header.append(hook_name, style="dim")
    return header


def _build_metadata_line(
    display: ConsoleDisplay,
    content: Text,
    *,
    prefix_style: str,
    width: int,
) -> Text:
    style_name = getattr(display.style, "name", None)
    if style_name == "a3":
        line = Text()
        line.append("▎• ", style=prefix_style)
        line.append_text(content)
        return line

    return display.style.metadata_line(content, width)


def show_hook_message(
    target: object,
    message: str | Text | None,
    *,
    hook_name: str | None,
    hook_kind: HookKind = "tool",
    style: str = "bright_yellow",
) -> None:
    """Render a hook status line using the active message style (A3 by default)."""
    try:
        agent = getattr(target, "agent", target)
        display = _resolve_display(agent)
        width = console.console.size.width
        prefix_style = f"bold {style}"
        style_name = getattr(display.style, "name", None)
        prefix_text = "▎• "
        indent = " " * len(prefix_text)

        header = _build_hook_header(hook_kind, hook_name, style=style)
        lines = _normalize_message_lines(message)

        if not lines:
            display.show_status_message(
                _build_metadata_line(
                    display,
                    header,
                    prefix_style=prefix_style,
                    width=width,
                )
            )
            return

        first_line = Text()
        first_line.append_text(header)
        first_line.append(" — ", style="dim")
        first_line.append_text(lines[0])
        display.show_status_message(
            _build_metadata_line(
                display,
                first_line,
                prefix_style=prefix_style,
                width=width,
            )
        )

        if style_name == "a3":
            for line in lines[1:]:
                indented = Text(indent, style="dim")
                indented.append_text(line)
                display.show_status_message(indented)
            return

        for line in lines[1:]:
            display.show_status_message(
                _build_metadata_line(
                    display,
                    line,
                    prefix_style=prefix_style,
                    width=width,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to render hook message", data={"error": str(exc)})


def show_hook_failure(
    target: object,
    *,
    hook_name: str | None,
    hook_kind: HookKind = "tool",
    error: Exception | None = None,
) -> None:
    """Render a bright-red hook failure notification (details are in logs)."""
    summary = "hook failure (see logs)"
    if error is not None:
        summary = f"{summary}: {error}"
    show_hook_message(
        target,
        summary,
        hook_name=hook_name,
        hook_kind=hook_kind,
        style="bright_red",
    )
