"""Shared UI helpers for hook output and failures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from rich.text import Text

from fast_agent.core.logging.logger import get_logger
from fast_agent.ui.console_display import ConsoleDisplay

if TYPE_CHECKING:
    from fast_agent.config import Settings

logger = get_logger(__name__)

HookKind = Literal[
    "tool",
    "agent",
    "extension",
    "agent_startup",
    "agent_shutdown",
]

HOOK_KIND_LABELS: dict[HookKind, str] = {
    "tool": "extension",
    "extension": "extension",
    "agent": "agent",
    "agent_startup": "agent startup",
    "agent_shutdown": "agent shutdown",
}

@runtime_checkable
class HookDisplayAgent(Protocol):
    @property
    def display(self) -> ConsoleDisplay: ...


@runtime_checkable
class HookContextCarrier(Protocol):
    @property
    def config(self) -> "Settings | None": ...


@runtime_checkable
class HookContextAgent(Protocol):
    @property
    def context(self) -> HookContextCarrier | None: ...


@runtime_checkable
class HookTargetCarrier(Protocol):
    @property
    def agent(self) -> object: ...


def _resolve_hook_agent(target: object) -> object:
    if isinstance(target, HookTargetCarrier):
        return target.agent
    return target


def _resolve_display(agent: object) -> ConsoleDisplay:
    if isinstance(agent, HookDisplayAgent):
        return agent.display

    config = None
    if isinstance(agent, HookContextAgent) and agent.context is not None:
        config = agent.context.config

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
    label = HOOK_KIND_LABELS.get(hook_kind, hook_kind)
    header.append(label, style=f"bold {style}")
    if hook_name:
        header.append(" ")
        header.append(hook_name, style="dim")
    return header


def _build_metadata_line(content: Text, *, prefix_style: str) -> Text:
    line = Text()
    line.append("▎ ", style=prefix_style)
    line.append_text(content)
    return line


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
        agent = _resolve_hook_agent(target)
        display = _resolve_display(agent)
        prefix_style = f"bold {style}"
        prefix_text = "  "
        indent = " " * len(prefix_text)

        header = _build_hook_header(hook_kind, hook_name, style=style)
        lines = _normalize_message_lines(message)

        if not lines:
            display.show_status_message(
                _build_metadata_line(
                    header,
                    prefix_style=prefix_style,
                )
            )
            return

        first_line = Text()
        first_line.append_text(header)
        first_line.append(" — ", style="dim")
        first_line.append_text(lines[0])
        display.show_status_message(
            _build_metadata_line(
                first_line,
                prefix_style=prefix_style,
            )
        )

        for line in lines[1:]:
            indented = Text(indent, style="dim")
            indented.append_text(line)
            display.show_status_message(indented)
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
