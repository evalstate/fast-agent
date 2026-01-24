"""Shared helper utilities for command handlers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import print as rich_print

from fast_agent.core.exceptions import AgentConfigError, format_fast_agent_error

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.core.logging.logger import Logger
    from fast_agent.types import PromptMessageExtended


def load_prompt_messages_from_file(
    filename: str, *, label: str
) -> list[PromptMessageExtended] | None:
    try:
        from fast_agent.mcp.prompts.prompt_load import load_prompt

        return load_prompt(Path(filename))
    except FileNotFoundError:
        rich_print(f"[red]File not found: {filename}[/red]")
    except AgentConfigError as exc:
        error_text = format_fast_agent_error(exc)
        rich_print(f"[red]Error loading {label}: {error_text}[/red]")
    except Exception as exc:  # noqa: BLE001
        rich_print(f"[red]Error loading {label}: {exc}[/red]")
    return None


def replace_agent_history(agent_obj: Any, messages: list[PromptMessageExtended]) -> None:
    if hasattr(agent_obj, "clear"):
        try:
            agent_obj.clear(clear_prompts=True)
        except TypeError:
            agent_obj.clear()
    history = getattr(agent_obj, "message_history", None)
    if isinstance(history, list):
        history.clear()
        history.extend(messages)


def clear_agent_histories(
    agents: Mapping[str, Any],
    logger: Logger | None = None,
) -> list[str]:
    """
    Clear in-memory histories for all agents.

    Args:
        agents: Dictionary mapping agent names to agent instances
        logger: Optional logger for error reporting

    Returns:
        List of agent names that were successfully cleared
    """
    cleared: list[str] = []
    for name, agent in agents.items():
        clear_fn = getattr(agent, "clear", None)
        if not callable(clear_fn):
            continue
        try:
            clear_fn()
            cleared.append(name)
        except Exception as exc:
            if logger:
                logger.warning(
                    "Failed to clear agent history",
                    data={"agent": name, "error": str(exc)},
                )
    return cleared
