"""Shared helper utilities for command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.commands.protocols import HistoryEditableAgent
from fast_agent.commands.results import CommandMessage
from fast_agent.core.exceptions import AgentConfigError, format_fast_agent_error
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from rich.text import Text

    from fast_agent.commands.context import CommandContext
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.logging.logger import Logger
    from fast_agent.types import PromptMessageExtended


@dataclass(frozen=True, slots=True)
class LoadedPromptMessagesResult:
    messages: list[PromptMessageExtended] | None = None
    error: str | None = None


def load_prompt_messages_result(
    filename: str,
    *,
    label: str,
    arguments: "Mapping[str, str] | None" = None,
) -> LoadedPromptMessagesResult:
    try:
        from fast_agent.mcp.prompts.prompt_load import load_prompt

        return LoadedPromptMessagesResult(messages=load_prompt(filename, arguments=arguments))
    except FileNotFoundError:
        return LoadedPromptMessagesResult(error=f"File not found: {filename}")
    except AgentConfigError as exc:
        error_text = format_fast_agent_error(exc)
        return LoadedPromptMessagesResult(error=f"Error loading {label}: {error_text}")
    except Exception as exc:
        return LoadedPromptMessagesResult(error=f"Error loading {label}: {exc}")


def replace_agent_history(agent_obj: object, messages: list[PromptMessageExtended]) -> None:
    if isinstance(agent_obj, HistoryEditableAgent):
        try:
            agent_obj.clear(clear_prompts=True)
        except TypeError:
            agent_obj.clear()
        agent_obj.load_message_history(messages)


def add_info_messages(
    outcome: "CommandOutcome",
    messages: "Sequence[str]",
    *,
    right_info: str,
    agent_name: str | None = None,
) -> None:
    for message in messages:
        outcome.add_message(
            message,
            channel="info",
            right_info=right_info,
            agent_name=agent_name,
        )


def unique_selection_options(options: "Iterable[str]") -> list[str]:
    """Return non-empty selection options, preserving first spelling case-insensitively."""
    return unique_preserve_order(
        (normalized for option in options if (normalized := strip_to_none(option)) is not None),
        key=str.casefold,
    )


async def prompt_selection_after_message(
    ctx: "CommandContext",
    *,
    content: "str | Text",
    right_info: str,
    agent_name: str,
    prompt: str,
    options: "Sequence[str]",
    allow_cancel: bool = True,
    default: str | None = None,
) -> str | None:
    await ctx.io.emit(CommandMessage(text=content, right_info=right_info, agent_name=agent_name))
    return await ctx.io.prompt_selection(
        prompt,
        options=options,
        allow_cancel=allow_cancel,
        default=default,
    )


def clear_agent_histories(
    agents: "Mapping[str, object]",
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
        if not isinstance(agent, HistoryEditableAgent):
            continue
        try:
            agent.clear()
            cleared.append(name)
        except Exception as exc:
            if logger:
                logger.warning(
                    "Failed to clear agent history",
                    data={"agent": name, "error": str(exc)},
                )
    return cleared
