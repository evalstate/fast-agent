"""Shared history command handlers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from fast_agent.commands.handlers.shared import (
    load_prompt_messages_from_file,
    replace_agent_history,
)
from fast_agent.commands.results import CommandOutcome
from fast_agent.constants import (
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    CONTROL_MESSAGE_SAVE_HISTORY,
)
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.types import LlmStopReason, PromptMessageExtended

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


def _group_turns_for_history_actions(
    history: list[PromptMessageExtended],
) -> list[tuple[int, list[PromptMessageExtended]]]:
    turns: list[tuple[int, list[PromptMessageExtended]]] = []
    current: list[PromptMessageExtended] = []
    current_start = 0
    saw_assistant = False

    for idx, message in enumerate(history):
        is_new_user = message.role == "user" and not message.tool_results
        if is_new_user:
            if not current:
                current = [message]
                current_start = idx
                saw_assistant = False
                continue
            if not saw_assistant:
                current.append(message)
                continue
            turns.append((current_start, current))
            current = [message]
            current_start = idx
            saw_assistant = False
            continue

        if current:
            current.append(message)
            if message.role == "assistant":
                saw_assistant = True

    if current:
        turns.append((current_start, current))
    return turns


def _collect_user_turns(
    history: list[PromptMessageExtended],
) -> list[tuple[int, PromptMessageExtended]]:
    turns = _group_turns_for_history_actions(list(history))
    user_turns: list[tuple[int, PromptMessageExtended]] = []
    for offset, turn in turns:
        if not turn:
            continue
        first = turn[0]
        if first.role != "user" or first.tool_results:
            continue
        user_turns.append((offset, first))
    return user_turns


def _trim_history_for_rewind(
    history: list[PromptMessageExtended],
    *,
    turn_start_index: int,
    template_messages: list[PromptMessageExtended] | None = None,
) -> list[PromptMessageExtended]:
    for idx in range(turn_start_index - 1, -1, -1):
        if history[idx].role == "assistant":
            return history[: idx + 1]
    if template_messages:
        return template_messages
    return history[:turn_start_index]


def _strip_web_metadata_channels(
    message: PromptMessageExtended,
) -> tuple[PromptMessageExtended, int]:
    channels = message.channels
    if not isinstance(channels, Mapping) or not channels:
        return message, 0

    removed_blocks = 0
    retained: dict[str, Sequence[Any]] = {}
    for channel_name, blocks in channels.items():
        if channel_name in {ANTHROPIC_SERVER_TOOLS_CHANNEL, ANTHROPIC_CITATIONS_CHANNEL}:
            removed_blocks += len(blocks)
            continue
        retained[channel_name] = blocks

    if removed_blocks == 0:
        return message, 0

    return message.model_copy(update={"channels": retained or None}), removed_blocks


def web_tools_enabled_for_agent(agent_obj: object) -> bool:
    """Return True when the agent's active LLM has web tools enabled."""
    llm = getattr(agent_obj, "llm", None) or getattr(agent_obj, "_llm", None)
    if llm is None:
        return False

    web_search_enabled = getattr(llm, "web_search_enabled", None)
    if isinstance(web_search_enabled, bool):
        return web_search_enabled

    enabled = getattr(llm, "web_tools_enabled", None)
    if isinstance(enabled, tuple) and len(enabled) >= 2:
        return bool(enabled[0] or enabled[1])
    if isinstance(enabled, bool):
        return enabled
    return False


async def handle_show_history(
    ctx: CommandContext,
    *,
    agent_name: str,
    target_agent: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    target = target_agent or agent_name
    agent_obj = ctx.agent_provider._agent(target)
    history = getattr(agent_obj, "message_history", [])
    usage = getattr(agent_obj, "usage_accumulator", None)
    await ctx.io.display_history_overview(target, list(history), usage)
    return outcome


async def handle_history_rewind(
    ctx: CommandContext,
    *,
    agent_name: str,
    turn_index: int | None,
    error: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    if error:
        outcome.add_message(error, channel="error", agent_name=agent_name)
        return outcome
    if turn_index is None:
        outcome.add_message(
            "Usage: /history rewind <turn>",
            channel="warning",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Tip: press Tab after '/history rewind ' to see turn options.",
            channel="info",
            agent_name=agent_name,
        )
        return outcome

    agent_obj = ctx.agent_provider._agent(agent_name)
    history = getattr(agent_obj, "message_history", [])
    user_turns = _collect_user_turns(list(history))
    if not user_turns:
        outcome.add_message(
            "No user turns available to rewind.",
            channel="warning",
            agent_name=agent_name,
        )
        return outcome
    if turn_index < 1 or turn_index > len(user_turns):
        outcome.add_message("Turn index out of range.", channel="error", agent_name=agent_name)
        return outcome

    turn_start, user_message = user_turns[turn_index - 1]
    content = getattr(user_message, "content", []) or []
    user_text = None
    if content:
        from fast_agent.mcp.helpers.content_helpers import get_text

        user_text = get_text(content[0])
    if not user_text or user_text == "<no text>":
        outcome.add_message(
            "Selected turn has no text content to rewind.",
            channel="error",
            agent_name=agent_name,
        )
        return outcome

    template_messages = getattr(agent_obj, "template_messages", None)
    trimmed = _trim_history_for_rewind(
        list(history),
        turn_start_index=turn_start,
        template_messages=template_messages,
    )
    load_history = getattr(agent_obj, "load_message_history", None)
    if callable(load_history):
        load_history(trimmed)
    else:
        existing_history = getattr(agent_obj, "message_history", None)
        if isinstance(existing_history, list):
            existing_history.clear()
            existing_history.extend(trimmed)

    outcome.buffer_prefill = user_text
    outcome.add_message(
        "History rewound. User turn loaded into input buffer.",
        channel="info",
        agent_name=agent_name,
    )
    return outcome


async def handle_history_review(
    ctx: CommandContext,
    *,
    agent_name: str,
    turn_index: int | None,
    error: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    if error:
        outcome.add_message(error, channel="error", agent_name=agent_name)
        return outcome
    if turn_index is None:
        outcome.add_message(
            "Usage: /history review <turn>",
            channel="warning",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Tip: press Tab after '/history review ' to see turn options.",
            channel="info",
            agent_name=agent_name,
        )
        return outcome

    agent_obj = ctx.agent_provider._agent(agent_name)
    history = getattr(agent_obj, "message_history", [])
    turns = [
        turn
        for _, turn in _group_turns_for_history_actions(list(history))
        if turn and turn[0].role == "user" and not turn[0].tool_results
    ]
    user_turns = turns
    if not user_turns:
        outcome.add_message(
            "No user turns available to review.",
            channel="warning",
            agent_name=agent_name,
        )
        return outcome
    if turn_index < 1 or turn_index > len(user_turns):
        outcome.add_message("Turn index out of range.", channel="error", agent_name=agent_name)
        return outcome

    selected_turn = user_turns[turn_index - 1]
    outcome.add_message(
        f"History review: turn {turn_index}",
        channel="info",
        agent_name=agent_name,
    )

    await ctx.io.display_history_turn(
        agent_obj.name if hasattr(agent_obj, "name") else agent_name,
        list(selected_turn),
        turn_index=turn_index,
        total_turns=len(user_turns),
    )

    return outcome


async def handle_history_fix(
    ctx: CommandContext,
    *,
    agent_name: str,
    target_agent: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    target = target_agent or agent_name

    agent_obj = ctx.agent_provider._agent(target)
    history = list(getattr(agent_obj, "message_history", []))
    if not history:
        outcome.add_message("No history to fix.", channel="warning", agent_name=target)
        return outcome

    last_msg = history[-1]
    if (
        last_msg.role == "assistant"
        and last_msg.tool_calls
        and last_msg.stop_reason == LlmStopReason.TOOL_USE
    ):
        trimmed = history[:-1]
        load_history = getattr(agent_obj, "load_message_history", None)
        if callable(load_history):
            load_history(trimmed)
        else:
            existing_history = getattr(agent_obj, "message_history", None)
            if isinstance(existing_history, list):
                existing_history.clear()
                existing_history.extend(trimmed)
        outcome.add_message(
            f"Removed pending tool call from '{target}'.",
            channel="info",
            agent_name=target,
        )
    else:
        outcome.add_message(
            "No pending tool call found at end of history.",
            channel="warning",
            agent_name=target,
        )

    return outcome


async def handle_history_webclear(
    ctx: CommandContext,
    *,
    agent_name: str,
    target_agent: str | None = None,
) -> CommandOutcome:
    """Strip Anthropic web-search/fetch metadata channels from history."""
    outcome = CommandOutcome()
    target = target_agent or agent_name

    agent_obj = ctx.agent_provider._agent(target)
    if not web_tools_enabled_for_agent(agent_obj):
        outcome.add_message(
            "Web metadata cleanup is only available when Anthropic web tools are enabled.",
            channel="warning",
            agent_name=target,
        )
        return outcome

    history = list(getattr(agent_obj, "message_history", []))
    if not history:
        outcome.add_message("No history to clean.", channel="warning", agent_name=target)
        return outcome

    cleaned_history: list[PromptMessageExtended] = []
    removed_blocks = 0
    touched_messages = 0
    for message in history:
        cleaned_message, removed = _strip_web_metadata_channels(message)
        cleaned_history.append(cleaned_message)
        if removed > 0:
            removed_blocks += removed
            touched_messages += 1

    if removed_blocks == 0:
        outcome.add_message(
            f"No web metadata channels found for agent '{target}'.",
            channel="warning",
            agent_name=target,
        )
        return outcome

    load_history = getattr(agent_obj, "load_message_history", None)
    if callable(load_history):
        load_history(cleaned_history)
    else:
        existing_history = getattr(agent_obj, "message_history", None)
        if isinstance(existing_history, list):
            existing_history.clear()
            existing_history.extend(cleaned_history)

    outcome.add_message(
        (
            f"Removed {removed_blocks} web metadata block(s) from {touched_messages} "
            f"message(s) for agent '{target}'."
        ),
        channel="info",
        agent_name=target,
    )
    return outcome


async def handle_history_clear_last(
    ctx: CommandContext,
    *,
    agent_name: str,
    target_agent: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    target = target_agent or agent_name

    agent_obj = ctx.agent_provider._agent(target)
    removed_message = None
    pop_callable = getattr(agent_obj, "pop_last_message", None)
    if callable(pop_callable):
        removed_message = pop_callable()
    else:
        history = getattr(agent_obj, "message_history", [])
        if history:
            try:
                removed_message = history.pop()
            except Exception:
                removed_message = None

    if removed_message:
        role = getattr(removed_message, "role", "message")
        role_display = role.capitalize() if isinstance(role, str) else "Message"
        outcome.add_message(
            f"Removed last {role_display} for agent '{target}'.",
            channel="info",
            agent_name=target,
        )
    else:
        outcome.add_message(
            f"No messages to remove for agent '{target}'.",
            channel="warning",
            agent_name=target,
        )

    return outcome


async def handle_history_clear_all(
    ctx: CommandContext,
    *,
    agent_name: str,
    target_agent: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    target = target_agent or agent_name

    agent_obj = ctx.agent_provider._agent(target)
    if hasattr(agent_obj, "clear"):
        try:
            agent_obj.clear()
            outcome.add_message(
                f"History cleared for agent '{target}'.",
                channel="info",
                agent_name=target,
            )
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(
                f"Failed to clear history for '{target}': {exc}",
                channel="error",
                agent_name=target,
            )
    else:
        outcome.add_message(
            f"Agent '{target}' does not support clearing history.",
            channel="warning",
            agent_name=target,
        )

    return outcome


async def handle_history_save(
    ctx: CommandContext,
    *,
    agent_name: str,
    filename: str | None,
    send_func: Any | None,
    history_exporter: type[HistoryExporter] | HistoryExporter | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    exporter = history_exporter or HistoryExporter

    try:
        agent_obj = ctx.agent_provider._agent(agent_name)
        saved_path = await exporter.save(agent_obj, filename)
        outcome.add_message(
            f"History saved to {saved_path}",
            channel="info",
            agent_name=agent_name,
        )
        return outcome
    except Exception as exc:  # noqa: BLE001
        if send_func:
            control = CONTROL_MESSAGE_SAVE_HISTORY + (f" {filename}" if filename else "")
            result = await send_func(control, agent_name)
            if result:
                outcome.add_message(result, channel="info", agent_name=agent_name)
            return outcome
        outcome.add_message(
            f"Failed to save history: {exc}",
            channel="error",
            agent_name=agent_name,
        )
        return outcome


async def handle_history_load(
    ctx: CommandContext,
    *,
    agent_name: str,
    filename: str | None,
    error: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    if error:
        outcome.add_message(error, channel="error", agent_name=agent_name)
        return outcome

    if filename is None:
        outcome.add_message("Filename required for /history load", channel="error")
        return outcome

    agent_obj = ctx.agent_provider._agent(agent_name)
    messages = load_prompt_messages_from_file(filename, label="history")
    if messages is None:
        return outcome

    replace_agent_history(agent_obj, messages)
    msg_count = len(messages)
    outcome.add_message(
        f"Loaded {msg_count} messages from {filename}",
        channel="info",
        agent_name=agent_name,
    )

    return outcome
