"""Shared history command handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from fast_agent.commands.handlers.shared import (
    load_prompt_messages_result,
    replace_agent_history,
)
from fast_agent.commands.history_summaries import HistoryTurn, collect_user_turns
from fast_agent.commands.model_capabilities import (
    resolve_web_fetch_enabled,
    resolve_web_search_enabled,
)
from fast_agent.commands.protocols import HistoryEditableAgent, TemplateMessageProvider
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.summary_utils import optional_string
from fast_agent.constants import (
    ANTHROPIC_ASSISTANT_RAW_CONTENT,
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    CONTROL_MESSAGE_SAVE_HISTORY,
)
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.types import LlmStopReason, PromptMessageExtended
from fast_agent.utils.count_display import format_count

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.commands.context import CommandContext
    from fast_agent.interfaces import FastAgentLLMProtocol, LlmCapableProtocol

    type _HistorySaveSendFunc = Callable[[str, str], Awaitable[str]]


@runtime_checkable
class _WebToolsTupleCapable(Protocol):
    @property
    def web_tools_enabled(self) -> tuple[bool, bool]: ...


WEB_METADATA_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
WEB_METADATA_TOOL_RESULT_TYPES = frozenset(
    f"{tool_name}_tool_result" for tool_name in WEB_METADATA_TOOL_NAMES
)


@dataclass(frozen=True, slots=True)
class _StrippedRawAssistantChannel:
    blocks: list[object]
    removed_count: int


@dataclass(frozen=True, slots=True)
class _StrippedWebMetadataMessage:
    message: PromptMessageExtended
    removed_count: int


@dataclass(frozen=True, slots=True)
class _HistoryRewindTarget:
    user_text: str
    turn_start_index: int


@dataclass(frozen=True, slots=True)
class _SelectedHistoryUserTurn:
    turn: HistoryTurn
    total_turns: int


def _history_target(agent_name: str, target_agent: str | None) -> str:
    return target_agent or agent_name


def _history_editable_agent(ctx: CommandContext, agent_name: str) -> HistoryEditableAgent:
    return cast("HistoryEditableAgent", ctx.agent_provider._agent(agent_name))


def _trim_history_for_rewind(
    history: list[PromptMessageExtended],
    *,
    turn_start_index: int,
    template_messages: list[PromptMessageExtended] | None = None,
) -> list[PromptMessageExtended]:
    previous_assistant_index = next(
        (
            index
            for index, message in reversed(tuple(enumerate(history[:turn_start_index])))
            if message.role == "assistant"
        ),
        None,
    )
    if previous_assistant_index is not None:
        return history[: previous_assistant_index + 1]
    if template_messages:
        return template_messages
    return history[:turn_start_index]


def _is_web_tool_trace_payload(payload: Mapping[str, object]) -> bool:
    block_type = payload.get("type")
    if not isinstance(block_type, str):
        return False

    if block_type == "server_tool_use":
        tool_name = payload.get("name")
        return isinstance(tool_name, str) and tool_name in WEB_METADATA_TOOL_NAMES

    return block_type in WEB_METADATA_TOOL_RESULT_TYPES


def _strip_web_tool_traces_from_raw_assistant_channel(
    blocks: Sequence[object],
) -> _StrippedRawAssistantChannel:
    retained: list[object] = []
    removed = 0

    for block in blocks:
        from fast_agent.mcp.helpers.content_helpers import get_text

        raw_text = get_text(block)
        if not isinstance(raw_text, str) or not raw_text:
            retained.append(block)
            continue

        try:
            payload = json.loads(raw_text)
        except JSONDecodeError:
            retained.append(block)
            continue

        if isinstance(payload, Mapping) and _is_web_tool_trace_payload(payload):
            removed += 1
            continue

        retained.append(block)

    return _StrippedRawAssistantChannel(blocks=retained, removed_count=removed)


def _strip_web_metadata_channels(
    message: PromptMessageExtended,
) -> _StrippedWebMetadataMessage:
    channels = message.channels
    if not isinstance(channels, Mapping) or not channels:
        return _StrippedWebMetadataMessage(message=message, removed_count=0)

    removed_blocks = 0
    retained: dict[str, Sequence[object]] = {}
    for channel_name, blocks in channels.items():
        if channel_name in {ANTHROPIC_SERVER_TOOLS_CHANNEL, ANTHROPIC_CITATIONS_CHANNEL}:
            removed_blocks += len(blocks)
            continue
        if channel_name == ANTHROPIC_ASSISTANT_RAW_CONTENT:
            cleaned_channel = _strip_web_tool_traces_from_raw_assistant_channel(blocks)
            removed_blocks += cleaned_channel.removed_count
            if cleaned_channel.blocks:
                retained[channel_name] = cleaned_channel.blocks
            continue
        retained[channel_name] = blocks

    if removed_blocks == 0:
        return _StrippedWebMetadataMessage(message=message, removed_count=0)

    return _StrippedWebMetadataMessage(
        message=message.model_copy(update={"channels": retained or None}),
        removed_count=removed_blocks,
    )


def web_tools_enabled_for_agent(agent_obj: LlmCapableProtocol | None) -> bool:
    """Return True when the agent's active LLM has web tools enabled."""
    llm = agent_obj.llm if agent_obj is not None else None
    return web_tools_enabled_for_llm(llm)


def web_tools_enabled_for_llm(llm: FastAgentLLMProtocol | None) -> bool:
    """Return True when the active LLM has web tools enabled."""
    if llm is None:
        return False
    if isinstance(llm, _WebToolsTupleCapable):
        web_search_enabled, web_fetch_enabled = llm.web_tools_enabled
        return bool(web_search_enabled or web_fetch_enabled)

    return bool(resolve_web_search_enabled(llm) or resolve_web_fetch_enabled(llm))


async def handle_show_history(
    ctx: CommandContext,
    *,
    agent_name: str,
    target_agent: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    target = _history_target(agent_name, target_agent)
    agent_obj = _history_editable_agent(ctx, target)
    history = agent_obj.message_history
    usage = agent_obj.usage_accumulator
    await ctx.io.display_history_overview(target, list(history), usage)
    return outcome


def _add_history_turn_request_error(
    outcome: CommandOutcome,
    *,
    agent_name: str,
    turn_index: int | None,
    error: str | None,
    command: str,
    usage_action: str,
) -> bool:
    if error:
        outcome.add_message(error, channel="error", agent_name=agent_name)
        return True
    if turn_index is None:
        outcome.add_message(
            f"Usage: /history {usage_action} <turn>",
            channel="warning",
            agent_name=agent_name,
        )
        outcome.add_message(
            f"Tip: press Tab after '/history {command} ' to see turn options.",
            channel="info",
            agent_name=agent_name,
        )
        return True
    return False


def _history_rewind_target(
    outcome: CommandOutcome,
    *,
    agent_name: str,
    history: Sequence[PromptMessageExtended],
    turn_index: int,
) -> _HistoryRewindTarget | None:
    selected = _selected_history_user_turn(
        outcome,
        agent_name=agent_name,
        history=history,
        turn_index=turn_index,
        empty_action="rewind",
    )
    if selected is None:
        return None

    selected_turn = selected.turn
    user_message = selected_turn.first_user_message
    if user_message is None:
        outcome.add_message(
            "Selected turn has no user message to rewind.",
            channel="error",
            agent_name=agent_name,
        )
        return None

    content = user_message.content
    user_text = None
    if content:
        from fast_agent.mcp.helpers.content_helpers import get_text

        user_text = optional_string(get_text(content[0]))
    if not user_text or user_text == "<no text>":
        outcome.add_message(
            "Selected turn has no text content to rewind.",
            channel="error",
            agent_name=agent_name,
        )
        return None

    return _HistoryRewindTarget(
        user_text=user_text,
        turn_start_index=selected_turn.start_index,
    )


def _selected_history_user_turn(
    outcome: CommandOutcome,
    *,
    agent_name: str,
    history: Sequence[PromptMessageExtended],
    turn_index: int,
    empty_action: str,
) -> _SelectedHistoryUserTurn | None:
    user_turns = collect_user_turns(list(history))
    if not user_turns:
        outcome.add_message(
            f"No user turns available to {empty_action}.",
            channel="warning",
            agent_name=agent_name,
        )
        return None
    if turn_index < 1 or turn_index > len(user_turns):
        outcome.add_message("Turn index out of range.", channel="error", agent_name=agent_name)
        return None
    return _SelectedHistoryUserTurn(
        turn=user_turns[turn_index - 1],
        total_turns=len(user_turns),
    )


async def handle_history_rewind(
    ctx: CommandContext,
    *,
    agent_name: str,
    turn_index: int | None,
    error: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    if _add_history_turn_request_error(
        outcome,
        agent_name=agent_name,
        turn_index=turn_index,
        error=error,
        command="rewind",
        usage_action="rewind",
    ):
        return outcome
    assert turn_index is not None

    agent_obj = _history_editable_agent(ctx, agent_name)
    history = agent_obj.message_history
    rewind_target = _history_rewind_target(
        outcome,
        agent_name=agent_name,
        history=history,
        turn_index=turn_index,
    )
    if rewind_target is None:
        return outcome

    template_messages = (
        agent_obj.template_messages if isinstance(agent_obj, TemplateMessageProvider) else None
    )
    trimmed = _trim_history_for_rewind(
        list(history),
        turn_start_index=rewind_target.turn_start_index,
        template_messages=template_messages,
    )
    agent_obj.load_message_history(trimmed)

    outcome.buffer_prefill = rewind_target.user_text
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

    if _add_history_turn_request_error(
        outcome,
        agent_name=agent_name,
        turn_index=turn_index,
        error=error,
        command="detail",
        usage_action="detail",
    ):
        return outcome
    assert turn_index is not None

    agent_obj = _history_editable_agent(ctx, agent_name)
    history = agent_obj.message_history
    selected = _selected_history_user_turn(
        outcome,
        agent_name=agent_name,
        history=history,
        turn_index=turn_index,
        empty_action="review",
    )
    if selected is None:
        return outcome

    outcome.add_message(
        f"History detail: turn {turn_index}",
        channel="info",
        agent_name=agent_name,
    )

    await ctx.io.display_history_turn(
        agent_obj.name,
        list(selected.turn.messages),
        turn_index=turn_index,
        total_turns=selected.total_turns,
    )

    return outcome


async def handle_history_fix(
    ctx: CommandContext,
    *,
    agent_name: str,
    target_agent: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    target = _history_target(agent_name, target_agent)

    agent_obj = _history_editable_agent(ctx, target)
    history = list(agent_obj.message_history)
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
        agent_obj.load_message_history(trimmed)
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
    target = _history_target(agent_name, target_agent)

    raw_agent = ctx.agent_provider._agent(target)
    if not web_tools_enabled_for_agent(cast("LlmCapableProtocol", raw_agent)):
        outcome.add_message(
            "Web metadata cleanup is only available when Anthropic web tools are enabled.",
            channel="warning",
            agent_name=target,
        )
        return outcome

    agent_obj = cast("HistoryEditableAgent", raw_agent)
    history = list(agent_obj.message_history)
    if not history:
        outcome.add_message("No history to clean.", channel="warning", agent_name=target)
        return outcome

    cleaned_history: list[PromptMessageExtended] = []
    removed_blocks = 0
    touched_messages = 0
    for message in history:
        cleaned_message = _strip_web_metadata_channels(message)
        cleaned_history.append(cleaned_message.message)
        if cleaned_message.removed_count > 0:
            removed_blocks += cleaned_message.removed_count
            touched_messages += 1

    if removed_blocks == 0:
        outcome.add_message(
            f"No web metadata channels found for agent '{target}'.",
            channel="warning",
            agent_name=target,
        )
        return outcome

    agent_obj.load_message_history(cleaned_history)

    outcome.add_message(
        (
            f"Removed {format_count(removed_blocks, 'web metadata block')} from "
            f"{format_count(touched_messages, 'message')} for agent '{target}'."
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
    target = _history_target(agent_name, target_agent)

    agent_obj = _history_editable_agent(ctx, target)
    removed_message = agent_obj.pop_last_message()

    if removed_message:
        role_display = str(removed_message.role).capitalize()
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
    target = _history_target(agent_name, target_agent)

    agent_obj = ctx.agent_provider._agent(target)
    if isinstance(agent_obj, HistoryEditableAgent):
        try:
            agent_obj.clear()
            outcome.add_message(
                f"History cleared for agent '{target}'.",
                channel="info",
                agent_name=target,
            )
        except Exception as exc:
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
    send_func: "_HistorySaveSendFunc | None",
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
    except Exception as exc:
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
    load_result = load_prompt_messages_result(filename, label="history")
    if load_result.error is not None:
        outcome.add_message(load_result.error, channel="error", agent_name=agent_name)
        return outcome

    messages = load_result.messages
    if messages is None:
        return outcome

    replace_agent_history(agent_obj, messages)
    outcome.add_message(
        f"Loaded {format_count(len(messages), 'message')} from {filename}",
        channel="info",
        agent_name=agent_name,
    )

    return outcome
