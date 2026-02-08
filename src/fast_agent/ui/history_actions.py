"""History display helpers shared by UI command handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fast_agent.config import Settings
    from fast_agent.types import PromptMessageExtended


async def display_history_turn(
    agent_name: str,
    turn: list[PromptMessageExtended],
    *,
    config: Settings | None,
    turn_index: int | None = None,
    total_turns: int | None = None,
) -> None:
    from fast_agent.ui.console_display import ConsoleDisplay
    from fast_agent.ui.message_display_helpers import (
        build_tool_use_additional_message,
        build_user_message_display,
    )

    display = ConsoleDisplay(config=config)
    user_group: list[PromptMessageExtended] = []
    tool_name_lookup: dict[str, str] = {}

    def _tool_name_from_call(call_id: str, call: object) -> str:
        params = getattr(call, "params", None)
        tool_name = getattr(params, "name", None) if params is not None else None
        tool_name = tool_name or getattr(call, "name", None)
        return tool_name or call_id

    def _tool_args_from_call(call: object) -> dict[str, Any] | None:
        params = getattr(call, "params", None)
        raw_args = getattr(params, "arguments", None) if params is not None else None
        if raw_args is None:
            raw_args = getattr(call, "arguments", None)
        if isinstance(raw_args, Mapping):
            return dict(raw_args)
        return None

    def flush_user_group() -> None:
        if not user_group:
            return
        message_text, attachments = build_user_message_display(user_group)
        part_count = len(user_group)
        turn_range = (turn_index, turn_index) if turn_index else None
        display.show_user_message(
            message=message_text,
            attachments=attachments,
            name=agent_name,
            part_count=part_count,
            turn_range=turn_range,
            total_turns=total_turns,
        )
        user_group.clear()

    for message in turn:
        tool_calls = message.tool_calls
        if tool_calls:
            for call_id, call in tool_calls.items():
                tool_name_lookup[call_id] = _tool_name_from_call(call_id, call)

        if message.role == "user" and not message.tool_results:
            user_group.append(message)
            continue

        flush_user_group()

        if message.role == "assistant":
            last_text = message.last_text()
            additional_message = build_tool_use_additional_message(message, last_text)
            message_payload: str | PromptMessageExtended = message
            if last_text is None and additional_message is None:
                message_payload = "<no text>"
            await display.show_assistant_message(
                message_text=message_payload,
                name=agent_name,
                additional_message=additional_message,
            )

            if tool_calls:
                for call_id, call in tool_calls.items():
                    tool_name = tool_name_lookup.get(call_id, call_id)
                    tool_args = _tool_args_from_call(call)
                    display.show_tool_call(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        name=agent_name,
                        tool_call_id=call_id,
                    )

        tool_results = message.tool_results
        if tool_results:
            for call_id, result in tool_results.items():
                tool_name = tool_name_lookup.get(call_id)
                display.show_tool_result(
                    result=result,
                    name=agent_name,
                    tool_name=tool_name,
                    tool_call_id=call_id,
                    truncate_content=False,
                )

    flush_user_group()
