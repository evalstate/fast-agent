"""History display helpers shared by UI command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from fast_agent.ui.message_display_helpers import build_user_message_display

    display = ConsoleDisplay(config=config)
    user_group: list[PromptMessageExtended] = []

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
        if message.role == "user":
            user_group.append(message)
            continue

        flush_user_group()

        if message.role == "assistant":
            await display.show_assistant_message(
                message_text=message.last_text() or "<no text>",
                name=agent_name,
            )
            continue

        if message.role == "tool":
            tool_name = None
            tool_args = None
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = getattr(tool_call, "name", None)
                tool_args = getattr(tool_call, "arguments", None)
            display.show_tool_call(
                content=message.last_text() or "<no text>",
                tool_name=tool_name,
                tool_args=tool_args,
                name=agent_name,
            )

        tool_results = getattr(message, "tool_results", None)
        if tool_results:
            for result in tool_results.values():
                display.show_tool_result(
                    result=result,
                    name=agent_name,
                    truncate_content=False,
                )

    flush_user_group()
