"""
Built-in history trimmer hook for tool loops.

After a turn completes with multiple tool calls, this hook trims the history
to keep only the essential messages: user message, last tool call, its result,
and the final response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.hooks.hook_context import HookContext
    from fast_agent.types import PromptMessageExtended


def _find_turn_start(history: list[PromptMessageExtended]) -> int:
    """
    Find the index where the current turn started.

    A turn starts with a user message that does NOT contain tool results.
    We search backwards from the end to find this boundary.

    Returns:
        Index of the turn start, or 0 if not found.
    """
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        if msg.role == "user":
            # Check if this is a tool result message
            if msg.tool_results:
                continue  # This is a tool result, keep searching
            # This is a user message without tool results - turn boundary
            return i
    return 0


def _trim_turn_messages(
    turn_messages: list[PromptMessageExtended],
) -> list[PromptMessageExtended]:
    """
    Trim a turn's messages to keep only essential context.

    Input structure (typical tool loop):
        - user message (the prompt)
        - assistant message (tool_call 1)
        - user message (tool_result 1)
        - assistant message (tool_call 2)
        - user message (tool_result 2)
        - ...
        - assistant message (final response, no tool calls)

    Output structure:
        - user message (the prompt)
        - assistant message (last tool_call)
        - user message (its tool_result)
        - assistant message (final response)

    If there's only one tool call or no tool calls, returns messages unchanged.
    """
    if len(turn_messages) < 4:
        # Not enough messages to trim (need at least: user, assistant, user, assistant)
        return turn_messages

    # Find all assistant messages with tool calls
    tool_call_indices: list[int] = []
    for i, msg in enumerate(turn_messages):
        if msg.role == "assistant" and msg.stop_reason == LlmStopReason.TOOL_USE:
            tool_call_indices.append(i)

    if len(tool_call_indices) <= 1:
        # Only one or no tool calls, nothing to trim
        return turn_messages

    # We want to keep:
    # 1. First user message (the original prompt)
    # 2. Last tool call assistant message
    # 3. Its corresponding tool result (next message)
    # 4. Final assistant response

    first_user_idx = 0
    last_tool_call_idx = tool_call_indices[-1]
    tool_result_idx = last_tool_call_idx + 1
    final_response_idx = len(turn_messages) - 1

    # Validate we have the expected structure
    if tool_result_idx >= len(turn_messages):
        return turn_messages  # Unexpected structure, don't trim

    trimmed = [
        turn_messages[first_user_idx],
        turn_messages[last_tool_call_idx],
    ]

    # Add tool result if it exists and is in the expected position
    if tool_result_idx < final_response_idx:
        trimmed.append(turn_messages[tool_result_idx])

    # Add final response if it's different from what we've added
    if final_response_idx > tool_result_idx:
        trimmed.append(turn_messages[final_response_idx])
    elif final_response_idx == tool_result_idx:
        # Edge case: tool result is the final message (shouldn't happen normally)
        pass

    return trimmed


async def trim_tool_loop_history(ctx: HookContext) -> None:
    """
    Built-in hook to trim tool loop history after a turn completes.

    After a turn with multiple tool calls, this removes intermediate tool calls
    and results, keeping only:
    - The original user prompt
    - The last tool call
    - Its result
    - The final assistant response

    This helps keep context clean for subsequent turns while preserving
    the most relevant tool interaction.

    Usage in agent card:
        trim_tool_history: true

    Or as a custom hook:
        tool_hooks:
          after_turn_complete: fast_agent.hooks.history_trimmer:trim_tool_loop_history
    """
    if not ctx.is_turn_complete:
        return

    history = ctx.message_history
    if len(history) < 4:
        # Not enough messages to need trimming
        return

    # Find where this turn started
    turn_start_idx = _find_turn_start(history)

    # Get messages before this turn and for this turn
    pre_turn_messages = history[:turn_start_idx]
    turn_messages = history[turn_start_idx:]

    # Trim the turn messages
    trimmed_turn = _trim_turn_messages(turn_messages)

    # Only update history if we actually trimmed something
    if len(trimmed_turn) < len(turn_messages):
        new_history = pre_turn_messages + trimmed_turn
        ctx.load_message_history(new_history)
