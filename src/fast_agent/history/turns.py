"""Where user turns start in a conversation history.

A turn is one user prompt and everything the agent did in reply. User-role
messages that fast-agent adds itself are not prompts: tool results, tool-media
continuations, tool-loop messages (hook additions, structured-output finalizers) and
compaction checkpoints all belong to (or summarise) other turns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp_types import TextContent

from fast_agent.constants import (
    FAST_AGENT_COMPACTION_CHANNEL,
    FAST_AGENT_TOOL_LOOP_MESSAGE,
    FAST_AGENT_TOOL_MEDIA_MESSAGE,
)

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended

_NON_PROMPT_CHANNELS = frozenset(
    {FAST_AGENT_COMPACTION_CHANNEL, FAST_AGENT_TOOL_LOOP_MESSAGE, FAST_AGENT_TOOL_MEDIA_MESSAGE}
)


def is_user_prompt(message: PromptMessageExtended) -> bool:
    """True for user input, as opposed to user-role messages fast-agent adds itself."""
    return (
        message.role == "user"
        and not message.tool_results
        and _NON_PROMPT_CHANNELS.isdisjoint(message.channels or {})
    )


def user_turn_starts(messages: list[PromptMessageExtended]) -> list[int]:
    """Indices where user turns start.

    Prompts sent before any assistant reply belong to the same turn.
    """
    starts: list[int] = []
    awaiting_reply = False
    for index, message in enumerate(messages):
        if is_user_prompt(message):
            if not awaiting_reply:
                starts.append(index)
                awaiting_reply = True
        elif message.role == "assistant":
            awaiting_reply = False
    return starts


def compacted_turn_count(message: PromptMessageExtended) -> int:
    """User turns folded into a compaction checkpoint; 0 for other messages."""
    blocks = (message.channels or {}).get(FAST_AGENT_COMPACTION_CHANNEL)
    if not blocks or not isinstance(blocks[0], TextContent):
        return 0
    try:
        metadata = json.loads(blocks[0].text)
    except ValueError:  # checkpoints can come from saved sessions on disk
        return 0
    count = metadata.get("turns_compacted") if isinstance(metadata, dict) else None
    return count if isinstance(count, int) else 0


def total_user_turns(messages: list[PromptMessageExtended]) -> int:
    """User turns in the conversation so far, including those compacted away."""
    return len(user_turn_starts(messages)) + sum(map(compacted_turn_count, messages))
