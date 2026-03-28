"""Hook to save message history at the end of each turn."""

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, cast

from fast_agent.hooks import HookContext
from fast_agent.mcp.prompt_serialization import save_messages

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


async def save_history_to_file(ctx: HookContext) -> None:
    """
    Save the turn's messages to a timestamped JSON file.

    File format: <agent_name>-yyyy-mm-dd-hh-mm-ss.json
    """
    agent_name = ctx.agent.name
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{agent_name}-{timestamp}.json"

    # Collect messages from the turn:
    # - delta_messages contains accumulated messages when use_history: false
    # - ctx.message is the final assistant response
    # - message_history is the agent's persisted history (empty if use_history: false)
    messages: list[PromptMessageExtended] = ctx.message_history
    if not messages:
        # Fall back to runner's turn messages + final response
        runner_messages = getattr(ctx.runner, "delta_messages", None)
        if isinstance(runner_messages, Iterable):
            messages = [cast("PromptMessageExtended", message) for message in runner_messages]
        else:
            messages = []
        if ctx.message and ctx.message not in messages:
            messages.append(ctx.message)

    save_messages(messages, filename)
