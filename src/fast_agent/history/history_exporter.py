"""
History export utilities for agents.

Provides a minimal, type-friendly way to save an agent's message history
without using control strings. Uses the existing serialization helpers
to choose JSON (for .json files) or Markdown-like delimited text otherwise.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.mcp.prompt_serialization import save_messages, to_json
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


def _write_text(path: str, content: str) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(content)


class HistoryExporter:
    """Utility for exporting agent history to a file."""

    @staticmethod
    async def save(
        agent: AgentProtocol,
        filename: str | None = None,
        *,
        compact: bool = False,
    ) -> str:
        """
        Save the given agent's message history to a file.

        If filename ends with ".json", the history is saved in MCP JSON format.
        Otherwise, it is saved in a human-readable Markdown-style format.

        Args:
            agent: The agent whose history will be saved.
            filename: Optional filename. If None, a default timestamped name is generated.
            compact: Write JSON without indentation (for frequent checkpoint saves).

        Returns:
            The path that was written to.
        """
        # Determine a default filename when not provided
        if not filename:
            from datetime import datetime

            timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")
            target = f"{timestamp}-conversation.json"
        else:
            target = filename

        messages = agent.message_history
        if strip_casefold(Path(target).suffix) == ".json":
            # Serialize on the event loop so the snapshot is consistent, then
            # hand the (potentially multi-MB) file write to a worker thread so
            # per-checkpoint disk latency does not block streaming or the UI.
            json_str = to_json(messages, indent=None if compact else 2)
            await asyncio.to_thread(_write_text, target, json_str)
        else:
            save_messages(messages, target)

        # Return and optionally print a small confirmation
        return target
