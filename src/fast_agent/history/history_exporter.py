"""
History export utilities for agents.

Provides a minimal, type-friendly way to save an agent's message history
without using control strings. Uses the existing serialization helpers
to choose JSON (for .json files) or Markdown-like delimited text otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fast_agent.mcp.prompt_serialization import save_messages

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class HistoryExporter:
    """Utility for exporting agent history to a file."""

    @staticmethod
    async def save(agent: AgentProtocol, filename: Optional[str] = None) -> str:
        """
        Save the given agent's message history to a file.

        If filename ends with ".json", the history is saved in MCP JSON format.
        Otherwise, it is saved in a human-readable Markdown-style format.

        Args:
            agent: The agent whose history will be saved.
            filename: Optional filename. If None, a default timestamped name is generated.

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
        save_messages(messages, target)

        # Return and optionally print a small confirmation
        return target
