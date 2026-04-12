"""Inbox Watcher Hook — Real-time Agent Communication (RTAC).

Injects unread inbox messages into the agent's LLM context between tool loops
using ToolRunnerHooks.before_llm_call + runner.append_messages().

This replaces the old pull model (manual inbox polling)
to a push model where messages are delivered automatically.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.spawn.message_bus import AgentMessage


class InboxWatcherHook:
    """Push unread emails into agent's LLM context between tool loops.

    - Checks inbox before each LLM call (throttled by check_interval)
    - Injects up to max_per_inject messages per check
    - Marks injected messages as done; remaining unread arrive in next batch
    - Agent decides priority — hook only delivers, never forces task switches
    """

    def __init__(
        self,
        agent_name: str,
        messages_dir: str | Path,
        check_interval: float = 5.0,
        max_per_inject: int = 3,
    ) -> None:
        self.agent_name = agent_name
        self._messages_dir = Path(messages_dir)
        self.check_interval = check_interval
        self.max_per_inject = max_per_inject
        self._last_check: float = 0.0
        self._injected_ids: set[str] = set()

    def _get_bus(self) -> Any:
        """Lazy-load MessageBus to avoid import cycles."""
        from fast_agent.spawn.message_bus import MessageBus

        return MessageBus(self._messages_dir)

    async def before_llm_call(
        self,
        runner: ToolRunner,
        messages: list[Any],
    ) -> None:
        """Hook: check inbox and inject unread messages before each LLM call."""
        now = time.time()

        # Throttle: skip if checked recently (bypass on first call)
        if self._last_check > 0 and (now - self._last_check) < self.check_interval:
            return
        self._last_check = now

        # Guard: need agent name and messages dir
        if not self.agent_name or not self._messages_dir.exists():
            return

        try:
            bus = self._get_bus()
            unread = bus.read_unread(self.agent_name)
        except Exception as exc:
            logger.debug("RTAC: failed to read inbox: %s", exc)
            return

        # Filter already-injected (safety dedup)
        new_messages = [
            m for m in unread if m.message_id not in self._injected_ids
        ]
        if not new_messages:
            return

        # Take batch, remaining stay unread for next inject cycle
        batch = new_messages[: self.max_per_inject]
        inject_text = self._format_messages(batch)
        runner.append_messages(inject_text)

        # Mark done only the injected batch
        for msg in batch:
            self._injected_ids.add(msg.message_id)
            try:
                bus.mark_done(self.agent_name, msg.message_id)
            except Exception:
                pass  # Best-effort

        remaining = len(new_messages) - len(batch)
        logger.info(
            "📨 RTAC: injected %d message(s) into %s (remaining unread: %d)",
            len(batch),
            self.agent_name,
            remaining,
        )

    @staticmethod
    def _format_messages(messages: list[AgentMessage]) -> str:
        """Format messages for LLM context injection."""
        priority_icons = {
            "urgent": "🔴",
            "high": "🟠",
            "normal": "🟢",
            "low": "⚪",
        }

        lines = [
            "\n━━━ 📬 NEW MESSAGES RECEIVED ━━━",
            f"You have {len(messages)} new message(s):\n",
        ]

        for msg in messages:
            icon = priority_icons.get(msg.priority, "🟢")
            lines.append(
                f"{icon} [{msg.message_type.upper()}] from {msg.from_name}:"
            )
            lines.append(f"  {msg.content}")
            if msg.reply_to:
                lines.append(f"  (reply to: {msg.reply_to})")
            lines.append("")

        lines.append(
            "→ Review these messages and decide how to handle them. "
            "You may continue your current task if it has higher priority, "
            "or switch to handle urgent messages first — use your judgment."
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━\n")

        return "\n".join(lines)


def create_inbox_watcher() -> InboxWatcherHook | None:
    """Factory: create InboxWatcherHook from environment variables.

    Returns None if TEAM_MY_NAME or TEAM_WORKSPACE are not set.
    """
    agent_name = os.environ.get("TEAM_MY_NAME", "")
    workspace = os.environ.get("TEAM_WORKSPACE", "")

    if not agent_name or not workspace:
        return None

    messages_dir = Path(workspace) / ".messages"
    if not messages_dir.exists():
        return None

    return InboxWatcherHook(
        agent_name=agent_name,
        messages_dir=messages_dir,
    )
