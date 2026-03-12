"""Message Bus — file-based inbox system for inter-agent communication.

Each agent has an inbox file: ``<messages_dir>/{role}_inbox.jsonl``
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """A message between agents."""

    message_id: str = ""
    from_role: str = ""
    to_role: str = ""
    content: str = ""
    message_type: str = "task"  # task | question | response | notification
    priority: str = "normal"  # low | normal | high | urgent
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    reply_to: str = ""

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> AgentMessage:
        return cls.from_dict(json.loads(line))


class MessageBus:
    """File-based JSONL inbox per agent role."""

    def __init__(self, messages_dir: str | Path) -> None:
        self._dir = Path(messages_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def messages_dir(self) -> Path:
        return self._dir

    def _inbox_path(self, role: str) -> Path:
        return self._dir / f"{role}_inbox.jsonl"

    def _notify_path(self, role: str) -> Path:
        return self._dir / f".notify_{role}"

    def send(
        self,
        from_role: str,
        to_role: str,
        content: str,
        message_type: str = "task",
        priority: str = "normal",
        context: dict[str, Any] | None = None,
        reply_to: str = "",
    ) -> AgentMessage:
        """Send a message to an agent's inbox."""
        msg = AgentMessage(
            from_role=from_role,
            to_role=to_role,
            content=content,
            message_type=message_type,
            priority=priority,
            context=context or {},
            reply_to=reply_to,
        )
        with open(self._inbox_path(to_role), "a", encoding="utf-8") as f:
            f.write(msg.to_line() + "\n")
        self._notify_path(to_role).touch()
        logger.info("Message sent: %s → %s (%s)", from_role, to_role, message_type)
        return msg

    def read_inbox(self, role: str) -> list[AgentMessage]:
        """Read all messages from an agent's inbox."""
        inbox = self._inbox_path(role)
        if not inbox.exists():
            return []
        messages: list[AgentMessage] = []
        for line in inbox.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line:
                try:
                    messages.append(AgentMessage.from_line(line))
                except (json.JSONDecodeError, KeyError):
                    continue
        return messages

    def read_inbox_formatted(self, role: str) -> str:
        """Read inbox as human-readable formatted string."""
        messages = self.read_inbox(role)
        if not messages:
            return "(no messages)"
        priority_icons = {
            "urgent": "🔴",
            "high": "🟠",
            "normal": "🟢",
            "low": "⚪",
        }
        lines = [f"📬 Inbox for {role} ({len(messages)} messages):"]
        for msg in messages:
            icon = priority_icons.get(msg.priority, "🟢")
            lines.append(
                f"\n{icon} [{msg.message_type.upper()}] "
                f"from {msg.from_role} (id: {msg.message_id}):"
            )
            lines.append(f"  {msg.content}")
            if msg.reply_to:
                lines.append(f"  (reply to: {msg.reply_to})")
        return "\n".join(lines)

    def get_unread_count(self, role: str) -> int:
        return len(self.read_inbox(role))

    def clear_inbox(self, role: str) -> int:
        """Clear all messages from an agent's inbox. Returns count cleared."""
        inbox = self._inbox_path(role)
        if not inbox.exists():
            return 0
        count = len(self.read_inbox(role))
        inbox.unlink()
        notify = self._notify_path(role)
        if notify.exists():
            notify.unlink()
        return count

    def has_pending(self, role: str) -> bool:
        return self._notify_path(role).exists()

    def list_inboxes(self) -> dict[str, int]:
        """List all inboxes and their message counts."""
        inboxes: dict[str, int] = {}
        for f in self._dir.glob("*_inbox.jsonl"):
            role = f.stem.replace("_inbox", "")
            inboxes[role] = len(self.read_inbox(role))
        return inboxes
