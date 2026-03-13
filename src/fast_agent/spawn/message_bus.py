"""Message Bus — file-based inbox queue for inter-agent communication.

Each agent has an inbox file: ``<messages_dir>/{agent_name}_inbox.jsonl``
Messages are FIFO — oldest first. Processed messages are tracked separately.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    """Convert agent name to a safe filename component."""
    return re.sub(r"[^\w\-]", "_", name).strip("_").lower()


@dataclass
class AgentMessage:
    """A message between agents."""

    message_id: str = ""
    from_name: str = ""
    to_name: str = ""
    content: str = ""
    message_type: str = "task"  # task | question | response | notification
    priority: str = "normal"  # low | normal | high | urgent
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    reply_to: str = ""  # message_id this replies to
    status: str = "unread"  # unread | processing | done

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        # Backward compat: accept from_role/to_role
        if "from_role" in data and "from_name" not in data:
            data["from_name"] = data.pop("from_role")
        if "to_role" in data and "to_name" not in data:
            data["to_name"] = data.pop("to_role")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> AgentMessage:
        return cls.from_dict(json.loads(line))


class MessageBus:
    """File-based JSONL inbox queue per agent name.

    Each agent has:
    - ``{name}_inbox.jsonl`` — all messages (append-only)
    - ``{name}_processed.json`` — set of processed message_ids

    Queue semantics:
    - ``send()`` — append message to target inbox
    - ``read_unread()`` — get unread messages (FIFO order)
    - ``pop_next()`` — get oldest unread, mark as processing
    - ``mark_done()`` — mark message as done
    """

    def __init__(self, messages_dir: str | Path) -> None:
        self._dir = Path(messages_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def messages_dir(self) -> Path:
        return self._dir

    def _inbox_path(self, agent_name: str) -> Path:
        return self._dir / f"{_safe_filename(agent_name)}_inbox.jsonl"

    def _processed_path(self, agent_name: str) -> Path:
        return self._dir / f"{_safe_filename(agent_name)}_processed.json"

    def _load_processed(self, agent_name: str) -> set[str]:
        """Load set of processed message IDs."""
        path = self._processed_path(agent_name)
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return set(data.get("processed", []))
        except (json.JSONDecodeError, KeyError):
            return set()

    def _save_processed(self, agent_name: str, processed: set[str]) -> None:
        """Save processed message IDs."""
        path = self._processed_path(agent_name)
        path.write_text(
            json.dumps({"processed": sorted(processed)}, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Send ──

    def send(
        self,
        from_name: str,
        to_name: str,
        content: str,
        message_type: str = "task",
        priority: str = "normal",
        context: dict[str, Any] | None = None,
        reply_to: str = "",
    ) -> AgentMessage:
        """Send a message to an agent's inbox queue."""
        msg = AgentMessage(
            from_name=from_name,
            to_name=to_name,
            content=content,
            message_type=message_type,
            priority=priority,
            context=context or {},
            reply_to=reply_to,
            status="unread",
        )
        with open(self._inbox_path(to_name), "a", encoding="utf-8") as f:
            f.write(msg.to_line() + "\n")
        logger.info(
            "📬 Message queued: %s → %s [%s] (id: %s)",
            from_name, to_name, message_type, msg.message_id,
        )
        return msg

    # ── Read ──

    def _read_all(self, agent_name: str) -> list[AgentMessage]:
        """Read ALL messages from inbox (including processed)."""
        inbox = self._inbox_path(agent_name)
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

    def read_inbox(self, agent_name: str) -> list[AgentMessage]:
        """Read all messages (for backward compat)."""
        return self._read_all(agent_name)

    def read_unread(self, agent_name: str) -> list[AgentMessage]:
        """Read only unread messages (FIFO order)."""
        all_msgs = self._read_all(agent_name)
        processed = self._load_processed(agent_name)
        return [m for m in all_msgs if m.message_id not in processed]

    def pop_next(self, agent_name: str) -> AgentMessage | None:
        """Get the oldest unread message (FIFO). Does NOT auto-mark as done."""
        unread = self.read_unread(agent_name)
        return unread[0] if unread else None

    # ── Mark ──

    def mark_done(self, agent_name: str, message_id: str) -> None:
        """Mark a message as processed."""
        processed = self._load_processed(agent_name)
        processed.add(message_id)
        self._save_processed(agent_name, processed)

    def mark_all_done(self, agent_name: str) -> int:
        """Mark all current messages as processed. Returns count."""
        all_msgs = self._read_all(agent_name)
        processed = self._load_processed(agent_name)
        count = 0
        for msg in all_msgs:
            if msg.message_id not in processed:
                processed.add(msg.message_id)
                count += 1
        if count:
            self._save_processed(agent_name, processed)
        return count

    # ── Query ──

    def unread_count(self, agent_name: str) -> int:
        """Count unread messages."""
        return len(self.read_unread(agent_name))

    def has_unread(self, agent_name: str) -> bool:
        """Check if agent has unread messages."""
        return self.unread_count(agent_name) > 0

    def read_inbox_formatted(self, agent_name: str) -> str:
        """Read inbox as human-readable formatted string."""
        messages = self._read_all(agent_name)
        processed = self._load_processed(agent_name)
        if not messages:
            return "(no messages)"
        priority_icons = {
            "urgent": "🔴",
            "high": "🟠",
            "normal": "🟢",
            "low": "⚪",
        }
        lines = [f"📬 Inbox for {agent_name} ({len(messages)} messages):"]
        for msg in messages:
            icon = priority_icons.get(msg.priority, "🟢")
            status_tag = " ✅" if msg.message_id in processed else " 🆕"
            lines.append(
                f"\n{icon}{status_tag} [{msg.message_type.upper()}] "
                f"from {msg.from_name} (id: {msg.message_id}):"
            )
            lines.append(f"  {msg.content}")
            if msg.reply_to:
                lines.append(f"  (reply to: {msg.reply_to})")
        return "\n".join(lines)

    def get_unread_count(self, agent_name: str) -> int:
        """Alias for unread_count (backward compat)."""
        return self.unread_count(agent_name)

    def clear_inbox(self, agent_name: str) -> int:
        """Clear all messages from an agent's inbox. Returns count cleared."""
        inbox = self._inbox_path(agent_name)
        if not inbox.exists():
            return 0
        count = len(self._read_all(agent_name))
        inbox.unlink()
        proc = self._processed_path(agent_name)
        if proc.exists():
            proc.unlink()
        return count

    def has_pending(self, agent_name: str) -> bool:
        """Check if agent has any unread messages."""
        return self.has_unread(agent_name)

    def list_inboxes(self) -> dict[str, int]:
        """List all inboxes and their unread message counts."""
        inboxes: dict[str, int] = {}
        for f in self._dir.glob("*_inbox.jsonl"):
            name = f.stem.replace("_inbox", "")
            inboxes[name] = self.unread_count(name)
        return inboxes
