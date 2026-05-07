"""Subagents Tool — peer-to-peer agent communication capabilities.

Actions: list, send, wait, steer, kill, status, inbox

Each spawned agent gets an instance of :class:`SubagentsTool`
with its agent_name pre-configured, enabling it to interact with
sibling agents via the :class:`SpawnRegistry` and :class:`MessageBus`.
"""

from __future__ import annotations

import json
import logging
import os
import signal as os_signal
from typing import Any

from fast_agent.spawn.message_bus import MessageBus
from fast_agent.spawn.spawn_registry import (
    SpawnRegistry,
    SpawnStatus,
)

logger = logging.getLogger(__name__)


class SubagentsTool:
    """Each spawned agent gets an instance with its name."""

    def __init__(
        self,
        my_name: str,
        workspace_dir: str,
        registry: SpawnRegistry | None = None,
        message_bus: MessageBus | None = None,
    ) -> None:
        self.my_name = my_name
        self.workspace_dir = workspace_dir
        self._registry = registry or SpawnRegistry()
        self._bus = message_bus or MessageBus(messages_dir=workspace_dir)

    def dispatch(self, action: str, **kwargs: Any) -> str:
        """Route an action to the appropriate handler."""
        actions = {
            "list": self._action_list,
            "send": self._action_send,
            "wait": self._action_wait,
            "steer": self._action_steer,
            "kill": self._action_kill,
            "status": self._action_status,
            "inbox": self._action_inbox,
        }
        handler = actions.get(action)
        if not handler:
            return json.dumps(
                {"error": (f"Unknown action '{action}'. Available: {list(actions.keys())}")}
            )
        try:
            return handler(**kwargs)
        except Exception as e:
            logger.error("subagents(%s) failed: %s", action, e)
            return json.dumps({"error": str(e), "action": action})

    def _action_list(self, **kwargs: Any) -> str:
        all_spawns = self._registry.list_all()
        agents = [
            {
                "agent_name": r.agent_name,
                "role": r.role,
                "run_id": r.run_id,
                "status": r.status,
                "lifecycle": r.lifecycle,
                "task": r.task[:80] if r.task else "",
                "is_me": r.agent_name == self.my_name,
            }
            for r in all_spawns
        ]
        return json.dumps(
            {
                "my_name": self.my_name,
                "count": len(agents),
                "agents": agents,
            }
        )

    def _action_send(
        self,
        target: str = "",
        message: str = "",
        message_type: str = "task",
        priority: str = "normal",
        reply_to: str = "",
        **kwargs: Any,
    ) -> str:
        if not target:
            return json.dumps({"error": "target is required"})
        if not message:
            return json.dumps({"error": "message is required"})
        msg = self._bus.send(
            from_name=self.my_name,
            to_name=target,
            content=message,
            message_type=message_type,
            priority=priority,
            reply_to=reply_to,
        )
        return json.dumps(
            {
                "status": "sent",
                "message_id": msg.message_id,
                "from": self.my_name,
                "to": target,
            }
        )

    def _action_wait(
        self,
        target: str = "",
        timeout_seconds: float = 300,
        **kwargs: Any,
    ) -> str:
        """Poll the spawn registry until target agent completes."""
        import time as _time

        if not target:
            return json.dumps({"error": "target is required"})

        poll_interval = 2.0
        start = _time.time()

        while _time.time() - start < timeout_seconds:
            record = self._registry.find_by_name(target)
            if record and record.status in ("completed", "error", "timeout", "cancelled"):
                return json.dumps(
                    {
                        "status": record.status,
                        "agent_name": record.agent_name,
                        "role": record.role,
                        "run_id": record.run_id,
                        "result_summary": record.result[:500] if record.result else "",
                        "error": record.error or "",
                    }
                )
            _time.sleep(poll_interval)

        return json.dumps(
            {
                "status": "timeout",
                "message": f"Agent '{target}' did not complete within {timeout_seconds}s",
            }
        )

    def _action_steer(
        self,
        target: str = "",
        new_instruction: str = "",
        **kwargs: Any,
    ) -> str:
        if not target:
            return json.dumps({"error": "target is required"})
        if not new_instruction:
            return json.dumps({"error": "new_instruction is required"})
        record = self._registry.find_by_name(target)
        if not record or record.is_terminal:
            return json.dumps({"error": f"No active agent '{target}' found"})
        if record.pid:
            try:
                os.kill(record.pid, os_signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        self._registry.update_status(record.run_id, SpawnStatus.KILLED)
        self._bus.send(
            from_name=self.my_name,
            to_name=target,
            content=f"[STEER] Previous task cancelled. New instruction:\n\n{new_instruction}",
            message_type="task",
            priority="urgent",
        )
        return json.dumps(
            {
                "status": "steered",
                "target": target,
                "killed_run_id": record.run_id,
            }
        )

    def _action_kill(self, target: str = "", **kwargs: Any) -> str:
        if not target:
            return json.dumps({"error": "target is required"})
        record = self._registry.find_by_name(target)
        if not record or record.is_terminal:
            return json.dumps({"error": f"No active agent '{target}' found"})
        if record.pid:
            try:
                os.kill(record.pid, os_signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        self._registry.update_status(record.run_id, SpawnStatus.KILLED)
        return json.dumps(
            {
                "status": "killed",
                "target": target,
                "killed_run_id": record.run_id,
            }
        )

    def _action_status(self, target: str = "", **kwargs: Any) -> str:
        if not target:
            return json.dumps({"error": "target is required"})
        record = self._registry.find_by_name(target)
        if not record:
            return json.dumps({"status": "not_found", "agent_name": target})
        return json.dumps(
            {
                "agent_name": record.agent_name,
                "role": record.role,
                "run_id": record.run_id,
                "status": record.status,
                "lifecycle": record.lifecycle,
                "task": record.task,
                "duration_seconds": record.duration_seconds,
                "error": record.error,
            }
        )

    def _action_inbox(self, **kwargs: Any) -> str:
        return self._bus.read_inbox_formatted(self.my_name)
