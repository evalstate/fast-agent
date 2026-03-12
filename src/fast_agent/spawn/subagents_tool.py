"""Subagents Tool — peer-to-peer agent communication capabilities.

Actions: list, send, wait, steer, kill, status, inbox

Each spawned agent gets an instance of :class:`SubagentsTool`
with its role pre-configured, enabling it to interact with
sibling agents via the :class:`SpawnRegistry`,
:class:`MessageBus`, and :class:`SignalStore`.
"""

from __future__ import annotations

import json
import logging
import os
import signal as os_signal
from typing import Any

from fast_agent.spawn.message_bus import MessageBus
from fast_agent.spawn.signal_store import SignalStore
from fast_agent.spawn.spawn_registry import (
    SpawnRegistry,
    SpawnStatus,
)

logger = logging.getLogger(__name__)


class SubagentsTool:
    """Each spawned agent gets an instance with its role."""

    def __init__(
        self,
        my_role: str,
        workspace_dir: str,
        registry: SpawnRegistry | None = None,
        message_bus: MessageBus | None = None,
        signal_store: SignalStore | None = None,
    ) -> None:
        self.my_role = my_role
        self.workspace_dir = workspace_dir
        self._registry = registry or SpawnRegistry()
        self._bus = message_bus or MessageBus(messages_dir=workspace_dir)
        self._signals = signal_store or SignalStore(signals_dir=workspace_dir)

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
                "role": r.role,
                "run_id": r.run_id,
                "status": r.status,
                "lifecycle": r.lifecycle,
                "task": r.task[:80] if r.task else "",
                "is_me": r.role == self.my_role,
            }
            for r in all_spawns
        ]
        return json.dumps(
            {
                "my_role": self.my_role,
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
            from_role=self.my_role,
            to_role=target,
            content=message,
            message_type=message_type,
            priority=priority,
            reply_to=reply_to,
        )
        return json.dumps(
            {
                "status": "sent",
                "message_id": msg.message_id,
                "from": self.my_role,
                "to": target,
            }
        )

    def _action_wait(
        self,
        target: str = "",
        timeout_seconds: float = 300,
        **kwargs: Any,
    ) -> str:
        if not target:
            return json.dumps({"error": "target is required"})
        sig = self._signals.wait_for_role(role=target, timeout_seconds=timeout_seconds)
        if sig:
            return json.dumps(
                {
                    "status": sig.status,
                    "role": sig.role,
                    "result_summary": sig.result_summary,
                    "output_files": sig.output_files,
                }
            )
        return json.dumps(
            {
                "status": "timeout",
                "message": (f"Agent '{target}' did not complete within {timeout_seconds}s"),
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
        targets = self._registry.find_by_role(target)
        active = [r for r in targets if not r.is_terminal]
        if not active:
            return json.dumps({"error": (f"No active agent with role '{target}' found")})
        record = active[0]
        if record.pid:
            try:
                os.kill(record.pid, os_signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        self._registry.update_status(record.run_id, SpawnStatus.KILLED)
        self._bus.send(
            from_role=self.my_role,
            to_role=target,
            content=(f"[STEER] Previous task cancelled. New instruction:\n\n{new_instruction}"),
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
        targets = self._registry.find_by_role(target)
        active = [r for r in targets if not r.is_terminal]
        if not active:
            return json.dumps({"error": (f"No active agent '{target}' found")})
        killed = []
        for record in active:
            if record.pid:
                try:
                    os.kill(record.pid, os_signal.SIGTERM)
                except (
                    ProcessLookupError,
                    PermissionError,
                ):
                    pass
            self._registry.update_status(record.run_id, SpawnStatus.KILLED)
            killed.append(record.run_id)
        return json.dumps(
            {
                "status": "killed",
                "target": target,
                "killed_run_ids": killed,
            }
        )

    def _action_status(self, target: str = "", **kwargs: Any) -> str:
        if not target:
            return json.dumps({"error": "target is required"})
        targets = self._registry.find_by_role(target)
        if not targets:
            return json.dumps({"status": "not_found", "role": target})
        rec = targets[-1]
        return json.dumps(
            {
                "role": rec.role,
                "run_id": rec.run_id,
                "status": rec.status,
                "lifecycle": rec.lifecycle,
                "task": rec.task,
                "duration_seconds": rec.duration_seconds,
                "error": rec.error,
            }
        )

    def _action_inbox(self, **kwargs: Any) -> str:
        return self._bus.read_inbox_formatted(self.my_role)
