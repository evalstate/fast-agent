from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fast_agent.event_progress import ProgressAction

if TYPE_CHECKING:
    from fast_agent.core.logging.logger import Logger


class OpenAIToolNotificationMixin:
    if TYPE_CHECKING:
        logger: Logger
        name: str | None

        def _notify_tool_stream_listeners(
            self, event_type: str, payload: dict[str, Any] | None = None
        ) -> None: ...

    def _emit_fallback_tool_notification_event(
        self,
        *,
        tool_name: str,
        tool_use_id: str,
        index: int,
        model: str,
    ) -> None:
        payload = {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "index": index,
        }

        self._notify_tool_stream_listeners("start", payload)
        self.logger.info(
            "Model emitted fallback tool notification",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_event": "start",
                "fallback": True,
            },
        )
        self._notify_tool_stream_listeners("stop", payload)
        self.logger.info(
            "Model emitted fallback tool notification",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_event": "stop",
                "fallback": True,
            },
        )
