"""Shared ACP slash command tool-call update helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from acp.helpers import text_block, tool_content
from acp.schema import ToolCallProgress, ToolCallStart

if TYPE_CHECKING:
    from acp.schema import ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent

    type ToolCallContent = (
        ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent
    )

ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]


class SessionUpdateSender(Protocol):
    async def send_session_update(self, update: object) -> None: ...


def _message_content(message: str | None) -> "list[ToolCallContent] | None":
    return [tool_content(text_block(message))] if message else None


async def send_fetch_tool_call_start(
    acp: SessionUpdateSender,
    *,
    tool_call_id: str,
    title: str,
    status: ToolCallStatus = "in_progress",
) -> bool:
    try:
        await acp.send_session_update(
            ToolCallStart(
                tool_call_id=tool_call_id,
                title=title,
                kind="fetch",
                status=status,
                session_update="tool_call",
            )
        )
    except Exception:
        return False
    return True


async def send_tool_call_progress(
    acp: SessionUpdateSender,
    *,
    tool_call_id: str,
    title: str,
    status: ToolCallStatus,
    message: str | None = None,
) -> bool:
    try:
        await acp.send_session_update(
            ToolCallProgress(
                tool_call_id=tool_call_id,
                title=title,
                status=status,
                content=_message_content(message),
                session_update="tool_call_update",
            )
        )
    except Exception:
        return False
    return True
