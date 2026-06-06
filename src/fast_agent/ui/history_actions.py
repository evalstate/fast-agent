"""History display helpers shared by UI command handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from mcp.types import TextContent
from rich.text import Text

from fast_agent.constants import FAST_AGENT_TOOL_METADATA
from fast_agent.history.tool_activities import display_remote_tool_activities
from fast_agent.ui.citation_display import (
    render_sources_pre_content,
    web_tool_badges,
)
from fast_agent.utils.tool_names import (
    is_read_text_file_tool_name as is_read_text_file_tool_name_shared,
)

if TYPE_CHECKING:
    from mcp.types import CallToolRequest

    from fast_agent.config import Settings
    from fast_agent.types import PromptMessageExtended

type JsonObject = dict[str, object]


class _HistoryDisplay(Protocol):
    def show_user_message(self, **kwargs: object) -> None: ...

    async def show_assistant_message(self, **kwargs: object) -> None: ...

    def show_tool_call(self, **kwargs: object) -> None: ...

    def show_tool_result(self, **kwargs: object) -> None: ...


def _json_object(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _stored_tool_metadata_from_turn(
    turn: list["PromptMessageExtended"],
) -> dict[str, JsonObject]:
    tool_metadata_lookup: dict[str, JsonObject] = {}
    for message in turn:
        channels = message.channels or {}
        payloads = channels.get(FAST_AGENT_TOOL_METADATA)
        if not isinstance(payloads, list):
            continue
        for payload in payloads:
            if not isinstance(payload, TextContent):
                continue
            try:
                data = json.loads(payload.text)
            except Exception:
                continue
            parsed = _json_object(data)
            if parsed is None:
                continue
            for call_id, metadata in parsed.items():
                parsed_metadata = _json_object(metadata)
                if parsed_metadata is not None:
                    tool_metadata_lookup[call_id] = parsed_metadata
    return tool_metadata_lookup


def _tool_name_from_call(call_id: str, call: "CallToolRequest") -> str:
    return call.params.name or call_id


def _tool_args_from_call(call: "CallToolRequest") -> JsonObject | None:
    arguments = call.params.arguments
    return _json_object(arguments) if isinstance(arguments, Mapping) else None


@dataclass(slots=True)
class _HistoryTurnDisplayContext:
    display: _HistoryDisplay
    agent_name: str
    turn_index: int | None
    total_turns: int | None
    tool_metadata_lookup: dict[str, JsonObject]
    user_group: list["PromptMessageExtended"] = field(default_factory=list)
    tool_name_lookup: dict[str, str] = field(default_factory=dict)

    def record_tool_calls(self, message: "PromptMessageExtended") -> None:
        tool_calls = message.tool_calls
        if not tool_calls:
            return
        for call_id, call in tool_calls.items():
            self.tool_name_lookup[call_id] = _tool_name_from_call(call_id, call)

    def queue_user_message(self, message: "PromptMessageExtended") -> bool:
        if message.role != "user" or message.tool_results:
            return False
        self.user_group.append(message)
        return True

    def flush_user_group(self) -> None:
        if not self.user_group:
            return

        from fast_agent.ui.message_display_helpers import (
            build_user_message_display,
            build_user_message_image_previews,
        )

        message_text, attachments = build_user_message_display(self.user_group)
        image_previews = build_user_message_image_previews(self.user_group)
        turn_range = (self.turn_index, self.turn_index) if self.turn_index else None
        self.display.show_user_message(
            message=message_text,
            attachments=attachments,
            image_previews=image_previews or None,
            name=self.agent_name,
            part_count=len(self.user_group),
            turn_range=turn_range,
            total_turns=self.total_turns,
        )
        self.user_group.clear()

    async def display_message(self, message: "PromptMessageExtended") -> None:
        self.flush_user_group()
        if message.role == "assistant":
            await self._display_assistant_message(message)
        self._display_tool_results(message)

    async def _display_assistant_message(self, message: "PromptMessageExtended") -> None:
        rendered_remote_activities = display_remote_tool_activities(
            self.display,
            message,
            name=self.agent_name,
            truncate_content=False,
        )
        assistant_payload = self._assistant_payload(message, rendered_remote_activities)
        if assistant_payload is not None:
            await self.display.show_assistant_message(
                message_text=assistant_payload["message_text"],
                name=self.agent_name,
                bottom_items=assistant_payload["bottom_items"],
                highlight_index=assistant_payload["highlight_index"],
                additional_message=assistant_payload["additional_message"],
                pre_content=assistant_payload["pre_content"],
            )

        self._display_tool_calls(message)

    def _assistant_payload(
        self,
        message: "PromptMessageExtended",
        rendered_remote_activities: bool,
    ) -> dict[str, object] | None:
        from fast_agent.ui.message_display_helpers import (
            build_tool_use_additional_message,
            tool_use_requests_file_read_access,
            tool_use_requests_shell_access,
        )

        last_text = message.last_text()
        shell_access = tool_use_requests_shell_access(
            message,
            # History replay has no runtime tool registry. Treating
            # "execute" as shell here matches the live local-shell UX.
            assume_execute_is_shell=True,
        )
        read_file_access = tool_use_requests_file_read_access(message)
        additional_message = build_tool_use_additional_message(
            message,
            last_text,
            shell_access=shell_access,
            file_read=read_file_access,
        )
        badges = web_tool_badges(message)
        additional_message = _append_web_activity_badges(additional_message, badges)
        pre_content = render_sources_pre_content(message)

        if not _should_render_assistant_message(
            rendered_remote_activities=rendered_remote_activities,
            last_text=last_text,
            additional_message=additional_message,
            pre_content=pre_content,
            badges=badges,
        ):
            return None

        bottom_items: list[str] | None = badges or None
        highlight_index = 0 if badges else None
        if shell_access or read_file_access:
            bottom_items = None
            highlight_index = None

        message_text: str | PromptMessageExtended = message
        if last_text is None and additional_message is None and not badges:
            message_text = "<no text>"

        return {
            "message_text": message_text,
            "bottom_items": bottom_items,
            "highlight_index": highlight_index,
            "additional_message": additional_message,
            "pre_content": pre_content,
        }

    def _display_tool_calls(self, message: "PromptMessageExtended") -> None:
        tool_calls = message.tool_calls
        if not tool_calls:
            return

        for call_id, call in tool_calls.items():
            tool_name = self.tool_name_lookup.get(call_id, call_id)
            if is_read_text_file_tool_name_shared(tool_name):
                continue
            self.display.show_tool_call(
                tool_name=tool_name,
                tool_args=_tool_args_from_call(call),
                name=self.agent_name,
                metadata=self.tool_metadata_lookup.get(call_id),
                tool_call_id=call_id,
            )

    def _display_tool_results(self, message: "PromptMessageExtended") -> None:
        tool_results = message.tool_results
        if not tool_results:
            return

        for call_id, result in tool_results.items():
            self.display.show_tool_result(
                result=result,
                name=self.agent_name,
                tool_name=self.tool_name_lookup.get(call_id),
                tool_call_id=call_id,
                truncate_content=False,
            )


def _append_web_activity_badges(additional_message: Text | None, badges: list[str]) -> Text | None:
    if not badges:
        return additional_message

    badge_text = Text(f"\n\nWeb activity: {', '.join(badges)}", style="bright_cyan")
    if additional_message is None:
        return badge_text
    return Text.assemble(additional_message, badge_text)


def _should_render_assistant_message(
    *,
    rendered_remote_activities: bool,
    last_text: str | None,
    additional_message: Text | None,
    pre_content: Any,
    badges: list[str],
) -> bool:
    return not (
        rendered_remote_activities
        and last_text is None
        and additional_message is None
        and pre_content is None
        and not badges
    )


async def display_history_turn(
    agent_name: str,
    turn: list[PromptMessageExtended],
    *,
    config: Settings | None,
    turn_index: int | None = None,
    total_turns: int | None = None,
) -> None:
    from fast_agent.ui.console_display import ConsoleDisplay

    context = _HistoryTurnDisplayContext(
        display=cast("_HistoryDisplay", ConsoleDisplay(config=config)),
        agent_name=agent_name,
        turn_index=turn_index,
        total_turns=total_turns,
        tool_metadata_lookup=_stored_tool_metadata_from_turn(turn),
    )
    for message in turn:
        context.record_tool_calls(message)
        if context.queue_user_message(message):
            continue
        await context.display_message(message)

    context.flush_user_group()
