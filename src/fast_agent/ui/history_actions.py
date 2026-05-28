"""History display helpers shared by UI command handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from rich.text import Text

from fast_agent.constants import FAST_AGENT_TOOL_METADATA
from fast_agent.history.tool_activities import display_remote_tool_activities
from fast_agent.ui.citation_display import (
    render_sources_pre_content,
    web_tool_badges,
)

if TYPE_CHECKING:
    from fast_agent.config import Settings
    from fast_agent.types import PromptMessageExtended


async def display_history_turn(
    agent_name: str,
    turn: list[PromptMessageExtended],
    *,
    config: Settings | None,
    turn_index: int | None = None,
    total_turns: int | None = None,
) -> None:
    from fast_agent.ui.console_display import ConsoleDisplay
    from fast_agent.ui.message_display_helpers import (
        build_tool_use_additional_message,
        build_user_message_display,
        build_user_message_image_previews,
        tool_use_requests_file_read_access,
        tool_use_requests_shell_access,
    )

    display = ConsoleDisplay(config=config)
    user_group: list[PromptMessageExtended] = []
    tool_name_lookup: dict[str, str] = {}
    tool_metadata_lookup: dict[str, dict[str, Any]] = {}

    def _tool_name_from_call(call_id: str, call: object) -> str:
        params = getattr(call, "params", None)
        tool_name = getattr(params, "name", None) if params is not None else None
        tool_name = tool_name or getattr(call, "name", None)
        return tool_name or call_id

    def _tool_args_from_call(call: object) -> dict[str, Any] | None:
        params = getattr(call, "params", None)
        raw_args = getattr(params, "arguments", None) if params is not None else None
        if raw_args is None:
            raw_args = getattr(call, "arguments", None)
        if isinstance(raw_args, Mapping):
            return dict(raw_args)
        return None

    def _is_read_text_file_tool_name(tool_name: str) -> bool:
        normalized = tool_name.lower()
        for sep in ("/", ".", ":"):
            if sep in normalized:
                normalized = normalized.rsplit(sep, 1)[-1]
        return normalized == "read_text_file" or normalized.endswith("__read_text_file")

    for message in turn:
        channels = getattr(message, "channels", None)
        if not isinstance(channels, Mapping):
            continue
        payloads = channels.get(FAST_AGENT_TOOL_METADATA)
        if not isinstance(payloads, list):
            continue
        for payload in payloads:
            text = getattr(payload, "text", None)
            if not isinstance(text, str):
                continue
            try:
                data = json.loads(text)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            for call_id, metadata in data.items():
                if isinstance(call_id, str) and isinstance(metadata, dict):
                    tool_metadata_lookup[call_id] = dict(metadata)

    def flush_user_group() -> None:
        if not user_group:
            return
        message_text, attachments = build_user_message_display(user_group)
        image_previews = build_user_message_image_previews(user_group)
        part_count = len(user_group)
        turn_range = (turn_index, turn_index) if turn_index else None
        display.show_user_message(
            message=message_text,
            attachments=attachments,
            image_previews=image_previews or None,
            name=agent_name,
            part_count=part_count,
            turn_range=turn_range,
            total_turns=total_turns,
        )
        user_group.clear()

    for message in turn:
        tool_calls = message.tool_calls
        if tool_calls:
            for call_id, call in tool_calls.items():
                tool_name_lookup[call_id] = _tool_name_from_call(call_id, call)

        if message.role == "user" and not message.tool_results:
            user_group.append(message)
            continue

        flush_user_group()

        if message.role == "assistant":
            rendered_remote_activities = display_remote_tool_activities(
                display,
                message,
                name=agent_name,
                truncate_content=False,
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
            pre_content = render_sources_pre_content(message)
            display_message = message

            badges = web_tool_badges(message)
            if badges:
                badge_text = Text(f"\n\nWeb activity: {', '.join(badges)}", style="bright_cyan")
                additional_message = (
                    badge_text
                    if additional_message is None
                    else Text.assemble(additional_message, badge_text)
                )

            bottom_items = badges or None
            highlight_index = 0 if badges else None
            if shell_access or read_file_access:
                bottom_items = None
                highlight_index = None

            should_render_assistant_message = not (
                rendered_remote_activities
                and last_text is None
                and additional_message is None
                and pre_content is None
                and not badges
            )
            if should_render_assistant_message:
                message_payload: str | PromptMessageExtended = display_message
                if last_text is None and additional_message is None and not badges:
                    message_payload = "<no text>"
                await display.show_assistant_message(
                    message_text=message_payload,
                    name=agent_name,
                    bottom_items=bottom_items,
                    highlight_index=highlight_index,
                    additional_message=additional_message,
                    pre_content=pre_content,
                )

            if tool_calls:
                for call_id, call in tool_calls.items():
                    tool_name = tool_name_lookup.get(call_id, call_id)
                    if _is_read_text_file_tool_name(tool_name):
                        continue
                    tool_args = _tool_args_from_call(call)
                    display.show_tool_call(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        name=agent_name,
                        metadata=tool_metadata_lookup.get(call_id),
                        tool_call_id=call_id,
                    )

        tool_results = message.tool_results
        if tool_results:
            for call_id, result in tool_results.items():
                tool_name = tool_name_lookup.get(call_id)
                display.show_tool_result(
                    result=result,
                    name=agent_name,
                    tool_name=tool_name,
                    tool_call_id=call_id,
                    truncate_content=False,
                )

    flush_user_group()
    display.show_pending_tool_result_images()
