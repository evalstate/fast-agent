"""Tool-result message and channel assembly helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from mcp.types import CallToolResult, ContentBlock, TextContent

from fast_agent.constants import (
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_PENDING_MEDIA_ATTACHMENTS,
    FAST_AGENT_TOOL_METADATA,
    FAST_AGENT_TOOL_TIMING,
    FAST_AGENT_URL_ELICITATION_CHANNEL,
)
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.tool_result_metadata import (
    fatal_tool_error,
    url_elicitation_required_payload,
)
from fast_agent.mcp.url_elicitation_required import URLElicitationRequiredDisplayPayload
from fast_agent.types import PromptMessageExtended, ToolTimingInfo

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def tool_result_channels(
    *,
    tool_timings: dict[str, ToolTimingInfo] | None,
    tool_metadata: dict[str, dict[str, object]] | None,
    tool_loop_error: str | None,
    tool_results: Mapping[str, CallToolResult] | None = None,
) -> tuple[dict[str, Sequence[ContentBlock]] | None, list[ContentBlock]]:
    channels: dict[str, Sequence[ContentBlock]] = {}
    content: list[ContentBlock] = []
    if tool_loop_error:
        content.append(text_content(tool_loop_error))
        channels[FAST_AGENT_ERROR_CHANNEL] = [text_content(tool_loop_error)]
    if tool_results:
        fatal_errors = [
            str(error)
            for result in tool_results.values()
            if (error := fatal_tool_error(result))
        ]
        if fatal_errors:
            content.extend(text_content(error) for error in fatal_errors)
            channels[FAST_AGENT_ERROR_CHANNEL] = [
                text_content("\n".join(fatal_errors))
            ]
    if tool_timings:
        channels[FAST_AGENT_TOOL_TIMING] = [
            TextContent(type="text", text=json.dumps(tool_timings))
        ]
    if tool_metadata:
        channels[FAST_AGENT_TOOL_METADATA] = [
            TextContent(type="text", text=json.dumps(tool_metadata))
        ]
    return channels or None, content


def build_tool_result_message(
    tool_results: dict[str, CallToolResult],
    *,
    tool_timings: dict[str, ToolTimingInfo] | None = None,
    tool_metadata: dict[str, dict[str, object]] | None = None,
    tool_loop_error: str | None = None,
    pending_media: Sequence[ContentBlock] = (),
) -> PromptMessageExtended:
    channels, content = tool_result_channels(
        tool_timings=tool_timings,
        tool_metadata=tool_metadata,
        tool_loop_error=tool_loop_error,
        tool_results=tool_results,
    )

    deferred_url_elicitations = _deferred_url_elicitation_payloads(tool_results)
    if deferred_url_elicitations:
        channels = _add_channel(
            channels,
            FAST_AGENT_URL_ELICITATION_CHANNEL,
            [TextContent(type="text", text=json.dumps(deferred_url_elicitations))],
        )

    if pending_media:
        channels = _add_channel(
            channels,
            FAST_AGENT_PENDING_MEDIA_ATTACHMENTS,
            pending_media,
        )

    return PromptMessageExtended(
        role="user",
        content=content,
        tool_results=tool_results,
        channels=channels,
    )


def _deferred_url_elicitation_payloads(
    tool_results: Mapping[str, CallToolResult],
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for result in tool_results.values():
        payload = url_elicitation_required_payload(result)
        if isinstance(payload, URLElicitationRequiredDisplayPayload):
            payloads.append(asdict(payload))
    return payloads


def _add_channel(
    channels: dict[str, Sequence[ContentBlock]] | None,
    name: str,
    content: Sequence[ContentBlock],
) -> dict[str, Sequence[ContentBlock]]:
    if channels is None:
        channels = {}
    channels[name] = content
    return channels
