"""Current-draft attachment indicator helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fast_agent.llm.model_info import ModelInfo
from fast_agent.mcp.helpers.content_helpers import resource_link
from fast_agent.mcp.mime_utils import guess_mime_type
from fast_agent.ui.binary_indicator import render_glyph_indicator

if TYPE_CHECKING:
    from fast_agent.llm.provider_types import Provider

ATTACHMENT_GLYPH = "▲"
ATTACHMENT_SUPPORTED_COLOR = "ansigreen"
ATTACHMENT_QUESTIONABLE_COLOR = "ansired"
ATTACHMENT_IDLE_COLOR = "ansibrightblack"
UNKNOWN_ATTACHMENT_MIME = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class DraftAttachmentSummary:
    count: int
    mime_types: tuple[str, ...]
    any_questionable: bool


@dataclass(frozen=True, slots=True)
class _AttachmentResource:
    mime_type: str
    source: Literal["embedded", "link"]


def _local_attachment_mime_type(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return UNKNOWN_ATTACHMENT_MIME
    return guess_mime_type(str(path))


def _attachment_resource(
    server_name: str,
    resource_uri: str,
    *,
    url_mention_server: str,
) -> _AttachmentResource:
    if server_name == url_mention_server:
        mime_type = resource_link(resource_uri).mimeType or UNKNOWN_ATTACHMENT_MIME
        return _AttachmentResource(mime_type=mime_type, source="link")
    return _AttachmentResource(
        mime_type=_local_attachment_mime_type(Path(resource_uri)),
        source="embedded",
    )


def _mime_type_is_questionable(
    mime_type: str,
    model_info: ModelInfo | None,
    *,
    resource_source: Literal["embedded", "link"],
) -> bool:
    if mime_type == UNKNOWN_ATTACHMENT_MIME:
        return True
    return model_info is not None and not model_info.supports_mime(
        mime_type,
        resource_source=resource_source,
    )


def summarize_draft_attachments(
    text: str,
    *,
    model_name: str | None,
    provider: Provider | None = None,
    cwd: Path | None = None,
) -> DraftAttachmentSummary | None:
    from fast_agent.ui.prompt.attachment_tokens import FILE_MENTION_SERVER, URL_MENTION_SERVER
    from fast_agent.ui.prompt.resource_mentions import parse_mentions

    parsed = parse_mentions(text, cwd=cwd)
    attachment_mentions = [
        mention
        for mention in parsed.mentions
        if mention.server_name in {FILE_MENTION_SERVER, URL_MENTION_SERVER}
    ]
    if not attachment_mentions:
        return None

    model_info = ModelInfo.from_name(model_name, provider=provider) if model_name else None
    mime_types: list[str] = []
    any_questionable = False
    for mention in attachment_mentions:
        resource = _attachment_resource(
            mention.server_name,
            mention.resource_uri,
            url_mention_server=URL_MENTION_SERVER,
        )

        mime_types.append(resource.mime_type)
        if _mime_type_is_questionable(
            resource.mime_type,
            model_info,
            resource_source=resource.source,
        ):
            any_questionable = True

    return DraftAttachmentSummary(
        count=len(attachment_mentions),
        mime_types=tuple(mime_types),
        any_questionable=any_questionable,
    )


def render_attachment_indicator(summary: DraftAttachmentSummary | None) -> str | None:
    if summary is None or summary.count <= 0:
        return render_glyph_indicator(
            glyph=f" {ATTACHMENT_GLYPH} ",
            color=ATTACHMENT_IDLE_COLOR,
        )

    color = (
        ATTACHMENT_QUESTIONABLE_COLOR if summary.any_questionable else ATTACHMENT_SUPPORTED_COLOR
    )
    return render_glyph_indicator(glyph=_attachment_count_label(summary.count), color=color)


def _attachment_count_label(count: int) -> str:
    suffix = "+" if count >= 10 else str(count)
    return f" {ATTACHMENT_GLYPH}{suffix}"
