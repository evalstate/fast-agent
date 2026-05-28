"""Current-draft attachment indicator helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.llm.model_info import ModelInfo
from fast_agent.mcp.helpers.content_helpers import resource_link
from fast_agent.mcp.mime_utils import guess_mime_type

if TYPE_CHECKING:
    from fast_agent.llm.provider_types import Provider

ATTACHMENT_GLYPH = "▲"
ATTACHMENT_SUPPORTED_COLOR = "ansigreen"
ATTACHMENT_QUESTIONABLE_COLOR = "ansired"
ATTACHMENT_IDLE_COLOR = "ansibrightblack"


@dataclass(frozen=True, slots=True)
class DraftAttachmentSummary:
    count: int
    mime_types: tuple[str, ...]
    any_questionable: bool


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
        if mention.server_name == URL_MENTION_SERVER:
            mime_type = resource_link(mention.resource_uri).mimeType or "application/octet-stream"
            mime_types.append(mime_type)
            if mime_type == "application/octet-stream":
                any_questionable = True
                continue
            if model_info and not model_info.supports_mime(
                mime_type,
                resource_source="link",
            ):
                any_questionable = True
            continue

        path = Path(mention.resource_uri)
        if not path.exists():
            any_questionable = True
            mime_types.append("application/octet-stream")
            continue
        if not path.is_file():
            any_questionable = True
            mime_types.append("application/octet-stream")
            continue

        mime_type = guess_mime_type(str(path))
        mime_types.append(mime_type)
        if mime_type == "application/octet-stream":
            any_questionable = True
            continue
        if model_info and not model_info.supports_mime(
            mime_type,
            resource_source="embedded",
        ):
            any_questionable = True

    return DraftAttachmentSummary(
        count=len(attachment_mentions),
        mime_types=tuple(mime_types),
        any_questionable=any_questionable,
    )


def render_attachment_indicator(summary: DraftAttachmentSummary | None) -> str | None:
    if summary is None or summary.count <= 0:
        return f"<style bg='{ATTACHMENT_IDLE_COLOR}'> {ATTACHMENT_GLYPH} </style>"

    if summary.count >= 10:
        label = f" {ATTACHMENT_GLYPH}+"
    else:
        label = f" {ATTACHMENT_GLYPH}{summary.count}"

    color = (
        ATTACHMENT_QUESTIONABLE_COLOR if summary.any_questionable else ATTACHMENT_SUPPORTED_COLOR
    )
    return f"<style bg='{color}'>{label}</style>"
